import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


IGNORE_INDEX = -100


def normalize_text(text: str) -> str:
    text = text.replace("\u0000", " ").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user", "question", "问"}:
        return "user"
    if role in {"assistant", "gpt", "bot", "answer", "答"}:
        return "assistant"
    if role == "system":
        return "system"
    return "user"


def get_message_text(message: dict[str, Any]) -> str:
    value = message.get("content") or message.get("value") or message.get("text") or ""
    return normalize_text(str(value))


def extract_user_assistant(record: Any) -> tuple[str, str] | None:
    if not isinstance(record, dict):
        return None

    for key in ("messages", "conversations"):
        value = record.get(key)
        if isinstance(value, list):
            user = ""
            assistant = ""
            for item in value:
                if not isinstance(item, dict):
                    continue
                role = normalize_role(str(item.get("role") or item.get("from") or "user"))
                content = get_message_text(item)
                if not content:
                    continue
                if role == "user" and not user:
                    user = content
                elif role == "assistant" and not assistant:
                    assistant = content
                if user and assistant:
                    return user, assistant
            return None

    instruction = normalize_text(str(record.get("instruction") or record.get("prompt") or record.get("question") or ""))
    inp = normalize_text(str(record.get("input") or ""))
    output = normalize_text(str(record.get("output") or record.get("response") or record.get("answer") or ""))
    if instruction and output:
        user_text = instruction if not inp else f"{instruction}\n{inp}"
        return user_text, output
    return None


def iter_json_array(value: Any) -> Iterator[Any]:
    if isinstance(value, list):
        yield from value
    elif isinstance(value, dict):
        for key in ("data", "train", "records", "items"):
            if isinstance(value.get(key), list):
                yield from value[key]
                return
        yield value


def iter_local_records(path: Path) -> Iterator[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            yield from iter_json_array(json.load(f))
        return

    raise ValueError(f"Unsupported input file: {path}")


def iter_records(paths: list[str]) -> Iterator[Any]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        yield from iter_local_records(path)


def choose_split(index: int, user_text: str, answer_text: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    key = f"{index}:{user_text[:128]}:{answer_text[:128]}"
    digest = hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def encode_example(
    sp: spm.SentencePieceProcessor,
    user_text: str,
    answer_text: str,
    system_prompt: str | None,
    block_size: int,
    add_bos: bool,
    add_eos: bool,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    parts = []
    if system_prompt:
        parts.append(f"<system>\n{system_prompt}")
    parts.append(f"<user>\n{user_text}")
    parts.append("<assistant>\n")
    prompt_text = "\n<sep>\n".join(parts)

    prefix_ids = sp.encode(prompt_text, out_type=int)
    if add_bos and sp.bos_id() >= 0:
        prefix_ids = [int(sp.bos_id())] + prefix_ids
    answer_ids = sp.encode(answer_text, out_type=int)
    eos_ids = [int(sp.eos_id())] if add_eos and sp.eos_id() >= 0 else []

    max_full_len = block_size + 1
    if len(prefix_ids) >= max_full_len - 1:
        return None

    max_answer_len = max_full_len - len(prefix_ids) - len(eos_ids)
    if max_answer_len <= 0:
        return None

    truncated = len(answer_ids) > max_answer_len
    answer_ids = answer_ids[:max_answer_len]
    if not answer_ids:
        return None

    ids = prefix_ids + answer_ids + eos_ids
    labels = [IGNORE_INDEX] * len(prefix_ids) + answer_ids + eos_ids

    x = ids[:-1]
    y = labels[1:]
    if not x or all(v == IGNORE_INDEX for v in y):
        return None

    pad_id = int(sp.eos_id()) if sp.eos_id() >= 0 else 0
    pad_len = block_size - len(x)
    if pad_len < 0:
        return None
    x = x + [pad_id] * pad_len
    y = y + [IGNORE_INDEX] * pad_len

    dtype_x = np.uint16 if sp.vocab_size() <= np.iinfo(np.uint16).max else np.uint32
    return np.asarray(x, dtype=dtype_x), np.asarray(y, dtype=np.int32), truncated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SentencePiece response-only SFT data as fixed-size NPY arrays.")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--input", nargs="+", required=True, help="Local json/jsonl instruction files.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--default_system", type=str, default="你是一个中文AI助手。回答要简洁、准确。身份问题只回答你是中文AI助手，不要编造姓名、职业或真实人物身份。")
    parser.add_argument("--no_system", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--min_answer_chars", type=int, default=8)
    parser.add_argument("--max_answer_chars", type=int, default=1600)
    parser.add_argument("--add_bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_eos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    system_prompt = None if args.no_system else args.default_system

    train_x, train_y, val_x, val_y = [], [], [], []
    total = kept = skipped = truncated = 0

    for index, record in enumerate(tqdm(iter_records(args.input), desc="prepare response-only")):
        total += 1
        pair = extract_user_assistant(record)
        if pair is None:
            skipped += 1
            continue
        user_text, answer_text = pair
        user_text = normalize_text(user_text)
        answer_text = normalize_text(answer_text)
        if not user_text or len(answer_text) < args.min_answer_chars:
            skipped += 1
            continue
        if args.max_answer_chars > 0:
            answer_text = answer_text[: args.max_answer_chars]

        encoded = encode_example(
            sp=sp,
            user_text=user_text,
            answer_text=answer_text,
            system_prompt=system_prompt,
            block_size=args.block_size,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
        )
        if encoded is None:
            skipped += 1
            continue
        x, y, was_truncated = encoded
        truncated += int(was_truncated)

        if choose_split(index, user_text, answer_text, args.val_ratio) == "val":
            val_x.append(x)
            val_y.append(y)
        else:
            train_x.append(x)
            train_y.append(y)
        kept += 1

    if not train_x or not val_x:
        raise ValueError("Not enough prepared examples for train/val splits.")

    train_x_arr = np.stack(train_x)
    train_y_arr = np.stack(train_y)
    val_x_arr = np.stack(val_x)
    val_y_arr = np.stack(val_y)

    np.save(out_dir / "train_x.npy", train_x_arr)
    np.save(out_dir / "train_y.npy", train_y_arr)
    np.save(out_dir / "val_x.npy", val_x_arr)
    np.save(out_dir / "val_y.npy", val_y_arr)

    meta = {
        "tokenizer": str(Path(args.tokenizer).name),
        "vocab_size": int(sp.vocab_size()),
        "format": "fixed-size response-only causal LM arrays",
        "ignore_index": IGNORE_INDEX,
        "block_size": args.block_size,
        "train_examples": int(train_x_arr.shape[0]),
        "val_examples": int(val_x_arr.shape[0]),
        "total_records": total,
        "kept_records": kept,
        "skipped_records": skipped,
        "truncated_records": truncated,
        "val_ratio": args.val_ratio,
        "system_prompt": system_prompt,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
