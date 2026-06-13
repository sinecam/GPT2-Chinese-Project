import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


TEXT_FIELDS = ("text", "content", "article", "paragraph", "passage", "document")


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


def extract_messages(record: dict[str, Any]) -> list[tuple[str, str]]:
    for key in ("messages", "conversations"):
        value = record.get(key)
        if isinstance(value, list):
            messages = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                role = normalize_role(str(item.get("role") or item.get("from") or "user"))
                content = get_message_text(item)
                if content:
                    messages.append((role, content))
            return messages

    instruction = normalize_text(str(record.get("instruction") or record.get("prompt") or record.get("question") or ""))
    inp = normalize_text(str(record.get("input") or ""))
    output = normalize_text(str(record.get("output") or record.get("response") or record.get("answer") or ""))
    if instruction and output:
        user_text = instruction if not inp else f"{instruction}\n{inp}"
        return [("user", user_text), ("assistant", output)]

    return []


def extract_plain_text(record: Any) -> str:
    if isinstance(record, str):
        return normalize_text(record)
    if not isinstance(record, dict):
        return ""
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str):
            value = normalize_text(value)
            if value:
                return value
    return ""


def format_dialogue(messages: list[tuple[str, str]], default_system: str | None = None) -> str:
    parts = []
    if default_system:
        parts.append(f"<system>\n{default_system}")

    for role, content in messages:
        if role not in {"system", "user", "assistant"}:
            role = "user"
        parts.append(f"<{role}>\n{content}")

    return "\n<sep>\n".join(parts)


def record_to_training_text(record: Any, mode: str, default_system: str | None) -> str:
    if mode in {"auto", "dialogue"} and isinstance(record, dict):
        messages = extract_messages(record)
        if messages:
            return format_dialogue(messages, default_system=default_system)
        if mode == "dialogue":
            return ""

    return extract_plain_text(record)


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
                    yield line
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            yield from iter_json_array(json.load(f))
        return

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = normalize_text(line)
            if line:
                yield line


def iter_hf_records(args: argparse.Namespace) -> Iterator[Any]:
    from datasets import load_dataset

    dataset = load_dataset(
        args.hf_dataset,
        args.hf_config,
        split=args.hf_split,
        streaming=args.streaming,
    )
    yield from dataset


def iter_records(args: argparse.Namespace) -> Iterator[Any]:
    if args.hf_dataset:
        yield from iter_hf_records(args)

    for raw_path in args.input:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        yield from iter_local_records(path)


def choose_split(index: int, text: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    digest = hashlib.md5(f"{index}:{text[:256]}".encode("utf-8", errors="ignore")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


class BinWriter:
    def __init__(self, path: Path, dtype: np.dtype):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.file = path.open("wb")
        self.tokens = 0
        self.records = 0

    def write(self, ids: list[int]) -> None:
        if not ids:
            return
        array = np.asarray(ids, dtype=self.dtype)
        array.tofile(self.file)
        self.tokens += int(array.size)
        self.records += 1

    def close(self) -> None:
        self.file.close()


def encode_text(sp: spm.SentencePieceProcessor, text: str, add_bos: bool, add_eos: bool) -> list[int]:
    ids = sp.encode(text, out_type=int)
    if add_bos and sp.bos_id() >= 0:
        ids = [int(sp.bos_id())] + ids
    if add_eos and sp.eos_id() >= 0:
        ids = ids + [int(sp.eos_id())]
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Chinese pretraining or dialogue SFT data into GPT train.bin/val.bin files.")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--input", nargs="*", default=[], help="Local txt/json/jsonl files.")
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--hf_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "pretrain", "dialogue"])
    parser.add_argument("--default_system", type=str, default="你是一个乐于助人的中文助手。")
    parser.add_argument("--no_system", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--min_chars", type=int, default=8)
    parser.add_argument("--max_chars", type=int, default=4096)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--add_bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_eos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input and not args.hf_dataset:
        raise ValueError("Provide --input files or --hf_dataset.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    dtype = np.uint16 if sp.vocab_size() <= np.iinfo(np.uint16).max else np.uint32

    train_writer = BinWriter(out_dir / "train.bin", dtype)
    val_writer = BinWriter(out_dir / "val.bin", dtype)

    default_system = None if args.no_system else args.default_system
    total_records = 0
    skipped = 0

    try:
        for index, record in enumerate(tqdm(iter_records(args), desc="encode records")):
            text = record_to_training_text(record, args.mode, default_system)
            text = normalize_text(text)
            if len(text) < args.min_chars:
                skipped += 1
                continue
            if args.max_chars > 0:
                text = text[: args.max_chars]

            ids = encode_text(sp, text, add_bos=args.add_bos, add_eos=args.add_eos)
            if len(ids) < 2:
                skipped += 1
                continue

            split = choose_split(index, text, args.val_ratio)
            if split == "val":
                val_writer.write(ids)
            else:
                train_writer.write(ids)

            total_records += 1
            if args.max_records and total_records >= args.max_records:
                break
    finally:
        train_writer.close()
        val_writer.close()

    meta = {
        "tokenizer": str(Path(args.tokenizer).name),
        "vocab_size": int(sp.vocab_size()),
        "dtype": np.dtype(dtype).name,
        "mode": args.mode,
        "train_tokens": train_writer.tokens,
        "val_tokens": val_writer.tokens,
        "train_records": train_writer.records,
        "val_records": val_writer.records,
        "skipped_records": skipped,
        "val_ratio": args.val_ratio,
        "format": "raw contiguous token ids for causal language modeling",
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
