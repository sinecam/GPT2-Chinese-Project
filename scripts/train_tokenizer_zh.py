import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Iterator

import sentencepiece as spm
from tqdm import tqdm


TEXT_FIELDS = (
    "text",
    "content",
    "article",
    "paragraph",
    "prompt",
    "response",
    "instruction",
    "input",
    "output",
)

ROLE_SYMBOLS = ["<user>", "<assistant>", "<system>", "<sep>"]


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def normalize_text(text: str) -> str:
    text = text.replace("\u0000", " ").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def stringify_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or message.get("from") or "").strip().lower()
    content = message.get("content") or message.get("value") or message.get("text") or ""
    content = normalize_text(str(content))
    if not content:
        return ""
    if role in {"human", "user", "问", "question"}:
        return f"<user>\n{content}"
    if role in {"assistant", "gpt", "bot", "答", "answer"}:
        return f"<assistant>\n{content}"
    if role in {"system"}:
        return f"<system>\n{content}"
    return content


def record_to_text(record: Any) -> str:
    if isinstance(record, str):
        return normalize_text(record)

    if not isinstance(record, dict):
        return ""

    for key in ("messages", "conversations"):
        value = record.get(key)
        if isinstance(value, list):
            parts = [stringify_message(item) for item in value if isinstance(item, dict)]
            return "\n<sep>\n".join(part for part in parts if part)

    instruction = normalize_text(str(record.get("instruction") or record.get("prompt") or ""))
    inp = normalize_text(str(record.get("input") or ""))
    output = normalize_text(str(record.get("output") or record.get("response") or record.get("answer") or ""))
    if instruction and output:
        question = instruction if not inp else f"{instruction}\n{inp}"
        return f"<user>\n{question}\n<sep>\n<assistant>\n{output}"

    values = []
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str):
            value = normalize_text(value)
            if value:
                values.append(value)
    return "\n".join(values)


def record_to_corpus_line(record: Any, min_chars: int, max_chars_per_line: int) -> str | None:
    text = normalize_text(record_to_text(record))
    if len(text) < min_chars:
        return None
    if max_chars_per_line > 0:
        text = text[:max_chars_per_line]
    return text.replace("\n", " ")


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


def iter_hf_corpus_lines(args: argparse.Namespace) -> Iterator[str]:
    if not args.hf_dataset:
        return
    for record in iter_hf_records(args):
        line = record_to_corpus_line(record, args.min_chars, args.max_chars_per_line)
        if line:
            yield line


def iter_local_corpus_lines(args: argparse.Namespace) -> Iterator[str]:
    if not args.input:
        return

    if args.num_workers <= 1:
        for raw_path in args.input:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Input not found: {path}")
            for record in iter_local_records(path):
                line = record_to_corpus_line(record, args.min_chars, args.max_chars_per_line)
                if line:
                    yield line
        return

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for raw_path in args.input:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Input not found: {path}")
            mapped = executor.map(
                record_to_corpus_line,
                iter_local_records(path),
                repeat(args.min_chars),
                repeat(args.max_chars_per_line),
                chunksize=args.worker_chunksize,
            )
            for line in mapped:
                if line:
                    yield line


def iter_corpus_lines(args: argparse.Namespace) -> Iterator[str]:
    yield from iter_hf_corpus_lines(args)
    yield from iter_local_corpus_lines(args)


def write_sentencepiece_corpus(args: argparse.Namespace, corpus_path: Path) -> int:
    written = 0
    with corpus_path.open("w", encoding="utf-8") as out:
        for line in tqdm(iter_corpus_lines(args), desc="collect tokenizer corpus"):
            out.write(line + "\n")
            written += 1
            if args.max_lines and written >= args.max_lines:
                break
    return written


def train_sentencepiece(args: argparse.Namespace, corpus_path: Path) -> None:
    model_prefix = str(Path(args.out_dir) / args.model_prefix)
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        byte_fallback=args.byte_fallback,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name=args.normalization_rule_name,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        num_threads=args.num_threads,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=ROLE_SYMBOLS,
    )


def write_config(args: argparse.Namespace, model_path: Path, corpus_lines: int) -> None:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    config = {
        "tokenizer": "sentencepiece",
        "model_file": model_path.name,
        "vocab_size": int(sp.vocab_size()),
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "byte_fallback": args.byte_fallback,
        "corpus_lines": corpus_lines,
        "num_threads": args.num_threads,
        "num_workers": args.num_workers,
        "special_tokens": {
            "pad_id": int(sp.pad_id()),
            "unk_id": int(sp.unk_id()),
            "bos_id": int(sp.bos_id()),
            "eos_id": int(sp.eos_id()),
            "user_id": int(sp.piece_to_id("<user>")),
            "assistant_id": int(sp.piece_to_id("<assistant>")),
            "system_id": int(sp.piece_to_id("<system>")),
            "sep_id": int(sp.piece_to_id("<sep>")),
        },
    }
    with (Path(args.out_dir) / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese SentencePiece tokenizer for GPT-style dialogue models.")
    parser.add_argument("--input", nargs="*", default=[], help="Local txt/json/jsonl files.")
    parser.add_argument("--hf_dataset", type=str, default=None, help="Optional Hugging Face dataset name.")
    parser.add_argument("--hf_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--out_dir", type=str, default="artifacts/tokenizer_zh")
    parser.add_argument("--model_prefix", type=str, default="spm_zh")
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--byte_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalization_rule_name", type=str, default="nmt_nfkc")
    parser.add_argument("--input_sentence_size", type=int, default=20000000)
    parser.add_argument("--max_lines", type=int, default=0)
    parser.add_argument("--min_chars", type=int, default=8)
    parser.add_argument("--max_chars_per_line", type=int, default=4096)
    parser.add_argument("--num_threads", type=int, default=cpu_count(), help="SentencePiece trainer threads.")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel local corpus formatting workers. HF streaming stays sequential.")
    parser.add_argument("--worker_chunksize", type=int, default=1024)
    parser.add_argument("--keep_corpus", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input and not args.hf_dataset:
        raise ValueError("Provide --input files or --hf_dataset.")
    if args.num_threads < 1:
        raise ValueError("--num_threads must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if args.worker_chunksize < 1:
        raise ValueError("--worker_chunksize must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CPU cores detected: {cpu_count()}")
    print(f"SentencePiece num_threads: {args.num_threads}")
    print(f"local corpus num_workers: {args.num_workers}")

    corpus_path = out_dir / "tokenizer_corpus.txt"
    corpus_lines = write_sentencepiece_corpus(args, corpus_path)
    if corpus_lines == 0:
        raise ValueError("Tokenizer corpus is empty after filtering.")

    train_sentencepiece(args, corpus_path)
    model_path = out_dir / f"{args.model_prefix}.model"
    write_config(args, model_path, corpus_lines)

    if not args.keep_corpus:
        corpus_path.unlink(missing_ok=True)

    print(f"trained tokenizer: {model_path}")
    print(f"corpus lines: {corpus_lines:,}")


if __name__ == "__main__":
    main()
