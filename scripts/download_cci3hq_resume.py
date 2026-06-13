from pathlib import Path
import argparse
import re
from datasets import load_dataset

ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = len(ZH_RE.findall(text))
    return zh / max(len(text), 1)


def pick_text_field(item):
    for key in ["text", "content", "raw_content", "document", "article"]:
        if key in item and isinstance(item[key], str):
            return item[key]
    for key, value in item.items():
        if isinstance(value, str) and len(value) > 50:
            return value
    return ""


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines)


def truncate_incomplete_last_line(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("rb+") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return

        pos = size - 1
        while pos > 0:
            pos -= 1
            f.seek(pos)
            if f.read(1) == b"\n":
                f.truncate(pos + 1)
                print(f"truncated incomplete last line at byte {pos + 1}")
                return

        f.truncate(0)
        print("file had no complete newline, truncated to empty")


def count_existing(path: Path):
    if not path.exists():
        return 0, 0

    lines = 0
    chars = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.rstrip("\n")
            if text:
                lines += 1
                chars += len(text)

    return lines, chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BAAI/CCI3-HQ")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="data/raw/chinese_pretrain_5b.txt")
    parser.add_argument("--max_chars", type=int, default=5_000_000_000)
    parser.add_argument("--min_chars", type=int, default=100)
    parser.add_argument("--min_zh_ratio", type=float, default=0.25)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    truncate_incomplete_last_line(out_path)
    existing_lines, existing_chars = count_existing(out_path)

    print(f"Dataset: {args.dataset}")
    print(f"Output: {out_path}")
    print(f"Target chars: {args.max_chars:,}")
    print(f"Existing kept docs(lines): {existing_lines:,}")
    print(f"Existing chars: {existing_chars:,}")
    print(f"Existing file size: {out_path.stat().st_size / 1024 / 1024:.1f} MB" if out_path.exists() else "Existing file size: 0 MB")

    if existing_chars >= args.max_chars:
        print("Already reached target chars. Nothing to do.")
        return

    print("Loading dataset stream...")
    ds = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        token=True,
    )

    total_chars = existing_chars
    accepted_docs = 0
    appended_docs = 0
    seen_docs = 0

    with out_path.open("a", encoding="utf-8") as f:
        for item in ds:
            seen_docs += 1

            text = pick_text_field(item)
            text = clean_text(text)

            if len(text) < args.min_chars:
                continue
            if chinese_ratio(text) < args.min_zh_ratio:
                continue

            if accepted_docs < existing_lines:
                accepted_docs += 1
                if accepted_docs % 50000 == 0:
                    print(f"skipped existing kept docs: {accepted_docs:,}/{existing_lines:,}")
                continue

            f.write(text + "\n")
            total_chars += len(text)
            accepted_docs += 1
            appended_docs += 1

            if appended_docs % 1000 == 0:
                print(
                    f"seen={seen_docs:,}, "
                    f"total_kept={accepted_docs:,}, "
                    f"appended={appended_docs:,}, "
                    f"chars={total_chars:,}, "
                    f"size≈{out_path.stat().st_size / 1024 / 1024:.1f}MB",
                    flush=True,
                )

            if total_chars >= args.max_chars:
                break

    print("Done.")
    print(f"seen docs: {seen_docs:,}")
    print(f"total kept docs: {accepted_docs:,}")
    print(f"appended docs: {appended_docs:,}")
    print(f"chars: {total_chars:,}")
    print(f"file size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
