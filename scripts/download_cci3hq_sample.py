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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BAAI/CCI3-HQ")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="data/raw/chinese_pretrain.txt")
    parser.add_argument("--max_chars", type=int, default=100_000_000)
    parser.add_argument("--min_chars", type=int, default=100)
    parser.add_argument("--min_zh_ratio", type=float, default=0.25)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    print(f"Streaming split: {args.split}")
    print(f"Target chars: {args.max_chars}")
    print(f"Output: {out_path}")

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    total_chars = 0
    kept_docs = 0
    seen_docs = 0

    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            seen_docs += 1
            text = pick_text_field(item)
            text = clean_text(text)

            if len(text) < args.min_chars:
                continue

            if chinese_ratio(text) < args.min_zh_ratio:
                continue

            f.write(text + "\n")
            total_chars += len(text)
            kept_docs += 1

            if kept_docs % 1000 == 0:
                print(
                    f"seen={seen_docs}, kept={kept_docs}, "
                    f"chars={total_chars:,}, size≈{out_path.stat().st_size / 1024 / 1024:.1f}MB"
                )

            if total_chars >= args.max_chars:
                break

    print("Done.")
    print(f"seen docs: {seen_docs}")
    print(f"kept docs: {kept_docs}")
    print(f"chars: {total_chars:,}")
    print(f"file size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
