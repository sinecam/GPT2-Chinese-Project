import json
import random
import argparse
from pathlib import Path


def clean_text(s):
    s = str(s).replace("\r", "\n").strip()
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s


def load_and_filter(path, max_samples=None, seed=1337):
    random.seed(seed)
    items = []
    total = 0
    skipped = 0

    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue

            title = clean_text(obj.get("title", ""))
            desc = clean_text(obj.get("desc", ""))
            answer = clean_text(obj.get("answer", ""))
            category = clean_text(obj.get("category", ""))

            if not title or not answer:
                skipped += 1
                continue

            if len(title) < 3 or len(title) > 120:
                skipped += 1
                continue

            if len(answer) < 5 or len(answer) > 500:
                skipped += 1
                continue

            bad_words = ["http://", "https://", "www.", "<br", "&nbsp"]
            text_all = title + desc + answer
            if any(w in text_all for w in bad_words):
                skipped += 1
                continue

            if desc and desc != title:
                instruction = f"{title}\n补充信息：{desc}"
            else:
                instruction = title

            items.append({
                "instruction": instruction,
                "output": answer,
                "category": category,
            })

    random.shuffle(items)

    if max_samples is not None:
        items = items[:max_samples]

    return items, total, skipped


def save_jsonl(items, path):
    with Path(path).open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", required=True)
    parser.add_argument("--valid_input", required=True)
    parser.add_argument("--out_dir", default="data/sft/baike_qa2019")
    parser.add_argument("--max_train", type=int, default=50000)
    parser.add_argument("--max_valid", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items, train_total, train_skipped = load_and_filter(
        args.train_input,
        args.max_train,
        args.seed,
    )

    valid_items, valid_total, valid_skipped = load_and_filter(
        args.valid_input,
        args.max_valid,
        args.seed + 1,
    )

    train_out = out_dir / f"baike_qa_train_{len(train_items)}.jsonl"
    valid_out = out_dir / f"baike_qa_valid_{len(valid_items)}.jsonl"

    save_jsonl(train_items, train_out)
    save_jsonl(valid_items, valid_out)

    print("train total:", train_total)
    print("train skipped:", train_skipped)
    print("train saved:", len(train_items))
    print("valid total:", valid_total)
    print("valid skipped:", valid_skipped)
    print("valid saved:", len(valid_items))
    print("train output:", train_out)
    print("valid output:", valid_out)


if __name__ == "__main__":
    main()
