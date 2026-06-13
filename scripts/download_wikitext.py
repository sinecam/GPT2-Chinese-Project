from pathlib import Path
from datasets import load_dataset


def main():
    out_path = Path("data/raw/pretrain.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading WikiText-2...")

    # 新版 Hugging Face 需要使用带 namespace 的数据集名
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    texts = []
    for split in ["train", "validation", "test"]:
        for item in ds[split]:
            text = item["text"].strip()
            if text:
                texts.append(text)

    final_text = "\n".join(texts)

    out_path.write_text(final_text, encoding="utf-8")

    print(f"Saved to {out_path}")
    print(f"Documents/lines: {len(texts)}")
    print(f"Characters: {len(final_text)}")


if __name__ == "__main__":
    main()
