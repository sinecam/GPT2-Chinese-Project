from pathlib import Path
import argparse
import json
import numpy as np
import tiktoken


def format_sample(instruction, output):
    return f"User: {instruction}\nAssistant: {output}\n<|endoftext|>\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sft/sft.jsonl")
    parser.add_argument("--out_dir", type=str, default="data/processed/sft")
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    all_texts = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            all_texts.append(format_sample(item["instruction"], item["output"]))

    text = "".join(all_texts * args.repeat)
    ids = enc.encode_ordinary(text)

    split_idx = int(len(ids) * (1 - args.val_ratio))
    train_ids = np.array(ids[:split_idx], dtype=np.uint16)
    val_ids = np.array(ids[split_idx:], dtype=np.uint16)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    print(f"samples: {len(all_texts)}")
    print(f"repeat: {args.repeat}")
    print(f"total tokens: {len(ids)}")
    print(f"train tokens: {len(train_ids)}")
    print(f"val tokens: {len(val_ids)}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
