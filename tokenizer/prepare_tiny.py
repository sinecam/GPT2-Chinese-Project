from pathlib import Path
import argparse
import numpy as np
import tiktoken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/tiny.txt")
    parser.add_argument("--out_dir", type=str, default="data/processed/tiny")
    parser.add_argument("--repeat", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    text = (text + "\n") * args.repeat

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)

    n = int(len(ids) * 0.9)
    train_ids = np.array(ids[:n], dtype=np.uint16)
    val_ids = np.array(ids[n:], dtype=np.uint16)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    print(f"input text chars: {len(text)}")
    print(f"total tokens: {len(ids)}")
    print(f"train tokens: {len(train_ids)}")
    print(f"val tokens: {len(val_ids)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
