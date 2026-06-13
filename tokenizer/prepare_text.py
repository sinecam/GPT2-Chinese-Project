from pathlib import Path
import argparse
import numpy as np
import tiktoken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/pretrain.txt")
    parser.add_argument("--out_dir", type=str, default="data/processed/pretrain")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    text = (text + "\n") * args.repeat

    print(f"Input file: {input_path}")
    print(f"Characters: {len(text)}")

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)

    print(f"Total tokens: {len(ids)}")
    print(f"Max token id: {max(ids)}")

    split_idx = int(len(ids) * (1 - args.val_ratio))

    train_ids = np.array(ids[:split_idx], dtype=np.uint16)
    val_ids = np.array(ids[split_idx:], dtype=np.uint16)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    print(f"Train tokens: {len(train_ids)}")
    print(f"Val tokens: {len(val_ids)}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
