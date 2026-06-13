from pathlib import Path
import argparse
import random
import numpy as np
import tiktoken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/chinese_pretrain.txt")
    parser.add_argument("--out_dir", type=str, default="data/processed/chinese_pretrain")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    if train_path.exists():
        train_path.unlink()
    if val_path.exists():
        val_path.unlink()

    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(args.seed)

    train_tokens = 0
    val_tokens = 0
    lines = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         train_path.open("ab") as ftrain, \
         val_path.open("ab") as fval:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            ids = enc.encode_ordinary(line + "\n")
            arr = np.array(ids, dtype=np.uint16)

            if rng.random() < args.val_ratio:
                arr.tofile(fval)
                val_tokens += len(arr)
            else:
                arr.tofile(ftrain)
                train_tokens += len(arr)

            lines += 1

            if lines % 10000 == 0:
                print(
                    f"lines={lines:,}, "
                    f"train_tokens={train_tokens:,}, "
                    f"val_tokens={val_tokens:,}"
                )

    print("Done.")
    print(f"input: {input_path}")
    print(f"lines: {lines:,}")
    print(f"train tokens: {train_tokens:,}")
    print(f"val tokens: {val_tokens:,}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
