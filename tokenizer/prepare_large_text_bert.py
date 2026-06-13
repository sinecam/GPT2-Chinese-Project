from pathlib import Path
import argparse
import random
import numpy as np
from transformers import BertTokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-chinese")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    eot = tokenizer.sep_token_id

    print(f"tokenizer: {args.tokenizer_name}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"eot token id: {eot}")

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    if train_path.exists():
        train_path.unlink()
    if val_path.exists():
        val_path.unlink()

    train_count = 0
    val_count = 0
    line_count = 0

    with train_path.open("ab") as train_f, val_path.open("ab") as val_f:
        with input_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                ids = tokenizer.encode(line, add_special_tokens=False)
                if not ids:
                    continue

                ids.append(eot)
                arr = np.array(ids, dtype=np.uint16)

                if random.random() < args.val_ratio:
                    arr.tofile(val_f)
                    val_count += len(arr)
                else:
                    arr.tofile(train_f)
                    train_count += len(arr)

                line_count += 1

                if line_count % 10000 == 0:
                    print(
                        f"lines={line_count:,}, "
                        f"train tokens={train_count:,}, "
                        f"val tokens={val_count:,}",
                        flush=True,
                    )

    print("Done.")
    print(f"lines: {line_count:,}")
    print(f"train tokens: {train_count:,}")
    print(f"val tokens: {val_count:,}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
