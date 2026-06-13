from pathlib import Path
import argparse
import json
import random
import numpy as np
import tiktoken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    samples = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()

            if not instruction or not output:
                continue

            text = f"User: {instruction}\nAssistant: {output}\n"
            ids = enc.encode_ordinary(text)
            ids.append(eot)
            samples.append(ids)

    print(f"loaded examples: {len(samples)}")

    all_samples = samples * args.repeat
    random.shuffle(all_samples)

    train_ids = []
    val_ids = []

    for ids in all_samples:
        if random.random() < args.val_ratio:
            val_ids.extend(ids)
        else:
            train_ids.extend(ids)

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    train_arr.tofile(out_dir / "train.bin")
    val_arr.tofile(out_dir / "val.bin")

    print(f"repeat: {args.repeat}")
    print(f"train tokens: {len(train_arr):,}")
    print(f"val tokens: {len(val_arr):,}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
