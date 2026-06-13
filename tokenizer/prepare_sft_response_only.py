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
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    train_x, train_y = [], []
    val_x, val_y = [], []

    total = 0
    kept = 0
    skipped = 0
    truncated = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            total += 1
            item = json.loads(line)

            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()

            if not instruction or not output:
                skipped += 1
                continue

            prompt = f"User: {instruction}\nAssistant:"
            answer = " " + output + "\n"

            prompt_ids = enc.encode_ordinary(prompt)
            answer_ids = enc.encode_ordinary(answer)
            answer_ids.append(eot)

            # 需要留出至少一个回答 token
            max_total_len = args.block_size + 1
            if len(prompt_ids) >= max_total_len - 1:
                skipped += 1
                continue

            if len(prompt_ids) + len(answer_ids) > max_total_len:
                max_answer_len = max_total_len - len(prompt_ids)
                answer_ids = answer_ids[:max_answer_len]
                truncated += 1

            ids = prompt_ids + answer_ids

            # labels_full 和 ids 等长：
            # prompt 部分用 -100，不参与 loss；
            # answer 部分是真实 token，参与 loss。
            labels_full = [-100] * len(prompt_ids) + answer_ids

            # 自回归训练：x 预测 y
            x = ids[:-1]
            y = labels_full[1:]

            if len(x) == 0 or all(v == -100 for v in y):
                skipped += 1
                continue

            pad_len = args.block_size - len(x)
            if pad_len < 0:
                skipped += 1
                continue

            x = x + [eot] * pad_len
            y = y + [-100] * pad_len

            x_arr = np.array(x, dtype=np.uint16)
            y_arr = np.array(y, dtype=np.int32)

            if random.random() < args.val_ratio:
                val_x.append(x_arr)
                val_y.append(y_arr)
            else:
                train_x.append(x_arr)
                train_y.append(y_arr)

            kept += 1

    train_x = np.stack(train_x)
    train_y = np.stack(train_y)
    val_x = np.stack(val_x)
    val_y = np.stack(val_y)

    np.save(out_dir / "train_x.npy", train_x)
    np.save(out_dir / "train_y.npy", train_y)
    np.save(out_dir / "val_x.npy", val_x)
    np.save(out_dir / "val_y.npy", val_y)

    print(f"input file: {in_path}")
    print(f"total examples: {total:,}")
    print(f"kept examples: {kept:,}")
    print(f"skipped examples: {skipped:,}")
    print(f"truncated examples: {truncated:,}")
    print(f"train examples: {len(train_x):,}")
    print(f"val examples: {len(val_x):,}")
    print(f"block_size: {args.block_size}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
