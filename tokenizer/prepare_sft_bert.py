from pathlib import Path
import argparse
import json
import random
import numpy as np
from transformers import BertTokenizerFast


def encode_jsonl(path, tokenizer, eot_id, seed=1337):
    random.seed(seed)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    random.shuffle(rows)

    all_ids = []
    total = 0
    kept = 0

    for item in rows:
        total += 1
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()

        if not instruction or not output:
            continue

        text = f"User: {instruction}\nAssistant: {output}\n"
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eot_id)

        if len(ids) < 10:
            continue

        all_ids.extend(ids)
        kept += 1

    return np.array(all_ids, dtype=np.uint16), total, kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", required=True)
    parser.add_argument("--valid_input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer_name", default="bert-base-chinese")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(
        args.tokenizer_name,
        model_max_length=1000000000,
    )
    eot_id = tokenizer.sep_token_id

    train_arr, train_total, train_kept = encode_jsonl(
        args.train_input,
        tokenizer,
        eot_id,
        args.seed,
    )

    val_arr, val_total, val_kept = encode_jsonl(
        args.valid_input,
        tokenizer,
        eot_id,
        args.seed + 1,
    )

    train_arr.tofile(out_dir / "train.bin")
    val_arr.tofile(out_dir / "val.bin")

    print("tokenizer:", args.tokenizer_name)
    print("vocab_size:", tokenizer.vocab_size)
    print("eot token id:", eot_id)
    print("train examples total:", train_total)
    print("train examples kept:", train_kept)
    print("val examples total:", val_total)
    print("val examples kept:", val_kept)
    print("train tokens:", len(train_arr))
    print("val tokens:", len(val_arr))
    print("saved to:", out_dir)


if __name__ == "__main__":
    main()
