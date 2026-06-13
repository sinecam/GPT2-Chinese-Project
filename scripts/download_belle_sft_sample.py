import json
import random
from pathlib import Path
from datasets import load_dataset

random.seed(1337)

out_path = Path("data/sft/belle_sft_100k.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("BelleGroup/train_1M_CN", split="train")

items = list(ds)
random.shuffle(items)

kept = 0
max_samples = 100000

with out_path.open("w", encoding="utf-8") as f:
    for item in items:
        instruction = str(item.get("instruction", "")).strip()
        inp = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()

        if not instruction or not output:
            continue

        # 简单过滤太短、太长、奇怪样本
        if len(instruction) < 4 or len(instruction) > 300:
            continue
        if len(output) < 10 or len(output) > 800:
            continue

        if inp:
            instruction = instruction + "\n" + inp

        f.write(json.dumps({
            "instruction": instruction,
            "output": output
        }, ensure_ascii=False) + "\n")

        kept += 1
        if kept >= max_samples:
            break

print(f"saved {kept} examples to {out_path}")
