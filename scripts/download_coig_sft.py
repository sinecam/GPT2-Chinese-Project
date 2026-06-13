import json
import random
from pathlib import Path
from datasets import load_dataset

subsets = [
    "coig_pc",
    "wiki",
    "wikihow",
    "zhihu",
    "douban",
    "chinese_traditional",
    "exam",
]

out_path = Path("data/sft/coig_sft_zh.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

max_per_subset = 5000
random.seed(1337)

total = 0

with out_path.open("w", encoding="utf-8") as fout:
    for subset in subsets:
        print(f"loading subset: {subset}")
        ds = load_dataset("m-a-p/COIG-CQIA", subset, split="train")

        items = list(ds)
        random.shuffle(items)
        items = items[:max_per_subset]

        kept = 0
        for item in items:
            instruction = str(item.get("instruction", "")).strip()
            inp = str(item.get("input", "")).strip()
            output = str(item.get("output", "")).strip()

            if not instruction or not output:
                continue

            if inp and inp not in instruction:
                instruction = instruction + "\n" + inp

            ex = {
                "instruction": instruction,
                "output": output,
            }

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1
            total += 1

        print(f"subset {subset}: kept {kept}")

print(f"saved {total} examples to {out_path}")
