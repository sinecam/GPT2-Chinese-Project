# 数据目录说明
本地数据目录结构如下：

```text
data/raw/pretrain_corpus.jsonl       # 原始中文预训练语料
data/raw/dialogue_sft.jsonl          # 原始指令微调/对话数据
data/processed/pretrain/train.bin    # 由 scripts/prepare_dialogue_data.py 生成的预训练训练集
data/processed/pretrain/val.bin      # 由 scripts/prepare_dialogue_data.py 生成的预训练验证集
data/processed/pretrain/meta.json    # 预训练数据的元信息
data/processed/sft/train.bin         # 由 scripts/prepare_dialogue_data.py 生成的指令微调训练集
data/processed/sft/val.bin           # 由 scripts/prepare_dialogue_data.py 生成的指令微调验证集
data/processed/sft/meta.json         # 指令微调数据的元信息