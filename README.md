# 中文 GPT-2 Small 最终版

这是项目最终交付目录，仅保留一条可以执行的数据处理、训练、评测、模型导出和 Web 演示流程。历史版本、实验脚本、旧评测集、命令行聊天界面和重复模型实现均已移除。

## 目录

```text
.
|-- model/
|   `-- gpt.py
|-- scripts/
|   |-- train_tokenizer.py
|   |-- prepare_pretrain.py
|   |-- train_pretrain.py
|   |-- sft_common.py
|   |-- build_sft_mix.py
|   |-- prepare_sft.py
|   |-- train_sft.py
|   `-- benchmark.py
|-- eval/
|   `-- zh_generation_v3.jsonl
|-- final_release/
|   |-- app.py
|   |-- engine.py
|   |-- export_release.py
|   |-- smoke_test.py
|   |-- config.json
|   |-- FINAL_REPORT.md
|   `-- static/index.html
|-- requirements.txt
`-- README.md
```

大型语料、二进制训练数据、checkpoint 和最终导出模型不提交到 Git。

## 环境

要求 Python 3.10 以上和 PyTorch 2.1 以上。

```bash
pip install -r requirements.txt
```

## 输入

全流程需要准备以下外部文件：

| 输入 | 建议路径 | 网盘中的文件 |
|---|---|---|
| 中文预训练语料，TXT/JSON/JSONL | `data/raw/` | chinese_pretrain_5b.txt |
| 中文基础 SFT 数据，JSON/JSONL | `data/raw/base_sft.jsonl` | |
| 最终演示使用的基础 checkpoint | `checkpoints/base/ckpt.pt` | |

仓库不会包含这些大型文件。已有 50k tokenizer 和基础 checkpoint 时，可以直接从“最终 SFT”开始。

## 1. Tokenizer （作业压缩包中已保存输出文件，此步骤可以跳过）

```bash
python scripts/train_tokenizer.py \
  --input data/raw/chinese_pretrain_5b.txt data/raw/base_sft.jsonl \
  --out_dir artifacts/tokenizer_zh_50k \
  --vocab_size 50000 \
  --model_type bpe \
  --max_corpus_chars 200000000
```

输出：`artifacts/tokenizer_zh_50k/spm_zh.model`。在提交的作业压缩包中已有此文件

## 2. 预训练数据

```bash
python scripts/prepare_pretrain.py \
  --tokenizer artifacts/tokenizer_zh_50k/spm_zh.model \
  --input data/raw/chinese_pretrain_5b.txt \
  --out_dir data/processed/pretrain \
  --mode pretrain \
  --val_ratio 0.002
```

输出：
```text
/root/autodl-tmp/com/data/processed/pretrain/
├── train.bin
├── val.bin
└── meta.json
```

## 3. 预训练

两张 GPU：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_pretrain.py \
  --data_dir data/processed/pretrain \
  --tokenizer artifacts/tokenizer_zh_50k/spm_zh.model \
  --out_dir checkpoints/base \
  --max_iters 200000 \
  --batch_size 8 \
  --grad_accum_steps 8 \
  --learning_rate 3e-4 \
  --warmup_iters 2000 \
  --dtype bfloat16
```

## 4. 最终 SFT 数据

```bash
python scripts/build_sft_mix.py \
  --input data/raw/base_sft.jsonl \
  --benchmark eval/zh_generation_v3.jsonl \
  --out data/raw/final_sft_mix.jsonl \
  --max_records 12000
```

构建器会生成最终定向样本、筛选基础 SFT 样本、去重，并排除与评测问题过度相似的数据。

```bash
python scripts/prepare_sft.py \
  --tokenizer artifacts/tokenizer_zh_50k/spm_zh.model \
  --input data/raw/final_sft_mix.jsonl \
  --out_dir data/processed/final_sft \
  --block_size 1024 \
  --val_ratio 0.01
```

## 5. 最终 SFT 训练

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_sft.py \
  --data_dir data/processed/final_sft \
  --out_dir checkpoints/final \
  --init_from checkpoints/base/ckpt.pt \
  --max_iters 1200 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --learning_rate 3e-6 \
  --min_lr 3e-7 \
  --warmup_iters 200 \
  --lr_decay_iters 1200 \
  --eval_interval 300 \
  --eval_iters 100 \
  --dtype bfloat16
```

最佳模型为 `checkpoints/final/ckpt.pt`。

## 6. 标准评测

```bash
python scripts/benchmark.py \
  --model final=checkpoints/final/ckpt.pt \
  --tokenizer artifacts/tokenizer_zh_50k/spm_zh.model \
  --suite eval/zh_generation_v3.jsonl \
  --out_dir reports/final \
  --max_new_tokens 120 \
  --temperature 0 \
  --top_k 30 \
  --top_p 0.85 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

评测输出包括 `results.json`、`summary.csv`、`report.md` 和可复核运行配置。

## 7. 导出并验证最终 Web 版本

```bash
python -m final_release.export_release \
  --checkpoint checkpoints/final/ckpt.pt \
  --tokenizer artifacts/tokenizer_zh_50k/spm_zh.model

python -m final_release.smoke_test \
  --device cuda \
  --dtype bfloat16
```

看到 `SMOKE TEST PASSED` 后启动唯一保留的前端：

```bash
python -m final_release.app \
  --device cuda \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 6006
```

访问 `http://<服务器地址>:6006`。

演示时不要修改 `final_release/config.json`，它保存了最终汇报所使用的确定性生成参数。
