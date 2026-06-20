# GPT-2 Small 中文对话模型

这个分支从零搭建一套 GPT-2 Small 级别的中文对话大语言模型训练流程：中文 SentencePiece tokenizer、数据预处理、约 124M 参数的 GPT-2 Small 训练脚本、交互测试脚本和验证集困惑度评估脚本。

## 模型规格

默认配置对齐 GPT-2 Small：

- `n_layer=12`
- `n_head=12`
- `n_embd=768`
- `block_size=1024`
- `vocab_size=50000`
- 参数量约 124M，使用 token embedding 和 lm head 权重共享


## 为什么用 SentencePiece

中文没有天然空格分词，直接用英文 GPT-2 的 byte-level BPE 会让常见汉字和中文词组的切分不够经济。这里使用 SentencePiece BPE：

- 不依赖预分词，适合中文、英文、数字和符号混合文本。
- `character_coverage=0.9995`，覆盖绝大多数中文字符。
- `byte_fallback=True`，遇到生僻字或特殊符号也能编码。
- 内置 `<user>`、`<assistant>`、`<system>`、`<sep>`，便于对话格式训练。

## 推荐数据集

建议分两阶段训练：

1. 中文通用预训练：用大规模中文网页、百科、新闻、书籍、问答文本训练基础语言能力。
2. 中文对话 SFT：用中文指令和多轮对话数据把模型调成聊天助手。

可选数据集：

- 预训练：`Skywork/SkyPile-150B`、`BAAI/CCI3-HQ`、`wikipedia` 中文子集、清洗后的 Common Crawl 中文语料。
- SFT/对话：`BelleGroup/train_2M_CN`、`YeungNLP/firefly-train-1.1M`、`FreedomIntelligence/alpaca-gpt4-chinese`、中文 ShareGPT 格式数据。

数据质量比数据量更重要。124M 模型建议至少准备 5B 到 20B 中文 token 做预训练，再用 50 万到 300 万条中文指令/对话样本做 SFT。小规模验证可以先用 1M 到 10M token 跑通流程。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 目录约定

```text
data/raw/                 # 原始数据，仓库不提交
data/processed/pretrain/  # 预训练 train.bin / val.bin
data/processed/sft/       # SFT train.bin / val.bin
artifacts/tokenizer_zh/   # SentencePiece tokenizer
checkpoints/zh_gpt2small/ # 模型 checkpoint，仓库不提交
```

## 1. 训练中文 tokenizer

本地文本或 JSONL：

```bash
python scripts/train_tokenizer_zh.py \
  --input data/raw/pretrain_corpus.jsonl data/raw/dialogue_sft.jsonl \
  --out_dir artifacts/tokenizer_zh \
  --vocab_size 50000 \
  --model_type bpe \
  --max_corpus_chars 200000000 \
  --max_chars_per_line 768 \
  --num_threads 50 \
  --num_workers 24 \
  --worker_chunksize 2048
```

Hugging Face 数据集示例：

```bash
python scripts/train_tokenizer_zh.py \
  --hf_dataset BelleGroup/train_2M_CN \
  --hf_split train \
  --out_dir artifacts/tokenizer_zh \
  --vocab_size 50000 \
  --max_corpus_chars 200000000 \
  --max_chars_per_line 768 \
  --num_threads 50
```

多核和语料大小参数说明：

- `--num_threads` 传给 SentencePiece C++ trainer，默认等于检测到的 CPU 核数。50 核服务器可以直接设 `50`。
- `--num_workers` 只加速本地 `txt/json/jsonl` 到 tokenizer 临时语料的格式化阶段；Hugging Face streaming 仍主要受网络、磁盘缓存和数据集迭代速度限制。
- SentencePiece 的 `Tokenizing input sentences with whitespace` 等部分阶段在一些版本里不能完全吃满多核。遇到这个卡点，优先减少 tokenizer 训练语料字符数，而不是继续堆线程。
- `--max_corpus_chars 200000000` 会把 tokenizer 临时语料控制在约 2 亿字符；对 50k 中文 BPE 通常已经足够。完整大语料应留给模型预训练。
- `--max_chars_per_line 512` 到 `1024` 通常比长篇整段输入更快、更稳。长文档不需要完整喂给 tokenizer 才能学到好词表。
- 如果 CPU 不满但磁盘 I/O 打满，先把 `--num_workers` 降到 `8` 或 `16`；如果内存压力不大，可以试 `24` 到 `32`。
- `--worker_chunksize` 控制每个进程一次处理多少条记录，JSONL 大文件通常 `1024` 到 `4096` 比较合适。

输出：

- `artifacts/tokenizer_zh/spm_zh.model`
- `artifacts/tokenizer_zh/spm_zh.vocab`
- `artifacts/tokenizer_zh/tokenizer_config.json`

## 2. 制作预训练数据

```bash
python scripts/prepare_dialogue_data.py \
  --tokenizer artifacts/tokenizer_zh/spm_zh.model \
  --input data/raw/pretrain_corpus.jsonl \
  --out_dir data/processed/pretrain \
  --mode pretrain \
  --val_ratio 0.002
```

也可以直接读 Hugging Face：

```bash
python scripts/prepare_dialogue_data.py \
  --tokenizer artifacts/tokenizer_zh/spm_zh.model \
  --hf_dataset Skywork/SkyPile-150B \
  --hf_split train \
  --streaming \
  --out_dir data/processed/pretrain \
  --mode pretrain \
  --max_records 10000000 \
  --val_ratio 0.002
```

## 3. 从零预训练 GPT-2 Small

单卡：

```bash
python scripts/train_gpt2_small_zh.py \
  --data_dir data/processed/pretrain \
  --tokenizer artifacts/tokenizer_zh/spm_zh.model \
  --out_dir checkpoints/zh_gpt2small_pretrain \
  --max_iters 200000 \
  --batch_size 8 \
  --grad_accum_steps 16 \
  --learning_rate 3e-4 \
  --warmup_iters 2000
```

多卡 DDP：

```bash
torchrun --nproc_per_node=4 scripts/train_gpt2_small_zh.py \
  --data_dir data/processed/pretrain \
  --tokenizer artifacts/tokenizer_zh/spm_zh.model \
  --out_dir checkpoints/zh_gpt2small_pretrain \
  --max_iters 200000 \
  --batch_size 4 \
  --grad_accum_steps 16
```

有效 tokens/update = `GPU 数 * batch_size * block_size * grad_accum_steps`。

## 4. 制作中文对话 SFT 数据

```bash
python scripts/prepare_dialogue_data.py \
  --tokenizer artifacts/tokenizer_zh/spm_zh.model \
  --hf_dataset BelleGroup/train_2M_CN \
  --hf_split train \
  --out_dir data/processed/sft \
  --mode dialogue \
  --val_ratio 0.01
```

脚本会自动识别常见字段：

- `messages`
- `conversations`
- `instruction` / `input` / `output`
- `text` / `content`

对话会被格式化为：

```text
<user>
你好
<sep>
<assistant>
你好！有什么我可以帮你？
```

## 5. 对预训练模型做 SFT

```bash
python scripts/train_gpt2_small_zh.py \
  --data_dir data/processed/sft \
  --tokenizer artifacts/tokenizer_zh/spm_zh.model \
  --out_dir checkpoints/zh_gpt2small_sft \
  --init_from checkpoints/zh_gpt2small_pretrain/ckpt.pt \
  --max_iters 20000 \
  --batch_size 8 \
  --grad_accum_steps 8 \
  --learning_rate 1e-5 \
  --warmup_iters 500
```

## 6. 快速测试

先跑一个随机小模型前后向，确认环境正常：

```bash
python scripts/smoke_test_zh.py --vocab_size 50000
```

交互聊天：

```bash
python scripts/chat_zh.py \
  --ckpt checkpoints/zh_gpt2small_sft/ckpt.pt \
  --tokenizer artifacts/tokenizer_zh/spm_zh.model
```

单条 prompt 评估和生成样例：

```bash
python scripts/evaluate_zh.py \
  --ckpt checkpoints/zh_gpt2small_sft/ckpt.pt \
  --tokenizer artifacts/tokenizer_zh/spm_zh.model \
  --data_dir data/processed/sft \
  --prompt "请用三句话解释什么是大语言模型。"
```

## 训练建议

- tokenizer 训练语料要混合预训练文本和对话数据，避免 SFT 阶段大量角色标记或口语表达切分不稳定。
- 预训练先看验证集 loss 是否平稳下降；124M 模型中文数据足够时，loss 通常比生成效果更早稳定。
- SFT 学习率要小，推荐 `1e-5` 到 `5e-5`，训练太久会过拟合固定问答格式。
- 每隔固定 step 保留 `latest.pt` 和最优 `ckpt.pt`，不要把 checkpoint 提交到 Git。
- 生成质量不足时，优先提升数据清洗质量和预训练 token 数，其次再调模型大小。
