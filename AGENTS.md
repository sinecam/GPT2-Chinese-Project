# Project Instructions for Codex

This is a GPT-2 style Chinese language model training project.

Important constraints:
- Do not commit or modify large files under data/raw, data/processed, checkpoints, or logs.
- Do not assume checkpoint files are available in the repository.
- Keep compatibility with the existing project structure.
- The main model implementation is in model/gpt.py.
- Pretraining code is in train/pretrain_ddp.py.
- Fine-tuning code is in train/finetune.py.
- Generation scripts are in eval/.
- Tokenization/preprocessing scripts are in tokenizer/.
- The Streamlit frontend is app_modern.py.

Current important model versions:
- V3.0: GPT-2 Small structure + bert-base-chinese tokenizer + 5B Chinese pretraining data.
- V3.1/V3.2/V3.3: SFT attempts based on baike_qa2019.

When making changes:
- Prefer small, focused patches.
- Add clear comments only where necessary.
- Run python -m py_compile on modified Python files.
- Do not require downloading large datasets or checkpoints.
