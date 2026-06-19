# V4.3 Codex Context

Current project path on server:

`/root/autodl-tmp/GPT2_Medium_Project`

## Version Summary

### V4.0 Pretrain Base

- Model: GPT-2 Small
- Parameters: about 124M
- Tokenizer: SentencePiece BPE
- vocab_size: 50000
- Data: about 2.47B Chinese pretraining tokens
- Best checkpoint:
  `checkpoints/V4.0_sp50k_gpt2small_pretrain_5b/ckpt.pt`

### V4.1 SFT

- Dataset: `FreedomIntelligence/alpaca-gpt4-chinese`
- About 50k samples
- Result:
  - val loss about 2.4940
  - ppl about 12.11
- Conclusion:
  - SFT pipeline works.
  - Response quality is still weak.

### V4.3 SFT

V4.3 uses a filtered mixed dataset:

- `FreedomIntelligence/alpaca-gpt4-chinese`
- `BelleGroup/train_2M_CN` filtered local samples
- small identity QA samples

Raw mixed file:

`data/raw/V4.3_sft_mix_filtered.jsonl`

Encoded data dir:

`data/processed/V4.3_sp50k_sft_mix_filtered`

Encoded stats:

- train_tokens: 27,887,907
- val_tokens: 279,553
- train_records: 134,166
- val_records: 1,352
- skipped_records: 0

Training setup:

- initialized from V4.0 checkpoint
- not continued from V4.1
- max_iters: 8000
- batch_size: 16
- grad_accum_steps: 2
- learning_rate: 1e-5
- dtype: bfloat16
- 2 GPUs

Final V4.3 result:

- iter 8000
- train loss 2.2651
- val loss 2.3987
- ppl 11.01

## What I want Codex to check

Please inspect the files in this folder and help with the next step:

1. Check whether `build_v43_sft_mix.py` filtering logic is reasonable.
2. Check whether `prepare_dialogue_data.py` correctly handles local JSONL and dialogue format.
3. Check whether `train_gpt2_small_zh.py` handles `--init_from` and `--resume` safely.
4. Check whether `evaluate_zh.py` and `chat_zh.py` are suitable for SFT dialogue prompting.
5. Modify generation to prevent SentencePiece `unk_id` from being sampled, because current outputs may contain `⁇`.
6. Add optional generation parameters:
   - `--repetition_penalty`
   - `--no_repeat_ngram_size`
7. Provide final evaluation commands for:
   - 请用三句话解释什么是大语言模型。
   - 什么是人工智能？
   - 深度学习和机器学习有什么区别？
   - 你是谁？
8. Judge whether V4.3 can be used as the final demo model.

Do not ask me to upload `.bin`, `.pt`, checkpoint files, raw Belle data, or full data files because they are too large.
