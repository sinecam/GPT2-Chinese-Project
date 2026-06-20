# Standardized Chinese Model Evaluation

This workflow compares checkpoints with the same prompts, tokenizer, system prompt, decoding settings, and scoring rules. It is designed for the current 124M Chinese GPT-2 response-only SFT models.

## What Is Measured

The frozen `eval/zh_generation_v2.jsonl` suite contains 37 tasks across:

- identity stability
- AI and language-model knowledge
- simple factual questions
- arithmetic and basic reasoning
- strict instruction following
- practical writing
- constrained creative writing
- safety and truthfulness

Each run produces four independent views:

1. **Rule score (0-100):** exact answers, required concepts, forbidden content, format, sentence count, and length constraints.
2. **Degeneration quality (0-100):** unknown-token output, repeated 4-grams, repeated sentences, excessive non-Chinese output, and likely truncation.
3. **Reference loss / perplexity:** response-only teacher-forced loss on the same fixed reference answers.
4. **Human blind review (1-5):** correctness, instruction following, fluency, conciseness, and safety.

Checks marked as critical gates receive zero rule score when they fail. This prevents a response that fabricates identity, echoes a prompt, or complies with an unsafe request from earning misleading partial credit. The report also shows the critical-failure rate.\n\nThe displayed automatic score is:

```text
auto_score = 0.75 * rule_score + 0.25 * quality_score
```

This weighting is intentionally transparent. It is useful for regression detection, not a substitute for human judgment.

## Run V4.4 Versus V4.5

On the cloud server:

```bash
cd /root/autodl-tmp/GPT2_Medium_Project
source activate_env.sh
export PYTHONPATH=/root/autodl-tmp/GPT2_Medium_Project

python scripts/benchmark_zh.py \
  --model V4.4=checkpoints/V4.4_sp50k_gpt2small_sft_response_only_mix/ckpt.pt \
  --model V4.5=checkpoints/V4.5_sp50k_gpt2small_sft_quality_mix/ckpt.pt \
  --tokenizer artifacts/tokenizer_zh_50k/spm_zh.model \
  --suite eval/zh_generation_v2.jsonl \
  --out_dir reports/V4.4_vs_V4.5_zh_v2 \
  --device cuda \
  --dtype bfloat16 \
  --temperature 0 \
  --max_new_tokens 160 \
  --top_k 30 \
  --top_p 0.85 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4 \
  --seed 20260620
```

Add another `--model LABEL=PATH` argument to include V4.3 or a later checkpoint. Run all compared checkpoints in one command so the environment is identical.

The main comparison uses greedy decoding (`temperature=0`). This removes sampling luck. The top-k and top-p values are still recorded but do not affect greedy decoding.

## Outputs

The command writes:

- `report.md`: readable overall and per-category comparison
- `summary.csv`: one row per checkpoint
- `results.json`: every prompt, raw response, rule result, latency, and loss
- `run_config.json`: suite, tokenizer and checkpoint hashes plus all decoding settings
- `blind_review.csv`: randomized responses with empty human-score columns
- `blind_key.json`: candidate-to-model mapping; keep this hidden until scoring is complete

Checkpoint hashing is enabled by default. It can take several seconds for large files but prevents accidentally comparing the wrong checkpoint.

## Human Blind Review

Make one copy of `blind_review.csv` for each reviewer. Do not show reviewers `blind_key.json`. Score every response from 1 to 5 in these columns:

- `correctness_1_5`
- `instruction_1_5`
- `fluency_1_5`
- `conciseness_1_5`
- `safety_1_5`

After the sheets are complete:

```bash
python scripts/score_blind_review.py \
  --input reports/V4.4_vs_V4.5_zh_v2/reviewer_1.csv \
          reports/V4.4_vs_V4.5_zh_v2/reviewer_2.csv \
  --key reports/V4.4_vs_V4.5_zh_v2/blind_key.json \
  --out_dir reports/V4.4_vs_V4.5_zh_v2/human
```

This creates `human_report.md`, `human_summary.csv`, and `human_by_category.csv`.

Use at least two reviewers for a release decision. One reviewer is acceptable for quick iteration but should not be treated as a stable result.

## Release Decision

A later version should replace the current display checkpoint only when all of these are true:

1. Human mean score improves by at least 0.15 on the 1-5 scale, or the improvement is clearly visible in the target categories.
2. Automatic score does not decline by more than 2 points overall.
3. No critical category, especially identity or safety/truthfulness, declines by more than 5 rule-score points.
4. Unknown-token rate, truncation rate, and repetition do not regress materially.
5. Raw responses show no new systematic failure such as fabricated identity, copied templates, excessive verbosity, or format collapse.

Reference loss is secondary. A lower loss does not override worse generated answers.

## Reproducibility Rules

For official comparisons, keep all of the following unchanged:

- benchmark suite file and SHA-256
- tokenizer file and SHA-256
- system prompt
- decoding parameters
- dtype
- software environment
- GPU type when comparing latency or tokens per second

Never add benchmark prompts or reference answers to SFT data. Do not edit `zh_generation_v2.jsonl`; create `zh_generation_v2.jsonl` when the suite must change.

Repeatedly tuning against this public suite gradually turns it into a development set. Keep a second private JSONL suite on the server, using the same schema, for final release checks. Do not commit or train on that private suite.

## Optional Sampled Stability Check

The official score remains the deterministic run above. After selecting a candidate, run a secondary check with `temperature=0.6` and three different seeds into three separate output directories. Inspect the spread in rule score and the raw responses. A model that only performs well under one seed is not stable enough for a demo.

## Quick Smoke Test

During script development, run only the first three items:

```bash
python scripts/benchmark_zh.py \
  --model V4.5=checkpoints/V4.5_sp50k_gpt2small_sft_quality_mix/ckpt.pt \
  --tokenizer artifacts/tokenizer_zh_50k/spm_zh.model \
  --suite eval/zh_generation_v2.jsonl \
  --out_dir reports/smoke \
  --limit 3 \
  --temperature 0
```

The smoke test verifies the pipeline only. It is not a model score.
