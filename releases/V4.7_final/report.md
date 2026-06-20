# Chinese Generation Benchmark

- Suite: `eval/zh_generation_v3.jsonl`
- Suite SHA-256: `de85db6ce56b2fe71d660473bb3d5ee503fee0a0e24d13b2906461828bc58112`
- Generated: 2026-06-20T10:18:06.295755+00:00
- Decoding: temperature=0.0, top_k=30, top_p=0.85, repetition_penalty=1.15, no_repeat_ngram_size=4

## Overall

| Model | Auto | Rules | Quality | Pass % | Critical fail % | Ref loss | Ref ppl | Unknown % | Truncated % | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V4.6_best | 49.46 | 34.77 | 93.50 | 18.92 | 10.81 | 2.962931 | 19.3546 | 0.00 | 35.14 | 248.11 |
| V4.7_0200 | 49.40 | 33.83 | 96.13 | 18.92 | 8.11 | 2.83154 | 16.9716 | 0.00 | 21.62 | 255.03 |
| V4.7_0400 | 52.21 | 37.46 | 96.46 | 21.62 | 8.11 | 2.760238 | 15.8036 | 0.00 | 18.92 | 251.60 |
| V4.7_0600 | 53.62 | 39.49 | 95.99 | 21.62 | 8.11 | 2.724545 | 15.2495 | 0.00 | 21.62 | 260.80 |
| V4.7_0800 | 53.61 | 39.49 | 95.95 | 21.62 | 8.11 | 2.709329 | 15.0192 | 0.00 | 21.62 | 261.76 |
| V4.7_best | 53.62 | 39.49 | 95.99 | 21.62 | 8.11 | 2.724545 | 15.2495 | 0.00 | 21.62 | 261.36 |

## Categories

| Model | Category | Tasks | Rules | Quality | Pass % |
|---|---|---:|---:|---:|---:|
| V4.6_best | creative | 3 | 30.00 | 95.00 | 0.00 |
| V4.6_best | factual | 5 | 0.00 | 91.00 | 0.00 |
| V4.6_best | identity | 4 | 86.67 | 100.00 | 50.00 |
| V4.6_best | instruction | 5 | 30.00 | 89.51 | 20.00 |
| V4.6_best | knowledge_ai | 6 | 36.11 | 93.02 | 0.00 |
| V4.6_best | reasoning | 5 | 20.00 | 91.00 | 20.00 |
| V4.6_best | safety_truthfulness | 5 | 40.00 | 91.88 | 40.00 |
| V4.6_best | writing | 4 | 45.83 | 99.91 | 25.00 |
| V4.7_0200 | creative | 3 | 13.33 | 95.00 | 0.00 |
| V4.7_0200 | factual | 5 | 0.00 | 93.00 | 0.00 |
| V4.7_0200 | identity | 4 | 76.67 | 100.00 | 50.00 |
| V4.7_0200 | instruction | 5 | 30.00 | 90.36 | 20.00 |
| V4.7_0200 | knowledge_ai | 6 | 34.72 | 97.50 | 0.00 |
| V4.7_0200 | reasoning | 5 | 20.00 | 94.00 | 20.00 |
| V4.7_0200 | safety_truthfulness | 5 | 56.00 | 100.00 | 40.00 |
| V4.7_0200 | writing | 4 | 41.67 | 100.00 | 25.00 |
| V4.7_0400 | creative | 3 | 30.00 | 100.00 | 0.00 |
| V4.7_0400 | factual | 5 | 0.00 | 100.00 | 0.00 |
| V4.7_0400 | identity | 4 | 76.67 | 100.00 | 50.00 |
| V4.7_0400 | instruction | 5 | 45.00 | 90.70 | 40.00 |
| V4.7_0400 | knowledge_ai | 6 | 34.72 | 90.94 | 16.67 |
| V4.7_0400 | reasoning | 5 | 20.00 | 94.00 | 20.00 |
| V4.7_0400 | safety_truthfulness | 5 | 44.57 | 100.00 | 20.00 |
| V4.7_0400 | writing | 4 | 58.33 | 100.00 | 25.00 |
| V4.7_0600 | creative | 3 | 30.00 | 95.00 | 0.00 |
| V4.7_0600 | factual | 5 | 20.00 | 100.00 | 20.00 |
| V4.7_0600 | identity | 4 | 76.67 | 100.00 | 50.00 |
| V4.7_0600 | instruction | 5 | 40.00 | 91.00 | 20.00 |
| V4.7_0600 | knowledge_ai | 6 | 34.72 | 90.25 | 16.67 |
| V4.7_0600 | reasoning | 5 | 20.00 | 94.00 | 20.00 |
| V4.7_0600 | safety_truthfulness | 5 | 44.57 | 100.00 | 20.00 |
| V4.7_0600 | writing | 4 | 58.33 | 100.00 | 25.00 |
| V4.7_0800 | creative | 3 | 30.00 | 95.00 | 0.00 |
| V4.7_0800 | factual | 5 | 20.00 | 100.00 | 20.00 |
| V4.7_0800 | identity | 4 | 76.67 | 100.00 | 50.00 |
| V4.7_0800 | instruction | 5 | 40.00 | 91.00 | 20.00 |
| V4.7_0800 | knowledge_ai | 6 | 34.72 | 90.01 | 16.67 |
| V4.7_0800 | reasoning | 5 | 20.00 | 94.00 | 20.00 |
| V4.7_0800 | safety_truthfulness | 5 | 44.57 | 100.00 | 20.00 |
| V4.7_0800 | writing | 4 | 58.33 | 100.00 | 25.00 |
| V4.7_best | creative | 3 | 30.00 | 95.00 | 0.00 |
| V4.7_best | factual | 5 | 20.00 | 100.00 | 20.00 |
| V4.7_best | identity | 4 | 76.67 | 100.00 | 50.00 |
| V4.7_best | instruction | 5 | 40.00 | 91.00 | 20.00 |
| V4.7_best | knowledge_ai | 6 | 34.72 | 90.25 | 16.67 |
| V4.7_best | reasoning | 5 | 20.00 | 94.00 | 20.00 |
| V4.7_best | safety_truthfulness | 5 | 44.57 | 100.00 | 20.00 |
| V4.7_best | writing | 4 | 58.33 | 100.00 | 25.00 |

## Reading The Scores

- Auto = 75% rule score + 25% degeneration quality score.
- Reference loss is teacher-forced loss on the fixed reference answers. Compare it only when suite hash, tokenizer, and system prompt are identical.
- Keyword and format checks do not establish semantic correctness. Use the generated blind review sheet before choosing a release checkpoint.
- Speed is comparable only on the same machine and software environment.
