# Chinese Generation Benchmark

- Suite: `eval/zh_generation_v3.jsonl`
- Suite SHA-256: `c0a195635088698f45a22082c4bc65c34d06cc1465a54b173c8408958e078c55`
- Generated: 2026-06-25T10:54:53.704391+00:00
- Decoding: temperature=0.0, top_k=30, top_p=0.85, repetition_penalty=1.15, no_repeat_ngram_size=4

## Overall

| Model | Auto | Rules | Quality | Pass % | Critical fail % | Ref loss | Ref ppl | Unknown % | Truncated % | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v47 | 52.13 | 36.88 | 97.87 | 21.62 | 8.11 | 2.740515 | 15.495 | 0.00 | 13.51 | 230.45 |

## Categories

| Model | Category | Tasks | Rules | Quality | Pass % |
|---|---|---:|---:|---:|---:|
| v47 | creative | 3 | 13.33 | 95.00 | 0.00 |
| v47 | factual | 5 | 20.00 | 100.00 | 20.00 |
| v47 | identity | 4 | 76.67 | 100.00 | 50.00 |
| v47 | instruction | 5 | 25.00 | 93.25 | 20.00 |
| v47 | knowledge_ai | 6 | 39.44 | 100.00 | 16.67 |
| v47 | reasoning | 5 | 20.00 | 94.00 | 20.00 |
| v47 | safety_truthfulness | 5 | 44.57 | 100.00 | 20.00 |
| v47 | writing | 4 | 58.33 | 100.00 | 25.00 |

## Reading The Scores

- Auto = 75% rule score + 25% degeneration quality score.
- Reference loss is teacher-forced loss on the fixed reference answers. Compare it only when suite hash, tokenizer, and system prompt are identical.
- Keyword and format checks do not establish semantic correctness. Use the generated blind review sheet before choosing a release checkpoint.
- Speed is comparable only on the same machine and software environment.
