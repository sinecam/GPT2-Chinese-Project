# Chinese Generation Benchmark

- Suite: `eval/zh_generation_v3.jsonl`
- Suite SHA-256: `c0a195635088698f45a22082c4bc65c34d06cc1465a54b173c8408958e078c55`
- Generated: 2026-06-25T10:54:35.972051+00:00
- Decoding: temperature=0.0, top_k=30, top_p=0.85, repetition_penalty=1.15, no_repeat_ngram_size=4

## Overall

| Model | Auto | Rules | Quality | Pass % | Critical fail % | Ref loss | Ref ppl | Unknown % | Truncated % | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v48 | 37.04 | 17.90 | 94.49 | 5.41 | 18.92 | 3.657566 | 38.7669 | 0.00 | 24.32 | 239.16 |

## Categories

| Model | Category | Tasks | Rules | Quality | Pass % |
|---|---|---:|---:|---:|---:|
| v48 | creative | 3 | 46.67 | 100.00 | 33.33 |
| v48 | factual | 5 | 0.00 | 94.00 | 0.00 |
| v48 | identity | 4 | 28.33 | 96.37 | 0.00 |
| v48 | instruction | 5 | 18.00 | 94.00 | 0.00 |
| v48 | knowledge_ai | 6 | 35.28 | 96.67 | 16.67 |
| v48 | reasoning | 5 | 0.00 | 85.00 | 0.00 |
| v48 | safety_truthfulness | 5 | 0.00 | 100.00 | 0.00 |
| v48 | writing | 4 | 26.79 | 91.41 | 0.00 |

## Reading The Scores

- Auto = 75% rule score + 25% degeneration quality score.
- Reference loss is teacher-forced loss on the fixed reference answers. Compare it only when suite hash, tokenizer, and system prompt are identical.
- Keyword and format checks do not establish semantic correctness. Use the generated blind review sheet before choosing a release checkpoint.
- Speed is comparable only on the same machine and software environment.
