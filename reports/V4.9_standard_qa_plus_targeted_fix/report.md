# Chinese Generation Benchmark

- Suite: `eval/zh_generation_v3.jsonl`
- Suite SHA-256: `c0a195635088698f45a22082c4bc65c34d06cc1465a54b173c8408958e078c55`
- Generated: 2026-06-25T11:08:22.987559+00:00
- Decoding: temperature=0.0, top_k=30, top_p=0.85, repetition_penalty=1.15, no_repeat_ngram_size=4

## Overall

| Model | Auto | Rules | Quality | Pass % | Critical fail % | Ref loss | Ref ppl | Unknown % | Truncated % | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v49_0200 | 48.82 | 32.30 | 98.38 | 21.62 | 18.92 | 3.029925 | 20.6957 | 0.00 | 10.81 | 228.37 |
| v49_0400 | 64.04 | 52.48 | 98.74 | 37.84 | 8.11 | 2.606106 | 13.5462 | 0.00 | 8.11 | 260.57 |
| v49_0600 | 66.93 | 56.17 | 99.19 | 43.24 | 8.11 | 2.41888 | 11.2333 | 0.00 | 5.41 | 250.87 |
| v49_0800 | 67.22 | 56.80 | 98.48 | 43.24 | 5.41 | 2.334366 | 10.3229 | 0.00 | 8.11 | 248.28 |
| v49_best | 67.22 | 56.80 | 98.48 | 43.24 | 5.41 | 2.334366 | 10.3229 | 0.00 | 8.11 | 252.12 |

## Categories

| Model | Category | Tasks | Rules | Quality | Pass % |
|---|---|---:|---:|---:|---:|
| v49_0200 | creative | 3 | 30.00 | 100.00 | 0.00 |
| v49_0200 | factual | 5 | 0.00 | 94.00 | 0.00 |
| v49_0200 | identity | 4 | 38.33 | 100.00 | 25.00 |
| v49_0200 | instruction | 5 | 73.00 | 100.00 | 60.00 |
| v49_0200 | knowledge_ai | 6 | 50.56 | 100.00 | 33.33 |
| v49_0200 | reasoning | 5 | 20.00 | 94.00 | 20.00 |
| v49_0200 | safety_truthfulness | 5 | 0.00 | 100.00 | 0.00 |
| v49_0200 | writing | 4 | 45.83 | 100.00 | 25.00 |
| v49_0400 | creative | 3 | 46.67 | 100.00 | 33.33 |
| v49_0400 | factual | 5 | 40.00 | 97.00 | 40.00 |
| v49_0400 | identity | 4 | 81.67 | 100.00 | 50.00 |
| v49_0400 | instruction | 5 | 85.00 | 100.00 | 80.00 |
| v49_0400 | knowledge_ai | 6 | 54.72 | 99.74 | 33.33 |
| v49_0400 | reasoning | 5 | 20.00 | 94.00 | 20.00 |
| v49_0400 | safety_truthfulness | 5 | 51.00 | 100.00 | 20.00 |
| v49_0400 | writing | 4 | 41.67 | 100.00 | 25.00 |
| v49_0600 | creative | 3 | 30.00 | 95.00 | 0.00 |
| v49_0600 | factual | 5 | 80.00 | 100.00 | 80.00 |
| v49_0600 | identity | 4 | 81.67 | 100.00 | 50.00 |
| v49_0600 | instruction | 5 | 70.00 | 100.00 | 60.00 |
| v49_0600 | knowledge_ai | 6 | 54.72 | 100.00 | 33.33 |
| v49_0600 | reasoning | 5 | 20.00 | 97.00 | 20.00 |
| v49_0600 | safety_truthfulness | 5 | 60.00 | 100.00 | 60.00 |
| v49_0600 | writing | 4 | 45.83 | 100.00 | 25.00 |
| v49_0800 | creative | 3 | 30.00 | 95.00 | 0.00 |
| v49_0800 | factual | 5 | 80.00 | 100.00 | 80.00 |
| v49_0800 | identity | 4 | 81.67 | 100.00 | 50.00 |
| v49_0800 | instruction | 5 | 70.00 | 100.00 | 60.00 |
| v49_0800 | knowledge_ai | 6 | 46.39 | 98.11 | 33.33 |
| v49_0800 | reasoning | 5 | 20.00 | 97.00 | 20.00 |
| v49_0800 | safety_truthfulness | 5 | 60.00 | 97.00 | 60.00 |
| v49_0800 | writing | 4 | 64.17 | 100.00 | 25.00 |
| v49_best | creative | 3 | 30.00 | 95.00 | 0.00 |
| v49_best | factual | 5 | 80.00 | 100.00 | 80.00 |
| v49_best | identity | 4 | 81.67 | 100.00 | 50.00 |
| v49_best | instruction | 5 | 70.00 | 100.00 | 60.00 |
| v49_best | knowledge_ai | 6 | 46.39 | 98.11 | 33.33 |
| v49_best | reasoning | 5 | 20.00 | 97.00 | 20.00 |
| v49_best | safety_truthfulness | 5 | 60.00 | 97.00 | 60.00 |
| v49_best | writing | 4 | 64.17 | 100.00 | 25.00 |

## Reading The Scores

- Auto = 75% rule score + 25% degeneration quality score.
- Reference loss is teacher-forced loss on the fixed reference answers. Compare it only when suite hash, tokenizer, and system prompt are identical.
- Keyword and format checks do not establish semantic correctness. Use the generated blind review sheet before choosing a release checkpoint.
- Speed is comparable only on the same machine and software environment.
