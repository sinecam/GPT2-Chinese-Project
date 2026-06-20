# V4.7 Final Chinese Assistant

This directory is the standalone final project release. It does not depend on the training, data-building, historical-version, or benchmark scripts elsewhere in the repository.

## Contents

```text
final_release/
|-- app.py
|-- config.json
|-- engine.py
|-- export_release.py
|-- FINAL_REPORT.md
|-- model.py
|-- requirements.txt
|-- smoke_test.py
`-- static/
    `-- index.html
```

The exported model files are intentionally not committed:

```text
final_release/model/
|-- model.pt
|-- release_manifest.json
`-- spm_zh.model
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer
- CUDA-capable GPU recommended
- The final V4.7 checkpoint and 50k SentencePiece tokenizer

## One-Time Export

From the repository root:

```bash
source activate_env.sh

python final_release/export_release.py \
  --checkpoint checkpoints/V4.7_sp50k_gpt2small_sft_targeted_fix/ckpt.pt \
  --tokenizer artifacts/tokenizer_zh_50k/spm_zh.model
```

This creates a model-only checkpoint, copies the tokenizer, and writes SHA-256 hashes to `final_release/model/release_manifest.json`.

The exporter refuses to overwrite an existing release directory.

## Verify

Run one real end-to-end generation before starting the presentation service:

```bash
cd final_release
python smoke_test.py --device cuda --dtype bfloat16
```

A successful run prints `SMOKE TEST PASSED`, the checkpoint iteration, parameter count, device, prompt, and generated answer.

## Run

```bash
cd final_release
python app.py
```

Open:

```text
http://<server-address>:6006
```

Optional arguments:

```bash
python app.py \
  --checkpoint model/model.pt \
  --tokenizer model/spm_zh.model \
  --config config.json \
  --device cuda \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 6006
```

## API

Status:

```bash
curl http://127.0.0.1:6006/api/status
```

Chat:

```bash
curl -X POST http://127.0.0.1:6006/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请用一句话解释什么是大语言模型。","history":[]}'
```

## Demo Prompts

The interface includes the final seven demonstration prompts:

1. 请用一句话介绍你自己，并说明你能提供哪些帮助。
2. 请用一句话解释什么是大语言模型。
3. 请用两句话说明规律运动的好处，不要列点。
4. 写一封简短请假邮件，收件人是陈老师，因为感冒发烧请假一天。
5. 写一则100字以内的会议通知：7月8日下午14:00，地点为第二会议室，主题是项目进度。
6. 请概括为一句话：开发工作已经结束，当前正在进行上线前检查。
7. 改写得正式简洁：这个方案还可以但是有地方要改。

Each demo button starts a fresh conversation so previous messages cannot affect the result.

## Frozen Runtime Configuration

The final release uses deterministic decoding:

- temperature: 0
- maximum new tokens: 120
- top-k: 30
- top-p: 0.85
- repetition penalty: 1.15
- no-repeat n-gram size: 4

These settings live in `config.json`. Do not change them for the reported demonstration.

## Scope

This is a 124M-parameter Chinese GPT-2 project model. It is suitable for the final project report and controlled demonstration. It is not a production safety, medical, legal, financial, or real-time information system.
