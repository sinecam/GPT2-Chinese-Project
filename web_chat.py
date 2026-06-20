import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Literal

import sentencepiece as spm
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpt import GPT, GPTConfig


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    system_prompt: str = "你是一个中文AI助手。回答要简洁、准确。身份问题只回答你是中文AI助手，不要编造姓名、职业或真实人物身份。"
    max_turns: int = Field(6, ge=1, le=20)
    max_new_tokens: int = Field(120, ge=1, le=600)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_k: int = Field(30, ge=0, le=500)
    top_p: float = Field(0.85, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    no_repeat_ngram_size: int = Field(4, ge=0, le=12)


class ChatResponse(BaseModel):
    answer: str


class ModelState:
    def __init__(self) -> None:
        self.model: GPT | None = None
        self.sp: spm.SentencePieceProcessor | None = None
        self.device = torch.device("cpu")
        self.ckpt_path = ""
        self.tokenizer_path = ""
        self.lock = threading.Lock()


STATE = ModelState()
app = FastAPI(title="V4.7 Chinese Assistant", version="4.7")


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        cleaned[key] = value
    return cleaned


def load_model(ckpt_path: str, tokenizer_path: str, device_name: str) -> None:
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    missing, unexpected = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    model.eval()

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    if int(sp.vocab_size()) != int(config.vocab_size):
        raise RuntimeError(f"tokenizer vocab_size={sp.vocab_size()} does not match checkpoint vocab_size={config.vocab_size}")

    STATE.model = model
    STATE.sp = sp
    STATE.device = device
    STATE.ckpt_path = ckpt_path
    STATE.tokenizer_path = tokenizer_path


def valid_piece_id(piece_id: int) -> bool:
    return piece_id is not None and int(piece_id) >= 0


def collect_stop_ids(sp: spm.SentencePieceProcessor) -> set[int]:
    stop_ids = set()
    if valid_piece_id(sp.eos_id()):
        stop_ids.add(int(sp.eos_id()))
    for piece in ("<user>", "<assistant>", "<system>", "<sep>"):
        piece_id = int(sp.piece_to_id(piece))
        if piece_id >= 0:
            stop_ids.add(piece_id)
    return stop_ids


def collect_bad_ids(sp: spm.SentencePieceProcessor, stop_ids: set[int]) -> set[int]:
    bad_ids = set()
    for piece_id in (sp.unk_id(), sp.bos_id(), sp.pad_id()):
        if valid_piece_id(piece_id) and int(piece_id) not in stop_ids:
            bad_ids.add(int(piece_id))
    return bad_ids


def ban_token_ids(logits: torch.Tensor, token_ids: set[int]) -> None:
    valid_ids = [token_id for token_id in set(token_ids) if 0 <= token_id < logits.size(-1)]
    if valid_ids:
        logits[:, valid_ids] = -float("inf")


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> None:
    if penalty <= 1.0 or not generated_ids:
        return
    for token_id in set(generated_ids):
        if 0 <= token_id < logits.size(-1):
            score = logits[:, token_id].clone()
            logits[:, token_id] = torch.where(score < 0, score * penalty, score / penalty)


def banned_ngram_tokens(generated_ids: list[int], no_repeat_ngram_size: int) -> set[int]:
    if no_repeat_ngram_size < 2 or len(generated_ids) + 1 < no_repeat_ngram_size:
        return set()
    prefix = tuple(generated_ids[-(no_repeat_ngram_size - 1) :])
    banned = set()
    for i in range(len(generated_ids) - no_repeat_ngram_size + 1):
        ngram = tuple(generated_ids[i : i + no_repeat_ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def top_k_filter(logits: torch.Tensor, top_k: int) -> None:
    if top_k <= 0:
        return
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < values[:, [-1]]] = -float("inf")


def top_p_filter(logits: torch.Tensor, top_p: float) -> None:
    if top_p <= 0.0 or top_p >= 1.0:
        return
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_remove, -float("inf"))
    logits.scatter_(1, sorted_indices, sorted_logits)


def clean_answer(text: str) -> str:
    for marker in ("<user>", "<assistant>", "<system>", "<sep>", "<bos>", "<eos>"):
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()


def format_prompt(history: list[ChatMessage], user_message: str, system_prompt: str, max_turns: int) -> str:
    parts = []
    if system_prompt.strip():
        parts.append(f"<system>\n{system_prompt.strip()}")

    trimmed_history = history[-max_turns * 2 :]
    for item in trimmed_history:
        content = item.content.strip()
        if not content:
            continue
        if item.role == "user":
            parts.append(f"<user>\n{content}")
        else:
            parts.append(f"<assistant>\n{content}")

    parts.append(f"<user>\n{user_message.strip()}")
    parts.append("<assistant>\n")
    return "\n<sep>\n".join(parts)


def encode_prompt(sp: spm.SentencePieceProcessor, prompt: str) -> list[int]:
    ids = sp.encode(prompt, out_type=int)
    if sp.bos_id() >= 0:
        ids = [int(sp.bos_id())] + ids
    return ids


@torch.inference_mode()
def generate_reply(request: ChatRequest) -> str:
    if STATE.model is None or STATE.sp is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    model = STATE.model
    sp = STATE.sp
    prompt = format_prompt(request.history, request.message, request.system_prompt, request.max_turns)
    input_ids = encode_prompt(sp, prompt)
    idx = torch.tensor(input_ids, dtype=torch.long, device=STATE.device)[None, ...]

    stop_ids = collect_stop_ids(sp)
    bad_ids = collect_bad_ids(sp, stop_ids)
    generated: list[int] = []

    for _ in range(request.max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        ban_token_ids(logits, bad_ids)
        ban_token_ids(logits, banned_ngram_tokens(generated, request.no_repeat_ngram_size))
        apply_repetition_penalty(logits, generated, request.repetition_penalty)

        if request.temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / request.temperature
            top_k_filter(logits, request.top_k)
            top_p_filter(logits, request.top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_id = int(next_id.item())
        if token_id in stop_ids:
            break
        generated.append(token_id)
        idx = torch.cat((idx, next_id), dim=1)

    return clean_answer(sp.decode(generated))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


@app.get("/api/status")
def status() -> dict[str, str]:
    return {
        "device": str(STATE.device),
        "checkpoint": STATE.ckpt_path,
        "tokenizer": STATE.tokenizer_path,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    with STATE.lock:
        answer = generate_reply(request)
    return ChatResponse(answer=answer)


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>V4.7 中文助手</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #637083;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --user: #e7f0ff;
      --assistant: #f2f6f3;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(260px, 320px) minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 20px;
      overflow-y: auto;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
    }
    header {
      background: rgba(255,255,255,0.92);
      border-bottom: 1px solid var(--line);
      padding: 16px 22px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.3;
    }
    .status {
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .section { margin-bottom: 18px; }
    .section h2 {
      margin: 0 0 10px;
      font-size: 14px;
      color: var(--muted);
      font-weight: 700;
    }
    label {
      display: block;
      margin: 10px 0 5px;
      color: var(--muted);
      font-size: 13px;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      font: inherit;
      outline: none;
    }
    textarea { min-height: 96px; resize: vertical; line-height: 1.55; }
    input:focus, textarea:focus { border-color: var(--accent); }
    .row { display: grid; grid-template-columns: 1fr 72px; gap: 8px; align-items: center; }
    .quick {
      display: grid;
      gap: 8px;
    }
    .quick button, .ghost {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      text-align: left;
      cursor: pointer;
      font: inherit;
    }
    .quick button:hover, .ghost:hover { border-color: var(--accent); }
    .chat {
      padding: 20px 22px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .bubble {
      max-width: 820px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 13px;
      line-height: 1.7;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .bubble.user {
      align-self: flex-end;
      background: var(--user);
    }
    .bubble.assistant {
      align-self: flex-start;
      background: var(--assistant);
    }
    .composer {
      background: var(--panel);
      border-top: 1px solid var(--line);
      padding: 14px 22px 18px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 112px;
      gap: 10px;
    }
    .composer textarea { min-height: 54px; max-height: 180px; }
    .send {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      font-size: 15px;
    }
    .send:hover { background: var(--accent-dark); }
    .send:disabled { opacity: .55; cursor: not-allowed; }
    .error { color: var(--danger); font-size: 13px; margin-top: 8px; }
    @media (max-width: 860px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      main { min-height: 70vh; }
      .composer { grid-template-columns: 1fr; }
      .send { height: 44px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="section">
        <h2>系统提示</h2>
        <textarea id="systemPrompt">你是一个中文AI助手。回答要简洁、准确。身份问题只回答你是中文AI助手，不要编造姓名、职业或真实人物身份。</textarea>
      </div>
      <div class="section">
        <h2>生成参数</h2>
        <label>最大 token</label>
        <div class="row"><input id="maxNewTokens" type="range" min="20" max="400" value="120" step="10"><input id="maxNewTokensValue" value="120"></div>
        <label>Temperature</label>
        <div class="row"><input id="temperature" type="range" min="0" max="1.2" value="0" step="0.05"><input id="temperatureValue" value="0"></div>
        <label>Top-k</label>
        <div class="row"><input id="topK" type="range" min="0" max="120" value="30" step="1"><input id="topKValue" value="30"></div>
        <label>Top-p</label>
        <div class="row"><input id="topP" type="range" min="0.1" max="1" value="0.85" step="0.05"><input id="topPValue" value="0.85"></div>
        <label>重复惩罚</label>
        <div class="row"><input id="repPenalty" type="range" min="1" max="1.6" value="1.15" step="0.01"><input id="repPenaltyValue" value="1.15"></div>
        <label>禁用重复 n-gram</label>
        <div class="row"><input id="ngram" type="range" min="0" max="8" value="4" step="1"><input id="ngramValue" value="4"></div>
      </div>
      <div class="section quick">
        <h2>快速测试</h2>
        <button data-prompt="请用一句话介绍你自己，并说明你能提供哪些帮助。">自我介绍</button>
        <button data-prompt="请用一句话解释什么是大语言模型。">解释大语言模型</button>
        <button data-prompt="请用两句话说明规律运动的好处，不要列点。">规律运动的好处</button>
        <button data-prompt="写一封简短请假邮件，收件人是陈老师，因为感冒发烧请假一天。">请假邮件</button>
        <button data-prompt="写一则100字以内的会议通知：7月8日下午14:00，地点为第二会议室，主题是项目进度。">会议通知</button>
        <button data-prompt="请概括为一句话：开发工作已经结束，当前正在进行上线前检查。">一句话摘要</button>
        <button data-prompt="改写得正式简洁：这个方案还可以但是有地方要改。">正式改写</button>
      </div>
      <button id="clear" class="ghost" type="button">清空对话</button>
      <div id="error" class="error"></div>
    </aside>
    <main>
      <header>
        <h1>V4.7 中文助手</h1>
        <div id="status" class="status">正在连接模型...</div>
      </header>
      <div id="chat" class="chat"></div>
      <form id="composer" class="composer">
        <textarea id="message" placeholder="输入你的问题，按 Ctrl+Enter 发送"></textarea>
        <button id="send" class="send" type="submit">发送</button>
      </form>
    </main>
  </div>
<script>
const history = [];
const chat = document.getElementById('chat');
const errorBox = document.getElementById('error');
const sendButton = document.getElementById('send');
const messageBox = document.getElementById('message');

function bindRange(id) {
  const slider = document.getElementById(id);
  const value = document.getElementById(id + 'Value');
  slider.addEventListener('input', () => value.value = slider.value);
  value.addEventListener('input', () => slider.value = value.value);
}
['maxNewTokens','temperature','topK','topP','repPenalty','ngram'].forEach(bindRange);

function addBubble(role, content) {
  const node = document.createElement('div');
  node.className = `bubble ${role}`;
  node.textContent = content;
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
}

function payload(message) {
  return {
    message,
    history,
    system_prompt: document.getElementById('systemPrompt').value,
    max_new_tokens: Number(document.getElementById('maxNewTokensValue').value),
    temperature: Number(document.getElementById('temperatureValue').value),
    top_k: Number(document.getElementById('topKValue').value),
    top_p: Number(document.getElementById('topPValue').value),
    repetition_penalty: Number(document.getElementById('repPenaltyValue').value),
    no_repeat_ngram_size: Number(document.getElementById('ngramValue').value)
  };
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message) return;
  errorBox.textContent = '';
  addBubble('user', message);
  messageBox.value = '';
  sendButton.disabled = true;
  sendButton.textContent = '生成中';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload(message))
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const answer = data.answer || '';
    addBubble('assistant', answer || '没有生成内容。');
    history.push({role: 'user', content: message});
    history.push({role: 'assistant', content: answer});
  } catch (err) {
    errorBox.textContent = String(err.message || err);
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = '发送';
    messageBox.focus();
  }
}

document.getElementById('composer').addEventListener('submit', (event) => {
  event.preventDefault();
  sendMessage(messageBox.value);
});
messageBox.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.key === 'Enter') sendMessage(messageBox.value);
});
document.querySelectorAll('[data-prompt]').forEach(button => {
  button.addEventListener('click', () => sendMessage(button.dataset.prompt));
});
document.getElementById('clear').addEventListener('click', () => {
  history.length = 0;
  chat.innerHTML = '';
  errorBox.textContent = '';
});

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('status').textContent = `设备：${data.device} ｜ checkpoint：${data.checkpoint}`;
  } catch (_) {
    document.getElementById('status').textContent = '模型状态读取失败';
  }
}
loadStatus();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a browser chat UI for the SentencePiece Chinese GPT model.")
    parser.add_argument("--ckpt", type=str, default="releases/V4.7_final/model.pt")
    parser.add_argument("--tokenizer", type=str, default="releases/V4.7_final/spm_zh.model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_model(args.ckpt, args.tokenizer, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
