import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpt import GPT, GPTConfig


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        cleaned[key] = value
    return cleaned


def load_model(ckpt_path: str, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.to(device)
    model.eval()
    return model


def load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"dataset is too small for block_size={block_size}: {len(data)} tokens")
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(np.asarray(data[i : i + block_size], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.asarray(data[i + 1 : i + 1 + block_size], dtype=np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_loss(model: GPT, data: np.memmap, args: argparse.Namespace, device: torch.device) -> tuple[float, float]:
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(min(mean_loss, 20))


def build_prompt(sp: spm.SentencePieceProcessor, user_prompt: str, system_prompt: str | None) -> list[int]:
    parts = []
    if system_prompt:
        parts.append(f"<system>\n{system_prompt}")
    parts.append(f"<user>\n{user_prompt}")
    parts.append("<assistant>\n")
    text = "\n<sep>\n".join(parts)
    ids = sp.encode(text, out_type=int)
    if sp.bos_id() >= 0:
        ids = [int(sp.bos_id())] + ids
    return ids


@torch.no_grad()
def generate_ids(
    model: GPT,
    ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    stop_ids: set[int],
    device: torch.device,
) -> list[int]:
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
        if int(next_id.item()) in stop_ids:
            break
    return idx[0].tolist()


def clean_answer(text: str) -> str:
    for marker in ("<user>", "<system>", "<sep>", "<bos>", "<eos>"):
        if marker in text:
            text = text.split(marker)[0]
    return text.replace("<assistant>", "").strip()


def iter_prompts(args: argparse.Namespace) -> Iterable[str]:
    if args.prompts_file:
        with Path(args.prompts_file).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield str(value.get("prompt") or value.get("instruction") or value.get("question") or line)
                    else:
                        yield str(value)
                except json.JSONDecodeError:
                    yield line
    elif args.prompt:
        yield args.prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chinese GPT checkpoint with loss/PPL and sample generations.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="请用三句话解释什么是大语言模型。")
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default="你是一个乐于助人的中文助手。")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = load_model(args.ckpt, device)

    if args.data_dir:
        data_dir = Path(args.data_dir)
        meta = load_meta(data_dir)
        dtype_name = meta.get("dtype", "uint16")
        val_data = np.memmap(data_dir / "val.bin", dtype=np.dtype(dtype_name), mode="r")
        loss, ppl = evaluate_loss(model, val_data, args, device)
        print(f"val loss: {loss:.4f}")
        print(f"val ppl:  {ppl:.2f}")

    stop_ids = {int(sp.eos_id())}
    for piece in ("<user>", "<system>", "<sep>"):
        piece_id = int(sp.piece_to_id(piece))
        if piece_id >= 0:
            stop_ids.add(piece_id)

    for prompt in iter_prompts(args):
        input_ids = build_prompt(sp, prompt, args.system_prompt)
        output_ids = generate_ids(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_ids=stop_ids,
            device=device,
        )
        answer_ids = output_ids[len(input_ids) :]
        answer = clean_answer(sp.decode(answer_ids))
        print("\n[Prompt]")
        print(prompt)
        print("[Answer]")
        print(answer)


if __name__ == "__main__":
    main()
