import argparse
import sys
from pathlib import Path

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


def format_history(history: list[tuple[str, str]], system_prompt: str | None) -> str:
    parts = []
    if system_prompt:
        parts.append(f"<system>\n{system_prompt}")
    for user_text, assistant_text in history:
        parts.append(f"<user>\n{user_text}")
        if assistant_text:
            parts.append(f"<assistant>\n{assistant_text}")
    parts.append("<assistant>\n")
    return "\n<sep>\n".join(parts)


def encode_prompt(sp: spm.SentencePieceProcessor, text: str) -> list[int]:
    ids = sp.encode(text, out_type=int)
    if sp.bos_id() >= 0:
        ids = [int(sp.bos_id())] + ids
    return ids


@torch.no_grad()
def generate(
    model: GPT,
    input_ids: list[int],
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    stop_ids = {int(sp.eos_id())}
    for piece in ("<user>", "<system>", "<sep>"):
        piece_id = int(sp.piece_to_id(piece))
        if piece_id >= 0:
            stop_ids.add(piece_id)

    idx = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
    generated = []
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

        token_id = int(next_id.item())
        if token_id in stop_ids:
            break
        generated.append(token_id)
        idx = torch.cat((idx, next_id), dim=1)

    text = sp.decode(generated)
    return text.replace("<assistant>", "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Chinese chat with a fine-tuned GPT-2 checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="你是一个乐于助人的中文助手。")
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = load_model(args.ckpt, device)

    history: list[tuple[str, str]] = []
    print("输入 exit / quit / q 结束。")
    while True:
        try:
            user_text = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"exit", "quit", "q"}:
            break
        if not user_text:
            continue

        history.append((user_text, ""))
        visible_history = history[-args.max_turns :]
        prompt = format_history(visible_history, args.system_prompt)
        input_ids = encode_prompt(sp, prompt)
        answer = generate(
            model,
            input_ids,
            sp,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        history[-1] = (user_text, answer)
        print(f"助手：{answer}")


if __name__ == "__main__":
    main()
