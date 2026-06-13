import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpt import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test model forward/backward/generate for the Chinese GPT pipeline.")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab_size = args.vocab_size
    if args.tokenizer:
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
        vocab_size = int(sp.vocab_size())
        sample = "<user>\n你好，请介绍一下你自己。\n<sep>\n<assistant>\n"
        ids = sp.encode(sample, out_type=int)
        print(f"tokenizer vocab_size={vocab_size}, sample_tokens={len(ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=True,
    )
    model = GPT(config).to(device)
    model.train()

    x = torch.randint(0, vocab_size, (args.batch_size, args.block_size), device=device)
    y = torch.randint(0, vocab_size, (args.batch_size, args.block_size), device=device)
    _, loss = model(x, y)
    loss.backward()

    model.eval()
    out = model.generate(x[:1, :8], max_new_tokens=8, temperature=1.0, top_k=20)

    assert out.shape == (1, 16)
    assert torch.isfinite(loss).item()
    print(f"smoke test ok: device={device}, loss={loss.item():.4f}, generated_shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
