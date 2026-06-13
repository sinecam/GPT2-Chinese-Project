import argparse
import torch
import tiktoken

from model.gpt import GPT, GPTConfig


def safe_decode(enc, token_ids):
    token_bytes = b"".join(
        enc.decode_single_token_bytes(int(t)) for t in token_ids
    )
    return token_bytes.decode("utf-8", errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/v4_sft/ckpt.pt")
    parser.add_argument("--prompt", type=str, default="User: What is Transformer?\nAssistant:")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--stop_eot", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**ckpt["config"])

    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    # 支持命令行里的 \n 自动转成真正换行
    prompt = args.prompt.replace("\\n", "\n")

    ids = enc.encode_ordinary(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    text = safe_decode(enc, y[0].tolist())

    if args.stop_eot and "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0]

    print(text)


if __name__ == "__main__":
    main()
