import argparse
from pathlib import Path
import numpy as np
import tiktoken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, default="data/processed/pretrain/train.bin")
    parser.add_argument("--num_tokens", type=int, default=300)
    args = parser.parse_args()

    path = Path(args.bin)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.memmap(path, dtype=np.uint16, mode="r")

    print(f"File: {path}")
    print(f"Total tokens: {len(data)}")
    print(f"First {args.num_tokens} token ids:")
    print(data[:args.num_tokens].tolist())

    enc = tiktoken.get_encoding("gpt2")
    text = enc.decode(data[:args.num_tokens].astype(int).tolist())

    print("\nDecoded text:")
    print(text)


if __name__ == "__main__":
    main()
