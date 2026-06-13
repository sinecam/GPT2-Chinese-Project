import argparse
import time
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from model.gpt import GPT, GPTConfig


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i : i + block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
        for i in ix
    ])

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, ctx, eval_iters=20):
    out = {}
    model.eval()

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/processed/sft")
    parser.add_argument("--init_from", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints/v4_sft")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_iters", type=int, default=20)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()

    torch.manual_seed(1337)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.memmap(Path(args.data_dir) / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(Path(args.data_dir) / "val.bin", dtype=np.uint16, mode="r")

    ckpt = torch.load(args.init_from, map_location=device)
    config = GPTConfig(**ckpt["config"])

    # SFT 可以用更短 block_size，但不能超过 checkpoint 配置
    config.block_size = min(args.block_size, config.block_size)

    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    best_val_loss = float("inf")
    t0 = time.time()

    for iter_num in range(args.max_iters + 1):
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                config.block_size,
                args.batch_size,
                device,
                ctx,
                eval_iters=args.eval_iters,
            )

            elapsed = time.time() - t0
            print(
                f"iter {iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"time {elapsed:.1f}s"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_ckpt = {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                torch.save(save_ckpt, out_dir / "ckpt.pt")
                print(f"saved checkpoint to {out_dir / 'ckpt.pt'}")

        xb, yb = get_batch(train_data, config.block_size, args.batch_size, device)

        with ctx:
            _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()


if __name__ == "__main__":
    main()
