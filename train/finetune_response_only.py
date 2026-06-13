import argparse
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from model.gpt import GPT, GPTConfig


def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        new_state_dict[k] = v
    return new_state_dict


def get_batch(x_data, y_data, batch_size, device):
    n = x_data.shape[0]
    ix = np.random.randint(0, n, size=(batch_size,))

    x = torch.from_numpy(np.array(x_data[ix], dtype=np.int64)).to(device)
    y = torch.from_numpy(np.array(y_data[ix], dtype=np.int64)).to(device)

    return x, y


def forward_full_logits(model, x):
    # 你的 GPT forward 很可能在 targets=None 时只返回最后一个 token 的 logits。
    # 这里传入 dummy targets，强制返回完整序列 logits。
    dummy_targets = torch.zeros_like(x)
    logits, _ = model(x, dummy_targets)
    return logits


@torch.no_grad()
def estimate_loss(model, train_x, train_y, val_x, val_y, batch_size, eval_iters, device, ctx):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = []

        x_data = train_x if split == "train" else val_x
        y_data = train_y if split == "train" else val_y

        for _ in range(eval_iters):
            x, y = get_batch(x_data, y_data, batch_size, device)

            with ctx:
                logits = forward_full_logits(model, x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                )

            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--init_from", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )

    print(f"device: {device}")
    print(f"loading data from: {args.data_dir}")

    train_x = np.load(os.path.join(args.data_dir, "train_x.npy"), mmap_mode="r")
    train_y = np.load(os.path.join(args.data_dir, "train_y.npy"), mmap_mode="r")
    val_x = np.load(os.path.join(args.data_dir, "val_x.npy"), mmap_mode="r")
    val_y = np.load(os.path.join(args.data_dir, "val_y.npy"), mmap_mode="r")

    print(f"train_x: {train_x.shape}, train_y: {train_y.shape}")
    print(f"val_x: {val_x.shape}, val_y: {val_y.shape}")

    print(f"loading checkpoint: {args.init_from}")
    ckpt = torch.load(args.init_from, map_location=device, weights_only=False)

    config = GPTConfig(**ckpt["config"])
    model = GPT(config)

    state_dict = clean_state_dict(ckpt["model"])
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.train()

    print(f"number of parameters: {model.get_num_params() / 1e6:.2f}M")

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
                model=model,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
                device=device,
                ctx=ctx,
            )

            elapsed = time.time() - t0
            print(
                f"iter {iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"time {elapsed:.1f}s",
                flush=True,
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                ckpt_out = {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                }

                out_path = os.path.join(args.out_dir, "ckpt.pt")
                torch.save(ckpt_out, out_path)
                print(f"saved checkpoint to {out_path}", flush=True)

        if iter_num == args.max_iters:
            break

        x, y = get_batch(train_x, train_y, args.batch_size, device)

        with ctx:
            logits = forward_full_logits(model, x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()


if __name__ == "__main__":
    main()
