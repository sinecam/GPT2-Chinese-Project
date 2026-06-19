import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpt import GPT, GPTConfig


IGNORE_INDEX = -100


def setup_ddp() -> tuple[bool, int, int, int, torch.device, bool]:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if not ddp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device, True

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, local_rank, world_size, device, rank == 0


def cleanup_ddp(ddp: bool) -> None:
    if ddp:
        dist.destroy_process_group()


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        cleaned[key] = value
    return cleaned


def load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_arrays(data_dir: Path) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    paths = [data_dir / name for name in ("train_x.npy", "train_y.npy", "val_x.npy", "val_y.npy")]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing response-only data file: {path}")
    return tuple(np.load(path, mmap_mode="r") for path in paths)  # type: ignore[return-value]


def get_batch(x_data: np.memmap, y_data: np.memmap, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = x_data.shape[0]
    if n <= 0:
        raise ValueError("empty dataset")
    ix = np.random.randint(0, n, size=(batch_size,))
    x = torch.from_numpy(np.asarray(x_data[ix], dtype=np.int64)).to(device, non_blocking=True)
    y = torch.from_numpy(np.asarray(y_data[ix], dtype=np.int64)).to(device, non_blocking=True)
    return x, y


def full_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    dummy_targets = torch.zeros_like(x)
    logits, _ = model(x, dummy_targets)
    return logits


def response_only_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_x: np.memmap,
    train_y: np.memmap,
    val_x: np.memmap,
    val_y: np.memmap,
    args: argparse.Namespace,
    device: torch.device,
    ctx,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, x_data, y_data in (("train", train_x, train_y), ("val", val_x, val_y)):
        losses = torch.zeros(args.eval_iters, device=device)
        for k in range(args.eval_iters):
            x, y = get_batch(x_data, y_data, args.batch_size, device)
            with ctx:
                logits = full_logits(model, x)
                loss = response_only_loss(logits, y)
            losses[k] = loss.detach()
        out[split] = float(losses.mean().item())
    model.train()
    return out


def get_lr(iter_num: int, args: argparse.Namespace) -> float:
    if iter_num < args.warmup_iters:
        return args.learning_rate * (iter_num + 1) / max(1, args.warmup_iters)
    if iter_num > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (iter_num - args.warmup_iters) / max(1, args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def load_checkpoint(path: str | None, device: torch.device) -> dict | None:
    if not path:
        return None
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(
    path: Path,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    iter_num: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Response-only SFT for the SentencePiece Chinese GPT-2 model.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--init_from", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=8, help="micro batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_iters", type=int, default=300)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.init_from:
        raise ValueError("Use only one of --resume or --init_from.")
    if not args.resume and not args.init_from:
        raise ValueError("Provide --init_from for a fresh run or --resume to continue.")

    ddp, rank, local_rank, world_size, device, master = setup_ddp()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type == "cuda" and args.dtype != "float32"))
        if device_type == "cuda"
        else nullcontext()
    )

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir)
    meta = load_meta(data_dir)
    train_x, train_y, val_x, val_y = load_arrays(data_dir)
    block_size = int(meta.get("block_size", train_x.shape[1]))

    out_dir = Path(args.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    ckpt = load_checkpoint(args.resume or args.init_from, device)
    config = GPTConfig(**ckpt["config"])
    if block_size > config.block_size:
        raise ValueError(f"data block_size={block_size} exceeds model block_size={config.block_size}")

    model = GPT(config).to(device)
    raw_model = model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    best_val_loss = float("inf")
    missing, unexpected = raw_model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    if master:
        print(f"loaded checkpoint weights from {args.resume or args.init_from}")
        if missing:
            print(f"missing keys: {missing[:5]} total={len(missing)}")
        if unexpected:
            print(f"unexpected keys: {unexpected[:5]} total={len(unexpected)}")
    if args.resume:
        start_iter = int(ckpt.get("iter_num", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module

    tokens_per_iter = world_size * args.batch_size * block_size * args.grad_accum_steps
    if master:
        print(f"device={device}, ddp={ddp}, world_size={world_size}")
        print(f"train examples={train_x.shape[0]:,}, val examples={val_x.shape[0]:,}, block_size={block_size}")
        print(f"tokens/iter={tokens_per_iter:,}, start_iter={start_iter}, max_iters={args.max_iters}")

    t0 = time.time()
    running_loss = 0.0
    try:
        for iter_num in range(start_iter, args.max_iters + 1):
            lr = get_lr(iter_num, args)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if iter_num % args.eval_interval == 0:
                losses = estimate_loss(model, train_x, train_y, val_x, val_y, args, device, ctx)
                if master:
                    print(
                        f"iter {iter_num}: train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}, lr {lr:.2e}",
                        flush=True,
                    )
                    save_checkpoint(out_dir / "latest.pt", raw_model, optimizer, config, iter_num, best_val_loss, args)
                    if losses["val"] < best_val_loss:
                        best_val_loss = losses["val"]
                        save_checkpoint(out_dir / "ckpt.pt", raw_model, optimizer, config, iter_num, best_val_loss, args)
                        print(f"saved best checkpoint to {out_dir / 'ckpt.pt'}", flush=True)
                if ddp:
                    dist.barrier()

            if iter_num == args.max_iters:
                break

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro_step in range(args.grad_accum_steps):
                x, y = get_batch(train_x, train_y, args.batch_size, device)
                if ddp:
                    model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1
                with ctx:
                    logits = full_logits(model, x)
                    loss = response_only_loss(logits, y) / args.grad_accum_steps
                total_loss += float(loss.detach().item())
                loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss = total_loss if running_loss == 0 else 0.9 * running_loss + 0.1 * total_loss
            if master and iter_num % args.log_interval == 0:
                dt = time.time() - t0
                toks_per_sec = tokens_per_iter * max(1, args.log_interval) / max(dt, 1e-6)
                print(f"iter {iter_num}: loss {running_loss:.4f}, lr {lr:.2e}, tok/s {toks_per_sec:,.0f}", flush=True)
                t0 = time.time()
    finally:
        cleanup_ddp(ddp)


if __name__ == "__main__":
    main()
