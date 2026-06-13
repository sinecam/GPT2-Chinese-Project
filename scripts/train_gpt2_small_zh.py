import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpt import GPT, GPTConfig


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


def load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_bin(path: Path, dtype_name: str) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return np.memmap(path, dtype=np.dtype(dtype_name), mode="r")


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"dataset is too small for block_size={block_size}: {len(data)} tokens")

    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([
        torch.from_numpy(np.asarray(data[i : i + block_size], dtype=np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(np.asarray(data[i + 1 : i + 1 + block_size], dtype=np.int64))
        for i in ix
    ])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: np.memmap,
    val_data: np.memmap,
    args: argparse.Namespace,
    device: torch.device,
    device_type: str,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_iters, device=device)
        for k in range(args.eval_iters):
            xb, yb = get_batch(data, args.block_size, args.batch_size, device)
            with torch.amp.autocast(
                device_type=device_type,
                dtype=amp_dtype,
                enabled=(device_type == "cuda" and args.dtype != "float32"),
            ):
                _, loss = model(xb, yb)
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


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        cleaned[key] = value
    return cleaned


def build_config(args: argparse.Namespace, vocab_size: int, ckpt: dict | None = None) -> GPTConfig:
    if ckpt and "config" in ckpt:
        return GPTConfig(**ckpt["config"])
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )


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
    parser = argparse.ArgumentParser(description="Train a GPT-2 Small style Chinese dialogue language model.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints/zh_gpt2small")

    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch_size", type=int, default=8, help="micro batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=200000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--lr_decay_iters", type=int, default=200000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--resume", type=str, default=None, help="Continue optimizer/model state from checkpoint.")
    parser.add_argument("--init_from", type=str, default=None, help="Initialize model weights from checkpoint, reset optimizer.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.init_from:
        raise ValueError("Use only one of --resume or --init_from.")

    ddp, rank, local_rank, world_size, device, master = setup_ddp()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir)
    meta = load_meta(data_dir)
    data_dtype = meta.get("dtype", "uint16")
    train_data = load_bin(data_dir / "train.bin", data_dtype)
    val_data = load_bin(data_dir / "val.bin", data_dtype)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = int(sp.vocab_size())

    out_dir = Path(args.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    ckpt = load_checkpoint(args.resume or args.init_from, device)
    config = build_config(args, vocab_size, ckpt)
    if args.block_size > config.block_size:
        raise ValueError(f"--block_size={args.block_size} exceeds model block_size={config.block_size}")
    if config.vocab_size != vocab_size and master:
        print(f"warning: checkpoint vocab_size={config.vocab_size}, tokenizer vocab_size={vocab_size}")

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
    if ckpt is not None:
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

    if args.compile:
        model = torch.compile(model)
        raw_model = model

    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module

    tokens_per_iter = world_size * args.batch_size * args.block_size * args.grad_accum_steps
    if master:
        print(f"device={device}, ddp={ddp}, world_size={world_size}")
        print(f"train tokens={len(train_data):,}, val tokens={len(val_data):,}, dtype={data_dtype}")
        print(f"vocab_size={vocab_size}, block_size={args.block_size}, tokens/iter={tokens_per_iter:,}")
        print(f"start_iter={start_iter}, max_iters={args.max_iters}")

    t0 = time.time()
    running_loss = 0.0

    try:
        for iter_num in range(start_iter, args.max_iters + 1):
            lr = get_lr(iter_num, args)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if iter_num % args.eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data, args, device, device_type, amp_dtype)
                if master:
                    print(
                        f"iter {iter_num}: train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}, ppl {math.exp(min(losses['val'], 20)):.2f}, lr {lr:.2e}"
                    )
                    save_checkpoint(out_dir / "latest.pt", raw_model, optimizer, config, iter_num, best_val_loss, args)
                    if losses["val"] < best_val_loss:
                        best_val_loss = losses["val"]
                        save_checkpoint(out_dir / "ckpt.pt", raw_model, optimizer, config, iter_num, best_val_loss, args)
                        print(f"saved best checkpoint to {out_dir / 'ckpt.pt'}")
                if ddp:
                    dist.barrier()

            if iter_num == args.max_iters:
                break

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro_step in range(args.grad_accum_steps):
                xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
                if ddp:
                    model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1
                with torch.amp.autocast(
                    device_type=device_type,
                    dtype=amp_dtype,
                    enabled=(device_type == "cuda" and args.dtype != "float32"),
                ):
                    _, loss = model(xb, yb)
                    loss = loss / args.grad_accum_steps
                total_loss += float(loss.detach().item())
                loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss = total_loss if running_loss == 0 else 0.9 * running_loss + 0.1 * total_loss
            if master and iter_num % args.log_interval == 0:
                dt = time.time() - t0
                toks_per_sec = tokens_per_iter * max(1, args.log_interval) / max(dt, 1e-6)
                print(f"iter {iter_num}: loss {running_loss:.4f}, lr {lr:.2e}, tok/s {toks_per_sec:,.0f}")
                t0 = time.time()
    finally:
        cleanup_ddp(ddp)


if __name__ == "__main__":
    main()
