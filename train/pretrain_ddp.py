import argparse
import os
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.gpt import GPT, GPTConfig, get_mini_config, get_gpt2_small_config, get_gpt2_medium_config


def parse_args():
    parser = argparse.ArgumentParser(description="GPT pretraining with DDP and resume support")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints/ddp")

    parser.add_argument("--config", type=str, default="gpt2_small", choices=["mini", "gpt2_small", "gpt2_medium"])
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=1024)

    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=20)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=8)

    # 新增：
    # --resume：严格续训，读取模型参数、iter_num、best_val_loss，如果有 optimizer 也读取
    # --init_from：只加载模型参数，从 iter 0 重新计数，优化器重新初始化
    parser.add_argument("--resume", type=str, default=None, help="resume training from checkpoint")
    parser.add_argument("--init_from", type=str, default=None, help="initialize model weights from checkpoint")

    return parser.parse_args()


def setup_ddp():
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        try:
            dist.init_process_group(
                backend="nccl",
                device_id=device,
            )
        except TypeError:
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        master_process = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True

    return ddp, rank, local_rank, world_size, master_process, device


def cleanup_ddp(ddp):
    if ddp:
        dist.destroy_process_group()


def get_config_by_name(name):
    if name == "mini":
        return get_mini_config()
    if name == "gpt2_small":
        return get_gpt2_small_config()
    if name == "gpt2_medium":
        return get_gpt2_medium_config()
    raise ValueError(f"Unknown config: {name}")

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        new_state_dict[k] = v
    return new_state_dict


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def get_batch(split, train_data, val_data, batch_size, block_size, device):
    data = train_data if split == "train" else val_data
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"{split}.bin is too small for block_size={block_size}")

    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i:i + block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
        for i in ix
    ])

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device, device_type):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters, device=device)
        for k in range(args.eval_iters):
            xb, yb = get_batch(
                split,
                train_data,
                val_data,
                args.batch_size,
                args.block_size,
                device,
            )
            with torch.amp.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=(device_type == "cuda"),
            ):
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


def save_checkpoint(path, raw_model, optimizer, config, iter_num, best_val_loss, args, master_process):
    if not master_process:
        return

    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)
    print(f"saved checkpoint to {path}")


def main():
    args = parse_args()

    if args.resume is not None and args.init_from is not None:
        raise ValueError("只能选择 --resume 或 --init_from 其中一个，不能同时使用。")

    ddp, rank, local_rank, world_size, master_process, device = setup_ddp()
    device_type = "cuda" if device.type == "cuda" else "cpu"

    if master_process:
        print(f"DDP: {ddp}, world_size: {world_size}")
        print(f"device: {device}")
        print(f"micro batch size per GPU: {args.batch_size}")
        print(f"grad_accum_steps: {args.grad_accum_steps}")
        print(
            "effective tokens/update: "
            f"{world_size * args.batch_size * args.block_size * args.grad_accum_steps}"
        )

    torch.manual_seed(1337 + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(1337 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if master_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    if ddp:
        dist.barrier()

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.bin: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val.bin: {val_path}")

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    if master_process:
        print(f"train tokens: {len(train_data):,}")
        print(f"val tokens: {len(val_data):,}")

    start_iter = 0
    best_val_loss = float("inf")
    ckpt = None

    if args.resume is not None:
        if master_process:
            print(f"resuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, device)

        if "config" in ckpt:
            config = GPTConfig(**ckpt["config"])
        else:
            config = get_config_by_name(args.config)
            if args.vocab_size is not None:
                config.vocab_size = args.vocab_size

        start_iter = int(ckpt.get("iter_num", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    elif args.init_from is not None:
        if master_process:
            print(f"initializing from checkpoint: {args.init_from}")
        ckpt = load_checkpoint(args.init_from, device)

        if "config" in ckpt:
            config = GPTConfig(**ckpt["config"])
        else:
            config = get_config_by_name(args.config)
            if args.vocab_size is not None:
                config.vocab_size = args.vocab_size

        start_iter = 0
        best_val_loss = float("inf")

    else:
        config = get_config_by_name(args.config)
        if args.vocab_size is not None:
            config.vocab_size = args.vocab_size
        config.block_size = args.block_size

    if args.block_size > config.block_size:
        raise ValueError(
            f"args.block_size={args.block_size} 大于模型最大 block_size={config.block_size}，"
            "请减小 --block_size，或者重新训练更大 block_size 的模型。"
        )

    model = GPT(config)
    raw_model = model

    if ckpt is not None:
        state_dict = clean_state_dict(ckpt["model"])
        missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)

        if master_process:
            print(f"loaded model weights")
            if missing:
                print(f"missing keys: {missing[:10]} ... total={len(missing)}")
            if unexpected:
                print(f"unexpected keys: {unexpected[:10]} ... total={len(unexpected)}")

    raw_model.to(device)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    if args.resume is not None and ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if master_process:
            print("loaded optimizer state")
    elif args.resume is not None and master_process:
        print("optimizer state not found in checkpoint, using a fresh optimizer")

    model = raw_model

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    if master_process:
        print(f"number of parameters: {raw_model.get_num_params() / 1e6:.2f}M")
        print(f"start_iter: {start_iter}")
        print(f"max_iters: {args.max_iters}")
        print(f"best_val_loss: {best_val_loss}")

    t0 = time.time()

    for iter_num in range(start_iter, args.max_iters + 1):
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args,
                device,
                device_type,
            )

            if master_process:
                elapsed = time.time() - t0
                print(
                    f"iter {iter_num}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}, "
                    f"time {elapsed:.1f}s"
                )

                latest_path = out_dir / "latest.pt"
                save_checkpoint(
                    latest_path,
                    raw_model,
                    optimizer,
                    config,
                    iter_num,
                    best_val_loss,
                    args,
                    master_process,
                )

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    best_path = out_dir / "ckpt.pt"
                    save_checkpoint(
                        best_path,
                        raw_model,
                        optimizer,
                        config,
                        iter_num,
                        best_val_loss,
                        args,
                        master_process,
                    )

            if ddp:
                dist.barrier()

        if iter_num == args.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(args.grad_accum_steps):
            xb, yb = get_batch(
                "train",
                train_data,
                val_data,
                args.batch_size,
                args.block_size,
                device,
            )

            if ddp:
                model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1

            with torch.amp.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=(device_type == "cuda"),
            ):
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum_steps

            loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

        optimizer.step()

    cleanup_ddp(ddp)


if __name__ == "__main__":
    main()
