import os
from pathlib import Path
from dataclasses import asdict

import torch
from transformers import GPT2LMHeadModel

from model.gpt import GPT, GPTConfig, get_gpt2_medium_config


def clean_hf_key(k):
    # HuggingFace GPT2LMHeadModel 的 transformer 部分前缀是 transformer.
    return k


def main():
    out_dir = Path("checkpoints/V2.1_official_gpt2medium")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading official HuggingFace GPT-2 Medium...")
    hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-medium")
    hf_sd = hf_model.state_dict()

    print("building our GPT-2 Medium model...")
    config = get_gpt2_medium_config()
    model = GPT(config)
    sd = model.state_dict()

    # HuggingFace 的 Conv1D 权重方向和 nn.Linear 相反，需要转置
    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    # HuggingFace 多出来的 attention mask buffer，不需要复制
    ignore_keys = [
        ".attn.bias",
        ".attn.masked_bias",
    ]

    copied = 0
    skipped = []

    for k in sd.keys():
        if k == "lm_head.weight":
            # 你自己的模型里 lm_head 和 wte 权重共享，一般不用单独复制
            continue

        if any(x in k for x in ignore_keys):
            continue

        hf_k = k

        if hf_k not in hf_sd:
            skipped.append(k)
            continue

        if any(hf_k.endswith(w) for w in transposed):
            if sd[k].shape != hf_sd[hf_k].shape[::-1]:
                raise RuntimeError(
                    f"shape mismatch for {k}: ours {sd[k].shape}, hf {hf_sd[hf_k].shape}"
                )
            with torch.no_grad():
                sd[k].copy_(hf_sd[hf_k].t())
        else:
            if sd[k].shape != hf_sd[hf_k].shape:
                raise RuntimeError(
                    f"shape mismatch for {k}: ours {sd[k].shape}, hf {hf_sd[hf_k].shape}"
                )
            with torch.no_grad():
                sd[k].copy_(hf_sd[hf_k])

        copied += 1

    # lm_head 和 wte 权重共享
    model.load_state_dict(sd, strict=True)

    ckpt = {
        "model": model.state_dict(),
        "config": asdict(config),
        "iter_num": 0,
        "best_val_loss": None,
        "source": "openai-community/gpt2-medium",
        "version": "V2.1_official_gpt2medium",
        "note": "Official HuggingFace GPT-2 Medium weights converted to this project's checkpoint format.",
    }

    out_path = out_dir / "ckpt.pt"
    torch.save(ckpt, out_path)

    print(f"copied tensors: {copied}")
    print(f"skipped tensors: {skipped}")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
