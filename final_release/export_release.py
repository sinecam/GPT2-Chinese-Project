import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the V4.7 final model-only release artifact."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.tokenizer.exists():
        raise FileNotFoundError(args.tokenizer)
    if args.output_dir.exists():
        raise FileExistsError(
            f"{args.output_dir} already exists; do not overwrite a frozen release"
        )

    args.output_dir.mkdir(parents=True)
    model_path = args.output_dir / "model.pt"
    tokenizer_path = args.output_dir / "spm_zh.model"

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    required = ("model", "config")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"checkpoint is missing required keys: {missing}")

    release_checkpoint = {
        key: checkpoint[key]
        for key in (
            "model",
            "config",
            "iter_num",
            "best_val_loss",
            "args",
        )
        if key in checkpoint
    }
    del checkpoint
    gc.collect()
    torch.save(release_checkpoint, model_path)
    shutil.copy2(args.tokenizer, tokenizer_path)

    manifest = {
        "release": "V4.7_final",
        "iteration": release_checkpoint.get("iter_num"),
        "best_val_loss": release_checkpoint.get("best_val_loss"),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": sha256(model_path),
        "tokenizer_sha256": sha256(tokenizer_path),
    }
    manifest_path = args.output_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"release written to: {args.output_dir}")


if __name__ == "__main__":
    main()
