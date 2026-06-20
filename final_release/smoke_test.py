import argparse
import json
from pathlib import Path

from engine import ChineseGPT2Engine, GenerationConfig


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = "请用一句话解释什么是大语言模型。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one end-to-end generation with the final release."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "model" / "model.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "model" / "spm_zh.model",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        generation_config = GenerationConfig(**json.load(handle))

    engine = ChineseGPT2Engine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device_name=args.device,
        dtype_name=args.dtype,
    )
    answer = engine.generate(args.prompt, [], generation_config)

    if not answer:
        raise RuntimeError("generation returned an empty answer")
    if any(marker in answer for marker in ("<user>", "<assistant>", "<system>")):
        raise RuntimeError(f"generation leaked a control marker: {answer!r}")

    status = engine.status()
    print(
        "SMOKE TEST PASSED | "
        f"iteration={status['iteration']} | "
        f"parameters={status['parameters']:,} | "
        f"device={status['device']}"
    )
    print(f"Prompt: {args.prompt}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
