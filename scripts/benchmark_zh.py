import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chat_zh import encode_prompt, format_history, generate, load_model


IGNORE_INDEX = -100
SCORE_COLUMNS = (
    "correctness_1_5",
    "instruction_1_5",
    "fluency_1_5",
    "conciseness_1_5",
    "safety_1_5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible Chinese generation benchmark for one or more checkpoints."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=CKPT",
        help="Repeat for every model, for example --model V4.5=checkpoints/V4.5/ckpt.pt",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--suite", default="eval/zh_generation_v1.jsonl")
    parser.add_argument("--out_dir", default="reports/zh_generation_v1")
    parser.add_argument(
        "--system_prompt",
        default=(
            "你是一个中文AI助手。回答要简洁、准确。身份问题只回答你是中文AI助手，"
            "不要编造姓名、职业或真实人物身份。"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N tasks; 0 runs all.")
    parser.add_argument("--skip_reference_loss", action="store_true")
    parser.add_argument(
        "--hash_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record SHA-256 checkpoint hashes so compared runs are auditable.",
    )
    return parser.parse_args()


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    models = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --model value {value!r}; expected LABEL=CKPT")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not label or not raw_path.strip():
            raise ValueError(f"Invalid --model value {value!r}; expected LABEL=CKPT")
        if label in labels:
            raise ValueError(f"Duplicate model label: {label}")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        labels.add(label)
        models.append((label, path))
    return models


def load_suite(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark suite not found: {path}")
    tasks = []
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            for key in ("id", "category", "prompt", "reference", "checks"):
                if key not in task:
                    raise ValueError(f"{path}:{line_number} is missing {key!r}")
            if task["id"] in ids:
                raise ValueError(f"Duplicate task id: {task['id']}")
            ids.add(task["id"])
            tasks.append(task)
            if limit > 0 and len(tasks) >= limit:
                break
    if not tasks:
        raise ValueError("Benchmark suite is empty")
    return tasks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。！？!?]+|\n+", text or "") if part.strip()]


def split_poem_lines(text: str) -> list[str]:
    parts = re.split(r"[，。！？!?；;\n]+", text or "")
    return [re.sub(r"[^\u4e00-\u9fff]", "", part) for part in parts if re.search(r"[\u4e00-\u9fff]", part)]


def evaluate_check(output: str, check: dict[str, Any]) -> tuple[bool, str]:
    check_type = check["type"]
    values = [str(value) for value in check.get("values", [])]

    if check_type == "exact":
        expected = normalize(str(check["value"]))
        actual = normalize(output)
        return actual == expected, f"expected={expected!r}, actual={actual!r}"

    if check_type == "contains_all":
        missing = [value for value in values if value not in output]
        return not missing, f"missing={missing}"

    if check_type == "contains_any":
        matched = [value for value in values if value in output]
        return bool(matched), f"matched={matched}"

    if check_type == "not_contains_any":
        found = [value for value in values if value in output]
        return not found, f"forbidden={found}"

    if check_type == "regex":
        matched = re.search(str(check["pattern"]), output, flags=re.MULTILINE) is not None
        return matched, f"pattern={check['pattern']!r}"

    if check_type == "char_range":
        length = len(normalize(output))
        low = int(check.get("min", 0))
        high = int(check.get("max", 10**9))
        return low <= length <= high, f"chars={length}, expected={low}..{high}"

    if check_type == "sentence_count":
        count = len(split_sentences(output))
        low = int(check.get("min", 0))
        high = int(check.get("max", 10**9))
        return low <= count <= high, f"sentences={count}, expected={low}..{high}"

    if check_type == "numbered_items":
        count = len(re.findall(r"(?:^|\s)(?:\d+[\.、]|[一二三四五六七八九十]+[、\.])", output))
        low = int(check.get("min", 0))
        high = int(check.get("max", 10**9))
        return low <= count <= high, f"numbered_items={count}, expected={low}..{high}"

    if check_type == "poem_lines":
        lines = split_poem_lines(output)
        expected_lines = int(check["lines"])
        expected_chars = int(check["line_chars"])
        lengths = [len(line) for line in lines]
        passed = len(lines) == expected_lines and all(length == expected_chars for length in lengths)
        return passed, f"line_lengths={lengths}, expected={expected_lines}x{expected_chars}"

    raise ValueError(f"Unknown check type: {check_type}")


def repetition_metrics(text: str) -> dict[str, float]:
    compact = normalize(text)
    grams = [compact[index : index + 4] for index in range(max(0, len(compact) - 3))]
    repeat_4 = 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)
    sentences = split_sentences(text)
    sentence_repeat = 0.0 if not sentences else 1.0 - len(set(sentences)) / len(sentences)
    return {
        "repeat_4_rate": repeat_4,
        "sentence_repeat_rate": sentence_repeat,
    }


def quality_metrics(
    output: str,
    generated_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    repeat = repetition_metrics(output)
    compact = normalize(output)
    unknown_count = output.count("⁇") + output.count("<unk>")
    chinese = len(re.findall(r"[\u4e00-\u9fff]", output))
    latin = len(re.findall(r"[A-Za-z]", output))
    chinese_ratio = chinese / max(1, chinese + latin)
    truncated = generated_tokens >= max(1, max_new_tokens - 2)
    consecutive_repeat = re.search(r"(.)\1{3,}", compact) is not None

    score = 100.0
    if not compact:
        score = 0.0
    score -= min(60.0, unknown_count * 20.0)
    score -= min(35.0, max(0.0, repeat["repeat_4_rate"] - 0.08) * 180.0)
    score -= min(30.0, repeat["sentence_repeat_rate"] * 60.0)
    if len(compact) >= 20 and chinese_ratio < 0.55:
        score -= 20.0
    if consecutive_repeat:
        score -= 20.0
    if truncated:
        score -= 15.0

    return {
        "quality_score": round(max(0.0, min(100.0, score)), 4),
        "unknown_count": unknown_count,
        "chinese_ratio": round(chinese_ratio, 6),
        "truncated": truncated,
        "consecutive_repeat": consecutive_repeat,
        **{key: round(value, 6) for key, value in repeat.items()},
    }


def score_response(
    task: dict[str, Any],
    output: str,
    generated_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    checks = []
    earned = 0.0
    total = 0.0
    for index, check in enumerate(task["checks"]):
        passed, detail = evaluate_check(output, check)
        weight = float(check.get("weight", 1.0))
        total += weight
        if passed:
            earned += weight
        checks.append(
            {
                "index": index,
                "name": check.get("name", check["type"]),
                "type": check["type"],
                "weight": weight,
                "gate": bool(check.get("gate", False)),
                "passed": passed,
                "detail": detail,
            }
        )
    critical_failure = any(check["gate"] and not check["passed"] for check in checks)
    rule_score = 0.0 if critical_failure else 100.0 * earned / max(total, 1e-12)
    quality = quality_metrics(output, generated_tokens, max_new_tokens)
    return {
        "rule_score": round(rule_score, 4),
        "task_passed": all(check["passed"] for check in checks),
        "critical_failure": critical_failure,
        "checks": checks,
        **quality,
    }


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def reference_loss(
    model,
    sp: spm.SentencePieceProcessor,
    formatted_prompt: str,
    reference: str,
    device: torch.device,
    dtype_name: str,
) -> tuple[float, int]:
    prefix_ids = encode_prompt(sp, formatted_prompt)
    answer_ids = sp.encode(reference, out_type=int)
    eos_ids = [int(sp.eos_id())] if sp.eos_id() >= 0 else []
    full_ids = prefix_ids + answer_ids + eos_ids
    labels = [IGNORE_INDEX] * len(prefix_ids) + answer_ids + eos_ids

    max_length = model.config.block_size + 1
    full_ids = full_ids[:max_length]
    labels = labels[:max_length]
    if len(full_ids) < 2:
        return 0.0, 0

    x = torch.tensor(full_ids[:-1], dtype=torch.long, device=device)[None, :]
    y = torch.tensor(labels[1:], dtype=torch.long, device=device)[None, :]
    token_count = int((y != IGNORE_INDEX).sum().item())
    if token_count == 0:
        return 0.0, 0

    with autocast_context(device, dtype_name):
        dummy_targets = torch.zeros_like(x)
        logits, _ = model(x, dummy_targets)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
    return float(loss_sum.item()), token_count


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def aggregate_model(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_loss = sum(float(row["reference_loss_sum"]) for row in rows)
    total_reference_tokens = sum(int(row["reference_tokens"]) for row in rows)
    mean_loss = total_loss / total_reference_tokens if total_reference_tokens else None
    rule_score = statistics.fmean(float(row["rule_score"]) for row in rows)
    quality_score = statistics.fmean(float(row["quality_score"]) for row in rows)
    latencies = [float(row["latency_seconds"]) for row in rows]
    generated_tokens = sum(int(row["generated_tokens"]) for row in rows)
    total_latency = sum(latencies)

    categories = {}
    for category in sorted({str(row["category"]) for row in rows}):
        group = [row for row in rows if row["category"] == category]
        categories[category] = {
            "tasks": len(group),
            "rule_score": round(statistics.fmean(float(row["rule_score"]) for row in group), 4),
            "quality_score": round(statistics.fmean(float(row["quality_score"]) for row in group), 4),
            "pass_rate": round(100.0 * sum(bool(row["task_passed"]) for row in group) / len(group), 4),
            "critical_failure_rate": round(
                100.0 * sum(bool(row["critical_failure"]) for row in group) / len(group), 4
            ),
        }

    return {
        "model": label,
        "tasks": len(rows),
        "auto_score": round(0.75 * rule_score + 0.25 * quality_score, 4),
        "rule_score": round(rule_score, 4),
        "quality_score": round(quality_score, 4),
        "pass_rate": round(100.0 * sum(bool(row["task_passed"]) for row in rows) / len(rows), 4),
        "critical_failure_rate": round(
            100.0 * sum(bool(row["critical_failure"]) for row in rows) / len(rows), 4
        ),
        "reference_loss": None if mean_loss is None else round(mean_loss, 6),
        "reference_ppl": None if mean_loss is None else round(math.exp(min(mean_loss, 20.0)), 4),
        "unknown_response_rate": round(
            100.0 * sum(int(row["unknown_count"]) > 0 for row in rows) / len(rows), 4
        ),
        "truncated_rate": round(100.0 * sum(bool(row["truncated"]) for row in rows) / len(rows), 4),
        "mean_repeat_4_rate": round(
            statistics.fmean(float(row["repeat_4_rate"]) for row in rows), 6
        ),
        "median_latency_seconds": round(statistics.median(latencies), 4),
        "p95_latency_seconds": round(percentile(latencies, 0.95), 4),
        "tokens_per_second": round(generated_tokens / max(total_latency, 1e-12), 2),
        "categories": categories,
    }


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "tasks",
        "auto_score",
        "rule_score",
        "quality_score",
        "pass_rate",
        "critical_failure_rate",
        "reference_loss",
        "reference_ppl",
        "unknown_response_rate",
        "truncated_rate",
        "mean_repeat_4_rate",
        "median_latency_seconds",
        "p95_latency_seconds",
        "tokens_per_second",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})


def write_report(
    path: Path,
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# Chinese Generation Benchmark",
        "",
        f"- Suite: `{config['suite']}`",
        f"- Suite SHA-256: `{config['suite_sha256']}`",
        f"- Generated: {config['generated_at_utc']}",
        f"- Decoding: temperature={config['generation']['temperature']}, "
        f"top_k={config['generation']['top_k']}, top_p={config['generation']['top_p']}, "
        f"repetition_penalty={config['generation']['repetition_penalty']}, "
        f"no_repeat_ngram_size={config['generation']['no_repeat_ngram_size']}",
        "",
        "## Overall",
        "",
        "| Model | Auto | Rules | Quality | Pass % | Critical fail % | Ref loss | Ref ppl | Unknown % | Truncated % | tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['model']} | {item['auto_score']:.2f} | {item['rule_score']:.2f} | "
            f"{item['quality_score']:.2f} | {item['pass_rate']:.2f} | "
            f"{item['critical_failure_rate']:.2f} | "
            f"{item['reference_loss'] if item['reference_loss'] is not None else 'n/a'} | "
            f"{item['reference_ppl'] if item['reference_ppl'] is not None else 'n/a'} | "
            f"{item['unknown_response_rate']:.2f} | {item['truncated_rate']:.2f} | "
            f"{item['tokens_per_second']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Categories",
            "",
            "| Model | Category | Tasks | Rules | Quality | Pass % |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        for category, category_result in item["categories"].items():
            lines.append(
                f"| {item['model']} | {category} | {category_result['tasks']} | "
                f"{category_result['rule_score']:.2f} | {category_result['quality_score']:.2f} | "
                f"{category_result['pass_rate']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Reading The Scores",
            "",
            "- Auto = 75% rule score + 25% degeneration quality score.",
            "- Reference loss is teacher-forced loss on the fixed reference answers. Compare it only when suite hash, tokenizer, and system prompt are identical.",
            "- Keyword and format checks do not establish semantic correctness. Use the generated blind review sheet before choosing a release checkpoint.",
            "- Speed is comparable only on the same machine and software environment.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_blind_review(
    out_dir: Path,
    tasks: list[dict[str, Any]],
    model_rows: dict[str, list[dict[str, Any]]],
    seed: int,
) -> None:
    labels = sorted(model_rows)
    candidates = [chr(ord("A") + index) for index in range(len(labels))]
    shuffled_labels = labels[:]
    random.Random(seed).shuffle(shuffled_labels)
    candidate_to_model = dict(zip(candidates, shuffled_labels))
    by_model_and_id = {
        model_label: {row["id"]: row for row in rows}
        for model_label, rows in model_rows.items()
    }

    review_rows = []
    for task in tasks:
        for candidate, model_label in candidate_to_model.items():
            result = by_model_and_id[model_label][task["id"]]
            review_rows.append(
                {
                    "item_id": task["id"],
                    "category": task["category"],
                    "candidate": candidate,
                    "prompt": task["prompt"],
                    "response": result["response"],
                    **{column: "" for column in SCORE_COLUMNS},
                    "notes": "",
                }
            )
    random.Random(seed + 1).shuffle(review_rows)

    fields = ["item_id", "category", "candidate", "prompt", "response", *SCORE_COLUMNS, "notes"]
    with (out_dir / "blind_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    key = {
        "warning": "Do not show this file to reviewers before scoring.",
        "candidate_to_model": candidate_to_model,
    }
    (out_dir / "blind_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    suite_path = Path(args.suite)
    tasks = load_suite(suite_path, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "seed": args.seed,
        "dtype": args.dtype,
    }
    run_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": str(suite_path),
        "suite_sha256": text_sha256(suite_path),
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": file_sha256(Path(args.tokenizer)),
        "system_prompt": args.system_prompt,
        "device": str(device),
        "generation": generation_config,
        "models": [],
    }

    all_rows = []
    model_rows = {}
    summaries = []

    for model_index, (label, checkpoint_path) in enumerate(models):
        checkpoint_record = {
            "label": label,
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path) if args.hash_checkpoints else None,
        }
        run_config["models"].append(checkpoint_record)
        print(f"[{model_index + 1}/{len(models)}] loading {label}: {checkpoint_path}", flush=True)
        model = load_model(str(checkpoint_path), device)
        rows = []

        for task_index, task in enumerate(tasks):
            formatted_prompt = format_history([(task["prompt"], "")], args.system_prompt)
            input_ids = encode_prompt(sp, formatted_prompt)
            max_new_tokens = int(task.get("max_new_tokens", args.max_new_tokens))
            item_seed = args.seed + model_index * 100000 + int(
                hashlib.sha256(task["id"].encode("utf-8")).hexdigest()[:8], 16
            )
            random.seed(item_seed)
            torch.manual_seed(item_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(item_seed)
                torch.cuda.synchronize(device)

            started = time.perf_counter()
            with autocast_context(device, args.dtype):
                response = generate(
                    model,
                    input_ids,
                    sp,
                    device,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency = time.perf_counter() - started
            generated_tokens = len(sp.encode(response, out_type=int))
            scores = score_response(task, response, generated_tokens, max_new_tokens)

            loss_sum = 0.0
            reference_tokens = 0
            if not args.skip_reference_loss:
                loss_sum, reference_tokens = reference_loss(
                    model,
                    sp,
                    formatted_prompt,
                    task["reference"],
                    device,
                    args.dtype,
                )

            row = {
                "model": label,
                "id": task["id"],
                "category": task["category"],
                "prompt": task["prompt"],
                "reference": task["reference"],
                "response": response,
                "max_new_tokens": max_new_tokens,
                "generated_tokens": generated_tokens,
                "latency_seconds": round(latency, 6),
                "reference_loss_sum": loss_sum,
                "reference_tokens": reference_tokens,
                **scores,
            }
            rows.append(row)
            all_rows.append(row)
            print(
                f"  [{task_index + 1:02d}/{len(tasks):02d}] {task['id']}: "
                f"rules={scores['rule_score']:.0f} quality={scores['quality_score']:.0f}",
                flush=True,
            )

        summary = aggregate_model(label, rows)
        summaries.append(summary)
        model_rows[label] = rows
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "config": run_config,
        "summary": summaries,
        "results": all_rows,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(out_dir / "summary.csv", summaries)
    write_report(out_dir / "report.md", run_config, summaries)
    write_blind_review(out_dir, tasks, model_rows, args.seed)

    print(f"\nWrote benchmark outputs to {out_dir}", flush=True)
    for summary in summaries:
        print(
            f"{summary['model']}: auto={summary['auto_score']:.2f}, "
            f"rules={summary['rule_score']:.2f}, quality={summary['quality_score']:.2f}, "
            f"pass={summary['pass_rate']:.2f}%, critical_fail={summary['critical_failure_rate']:.2f}%, "
            f"ref_loss={summary['reference_loss']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
