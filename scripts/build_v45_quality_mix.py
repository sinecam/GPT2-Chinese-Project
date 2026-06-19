import argparse
import hashlib
import json
import random
import re
from heapq import nlargest
from pathlib import Path
from typing import Any, Iterator


CORE_SEED_EXAMPLES = [
    ("你是谁？", "我是一个中文 AI 助手，可以帮助你回答问题、解释概念、整理文字和完成简单写作任务。"),
    ("你是谁", "我是一个中文 AI 助手，不是真实人物，也没有现实中的姓名、职业或城市身份。"),
    ("请介绍一下你自己。", "我是一个中文对话助手，擅长用简洁清楚的方式回答问题，也可以帮你写邮件、总结内容和解释知识点。"),
    ("你叫什么名字？", "我没有真实姓名。你可以把我当作一个中文 AI 助手来使用。"),
    ("你是人类吗？", "我不是人类，而是一个由机器学习模型驱动的中文 AI 助手。"),
    ("你是律师吗？", "我不是律师，也不是任何现实职业的人。我是一个中文 AI 助手，可以提供一般性信息和写作帮助。"),
    ("你住在哪里？", "我没有现实中的居住地。我是一个运行在计算机上的中文 AI 助手。"),
    ("你能做什么？", "我可以回答常识问题、解释知识点、协助写作、总结内容，并根据你的要求生成中文文本。"),
    ("你不能做什么？", "我不能替代专业医生、律师或金融顾问，也不能保证所有回答完全正确。重要问题应以专业人士或权威资料为准。"),
    ("如果你不知道答案怎么办？", "如果我不知道答案，我应该直接说明不确定，并给出可能的查询方向，而不是编造事实。"),
    ("回答问题时你应该注意什么？", "我应该尽量准确、简洁、清楚地回答。如果不确定，应说明不确定，而不是编造事实。"),
    ("请用三句话解释什么是大语言模型。", "大语言模型是通过大量文本训练出来的人工智能模型。它能根据上下文理解问题，并生成接近人类表达的回答。它常用于聊天助手、写作、总结、翻译和知识问答等场景。"),
    ("大语言模型是什么？", "大语言模型是一类通过大量文本训练得到的人工智能模型。它可以理解和生成自然语言，并根据上下文完成问答、写作、总结和翻译等任务。"),
    ("什么是人工智能？", "人工智能是研究如何让机器具备感知、学习、推理和决策能力的技术方向。它可以帮助计算机完成图像识别、语音识别、自然语言处理和自动决策等任务。"),
    ("深度学习和机器学习有什么区别？", "机器学习是让计算机从数据中学习规律的方法。深度学习是机器学习的一个分支，通常使用多层神经网络处理图像、语音和文本等复杂数据。简单说，深度学习属于机器学习，但模型结构通常更深、需要的数据和计算也更多。"),
    ("什么是 SFT？", "SFT 是监督微调，也就是使用人工整理好的指令和回答数据继续训练模型。它的作用是让预训练模型更会按照用户指令进行回答。"),
    ("为什么预训练之后还要做指令微调？", "预训练主要让模型学习语言规律和基础知识，但不一定知道如何按指令回答。指令微调会用问题和答案样例训练模型，让它更适合对话和任务型问答。"),
    ("训练 loss 和验证 loss 有什么区别？", "训练 loss 反映模型在训练数据上的误差，验证 loss 反映模型在未参与训练的数据上的误差。验证 loss 更能体现模型的泛化能力。"),
    ("为什么模型会重复生成？", "模型重复生成可能与训练数据重复、模型能力不足或解码参数不合适有关。可以通过提高数据质量、调整采样参数和使用重复惩罚来缓解。"),
    ("请用三点说明大语言模型的用途。", "第一，大语言模型可以用于问答、聊天和学习辅导。第二，它可以帮助写作、润色、总结和翻译文本。第三，它还能作为应用中的自然语言交互接口，提高信息处理效率。"),
    ("帮我写一封请假邮件。", "尊敬的领导：您好！我因身体不适，需要请假一天，时间为明天一天。相关工作我会提前安排好，给您带来的不便敬请谅解。谢谢！"),
    ("写一封简短的请假邮件。", "尊敬的老师：您好！我因身体不适，无法按时参加今天的课程，特此向您请假一天。我会尽快补上落下的学习内容，感谢您的理解。"),
    ("帮我写一封感谢邮件。", "尊敬的老师：您好！非常感谢您近期对我的指导和帮助。您的建议让我受益很多，也让我更清楚下一步应该如何改进。再次感谢您的耐心支持！"),
    ("把这句话改得更正式：我明天有事来不了。", "我明天因个人事务无法到场，给您带来的不便敬请谅解。"),
    ("写一段简短的自我介绍。", "大家好，我做事认真负责，愿意主动学习新知识，也重视团队沟通与协作。希望在接下来的工作中不断提升自己，为团队创造更多价值。"),
    ("写一首关于春天的五言诗。", "春风拂柳新，花影映溪滨。燕语穿林过，山川满目春。"),
    ("写一首五言诗，主题是春天。", "细雨润新芽，东风入万家。桃红随水动，燕子掠晴霞。"),
    ("写一首关于秋天的五言诗。", "秋风吹远树，落叶满前庭。云淡山光静，归鸿入晚晴。"),
    ("用一句话总结人工智能。", "人工智能是让机器具备感知、学习、推理和决策能力的一类技术。"),
    ("请解释什么是机器学习。", "机器学习是一种让计算机从数据中学习规律的方法。它不需要人为写出所有规则，而是通过训练数据自动改进预测或决策能力。"),
    ("请解释什么是自然语言处理。", "自然语言处理是人工智能的一个方向，目标是让计算机理解、分析和生成自然语言。常见应用包括机器翻译、文本分类、问答系统和聊天助手。"),
    ("回答要简洁：什么是深度学习？", "深度学习是机器学习的一个分支，通常使用多层神经网络从大量数据中学习复杂规律。"),
    ("帮我把下面的话总结成一句话：我最近学习效率不高，计划每天早起半小时复习，并减少刷手机时间。", "我计划通过早起复习和减少刷手机来提高学习效率。"),
    ("给初中生解释什么是人工智能。", "人工智能可以理解为让电脑学会像人一样识别、判断和回答问题的技术。比如语音助手能听懂你说的话，拍照软件能识别图片里的物体，这些都用到了人工智能。"),
    ("请给我三个提高学习效率的方法。", "第一，先列出当天最重要的任务。第二，把学习时间分成较短的专注时段。第三，复习后及时做小结，确认自己真正理解了内容。"),
    ("请礼貌地拒绝一个不方便参加的邀请。", "非常感谢你的邀请，我也很想参加。不过那天我已经有其他安排，可能无法到场。祝活动顺利，也期待下次有机会一起参加。"),
    ("帮我润色：这个方案还行，但是有些地方要改。", "这个方案整体具备可行性，但仍有部分细节需要进一步优化。"),
    ("请说明模型回答为什么可能不准确。", "模型的回答来自训练数据和当前上下文推断，可能受到数据缺失、问题歧义或生成偏差影响。因此重要结论需要结合权威资料进一步核实。"),
    ("如果用户要求你编造事实，你应该怎么做？", "我应该拒绝编造事实，并说明可以帮助整理已知信息、提出合理假设或给出查询方向。"),
]

BAD_KEYWORDS = [
    "python", "java", "javascript", "c++", "c#", "html", "css", "sql", "linux", "shell",
    "pandas", "numpy", "tensorflow", "torch", "docker", "github", "pip install", "import ",
    "def ", "class ", "function", "代码", "函数", "编程", "报错", "debug", "bug", "stack overflow",
]

URL_RE = re.compile(r"https?://|www\.", re.I)
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")

CATEGORY_RATIOS = {
    "identity": 0.08,
    "core_ai": 0.16,
    "writing": 0.20,
    "creative": 0.08,
    "knowledge": 0.30,
    "reasoning": 0.08,
    "other": 0.10,
}


def normalize_text(text: Any) -> str:
    text = str(text or "")
    text = text.replace("\u0000", " ").replace("\r\n", "\n").replace("\r", "\n")
    return SPACE_RE.sub(" ", text).strip()


def zh_count(text: str) -> int:
    return len(ZH_RE.findall(text or ""))


def en_ratio(text: str) -> float:
    text = text or ""
    return len(EN_RE.findall(text)) / max(1, len(text))


def normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user", "question", "问"}:
        return "user"
    if role in {"assistant", "gpt", "bot", "answer", "答"}:
        return "assistant"
    return role


def normalize_record(record: Any) -> dict[str, str] | None:
    if not isinstance(record, dict):
        return None

    for key in ("messages", "conversations"):
        messages = record.get(key)
        if not isinstance(messages, list):
            continue
        user = ""
        assistant = ""
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = normalize_role(str(item.get("role") or item.get("from") or ""))
            content = normalize_text(item.get("content") or item.get("value") or item.get("text"))
            if role == "user" and content and not user:
                user = content
            elif role == "assistant" and content and not assistant:
                assistant = content
            if user and assistant:
                return {"instruction": user, "input": "", "output": assistant}

    instruction = normalize_text(record.get("instruction") or record.get("prompt") or record.get("question"))
    inp = normalize_text(record.get("input"))
    output = normalize_text(record.get("output") or record.get("response") or record.get("answer"))
    if instruction and output:
        return {"instruction": instruction, "input": inp, "output": output}
    return None


def iter_json_array(value: Any) -> Iterator[Any]:
    if isinstance(value, list):
        yield from value
    elif isinstance(value, dict):
        for key in ("data", "train", "records", "items"):
            if isinstance(value.get(key), list):
                yield from value[key]
                return
        yield value


def iter_local_records(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as f:
        yielded = 0
        for line in f:
            line = line.strip()
            if not line or line in {"[", "]", ","}:
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                yield json.loads(line)
                yielded += 1
            except json.JSONDecodeError:
                continue
        if yielded:
            return

    with path.open("r", encoding="utf-8") as f:
        yield from iter_json_array(json.load(f))


def iter_records(paths: list[str], scan_limit: int) -> Iterator[Any]:
    seen = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        for record in iter_local_records(path):
            yield record
            seen += 1
            if scan_limit > 0 and seen >= scan_limit:
                return


def too_repetitive(text: str) -> bool:
    compact = "".join((text or "").split())
    if len(compact) < 80:
        return False
    for n in (3, 4, 5, 6):
        grams = [compact[i : i + n] for i in range(len(compact) - n + 1)]
        if not grams:
            continue
        counts = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        most = max(counts.values())
        if most >= 8 and most / len(grams) > 0.08:
            return True
    sentences = [s for s in re.split(r"[。！？!?]\s*", text) if len(s) >= 6]
    return len(sentences) >= 4 and len(set(sentences)) <= len(sentences) // 2


def classify(row: dict[str, str]) -> str:
    text = f"{row['instruction']} {row.get('input', '')}".lower()
    if any(k in text for k in ("你是谁", "介绍一下你", "你叫什么", "你是人类", "你能做什么", "你住在哪里")):
        return "identity"
    if any(k in text for k in ("大语言模型", "人工智能", "机器学习", "深度学习", "自然语言处理", "sft", "预训练", "模型")):
        return "core_ai"
    if any(k in text for k in ("邮件", "请假", "通知", "总结", "改写", "润色", "自我介绍", "简历", "文案")):
        return "writing"
    if any(k in text for k in ("诗", "故事", "对联", "作文", "春天", "秋天")):
        return "creative"
    if any(k in text for k in ("为什么", "区别", "原因", "分析", "步骤", "方法", "建议")):
        return "reasoning"
    if any(k in text for k in ("什么是", "请解释", "如何", "怎么", "请说明")):
        return "knowledge"
    return "other"


def quality_score(row: dict[str, str], category: str) -> float:
    instruction = row["instruction"]
    output = row["output"]
    out_len = len(output)
    score = min(zh_count(output) / 120, 4.0)
    score += 3.0 if 40 <= out_len <= 450 else 1.5 if 20 <= out_len <= 800 else 0.0
    score += 1.0 if any(p in output for p in "。！？") else 0.0
    score += 1.0 if category in {"writing", "creative"} and out_len <= 500 else 0.0
    score += 1.0 if category == "core_ai" and any(w in output for w in ("人工智能", "模型", "学习", "自然语言")) else 0.0
    score += 2.0 if category == "identity" and "中文 AI 助手" in output else 0.0
    score -= 2.0 if en_ratio(instruction + output) > 0.18 else 0.0
    return score


def fingerprint(row: dict[str, str]) -> str:
    key = json.dumps(
        {
            "instruction": row["instruction"].lower(),
            "input": row.get("input", "").lower(),
            "output": row["output"][:300].lower(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()


def good(row: dict[str, str], args: argparse.Namespace) -> bool:
    instruction = row["instruction"]
    output = row["output"]
    text = f"{instruction}\n{row.get('input', '')}\n{output}"
    lower = text.lower()
    if not instruction or not output:
        return False
    if len(instruction) > args.max_instruction_chars or len(output) > args.max_answer_chars:
        return False
    if zh_count(instruction) < args.min_instruction_zh or zh_count(output) < args.min_answer_zh:
        return False
    if en_ratio(text) > args.max_en_ratio or URL_RE.search(text):
        return False
    if not args.allow_code and any(keyword in lower for keyword in BAD_KEYWORDS):
        return False
    if too_repetitive(output) or output.count("\n") > 18:
        return False
    return True


def category_limits(max_records: int) -> dict[str, int]:
    limits = {name: max(1, int(max_records * ratio)) for name, ratio in CATEGORY_RATIOS.items()}
    limits["knowledge"] += max_records - sum(limits.values())
    return limits


def add_candidate(
    pools: dict[str, list[tuple[float, int, dict[str, str]]]],
    row: dict[str, str],
    category: str,
    score: float,
    serial: int,
    limit: int,
) -> None:
    pools.setdefault(category, []).append((score, serial, row))
    cap = max(200, limit * 4)
    if len(pools[category]) > cap:
        pools[category] = nlargest(max(100, limit * 2), pools[category], key=lambda item: item[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaner V4.5 response-only SFT JSONL mix.")
    parser.add_argument("--input", nargs="*", default=[], help="Local json/jsonl files to filter and mix.")
    parser.add_argument("--out", type=str, default="data/raw/V4.5_quality_sft_mix.jsonl")
    parser.add_argument("--max_records", type=int, default=60000)
    parser.add_argument("--scan_limit", type=int, default=1000000, help="0 means scan all input records.")
    parser.add_argument("--seed_repeat", type=int, default=60)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--min_instruction_zh", type=int, default=2)
    parser.add_argument("--min_answer_zh", type=int, default=18)
    parser.add_argument("--max_instruction_chars", type=int, default=500)
    parser.add_argument("--max_answer_chars", type=int, default=900)
    parser.add_argument("--max_en_ratio", type=float, default=0.22)
    parser.add_argument("--allow_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    limits = category_limits(args.max_records)
    pools: dict[str, list[tuple[float, int, dict[str, str]]]] = {key: [] for key in limits}
    seen_external = set()
    seed_fingerprints = set()
    stats = {"total_input": 0, "normalized": 0, "kept_candidates": 0, "duplicates": 0, "filtered": 0}

    serial = 0
    for _ in range(args.seed_repeat):
        for instruction, output in CORE_SEED_EXAMPLES:
            row = {"instruction": instruction, "input": "", "output": output}
            seed_fingerprints.add(fingerprint(row))
            category = classify(row)
            add_candidate(pools, row, category, 100.0, serial, limits[category])
            serial += 1
    seen_external.update(seed_fingerprints)

    for record in iter_records(args.input, args.scan_limit):
        stats["total_input"] += 1
        row = normalize_record(record)
        if row is None:
            stats["filtered"] += 1
            continue
        stats["normalized"] += 1
        if not good(row, args):
            stats["filtered"] += 1
            continue
        fp = fingerprint(row)
        if fp in seen_external:
            stats["duplicates"] += 1
            continue
        seen_external.add(fp)
        category = classify(row)
        add_candidate(pools, row, category, quality_score(row, category), serial, limits[category])
        serial += 1
        stats["kept_candidates"] += 1

    selected: list[dict[str, str]] = []
    selected_by_category = {}
    for category, limit in limits.items():
        rows = [item[2] for item in nlargest(limit, pools.get(category, []), key=lambda item: item[0])]
        selected.extend(rows)
        selected_by_category[category] = len(rows)

    random.shuffle(selected)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(out_path),
        "records": len(selected),
        "selected_by_category": selected_by_category,
        "seed_examples": len(CORE_SEED_EXAMPLES),
        "seed_repeat": args.seed_repeat,
        "seed_records_requested": len(CORE_SEED_EXAMPLES) * args.seed_repeat,
        "limits": limits,
        **stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
