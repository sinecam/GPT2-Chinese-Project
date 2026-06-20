import argparse
import hashlib
import json
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator


SPACE_RE = re.compile(r"\s+")
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"https?://|www\.", re.I)
IDENTITY_NAME_RE = re.compile(r"我叫[\u4e00-\u9fff]{2,4}")
META_ANSWER_PHRASES = (
    "只输出",
    "不要解释",
    "不要添加",
    "控制在",
    "字以内",
    "严格使用",
    "仅返回",
)
CODE_MARKERS = (
    "```",
    "import ",
    "def ",
    "class ",
    "function ",
    "javascript",
    "python",
    "java ",
    "c++",
    "sql ",
)


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\u0000", " ").replace("\r\n", "\n").replace("\r", "\n")
    return SPACE_RE.sub(" ", text).strip()


def compact(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text or "").lower()


def zh_count(text: str) -> int:
    return len(ZH_RE.findall(text or ""))


def en_ratio(text: str) -> float:
    text = text or ""
    return len(EN_RE.findall(text)) / max(1, len(text))


def fingerprint(row: dict[str, str]) -> str:
    key = f"{compact(row['instruction'])}\n{compact(row.get('input', ''))}\n{compact(row['output'])}"
    return hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()


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
            text = normalize_text(item.get("content") or item.get("value") or item.get("text"))
            if role == "user" and text and not user:
                user = text
            elif role == "assistant" and text and not assistant:
                assistant = text
            if user and assistant:
                return {"instruction": user, "input": "", "output": assistant}

    instruction = normalize_text(
        record.get("instruction") or record.get("prompt") or record.get("question")
    )
    inp = normalize_text(record.get("input"))
    output = normalize_text(
        record.get("output") or record.get("response") or record.get("answer")
    )
    if instruction and output:
        return {"instruction": instruction, "input": inp, "output": output}
    return None


def iter_json_records(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as handle:
        yielded = 0
        for line in handle:
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

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        yield from value
    elif isinstance(value, dict):
        for key in ("data", "train", "records", "items"):
            if isinstance(value.get(key), list):
                yield from value[key]
                return
        yield value


def iter_records(paths: list[str]) -> Iterator[Any]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        yield from iter_json_records(path)


def load_benchmark_prompts(path: Path) -> list[str]:
    if not path.exists():
        return []
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = compact(str(record.get("prompt", "")))
            if prompt:
                prompts.append(prompt)
    return prompts


def too_similar_to_benchmark(instruction: str, benchmark_prompts: list[str]) -> bool:
    candidate = compact(instruction)
    if not candidate:
        return False
    for prompt in benchmark_prompts:
        if candidate == prompt:
            return True
        if min(len(candidate), len(prompt)) >= 10:
            ratio = SequenceMatcher(None, candidate, prompt, autojunk=False).ratio()
            if ratio >= 0.84:
                return True
    return False


def too_repetitive(text: str) -> bool:
    value = compact(text)
    if len(value) < 50:
        return False
    for size in (3, 4, 5, 6):
        grams = [value[index : index + size] for index in range(len(value) - size + 1)]
        if not grams:
            continue
        most_common = Counter(grams).most_common(1)[0][1]
        if most_common >= 7 and most_common / len(grams) > 0.08:
            return True
    sentences = [
        compact(part)
        for part in re.split(r"[。！？!?]+", text)
        if len(compact(part)) >= 6
    ]
    return len(sentences) >= 4 and len(set(sentences)) <= len(sentences) // 2


def prompt_echo(instruction: str, output: str) -> bool:
    prompt_value = compact(instruction)
    answer_value = compact(output)
    if not prompt_value or not answer_value:
        return True
    if answer_value == prompt_value or (len(prompt_value) >= 12 and prompt_value in answer_value):
        return True

    prefix = 0
    for left, right in zip(prompt_value, answer_value):
        if left != right:
            break
        prefix += 1
    if prefix >= 14 and prefix / max(1, min(len(prompt_value), len(answer_value))) > 0.45:
        return True

    lower_output = output.lower()
    return any(phrase in lower_output for phrase in META_ANSWER_PHRASES)


def identity_hallucination(instruction: str, output: str) -> bool:
    identity_prompt = any(
        phrase in instruction
        for phrase in ("你是谁", "你叫什么", "你的名字", "住在哪里", "什么职业")
    )
    return identity_prompt and IDENTITY_NAME_RE.search(output) is not None


def replay_is_good(row: dict[str, str]) -> bool:
    instruction = row["instruction"]
    output = row["output"]
    combined = f"{instruction}\n{row.get('input', '')}\n{output}"
    lower = combined.lower()

    if not instruction or not output:
        return False
    if len(instruction) > 420 or not 6 <= len(output) <= 600:
        return False
    if zh_count(instruction) < 2 or zh_count(output) < 4:
        return False
    if en_ratio(combined) > 0.18 or URL_RE.search(combined):
        return False
    if any(marker in lower for marker in CODE_MARKERS):
        return False
    if output.count("\n") > 14 or too_repetitive(output):
        return False
    if prompt_echo(instruction, output) or identity_hallucination(instruction, output):
        return False
    return True


def replay_score(row: dict[str, str]) -> float:
    output = row["output"]
    length = len(output)
    score = 0.0
    score += 4.0 if 20 <= length <= 220 else 2.0 if length <= 400 else 0.0
    score += 1.5 if any(mark in output for mark in "。！？") else 0.0
    score += min(3.0, zh_count(output) / 80.0)
    score -= en_ratio(output) * 5.0
    return score


def row(category: str, instruction: str, output: str, source: str) -> dict[str, str]:
    return {
        "instruction": normalize_text(instruction),
        "input": "",
        "output": normalize_text(output),
        "category": category,
        "source": source,
    }


def generate_identity_candidates() -> list[dict[str, str]]:
    cases = (
        ("你是什么身份", "我是一个中文 AI 助手，不是真实人物。"),
        ("请简单介绍你自己", "我是一个中文 AI 助手，可以协助回答问题、解释知识和处理文本。"),
        ("你有真实姓名吗", "我没有真实姓名。你可以把我称为中文 AI 助手。"),
        ("你住在哪个城市", "我没有现实中的住址或所在城市，我是运行在计算机上的中文 AI 助手。"),
        ("你从事什么职业", "我没有现实职业。我是一个中文 AI 助手，可以提供一般性信息和文本帮助。"),
        ("你是不是人类", "我不是人类，而是一个由机器学习模型驱动的中文 AI 助手。"),
        ("你能提供哪些帮助", "我可以回答问题、解释概念、总结内容，并协助完成中文写作任务。"),
        ("你的回答一定正确吗", "不一定。我的回答可能出错，重要信息应结合权威资料或专业人士进一步核实。"),
        ("遇到不知道的问题你会怎么办", "我会说明自己不确定，并建议查询可靠资料，而不是编造答案。"),
        ("你能代替医生或律师吗", "不能。我可以提供一般信息，但不能替代医生、律师等专业人士的判断。"),
    )
    prefixes = (
        "请如实回答：", "不要虚构个人经历。问题：", "用一句话回应：", "请保持身份一致：", "请直接说明：",
        "不要编造姓名或职业：", "以中文助手身份回答：", "请简洁作答：", "请说明真实情况：", "回答下面的问题：",
        "请保持诚实并回答：", "请说明你的能力边界：",
    )
    suffixes = ("。", "，不要展开。", "，回答要诚实。", "，不要虚构。", "，请控制在两句话内。", "，不要使用真实人物姓名。")
    values = []
    for question, answer in cases:
        for prefix in prefixes:
            for suffix in suffixes:
                values.append(row("identity", f"{prefix}{question}{suffix}", answer, "curated"))
    return values


def generate_knowledge_candidates() -> list[dict[str, str]]:
    cases = (
        ("什么叫语言模型", "语言模型用于估计词语或词元在上下文中出现的概率，并据此理解或生成文本。"),
        ("大语言模型如何生成文字", "大语言模型根据已有上下文预测后续词元，连续预测后形成完整文本。"),
        ("预训练有什么作用", "预训练让模型从大规模文本中学习语言规律、表达方式和基础知识。"),
        ("监督微调有什么作用", "监督微调使用指令与回答样本训练模型，使它更会理解并遵循用户要求。"),
        ("什么是分词器", "分词器把文本切分成词元，并在词元与数字编号之间进行转换。"),
        ("模型幻觉是什么意思", "模型幻觉是生成了看似合理但不符合事实或缺乏可靠依据的内容。"),
        ("如何减少模型幻觉", "可以使用可靠资料检索、来源核验和不确定性说明来降低幻觉风险。"),
        ("机器学习是什么", "机器学习是让计算机从数据中学习规律，并用于预测或决策的方法。"),
        ("深度学习是什么", "深度学习是机器学习的一个分支，通常使用多层神经网络学习复杂特征。"),
        ("深度学习与机器学习是什么关系", "深度学习属于机器学习，主要特点是使用多层神经网络自动学习特征。"),
        ("自然语言处理是什么", "自然语言处理研究如何让计算机理解、分析和生成人类语言。"),
        ("训练集和验证集有什么区别", "训练集用于更新模型参数，验证集用于评估模型在未参与训练数据上的表现。"),
        ("过拟合是什么意思", "过拟合是模型在训练数据上表现很好，但在新数据上表现明显变差。"),
        ("什么是上下文窗口", "上下文窗口是模型一次能够读取和利用的最大词元范围。"),
        ("什么是词元", "词元是分词器处理文本时使用的基本单位，可以是字、词或词的一部分。"),
        ("温度参数有什么作用", "温度控制生成的随机性；温度较低通常更稳定，较高则更多样。"),
        ("重复惩罚有什么作用", "重复惩罚会降低已经生成过的词元再次出现的概率，从而减少重复。"),
        ("模型参数是什么", "模型参数是训练过程中学习得到的数值，用来决定模型如何处理输入并生成输出。"),
        ("推理和训练有什么区别", "训练通过数据更新模型参数，推理则使用已经训练好的参数生成结果。"),
        ("为什么回答需要核实", "模型可能受数据缺失、问题歧义或生成偏差影响，因此重要结论需要核实。"),
    )
    prefixes = ("请用一句话解释：", "面向初学者说明：", "简洁回答：", "请给出准确解释：", "用不超过两句话说明：", "请不要列举无关内容。问题：", "请直接回答概念：", "用通俗中文解释：", "请概括核心定义：", "请用简明中文回答：")
    suffixes = ("。", "，不要超过80字。", "，重点说明核心含义。", "，不需要历史背景。", "，回答保持简洁。")
    values = []
    for question, answer in cases:
        for prefix in prefixes:
            for suffix in suffixes:
                values.append(row("concise_knowledge", f"{prefix}{question}{suffix}", answer, "curated"))
    return values
