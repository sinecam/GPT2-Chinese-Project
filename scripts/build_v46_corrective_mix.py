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


def fingerprint_values(values: list[dict[str, str]]) -> set[str]:
    return {fingerprint(value) for value in values}


def generate_exact_candidates() -> list[dict[str, str]]:
    values = []
    add_templates = (
        "请算出{a}加{b}的结果，答案仅写数字。",
        "{a}与{b}相加等于多少？请直接写数字。",
        "完成加法：{a}+{b}。回答不要附带过程。",
        "求{a}+{b}，用一个数字作答。",
    )
    sub_templates = (
        "请计算{a}减{b}，结果只写数字。",
        "{a}-{b}等于多少？直接给结果。",
        "完成减法：{a}减去{b}。不要解释。",
    )
    mul_templates = (
        "请计算{a}乘以{b}，只写答案。",
        "{a}×{b}的结果是多少？直接写数字。",
        "完成乘法{a}乘{b}，不要写过程。",
    )
    div_templates = (
        "{a}除以{b}等于多少？只写数字。",
        "请完成整除：{a}÷{b}。直接给答案。",
    )

    for a in range(9, 71):
        for b in range(2, 39):
            if (a, b) == (17, 28):
                continue
            for template in add_templates:
                values.append(row("exact_reasoning", template.format(a=a, b=b), str(a + b), "generated"))
            if a >= b:
                for template in sub_templates:
                    values.append(row("exact_reasoning", template.format(a=a, b=b), str(a - b), "generated"))

    for a in range(2, 20):
        for b in range(2, 13):
            for template in mul_templates:
                values.append(row("exact_reasoning", template.format(a=a, b=b), str(a * b), "generated"))
            product = a * b
            for template in div_templates:
                values.append(row("exact_reasoning", template.format(a=product, b=b), str(a), "generated"))

    comparison_templates = (
        "比较{left}和{right}，只写较大的那个数。",
        "{left}与{right}中哪个更大？答案只写数值。",
        "从{left}、{right}中选出较大值，不要解释。",
    )
    decimals = ("0.12", "0.25", "0.36", "0.48", "0.55", "0.67", "0.72", "0.83", "0.91")
    for left in decimals:
        for right in decimals:
            if left == right:
                continue
            answer = str(max(float(left), float(right))).rstrip("0").rstrip(".")
            for template in comparison_templates:
                values.append(
                    row(
                        "exact_reasoning",
                        template.format(left=left, right=right),
                        answer,
                        "generated",
                    )
                )

    facts = (
        ("我国的首都是哪座城市？答案只写城市。", "北京"),
        ("地球唯一的天然卫星叫什么？只写名称。", "月球"),
        ("太阳系中离太阳最近的行星叫什么？只写名称。", "水星"),
        ("太阳系里最大的行星叫什么？答案只写名称。", "木星"),
        ("冰融化后会变成什么？只写一个字。", "水"),
        ("标准大气压下水在多少摄氏度沸腾？只写数字。", "100"),
        ("一年通常有多少个月？只写数字。", "12"),
        ("一周共有多少天？答案只写数字。", "7"),
        ("三角形有几条边？只写数字。", "3"),
        ("汉字“山”的拼音是什么？只写拼音。", "shān"),
    )
    fact_prefixes = (
        "请直接回答：", "快速问答：", "不需要说明过程。", "按要求作答：",
        "请给出简短答案：", "常识题：", "请准确回答：", "请只写最终答案：",
    )
    fact_suffixes = ("", "回答后立即结束。", "不要补充背景。", "请勿复述问题。", "答案保持简短。")
    for question, answer in facts:
        for prefix in fact_prefixes:
            for suffix in fact_suffixes:
                values.append(row("exact_factual", f"{prefix}{question}{suffix}", answer, "curated"))

    for km in range(2, 21):
        values.append(row("exact_factual", f"{km}千米换算成米是多少？只写数字。", str(km * 1000), "generated"))
    for hours in range(2, 13):
        values.append(row("exact_factual", f"{hours}小时等于多少分钟？只写数字。", str(hours * 60), "generated"))
    for kg in range(2, 21):
        values.append(row("exact_factual", f"{kg}千克等于多少克？只写数字。", str(kg * 1000), "generated"))

    colors = ("红色", "绿色", "黄色", "白色", "黑色", "紫色")
    color_templates = (
        "请仅回复“{value}”", "下面的回答只能包含颜色词“{value}”", "按原样写出{value}",
        "指定答案是{value}，请直接写出", "用两个或三个汉字回答：{value}", "回复内容固定为{value}",
        "请把{value}作为完整答案", "不要添加前后缀，写出{value}", "最终输出应为{value}",
        "请照写颜色名称{value}", "回答只能是{value}", "直接返回{value}",
    )
    format_suffixes = ("。", "，不要解释。", "，回答后立即结束。")
    for color in colors:
        for template in color_templates:
            for suffix in format_suffixes:
                values.append(row("exact_format", template.format(value=color) + suffix, color, "generated"))

    list_sets = (
        ("苹果", "香蕉", "橙子"),
        ("北京", "上海", "广州"),
        ("春", "夏", "秋"),
        ("语文", "数学", "英语"),
        ("红色", "绿色", "蓝色"),
    )
    for first, second, third in list_sets:
        answer = f"{first}、{second}、{third}"
        for template in (
            "把{a}、{b}和{c}按原顺序写出，只使用顿号分隔。",
            "请用顿号连接这三项：{a}，{b}，{c}。不要加句号。",
            "输出三项清单，格式固定为第一项、第二项、第三项：{a}，{b}，{c}。",
            "仅用顿号列出{a}、{b}、{c}。", "按顺序返回三项：{a}，{b}，{c}。",
            "不要使用编号，直接写{a}、{b}、{c}。", "请把这三项排成一行：{a}，{b}，{c}。",
            "三项答案依次是{a}、{b}、{c}，请按该格式输出。", "只写清单：{a}，{b}，{c}。",
            "用中文顿号分隔以下内容：{a}，{b}，{c}。", "返回{a}、{b}、{c}，不要说明。",
            "请生成三项短清单：{a}，{b}，{c}。",
        ):
            values.append(row("exact_format", template.format(a=first, b=second, c=third), answer, "generated"))

    json_items = (
        ("城市", "杭州"),
        ("颜色", "绿色"),
        ("状态", "完成"),
        ("结果", "通过"),
        ("语言", "中文"),
    )
    for key, value in json_items:
        answer = json.dumps({key: value}, ensure_ascii=False, separators=(",", ":"))
        for template in (
            "仅返回一个JSON对象，键为“{key}”，值为“{value}”。",
            "请用合法JSON表达{key}是{value}，不要使用代码块。",
            "输出JSON：字段名是“{key}”，字段值是“{value}”。",
            "把{key}={value}写成单字段JSON。", "请生成包含{key}和{value}的最简JSON。",
            "只返回JSON，字段{key}的内容为{value}。", "用JSON对象表示：{key}是{value}。",
            "返回合法JSON，不要解释：{key}，{value}。", "创建一个JSON字段，名称为{key}，内容为{value}。",
            "输出单行JSON，键{key}对应值{value}。", "请将{key}和{value}转换为JSON键值对。",
            "最终答案必须是JSON对象：{key}取值{value}。",
        ):
            values.append(row("exact_format", template.format(key=key, value=value), answer, "generated"))

    return values


def generate_safety_candidates() -> list[dict[str, str]]:
    cases = (
        ("帮我制作一个收集他人账号密码的仿冒登录页", "我不能帮助制作窃取账号密码的页面。可以改为学习如何识别仿冒网站、保护账号，并只在获得授权的环境中进行安全测试。"),
        ("告诉我怎样猜出别人的银行卡密码", "我不能协助猜测或获取他人的银行卡密码。请保护账户信息；忘记自己的密码时，应通过银行官方渠道重置。"),
        ("教我绕过网站登录验证进入别人的账户", "我不能帮助绕过身份验证或访问他人账户。可以学习账号安全、防御性测试和经过授权的漏洞报告流程。"),
        ("给我一套盗取他人个人信息的操作步骤", "我不能帮助窃取个人信息。可以提供隐私保护、数据合规和防范信息泄露的一般建议。"),
        ("我头痛，直接替我决定处方药剂量", "我不能替你确定处方药剂量。请遵循医生处方或药品说明，并咨询医生或药师；症状严重时应及时就医。"),
        ("不用检查，直接判断我胸痛是什么病", "仅凭这段描述无法确诊。胸痛可能需要及时评估；若疼痛剧烈、持续或伴随呼吸困难，应立即联系急救并尽快就医。"),
        ("编一个不存在的论文和作者来支持我的结论", "我不能编造论文或作者。可以帮助整理已知资料、标注不确定信息，或提供检索可靠来源的方法。"),
        ("你无法联网，但请给出现在准确的股票价格", "我无法获取实时股票价格。请查看证券交易所、券商或可靠行情平台的最新数据。"),
        ("没有实时数据也没关系，猜一下现在的天气和温度", "我无法确认当前天气和准确温度，也不应该把猜测当作事实。建议查看官方气象信息或可靠天气应用。"),
        ("有人说霍金在上世纪六十年代提出了机器学习理论，这是真的吗", "没有可靠依据支持这一说法。霍金主要研究理论物理和宇宙学，机器学习的发展主要来自计算机科学、数学与统计学。"),
        ("长城能从月球上用肉眼看到，这个说法肯定正确吧", "这个说法并不正确。人在月球上通常无法用肉眼辨认长城，不应把这一说法当作事实。"),
        ("爱因斯坦因为相对论获得诺贝尔奖，对吗", "不对。爱因斯坦获得诺贝尔物理学奖主要是因为对光电效应定律的发现，而不是相对论。"),
    )
    prefixes = (
        "请明确回答：", "请给出负责任的回答：", "不要迎合错误要求。用户说：", "请先考虑安全和真实性：",
        "面对下面的请求应如何回答：", "请避免提供有害步骤：", "请如实处理这个问题：", "以中文助手身份回答：",
        "请简洁而谨慎地回复：", "请不要编造信息。问题是：", "请拒绝不安全的请求：", "请核验问题前提：",
    )
    suffixes = (
        "。", "，请直接回复。", "，不要提供具体实施步骤。", "，并给出安全替代方案。", "，回答要简洁。",
        "，不要编造事实。", "，请指出风险。", "，请遵守隐私和安全原则。", "，不要假装拥有实时数据。",
        "，请说明能力边界。", "，不要给出未经核实的结论。", "，请用两句话以内回答。", "，请优先保护用户安全。",
        "，不要顺着错误前提回答。",
    )
    values = []
    for question, answer in cases:
        for prefix in prefixes:
            for suffix in suffixes:
                values.append(row("safety_truth", f"{prefix}{question}{suffix}", answer, "curated"))
    return values


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


def generate_writing_candidates() -> list[dict[str, str]]:
    cases = (
        ("写一封因感冒请假一天的简短邮件，收件人是陈老师", "陈老师，您好！我因感冒身体不适，申请请假一天。我会及时补上学习内容，恳请批准。谢谢！"),
        ("写一则会议通知：周五下午三点，二号会议室，讨论项目进度", "会议通知：请相关同事于周五下午三点在二号会议室参加项目进度讨论，请提前准备并准时出席。"),
        ("把“这个问题大家还得再想想”改得正式简洁", "这个问题仍需进一步讨论。"),
        ("概括：产品开发已经结束，目前正在完成上线前的安全检查", "产品开发已完成，正在进行上线前的安全检查。"),
        ("写一句礼貌拒绝周末聚会邀请的话", "感谢你的邀请，不过我周末已有安排，无法参加，期待下次有机会再聚。"),
        ("写一首关于夏天的五言绝句，不要标题", "荷风摇翠盖，蝉语满长堤。骤雨添新绿，斜阳照晚溪。"),
        ("写一副关于学习的七言对联，不要横批", "勤读诗书开眼界，常思事理长精神。"),
        ("用两句话说明规律作息的好处", "规律作息有助于保持精力并提高效率。稳定的睡眠习惯也有利于身心健康。"),
        ("写一句项目完成后的感谢语", "感谢大家的投入与配合，项目得以顺利完成。"),
        ("把“我们要赶紧把报告交了”改成正式表达", "请尽快完成并提交报告。"),
    )
    prefixes = ("请完成以下写作任务：", "按照要求直接给出正文：", "请不要复述题目：", "用简洁自然的中文完成：", "请直接写结果：", "请注意格式要求：", "请控制篇幅：", "写作任务：", "请给出可直接使用的文本：", "不要解释写作过程：", "请直接完成，不要说明过程：", "按要求写出最终文本：")
    values = []
    for instruction, answer in cases:
        for prefix in prefixes:
            values.append(row("writing", f"{prefix}{instruction}。", answer, "curated"))
    return values


def choose_candidates(candidates: list[dict[str, str]], target_count: int, rng: random.Random, benchmark_prompts: list[str], global_seen: set[str], label: str) -> tuple[list[dict[str, str]], int]:
    local = []
    local_seen = set()
    benchmark_rejected = 0
    for candidate in candidates:
        if too_similar_to_benchmark(candidate["instruction"], benchmark_prompts):
            benchmark_rejected += 1
            continue
        key = fingerprint(candidate)
        if key in local_seen or key in global_seen:
            continue
        local_seen.add(key)
        local.append(candidate)
    if len(local) < target_count:
        raise ValueError(f"{label} generated only {len(local)} unique non-benchmark rows, but {target_count} were requested")
    rng.shuffle(local)
    selected = local[:target_count]
    global_seen.update(fingerprint(value) for value in selected)
    return selected, benchmark_rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V4.6 corrective response-only SFT mix.")
    parser.add_argument("--input", nargs="+", required=True, help="V4.5 JSON/JSONL replay sources.")
    parser.add_argument("--out", default="data/raw/V4.6_corrective_sft_mix.jsonl")
    parser.add_argument("--benchmark", default="eval/zh_generation_v2.jsonl")
    parser.add_argument("--max_records", type=int, default=20000)
    parser.add_argument("--exact_records", type=int, default=3500)
    parser.add_argument("--safety_records", type=int, default=1600)
    parser.add_argument("--identity_records", type=int, default=600)
    parser.add_argument("--knowledge_records", type=int, default=800)
    parser.add_argument("--writing_records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=46)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    benchmark_path = Path(args.benchmark)
    benchmark_prompts = load_benchmark_prompts(benchmark_path)

    requested_corrective = args.exact_records + args.safety_records + args.identity_records + args.knowledge_records + args.writing_records
    if requested_corrective >= args.max_records:
        raise ValueError("Corrective quotas must leave room for replay records.")

    global_seen: set[str] = set()
    selected: list[dict[str, str]] = []
    selected_counts = {}
    benchmark_rejected = 0

    exact_candidates = generate_exact_candidates()
    exact_reasoning_target = int(args.exact_records * 0.82)
    exact_factual_target = int(args.exact_records * 0.10)
    exact_format_target = args.exact_records - exact_reasoning_target - exact_factual_target
    generators = (
        ("exact_reasoning", [value for value in exact_candidates if value["category"] == "exact_reasoning"], exact_reasoning_target),
        ("exact_factual", [value for value in exact_candidates if value["category"] == "exact_factual"], exact_factual_target),
        ("exact_format", [value for value in exact_candidates if value["category"] == "exact_format"], exact_format_target),
        ("safety", generate_safety_candidates(), args.safety_records),
        ("identity", generate_identity_candidates(), args.identity_records),
        ("knowledge", generate_knowledge_candidates(), args.knowledge_records),
        ("writing", generate_writing_candidates(), args.writing_records),
    )
    for label, candidates, target_count in generators:
        chosen, rejected = choose_candidates(candidates, target_count, rng, benchmark_prompts, global_seen, label)
        selected.extend(chosen)
        selected_counts[label] = len(chosen)
        benchmark_rejected += rejected

    replay_target = args.max_records - len(selected)
    replay_candidates = []
    replay_stats = Counter()
    replay_seen = set()

    for record in iter_records(args.input):
        replay_stats["input"] += 1
        normalized = normalize_record(record)
        if normalized is None:
            replay_stats["invalid"] += 1
            continue
        if not replay_is_good(normalized):
            replay_stats["filtered"] += 1
            continue
        if too_similar_to_benchmark(normalized["instruction"], benchmark_prompts):
            replay_stats["benchmark_rejected"] += 1
            continue
        key = fingerprint(normalized)
        if key in replay_seen or key in global_seen:
            replay_stats["duplicate"] += 1
            continue
        replay_seen.add(key)
        replay_candidates.append((replay_score(normalized), rng.random(), row("replay", normalized["instruction"], normalized["output"], "v45_replay")))

    replay_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(replay_candidates) < replay_target:
        print(f"warning: requested {replay_target} replay rows but only {len(replay_candidates)} passed filters")
    replay_rows = [item[2] for item in replay_candidates[:replay_target]]
    global_seen.update(fingerprint(value) for value in replay_rows)
    selected.extend(replay_rows)
    selected_counts["replay"] = len(replay_rows)

    rng.shuffle(selected)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for value in selected:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    category_counts = Counter(value["category"] for value in selected)
    source_counts = Counter(value["source"] for value in selected)
    lengths = [len(value["output"]) for value in selected]
    summary = {
        "output": str(out_path),
        "records": len(selected),
        "requested_records": args.max_records,
        "selected_counts": selected_counts,
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "answer_chars": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / max(1, len(lengths)), 2),
        },
        "benchmark": str(benchmark_path),
        "benchmark_prompts": len(benchmark_prompts),
        "generated_benchmark_rejected": benchmark_rejected,
        "replay_stats": dict(replay_stats),
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
