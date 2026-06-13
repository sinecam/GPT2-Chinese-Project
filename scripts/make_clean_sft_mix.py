import json
import random
import re
from pathlib import Path

random.seed(1337)

src_path = Path("data/sft/coig_sft_zh.jsonl")
out_path = Path("data/sft/sft_mix_clean_zh.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

seed_examples = [
    {
        "instruction": "什么是人工智能？",
        "output": "人工智能是研究如何让机器具备感知、学习、推理和决策能力的技术方向。它可以帮助计算机完成图像识别、语音识别、自然语言处理和自动决策等任务。"
    },
    {
        "instruction": "人工智能的发展主要体现在哪些方面？",
        "output": "人工智能的发展主要体现在四个方面：第一，数据规模不断扩大；第二，计算能力持续提升；第三，深度学习和 Transformer 等算法不断进步；第四，人工智能在教育、医疗、交通、工业等领域的应用越来越广泛。"
    },
    {
        "instruction": "为什么预训练之后还要做指令微调？",
        "output": "预训练模型主要学习通用语言规律，擅长文本续写，但不一定知道如何按照用户问题进行回答。指令微调通过问题和答案样例训练模型，使模型更适合按照 User 和 Assistant 的格式完成问答。"
    },
    {
        "instruction": "什么是 SFT？",
        "output": "SFT 是监督微调，也就是使用人工整理好的指令和回答数据继续训练模型。它的作用是让预训练模型学会理解用户指令，并生成更符合要求的回答。"
    },
    {
        "instruction": "什么是 Transformer？",
        "output": "Transformer 是一种基于自注意力机制的神经网络结构。它能够让模型在处理一个词时关注上下文中的其他词，从而更好地理解长距离依赖关系。"
    },
    {
        "instruction": "什么是 GPT 模型？",
        "output": "GPT 是一种自回归语言模型，它根据前面的 token 逐步预测下一个 token。通过不断预测下一个 token，GPT 可以生成连续的自然语言文本。"
    },
    {
        "instruction": "请介绍本项目。",
        "output": "本项目从零实现了一个 GPT-2 Small 风格的语言模型，并完成了中文语料预训练和中文指令微调。实验展示了数据规模、训练步数和 SFT 对生成效果的影响。"
    },
    {
        "instruction": "本项目为什么要先预训练再微调？",
        "output": "预训练可以让模型学习中文语言规律和基础表达能力，微调则让模型学会按照用户指令进行回答。两者结合后，模型既有基本语言能力，也能更好地完成问答任务。"
    },
    {
        "instruction": "训练 loss 和验证 loss 有什么区别？",
        "output": "训练 loss 反映模型在训练数据上的误差，验证 loss 反映模型在未参与训练的数据上的误差。验证 loss 更能体现模型的泛化能力。"
    },
    {
        "instruction": "为什么大语言模型会重复生成？",
        "output": "大语言模型重复生成可能与训练不足、数据重复、解码参数不合适有关。例如温度太低时，模型容易反复选择高概率词，从而陷入重复循环。"
    },
    {
        "instruction": "如何缓解重复生成？",
        "output": "可以通过增加训练数据、提高数据质量、调整 temperature 和 top-k 等采样参数，以及使用 repetition penalty 或 no-repeat ngram 等反重复解码方法来缓解。"
    },
    {
        "instruction": "中文 GPT-2 使用英文 tokenizer 有什么问题？",
        "output": "英文 GPT-2 tokenizer 对中文不够高效，一个汉字可能被拆成多个 byte-level token，导致相同文本需要更多 token 表示。因此中文训练效率会受到一定影响。"
    },
]

bad_patterns = [
    "下列", "单项选择", "多项选择", "正确答案", "答案是", "试题", "考试",
    "A.", "B.", "C.", "D.", "A、", "B、", "C、", "D、",
    "法律", "法院", "判决", "罪", "案件", "原告", "被告",
    "豆瓣", "评分", "电影", "演员表",
    "什么意思", "翻译成英文",
]

def has_chinese(s):
    return re.search(r"[\u4e00-\u9fff]", s) is not None

def clean_text(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def good_item(ins, out):
    ins = clean_text(ins)
    out = clean_text(out)

    if not ins or not out:
        return False
    if not has_chinese(ins) or not has_chinese(out):
        return False
    if len(ins) < 4 or len(ins) > 120:
        return False
    if len(out) < 30 or len(out) > 350:
        return False
    text = ins + out
    if any(p in text for p in bad_patterns):
        return False
    return True

coig = []
with src_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        ins = clean_text(item.get("instruction", ""))
        out = clean_text(item.get("output", ""))
        if good_item(ins, out):
            coig.append({"instruction": ins, "output": out})

random.shuffle(coig)
coig = coig[:3000]

final = []

# 项目相关高质量样例重复，提高问答稳定性
for _ in range(80):
    final.extend(seed_examples)

# 筛选后的 COIG 只作为泛化补充
final.extend(coig)

random.shuffle(final)

with out_path.open("w", encoding="utf-8") as f:
    for ex in final:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"seed examples repeated: {len(seed_examples) * 80}")
print(f"filtered coig examples: {len(coig)}")
print(f"total examples: {len(final)}")
print(f"saved to: {out_path}")
