import json
import random
import re
from pathlib import Path
from datasets import load_dataset

random.seed(42)

OUT = Path("data/raw/V4.3_sft_mix_filtered.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

BAD_KEYWORDS = [
    "python", "java", "javascript", "c++", "c#", "html", "css", "sql",
    "linux", "shell", "pandas", "numpy", "torch", "tensorflow",
    "docker", "github", "代码", "函数", "编程", "报错", "debug",
    "class ", "def ", "import ", "pip install"
]

def zh_count(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s or ""))

def en_ratio(s):
    s = s or ""
    if not s:
        return 0
    letters = len(re.findall(r"[A-Za-z]", s))
    return letters / max(1, len(s))

def normalize(record):
    if isinstance(record, dict):
        conv = record.get("conversations") or record.get("messages")
        if isinstance(conv, list):
            user = ""
            assistant = ""
            for m in conv:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("from") or m.get("role") or "").lower()
                text = str(m.get("value") or m.get("content") or m.get("text") or "").strip()
                if not text:
                    continue
                if role in ("human", "user", "question", "问") and not user:
                    user = text
                elif role in ("gpt", "assistant", "bot", "answer", "答") and not assistant:
                    assistant = text
            if user and assistant:
                return {"instruction": user, "input": "", "output": assistant}

        instruction = str(record.get("instruction") or record.get("prompt") or record.get("question") or "").strip()
        inp = str(record.get("input") or "").strip()
        output = str(record.get("output") or record.get("response") or record.get("answer") or "").strip()
        if instruction and output:
            return {"instruction": instruction, "input": inp, "output": output}
    return None

def too_repetitive(text):
    chars = "".join((text or "").split())
    if len(chars) < 80:
        return False
    for n in (3, 4, 5, 6):
        grams = [chars[i:i+n] for i in range(len(chars)-n+1)]
        if not grams:
            continue
        most = max(grams.count(g) for g in set(grams))
        if most >= 8 and most / len(grams) > 0.08:
            return True
    return False

def good(x):
    ins = x["instruction"]
    out = x["output"]
    text = (ins + "\n" + x.get("input", "") + "\n" + out).lower()

    if zh_count(ins) < 2 or zh_count(out) < 20:
        return False
    if len(out) > 1600 or len(ins) > 800:
        return False
    if en_ratio(text) > 0.25:
        return False
    if any(k in text for k in BAD_KEYWORDS):
        return False
    if too_repetitive(out):
        return False
    return True

identity = [
    {"instruction": "你是谁？", "input": "", "output": "我是一个中文 AI 助手，可以帮助你回答问题、整理文字、写作和解释概念。"},
    {"instruction": "请介绍一下你自己。", "input": "", "output": "我是一个中文对话助手，擅长用简洁清楚的方式回答问题，也可以帮你写邮件、总结内容和提供学习建议。"},
    {"instruction": "你能做什么？", "input": "", "output": "我可以回答常识问题、解释知识点、协助写作、整理信息，也可以根据你的要求生成中文文本。"},
    {"instruction": "你是人类吗？", "input": "", "output": "我不是人类，我是一个由机器学习模型驱动的中文 AI 助手。"},
    {"instruction": "大语言模型是什么？", "input": "", "output": "大语言模型是一类通过大量文本训练出来的人工智能模型。它可以理解和生成自然语言，并根据上下文完成问答、写作、总结和翻译等任务。"},
    {"instruction": "请用三句话解释什么是大语言模型。", "input": "", "output": "大语言模型是通过大量文本训练出来的人工智能模型。它能根据上下文理解问题，并生成接近人类表达的回答。它常用于聊天助手、写作、总结、翻译和知识问答等场景。"},
]

rows = []

print("loading alpaca-gpt4-chinese...")
alpaca = load_dataset("FreedomIntelligence/alpaca-gpt4-chinese", split="train")
for rec in alpaca.shuffle(seed=42):
    x = normalize(rec)
    if x and good(x):
        rows.append(x)

print("alpaca kept:", len(rows))

target_belle = 100000
belle_kept = 0
print("loading BelleGroup/train_2M_CN streaming...")
belle = load_dataset("json", data_files="data/raw/BelleGroup_train_2M_CN/train_2M_CN.json", split="train", streaming=True).shuffle(seed=42, buffer_size=20000)
for i, rec in enumerate(belle):
    if i >= 900000:
        break
    x = normalize(rec)
    if x and good(x):
        rows.append(x)
        belle_kept += 1
        if belle_kept >= target_belle:
            break

rows.extend(identity * 30)
random.shuffle(rows)

with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(json.dumps({
    "output": str(OUT),
    "total_records": len(rows),
    "belle_kept": belle_kept,
    "identity_records": len(identity) * 30,
}, ensure_ascii=False, indent=2))
