import json
import random
import argparse
import re
from pathlib import Path
from collections import Counter


ALLOW_PREFIX = [
    "教育/科学",
    "电脑/网络",
    "文化/艺术",
    "生活-生活常识",
    "生活-礼节礼仪",
    "生活-服装/首饰",
    "电子数码",
]

BAD_CATEGORY_PREFIX = [
    "健康",
    "游戏",
    "烦恼",
    "娱乐-博彩",
    "商业/理财-股票",
    "商业/理财-基金",
]

BAD_WORDS = [
    "好评", "采纳", "楼主", "呵呵", "亲，", "亲,", "谢谢",
    "满意", "百度", "转载", "链接", "点击", "如图", "图片",
    "http://", "https://", "www.", "<br", "&nbsp",
    "加我", "QQ", "qq", "微信", "广告",
    "医院", "服用", "用药", "处方", "治疗", "病情分析", "指导意见",
    "彩票", "中奖", "盗号", "充值", "私服",
]

QUESTION_WORDS = [
    "什么", "怎么", "如何", "为什么", "区别", "原因", "方法",
    "作用", "原理", "定义", "怎么办", "怎么养", "有哪些", "多少",
    "是否", "能不能", "可以吗"
]


def clean_text(s):
    s = str(s).replace("\r", "\n")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def has_bad_repetition(text):
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 30:
        return False

    # 字符多样性过低，通常是重复句
    unique_ratio = len(set(chars)) / len(chars)
    if unique_ratio < 0.22:
        return True

    # 4-gram 重复过多
    grams = ["".join(chars[i:i+4]) for i in range(len(chars) - 3)]
    if grams:
        most_common_count = Counter(grams).most_common(1)[0][1]
        if most_common_count >= 5:
            return True

    # 明显短句重复
    parts = re.split(r"[，。！？；,.!?;]", text)
    parts = [p.strip() for p in parts if len(p.strip()) >= 4]
    if parts:
        c = Counter(parts)
        if c.most_common(1)[0][1] >= 3:
            return True

    return False


def is_good_item(obj):
    category = clean_text(obj.get("category", ""))
    title = clean_text(obj.get("title", ""))
    desc = clean_text(obj.get("desc", ""))
    answer = clean_text(obj.get("answer", ""))

    if not title or not answer:
        return None

    if any(category.startswith(x) for x in BAD_CATEGORY_PREFIX):
        return None

    if not any(category.startswith(x) for x in ALLOW_PREFIX):
        return None

    question_text = title + desc
    all_text = question_text + answer

    if not any(w in question_text for w in QUESTION_WORDS):
        return None

    if any(w in all_text for w in BAD_WORDS):
        return None

    if len(title) < 4 or len(title) > 80:
        return None

    if len(answer) < 30 or len(answer) > 260:
        return None

    if has_bad_repetition(answer):
        return None

    # 过滤乱码比例高的样本
    bad_chars = sum(1 for c in all_text if c in "�□�")
    if bad_chars > 0:
        return None

    if desc and desc != title and len(desc) <= 120:
        instruction = f"{title}\n补充信息：{desc}"
    else:
        instruction = title

    return {
        "instruction": instruction,
        "output": answer,
        "category": category,
    }


def load_items(path):
    items = []
    total = 0

    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            item = is_good_item(obj)
            if item is not None:
                items.append(item)

    return items, total


def save_jsonl(items, path):
    with Path(path).open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", required=True)
    parser.add_argument("--valid_input", required=True)
    parser.add_argument("--out_dir", default="data/sft/baike_qa2019_clean")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_valid", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items, train_total = load_items(args.train_input)
    valid_items, valid_total = load_items(args.valid_input)

    random.shuffle(train_items)
    random.shuffle(valid_items)

    train_items = train_items[:args.max_train]
    valid_items = valid_items[:args.max_valid]

    train_out = out_dir / f"baike_qa_train_clean_{len(train_items)}.jsonl"
    valid_out = out_dir / f"baike_qa_valid_clean_{len(valid_items)}.jsonl"

    save_jsonl(train_items, train_out)
    save_jsonl(valid_items, valid_out)

    print("train total:", train_total)
    print("train clean kept:", len(train_items))
    print("valid total:", valid_total)
    print("valid clean kept:", len(valid_items))
    print("train output:", train_out)
    print("valid output:", valid_out)

    print("\nexamples:")
    for item in train_items[:5]:
        print(json.dumps(item, ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
