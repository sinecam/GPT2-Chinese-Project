import json
import random
from pathlib import Path

random.seed(1337)

base_examples = [
    {
        "instruction": "什么是人工智能？",
        "output": "人工智能是研究如何让机器具备感知、学习、推理和决策能力的技术方向。它可以帮助计算机完成图像识别、语音识别、自然语言处理、自动决策等任务。"
    },
    {
        "instruction": "人工智能的发展主要体现在哪些方面？",
        "output": "人工智能的发展主要体现在四个方面：第一，数据规模不断扩大；第二，计算能力持续提升；第三，深度学习和 Transformer 等算法不断进步；第四，人工智能在教育、医疗、交通、工业等领域的应用越来越广泛。"
    },
    {
        "instruction": "人工智能有哪些应用场景？",
        "output": "人工智能可以应用在医疗诊断、智能交通、自动驾驶、教育辅助、工业质检、金融风控、智能客服、内容推荐和科学研究等领域。"
    },
    {
        "instruction": "人工智能有哪些局限？",
        "output": "人工智能的局限主要包括三点：第一，模型依赖高质量数据；第二，模型可能产生错误或偏见；第三，模型缺乏真正的人类常识和价值判断能力。"
    },
    {
        "instruction": "什么是语言模型？",
        "output": "语言模型是用来建模文本概率分布的模型。它可以根据前面的 token 预测下一个 token，因此可以用于文本生成、自动补全、问答和摘要等任务。"
    },
    {
        "instruction": "什么是 GPT 模型？",
        "output": "GPT 是一种自回归语言模型。它根据前面的 token 逐步预测下一个 token，并通过不断重复这个过程生成连续的自然语言文本。"
    },
    {
        "instruction": "GPT 模型为什么使用因果注意力？",
        "output": "GPT 是从左到右生成文本的模型，在预测当前位置时只能看到前面的内容，不能看到未来内容。因果注意力通过遮挡未来 token，保证模型按照自回归方式训练和生成。"
    },
    {
        "instruction": "Transformer 的核心思想是什么？",
        "output": "Transformer 的核心思想是自注意力机制。它可以让模型在处理一个词时关注上下文中其他位置的信息，从而更好地建模长距离依赖关系，并且比循环神经网络更适合并行训练。"
    },
    {
        "instruction": "请简单解释注意力机制。",
        "output": "注意力机制可以理解为模型在处理一个词时，会自动判断上下文中哪些词更重要。重要的词会获得更高权重，从而帮助模型理解词语之间的关系。"
    },
    {
        "instruction": "什么是 tokenizer？",
        "output": "tokenizer 是把文本转换成 token 序列的工具。模型不能直接处理原始文字，需要先把文字切分并映射成数字编号，再输入神经网络进行训练或生成。"
    },
    {
        "instruction": "为什么中文 GPT-2 使用英文 tokenizer 效率不高？",
        "output": "英文 GPT-2 tokenizer 主要针对英文设计。处理中文时，一个汉字可能会被拆成多个 byte-level token，因此相同长度的中文文本会产生更多 token，训练效率会受到影响。"
    },
    {
        "instruction": "什么是预训练？",
        "output": "预训练是指先让模型在大规模通用文本上学习语言规律和基础表达能力。完成预训练后，模型可以在较小的任务数据上继续微调，从而适应具体应用。"
    },
    {
        "instruction": "什么是 SFT？",
        "output": "SFT 是监督微调，也就是使用整理好的指令和回答数据继续训练模型。它的作用是让预训练模型学会理解用户指令，并按照问答格式生成更符合要求的回复。"
    },
    {
        "instruction": "为什么预训练之后还要做指令微调？",
        "output": "预训练模型主要学习普通文本续写能力，但不一定知道如何回答用户问题。指令微调通过问题和答案样例训练模型，使模型更适合按照 User 和 Assistant 的格式进行问答。"
    },
    {
        "instruction": "预训练和微调有什么区别？",
        "output": "预训练通常使用大规模通用文本，让模型学习语言规律；微调使用更小、更有针对性的数据，让模型适应具体任务。简单来说，预训练提供基础能力，微调改变模型的使用方式。"
    },
    {
        "instruction": "训练 loss 和验证 loss 有什么区别？",
        "output": "训练 loss 反映模型在训练集上的误差，验证 loss 反映模型在未参与训练的数据上的误差。验证 loss 更能体现模型的泛化能力。"
    },
    {
        "instruction": "为什么验证集 loss 下降说明模型在变好？",
        "output": "验证集 loss 下降说明模型在没有参与训练的数据上预测得更准确。这通常表示模型不仅记住了训练数据，还学到了一定的语言规律。"
    },
    {
        "instruction": "什么是 checkpoint？",
        "output": "checkpoint 是训练过程中保存的模型状态文件，通常包含模型参数、训练步数、优化器状态和验证损失等信息。保存 checkpoint 后，即使训练中断，也可以继续训练或用于生成测试。"
    },
    {
        "instruction": "为什么模型会重复生成？",
        "output": "模型重复生成可能与训练不足、数据重复、模型规模较小或解码参数不合适有关。例如 temperature 太低时，模型容易反复选择高概率词，从而陷入重复循环。"
    },
    {
        "instruction": "如何缓解重复生成？",
        "output": "可以通过增加训练数据、提高数据质量、调整 temperature 和 top-k 等采样参数，以及使用 repetition penalty 或 no-repeat ngram 等反重复解码策略来缓解重复生成。"
    },
    {
        "instruction": "请介绍本项目。",
        "output": "本项目从零实现了一个 GPT-2 Small 风格的语言模型，并完成了中文语料预训练和中文指令微调。实验展示了数据规模、训练步数和 SFT 对生成效果的影响。"
    },
    {
        "instruction": "本项目的训练流程是什么？",
        "output": "本项目的训练流程包括数据收集、文本清洗、tokenizer 分词、构建 train.bin 和 val.bin、GPT-2 Small 预训练、验证集评估、指令微调和最终生成测试。"
    },
    {
        "instruction": "本项目为什么要先训练 v8 再做 SFT？",
        "output": "v8 是在 1B 字符中文语料上训练得到的预训练模型，已经具备基本中文表达能力。在此基础上做 SFT，可以让模型从普通文本续写转向指令问答，更适合最终演示。"
    },
    {
        "instruction": "v6、v7、v8 的效果有什么变化？",
        "output": "v6 已经能生成中文，但重复和语义混乱比较明显；v7 的句子更自然，但仍有主题漂移；v8 在 1B 字符语料上训练后，文本结构和连贯性进一步提升。"
    },
]

augmented = []
for ex in base_examples:
    q = ex["instruction"]
    a = ex["output"]
    augmented.append({"instruction": q, "output": a})
    augmented.append({"instruction": "请回答：" + q, "output": a})
    augmented.append({"instruction": q + " 请简要说明。", "output": a})
    augmented.append({"instruction": "用通俗的话解释：" + q, "output": a})

random.shuffle(augmented)

out_path = Path("data/sft/sft_demo_clean_zh.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    for ex in augmented:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"saved {len(augmented)} examples to {out_path}")
