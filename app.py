import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import tiktoken
import streamlit as st

from model.gpt import GPT, GPTConfig


@st.cache_resource
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])

    model = GPT(config)

    state_dict = ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    return model, enc


def generate_text(model, enc, prompt, device, max_new_tokens, temperature, top_k, stop_eot):
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    out = enc.decode(y[0].tolist())

    if stop_eot and "<|endoftext|>" in out:
        out = out.split("<|endoftext|>")[0]

    return out


st.set_page_config(page_title="GPT-2 中文模型演示", layout="wide")

st.title("GPT-2 中文模型交互式演示")

st.markdown(
    """
    本页面用于测试从零训练的 GPT-2 中文模型。
    训练过程中建议使用 CPU 推理，避免和训练任务抢占 GPU 显存。
    """
)

default_v12 = "/root/autodl-tmp/GPT2_Small_Project/checkpoints/v12_sft_demo_overfit_zh/ckpt.pt"
default_v16 = "/root/autodl-tmp/GPT2_Medium_Project/checkpoints/eval_snapshots/v16_medium_current_test.pt"

model_choice = st.selectbox(
    "选择模型",
    [
        "v12 小模型 SFT 演示版",
        "v16 Medium 训练快照",
        "自定义 checkpoint 路径",
    ],
)

if model_choice == "v12 小模型 SFT 演示版":
    ckpt_path = default_v12
elif model_choice == "v16 Medium 训练快照":
    ckpt_path = default_v16
else:
    ckpt_path = st.text_input("输入 checkpoint 路径", default_v12)

device_mode = st.radio("推理设备", ["CPU", "CUDA"], horizontal=True)

if device_mode == "CUDA" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

st.write(f"当前 checkpoint：`{ckpt_path}`")
st.write(f"当前推理设备：`{device}`")

mode = st.radio("输入模式", ["问答模式", "续写模式"], horizontal=True)

if mode == "问答模式":
    user_input = st.text_area(
        "输入问题",
        value="什么是人工智能？",
        height=120,
    )
    prompt = f"User: {user_input.strip()}\nAssistant:"
else:
    prompt = st.text_area(
        "输入续写开头",
        value="人工智能的发展主要体现在",
        height=120,
    )

col1, col2, col3 = st.columns(3)

with col1:
    max_new_tokens = st.slider("最大生成 token 数", 20, 400, 180, 10)

with col2:
    temperature = st.slider("temperature", 0.1, 1.2, 0.5, 0.05)

with col3:
    top_k = st.slider("top_k", 1, 100, 40, 1)

stop_eot = st.checkbox("遇到 <|endoftext|> 停止显示", value=True)

if st.button("生成", type="primary"):
    ckpt = Path(ckpt_path)

    if not ckpt.exists():
        st.error(f"checkpoint 不存在：{ckpt_path}")
    else:
        with st.spinner("正在加载模型并生成..."):
            model, enc = load_model(str(ckpt), device)
            output = generate_text(
                model=model,
                enc=enc,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_eot=stop_eot,
            )

        st.subheader("生成结果")
        st.text_area("输出", value=output, height=300)
