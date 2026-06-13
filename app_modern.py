import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import tiktoken
import streamlit as st

from model.gpt import GPT, GPTConfig


# =========================
# 页面基础配置
# =========================
st.set_page_config(
    page_title="GPT-2 中文模型演示平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 自定义 CSS
# =========================
st.markdown(
    """
<style>
    .main {
        background: linear-gradient(135deg, #f6f9ff 0%, #eef4ff 45%, #ffffff 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    .hero-card {
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #1f5eff 0%, #4e8cff 55%, #7bb4ff 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(31, 94, 255, 0.22);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: 0.02em;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        opacity: 0.92;
        line-height: 1.8;
    }

    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(91, 141, 255, 0.22);
        box-shadow: 0 10px 28px rgba(30, 75, 160, 0.08);
        height: 100%;
    }

    .metric-label {
        font-size: 0.86rem;
        color: #64748b;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: #1e3a8a;
    }

    .section-card {
        padding: 1.35rem 1.5rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.8rem;
    }

    .small-tip {
        padding: 0.75rem 0.95rem;
        border-radius: 14px;
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        color: #1e3a8a;
        line-height: 1.7;
        font-size: 0.92rem;
        margin-top: 0.7rem;
    }

    .output-box {
        padding: 1.2rem 1.3rem;
        border-radius: 18px;
        background: #0f172a;
        color: #e5e7eb;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 16px 35px rgba(15, 23, 42, 0.18);
        white-space: pre-wrap;
        line-height: 1.9;
        font-size: 1rem;
    }

    .stButton > button {
        width: 100%;
        height: 3.1rem;
        border-radius: 16px;
        font-weight: 800;
        font-size: 1rem;
        border: none;
        background: linear-gradient(135deg, #2563eb 0%, #60a5fa 100%);
        color: white;
        box-shadow: 0 12px 25px rgba(37, 99, 235, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
        color: white;
        border: none;
    }

    div[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    textarea {
        border-radius: 16px !important;
    }

    .footer-note {
        color: #64748b;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# 模型加载函数
# =========================
def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        new_state_dict[k] = v
    return new_state_dict


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])

    model = GPT(config)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=True)
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    return model, enc, ckpt, config


def get_param_text(config):
    if config.n_layer == 12 and config.n_embd == 768:
        return "GPT-2 Small / 约 1.24 亿参数"
    if config.n_layer == 24 and config.n_embd == 1024:
        return "GPT-2 Medium / 约 3.55 亿参数"
    return f"{config.n_layer} 层 / {config.n_head} heads / hidden {config.n_embd}"


def build_prompt(mode, user_input):
    text = user_input.strip()
    if mode == "问答模式":
        return f"User: {text}\nAssistant:"
    return text


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


# =========================
# 默认 checkpoint
# =========================
DEFAULT_V12 = "/root/autodl-tmp/GPT2_Small_Project/checkpoints/v12_sft_demo_overfit_zh/ckpt.pt"
DEFAULT_V16_BEST = "/root/autodl-tmp/GPT2_Medium_Project/checkpoints/v16_gpt2medium_chinese_1b/ckpt.pt"
DEFAULT_V16_LATEST = "/root/autodl-tmp/GPT2_Medium_Project/checkpoints/v16_gpt2medium_chinese_1b/latest.pt"
DEFAULT_V21_OFFICIAL = "/root/autodl-tmp/GPT2_Medium_Project/checkpoints/V2.1_official_gpt2medium/ckpt.pt"


# =========================
# 顶部 Hero
# =========================
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">GPT-2 中文模型交互式演示平台</div>
    <div class="hero-subtitle">
        支持 GPT-2 Small SFT 演示版与 GPT-2 Medium 预训练版模型测试。
        可进行问答生成、中文续写、参数调节和不同 checkpoint 对比。
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# 侧边栏
# =========================
with st.sidebar:
    st.markdown("## ⚙️ 模型配置")

    model_choice = st.selectbox(
        "选择模型",
        [
            "v12 Small SFT 演示版",
            "v16 Medium 最优预训练版",
            "v16 Medium 最后一轮预训练版",
            "V2.1 官方 GPT-2 Medium 权重版",
            "自定义 checkpoint 路径",
        ],
    )

    if model_choice == "v12 Small SFT 演示版":
        ckpt_path = DEFAULT_V12
    elif model_choice == "v16 Medium 最优预训练版":
        ckpt_path = DEFAULT_V16_BEST
    elif model_choice == "v16 Medium 最后一轮预训练版":
        ckpt_path = DEFAULT_V16_LATEST
    elif model_choice == "V2.1 官方 GPT-2 Medium 权重版":
        ckpt_path = DEFAULT_V21_OFFICIAL
    else:
        ckpt_path = st.text_input("checkpoint 路径", value=DEFAULT_V12)

    device_mode = st.radio(
        "推理设备",
        ["CPU", "CUDA"],
        horizontal=True,
    )

    if device_mode == "CUDA" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    st.markdown("---")
    st.markdown("## 🎛️ 生成参数")

    max_new_tokens = st.slider("最大生成 token 数", 20, 500, 220, 10)
    temperature = st.slider("Temperature", 0.1, 1.2, 0.5, 0.05)
    top_k = st.slider("Top-k", 1, 120, 40, 1)
    stop_eot = st.checkbox("遇到 <|endoftext|> 后截断显示", value=True)

    st.markdown("---")
    st.caption("建议：v12 用问答模式；v16 预训练模型更适合续写模式。")


# =========================
# 状态卡片
# =========================
ckpt_exists = Path(ckpt_path).exists()

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">当前模型</div>
    <div class="metric-value">{model_choice}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">推理设备</div>
    <div class="metric-value">{device.upper()}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_c:
    status_text = "可用" if ckpt_exists else "路径不存在"
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">Checkpoint 状态</div>
    <div class="metric-value">{status_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# =========================
# 主体输入区
# =========================
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 输入设置</div>', unsafe_allow_html=True)

    mode = st.radio(
        "选择输入模式",
        ["问答模式", "续写模式"],
        horizontal=True,
    )

    if mode == "问答模式":
        default_text = "什么是人工智能？"
        placeholder = "请输入一个问题，例如：什么是人工智能？"
        label = "用户问题"
    else:
        default_text = "人工智能的发展主要体现在"
        placeholder = "请输入一段开头，例如：人工智能的发展主要体现在"
        label = "续写开头"

    user_input = st.text_area(
        label,
        value=default_text,
        height=160,
        placeholder=placeholder,
    )

    prompt = build_prompt(mode, user_input)

    st.markdown(
        f"""
<div class="small-tip">
    <b>当前 Prompt：</b><br>
    <code>{prompt.replace("<", "&lt;").replace(">", "&gt;").replace(chr(10), "<br>")}</code>
</div>
""",
        unsafe_allow_html=True,
    )

    generate_button = st.button("开始生成")

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 当前 checkpoint</div>', unsafe_allow_html=True)

    st.code(ckpt_path, language="text")

    if not ckpt_exists:
        st.error("当前 checkpoint 路径不存在，请检查路径。")
    else:
        try:
            with st.spinner("正在读取模型信息..."):
                _, _, ckpt_meta, config_meta = load_model(ckpt_path, device)

            st.markdown(
                f"""
<div class="small-tip">
    <b>模型结构：</b>{get_param_text(config_meta)}<br>
    <b>层数：</b>{config_meta.n_layer}　
    <b>Heads：</b>{config_meta.n_head}　
    <b>隐藏维度：</b>{config_meta.n_embd}<br>
    <b>block size：</b>{config_meta.block_size}　
    <b>best val loss：</b>{ckpt_meta.get("best_val_loss", "N/A")}　
    <b>iter：</b>{ckpt_meta.get("iter_num", "N/A")}
</div>
""",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"模型信息读取失败：{e}")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# 输出区
# =========================
if generate_button:
    if not ckpt_exists:
        st.error(f"checkpoint 不存在：{ckpt_path}")
    elif not user_input.strip():
        st.warning("请输入内容后再生成。")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚀 生成结果</div>', unsafe_allow_html=True)

        with st.spinner("模型正在生成，请稍候..."):
            model, enc, _, _ = load_model(ckpt_path, device)
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

        st.markdown(
            f"""
<div class="output-box">{output.replace("<", "&lt;").replace(">", "&gt;")}</div>
""",
            unsafe_allow_html=True,
        )

        st.download_button(
            label="下载生成结果",
            data=output,
            file_name="generation_result.txt",
            mime="text/plain",
        )

        st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
<div class="footer-note">
    GPT-2 中文模型演示平台 · From Scratch Pretraining & SFT Demo
</div>
""",
    unsafe_allow_html=True,
)
