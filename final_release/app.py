import argparse
import json
import threading
import time
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from final_release.engine import ChineseGPT2Engine, GenerationConfig


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "static" / "index.html"
LOCK = threading.Lock()
ENGINE: ChineseGPT2Engine | None = None
GENERATION_CONFIG: GenerationConfig | None = None

app = FastAPI(
    title="V4.7 Chinese Assistant",
    version="4.7-final",
    docs_url=None,
    redoc_url=None,
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    latency_ms: int


def load_generation_config(path: Path) -> GenerationConfig:
    if not path.exists():
        raise FileNotFoundError(f"generation config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return GenerationConfig(**data)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> dict:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return ENGINE.status()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if ENGINE is None or GENERATION_CONFIG is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    history = [
        {"role": item.role, "content": item.content}
        for item in request.history
    ]
    started = time.perf_counter()
    with LOCK:
        answer = ENGINE.generate(
            request.message,
            history,
            GENERATION_CONFIG,
        )
    latency_ms = round((time.perf_counter() - started) * 1000)
    return ChatResponse(answer=answer, latency_ms=latency_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone V4.7 final Chinese assistant."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "model" / "model.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "model" / "spm_zh.model",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.json",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    return parser.parse_args()


def main() -> None:
    global ENGINE, GENERATION_CONFIG

    args = parse_args()
    GENERATION_CONFIG = load_generation_config(args.config)
    ENGINE = ChineseGPT2Engine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device_name=args.device,
        dtype_name=args.dtype,
    )

    state = ENGINE.status()
    print(
        f"loaded V4.7 final: iteration={state['iteration']}, "
        f"parameters={state['parameters']:,}, device={state['device']}"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
