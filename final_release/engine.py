from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


@dataclass(frozen=True)
class GenerationConfig:
    system_prompt: str
    max_turns: int = 6
    max_new_tokens: int = 120
    temperature: float = 0.0
    top_k: int = 30
    top_p: float = 0.85
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 4


def clean_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        cleaned[key] = value
    return cleaned


class ChineseGPT2Engine:
    def __init__(
        self,
        checkpoint_path: Path,
        tokenizer_path: Path,
        device_name: str = "cuda",
        dtype_name: str = "bfloat16",
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

        use_cuda = device_name == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dtype_name = dtype_name if use_cuda else "float32"
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.iter_num = int(checkpoint.get("iter_num", -1))
        self.best_val_loss = checkpoint.get("best_val_loss")

        config = GPTConfig(**checkpoint["config"])
        model = GPT(config)
        missing, unexpected = model.load_state_dict(
            clean_state_dict(checkpoint["model"]),
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint mismatch: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        model.to(self.device)
        model.eval()
        self.model = model

        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        if int(tokenizer.vocab_size()) != int(config.vocab_size):
            raise RuntimeError(
                f"tokenizer vocab_size={tokenizer.vocab_size()} does not match "
                f"checkpoint vocab_size={config.vocab_size}"
            )
        self.tokenizer = tokenizer
        self.stop_ids = self._collect_stop_ids()
        self.bad_ids = self._collect_bad_ids()

    def _piece_id(self, piece: str) -> int | None:
        piece_id = int(self.tokenizer.piece_to_id(piece))
        if piece_id < 0:
            return None
        if self.tokenizer.id_to_piece(piece_id) != piece:
            return None
        return piece_id

    def _collect_stop_ids(self) -> set[int]:
        stop_ids = set()
        eos_id = int(self.tokenizer.eos_id())
        if eos_id >= 0:
            stop_ids.add(eos_id)
        for piece in ("<user>", "<assistant>", "<system>", "<sep>"):
            piece_id = self._piece_id(piece)
            if piece_id is not None:
                stop_ids.add(piece_id)
        return stop_ids

    def _collect_bad_ids(self) -> set[int]:
        bad_ids = set()
        for piece_id in (
            self.tokenizer.unk_id(),
            self.tokenizer.bos_id(),
            self.tokenizer.pad_id(),
        ):
            piece_id = int(piece_id)
            if piece_id >= 0 and piece_id not in self.stop_ids:
                bad_ids.add(piece_id)
        return bad_ids

    def _precision_context(self):
        if self.device.type != "cuda" or self.dtype_name == "float32":
            return nullcontext()
        dtype = (
            torch.float16
            if self.dtype_name == "float16"
            else torch.bfloat16
        )
        return torch.amp.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _ban_token_ids(
        logits: torch.Tensor,
        token_ids: set[int],
    ) -> None:
        valid = [
            token_id
            for token_id in token_ids
            if 0 <= token_id < logits.size(-1)
        ]
        if valid:
            logits[:, valid] = -float("inf")

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: list[int],
        penalty: float,
    ) -> None:
        if penalty <= 1.0:
            return
        for token_id in set(generated_ids):
            if 0 <= token_id < logits.size(-1):
                score = logits[:, token_id].clone()
                logits[:, token_id] = torch.where(
                    score < 0,
                    score * penalty,
                    score / penalty,
                )

    @staticmethod
    def _banned_ngram_tokens(
        generated_ids: list[int],
        ngram_size: int,
    ) -> set[int]:
        if ngram_size < 2 or len(generated_ids) + 1 < ngram_size:
            return set()

        prefix = tuple(generated_ids[-(ngram_size - 1) :])
        banned = set()
        for index in range(len(generated_ids) - ngram_size + 1):
            ngram = tuple(generated_ids[index : index + ngram_size])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])
        return banned

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, top_k: int) -> None:
        if top_k <= 0:
            return
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = -float("inf")

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> None:
        if top_p <= 0.0 or top_p >= 1.0:
            return

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits.scatter_(1, sorted_indices, sorted_logits)

    @staticmethod
    def _clean_answer(text: str) -> str:
        for marker in (
            "<user>",
            "<assistant>",
            "<system>",
            "<sep>",
            "<bos>",
            "<eos>",
        ):
            if marker in text:
                text = text.split(marker)[0]
        return text.strip()

    @staticmethod
    def _format_prompt(
        history: list[dict[str, str]],
        message: str,
        system_prompt: str,
        max_turns: int,
    ) -> str:
        parts = []
        if system_prompt.strip():
            parts.append(f"<system>\n{system_prompt.strip()}")

        for item in history[-max_turns * 2 :]:
            role = item.get("role", "")
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                parts.append(f"<{role}>\n{content}")

        parts.append(f"<user>\n{message.strip()}")
        parts.append("<assistant>\n")
        return "\n<sep>\n".join(parts)

    def _encode_prompt(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer.encode(prompt, out_type=int)
        bos_id = int(self.tokenizer.bos_id())
        if bos_id >= 0:
            token_ids = [bos_id] + token_ids
        return token_ids

    @torch.inference_mode()
    def generate(
        self,
        message: str,
        history: list[dict[str, str]],
        config: GenerationConfig,
    ) -> str:
        prompt = self._format_prompt(
            history,
            message,
            config.system_prompt,
            config.max_turns,
        )
        input_ids = self._encode_prompt(prompt)
        token_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )[None, :]
        generated: list[int] = []

        for _ in range(config.max_new_tokens):
            model_input = token_ids[:, -self.model.config.block_size :]
            with self._precision_context():
                logits = self.model(model_input)[:, -1, :]

            self._ban_token_ids(logits, self.bad_ids)
            self._ban_token_ids(
                logits,
                self._banned_ngram_tokens(
                    generated,
                    config.no_repeat_ngram_size,
                ),
            )
            self._apply_repetition_penalty(
                logits,
                generated,
                config.repetition_penalty,
            )

            if config.temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / config.temperature
                self._top_k_filter(logits, config.top_k)
                self._top_p_filter(logits, config.top_p)
                probabilities = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)

            next_token = int(next_id.item())
            if next_token in self.stop_ids:
                break

            generated.append(next_token)
            token_ids = torch.cat((token_ids, next_id), dim=1)

        return self._clean_answer(self.tokenizer.decode(generated))

    def status(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "dtype": self.dtype_name,
            "parameters": self.model.num_parameters(),
            "iteration": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "checkpoint": str(self.checkpoint_path),
            "tokenizer": str(self.tokenizer_path),
        }
