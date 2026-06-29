from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head

        query = query.view(
            batch_size,
            sequence_length,
            self.n_head,
            head_size,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            sequence_length,
            self.n_head,
            head_size,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.n_head,
            head_size,
        ).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            channels,
        )
        return self.resid_dropout(self.c_proj(output))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
        )
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block size "
                f"{self.config.block_size}"
            )

        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=token_ids.device,
        )
        x = (
            self.transformer.wte(token_ids)
            + self.transformer.wpe(positions)
        )
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, -1:, :]), None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
        )
        return logits, loss

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
