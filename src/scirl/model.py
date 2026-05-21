"""A very small causal Transformer language model.

这个模型不是为了追求 SOTA，而是为了把“大模型后训练”的核心结构缩小到
一台普通电脑可以理解和运行的尺度。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 384
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.10

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int | float]) -> "ModelConfig":
        return cls(**data)


class TinyTransformerLM(nn.Module):
    """Causal Transformer for next-character prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Weight tying is common in language models: input and output token
        # embeddings share parameters, saving memory and often improving quality.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small GPT-style initialization.

        PyTorch's default Embedding init is much wider than tiny LM training
        needs, which can make the first losses enormous. A small normal init
        gives the optimizer a calmer start.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Causal mask: token t can attend to <=t, never to future tokens.
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=mask)
        logits = self.lm_head(self.norm(hidden))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
