"""Shared utilities for training scripts."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .model import ModelConfig, TinyTransformerLM
from .tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: str | Path,
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model weights plus tokenizer and config in one file."""

    payload = {
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[TinyTransformerLM, CharTokenizer, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    config = ModelConfig.from_dict(payload["model_config"])
    model = TinyTransformerLM(config)
    model.load_state_dict(payload["model_state"])
    return model, tokenizer, payload.get("extra", {})

