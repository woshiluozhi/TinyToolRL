"""A tiny character-level tokenizer.

真实 LLM 使用 BPE/SentencePiece 等子词 tokenizer。为了让你能看清楚训练
逻辑，本项目用最简单的 char tokenizer：一个字符就是一个 token。

优点：完全自包含，不需要下载任何外部模型。
缺点：效率较低，长文本训练会慢；但本项目的文本很短，足够教学使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @classmethod
    def build(cls, texts: list[str]) -> "CharTokenizer":
        special = ["<pad>", "<eos>", "<unk>"]
        chars = sorted(set("".join(texts)))
        itos = special + chars
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                token = self.unk_token
            else:
                token = self.itos[idx]
            if skip_special and token in {self.pad_token, self.eos_token}:
                continue
            pieces.append(token)
        return "".join(pieces)

    def to_dict(self) -> dict[str, object]:
        return {
            "itos": self.itos,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CharTokenizer":
        itos = list(data["itos"])  # type: ignore[arg-type]
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(
            stoi=stoi,
            itos=itos,
            pad_token=str(data.get("pad_token", "<pad>")),
            eos_token=str(data.get("eos_token", "<eos>")),
            unk_token=str(data.get("unk_token", "<unk>")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

