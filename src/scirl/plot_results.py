"""Plot training curves from JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dir


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def plot_quickstart(output_root: str | Path = "outputs/quickstart") -> list[Path]:
    root = Path(output_root)
    figure_dir = ensure_dir(root / "figures")
    saved: list[Path] = []

    sft_rows = read_jsonl(root / "sft" / "train_metrics.jsonl")
    if sft_rows:
        plt.figure(figsize=(7, 4))
        plt.plot([r["epoch"] for r in sft_rows], [r["train_loss"] for r in sft_rows], marker="o")
        plt.xlabel("SFT epoch")
        plt.ylabel("masked LM loss")
        plt.title("SFT learning curve")
        plt.tight_layout()
        path = figure_dir / "sft_loss.png"
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(path)

        plt.figure(figsize=(7, 4))
        plt.plot([r["epoch"] for r in sft_rows], [r["accuracy"] for r in sft_rows], marker="o")
        plt.xlabel("SFT epoch")
        plt.ylabel("greedy tool-call accuracy")
        plt.ylim(0, 1)
        plt.title("Validation accuracy during SFT")
        plt.tight_layout()
        path = figure_dir / "sft_accuracy.png"
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(path)

    rl_rows = read_jsonl(root / "rl" / "train_metrics.jsonl")
    if rl_rows:
        plt.figure(figsize=(7, 4))
        plt.plot([r["update"] for r in rl_rows], [r["reward"] for r in rl_rows], alpha=0.8)
        plt.xlabel("RL update")
        plt.ylabel("sample reward")
        plt.title("RL reward curve")
        plt.tight_layout()
        path = figure_dir / "rl_reward.png"
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/quickstart")
    args = parser.parse_args()
    saved = plot_quickstart(args.output_root)
    print("saved figures:", [str(path) for path in saved])


if __name__ == "__main__":
    main()
