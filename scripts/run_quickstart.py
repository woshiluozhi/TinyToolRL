"""Run the whole mini research pipeline.

Usage:
    python scripts/run_quickstart.py --fast

The script trains:
1. SFT policy
2. DPO policy initialized from SFT
3. RL policy initialized from SFT
4. Simple plots
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scirl.plot_results import plot_quickstart  # noqa: E402
from scirl.train_dpo import train_dpo  # noqa: E402
from scirl.train_rl import train_rl  # noqa: E402
from scirl.train_sft import train_sft  # noqa: E402
from scirl.utils import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(ROOT / "outputs" / "quickstart"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fast", action="store_true", help="Short smoke-test run.")
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    if args.fast:
        sft_args = Namespace(
            out_dir=str(root / "sft"),
            train_size=220,
            val_size=80,
            eval_size=40,
            seed=0,
            device=args.device,
            epochs=2,
            batch_size=32,
            lr=5e-4,
            weight_decay=0.01,
            grad_clip=1.0,
            max_seq_len=384,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.10,
        )
        dpo_args = Namespace(
            base_checkpoint=str(root / "sft" / "checkpoint.pt"),
            out_dir=str(root / "dpo"),
            train_size=80,
            val_size=80,
            eval_size=40,
            seed=1,
            device=args.device,
            epochs=1,
            batch_size=8,
            lr=1e-4,
            beta=0.10,
            grad_clip=1.0,
        )
        rl_args = Namespace(
            base_checkpoint=str(root / "sft" / "checkpoint.pt"),
            out_dir=str(root / "rl"),
            train_size=100,
            val_size=80,
            eval_size=40,
            seed=2,
            device=args.device,
            updates=12,
            batch_size=4,
            lr=3e-5,
            temperature=0.9,
            top_k=20,
            max_new_tokens=80,
            kl_coef=0.02,
            entropy_coef=0.001,
            baseline_momentum=0.90,
            grad_clip=1.0,
            eval_every=6,
        )
    else:
        sft_args = Namespace(
            out_dir=str(root / "sft"),
            train_size=600,
            val_size=180,
            eval_size=100,
            seed=0,
            device=args.device,
            epochs=18,
            batch_size=32,
            lr=8e-4,
            weight_decay=0.01,
            grad_clip=1.0,
            max_seq_len=384,
            d_model=96,
            n_heads=4,
            n_layers=2,
            dropout=0.10,
        )
        dpo_args = Namespace(
            base_checkpoint=str(root / "sft" / "checkpoint.pt"),
            out_dir=str(root / "dpo"),
            train_size=240,
            val_size=180,
            eval_size=100,
            seed=1,
            device=args.device,
            epochs=1,
            batch_size=8,
            lr=3e-5,
            beta=0.10,
            grad_clip=1.0,
        )
        rl_args = Namespace(
            base_checkpoint=str(root / "sft" / "checkpoint.pt"),
            out_dir=str(root / "rl"),
            train_size=400,
            val_size=180,
            eval_size=100,
            seed=2,
            device=args.device,
            updates=40,
            batch_size=8,
            lr=8e-6,
            temperature=0.9,
            top_k=20,
            max_new_tokens=80,
            kl_coef=0.02,
            entropy_coef=0.001,
            baseline_momentum=0.90,
            grad_clip=1.0,
            eval_every=20,
        )

    print("\n=== Stage 1: SFT ===")
    sft_metrics = train_sft(sft_args)

    print("\n=== Stage 2: DPO ===")
    dpo_metrics = train_dpo(dpo_args)

    print("\n=== Stage 3: RL ===")
    rl_metrics = train_rl(rl_args)

    figures = plot_quickstart(root)
    summary = {
        "sft": sft_metrics,
        "dpo": dpo_metrics,
        "rl": rl_metrics,
        "figures": [str(path) for path in figures],
    }
    write_json(root / "summary.json", summary)
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
