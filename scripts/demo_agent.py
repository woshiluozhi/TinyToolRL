"""Show a few generated tool-use traces from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scirl.agent import run_agent  # noqa: E402
from scirl.tasks import make_dataset  # noqa: E402
from scirl.utils import get_device, load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "quickstart" / "sft" / "checkpoint.pt"))
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    examples = make_dataset(args.n, seed=args.seed)

    for i, example in enumerate(examples, start=1):
        trace = run_agent(model, tokenizer, example, device=device, greedy=True)
        print(f"\n--- Example {i}: {example.kind} ---")
        print(example.prompt)
        print("Generated trace:")
        print(trace.final_text)
        print("Oracle trace:")
        print(example.oracle_trace())


if __name__ == "__main__":
    main()

