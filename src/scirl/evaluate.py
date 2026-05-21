"""Evaluation for the tool-using scientific agent."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from .agent import run_agent
from .reward import is_correct, score_tool_call
from .tasks import make_dataset
from .utils import ensure_dir, get_device, load_checkpoint, write_json


def evaluate_examples(
    model,
    tokenizer,
    examples,
    device,
    greedy: bool = True,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run the agent on examples and compute accuracy/reward metrics."""

    rows: list[dict[str, object]] = []
    rewards: list[float] = []
    correct_flags: list[bool] = []
    format_flags: list[bool] = []

    for idx, example in enumerate(examples):
        trace = run_agent(model, tokenizer, example, device=device, greedy=greedy)
        reward = score_tool_call(trace.call_text, example)
        correct = is_correct(trace.call_text, example)
        rows.append(
            {
                "idx": idx,
                "kind": example.kind,
                "target_expression": example.expression,
                "target_result": example.result,
                "generated_call": trace.call_text.replace("\n", "\\n"),
                "generated_expression": trace.expression or "",
                "observation": trace.observation if trace.observation is not None else "",
                "correct": correct,
                "reward": reward.reward,
                "error": trace.error or reward.error or "",
            }
        )
        rewards.append(reward.reward)
        correct_flags.append(correct)
        format_flags.append(trace.expression is not None)

    by_kind: dict[str, list[bool]] = {}
    for example, correct in zip(examples, correct_flags):
        by_kind.setdefault(example.kind, []).append(correct)

    metrics = {
        "n": len(examples),
        "accuracy": mean(correct_flags) if correct_flags else 0.0,
        "format_rate": mean(format_flags) if format_flags else 0.0,
        "mean_reward": mean(rewards) if rewards else 0.0,
    }
    for kind, flags in sorted(by_kind.items()):
        metrics[f"accuracy/{kind}"] = mean(flags)
    return metrics, rows


def save_rows_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="outputs/eval")
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of greedy decoding.")
    args = parser.parse_args()

    device = get_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    examples = make_dataset(args.eval_size, seed=args.seed)
    metrics, rows = evaluate_examples(
        model, tokenizer, examples, device=device, greedy=not args.sample
    )
    write_json(out_dir / "metrics.json", metrics)
    save_rows_csv(out_dir / "samples.csv", rows)
    print(metrics)


if __name__ == "__main__":
    main()

