"""Direct Preference Optimization (DPO) on synthetic tool-call preferences."""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .agent import sequence_logprob
from .evaluate import evaluate_examples
from .tasks import make_dataset, make_preference_pairs
from .utils import (
    append_jsonl,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_json,
)


def train_dpo(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    policy, tokenizer, _ = load_checkpoint(args.base_checkpoint, map_location=device)
    reference, _, _ = load_checkpoint(args.base_checkpoint, map_location=device)
    policy.to(device)
    reference.to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    examples = make_dataset(args.train_size, seed=args.seed + 20_000)
    pairs = make_preference_pairs(examples, seed=args.seed + 30_000)
    val_examples = make_dataset(args.val_size, seed=args.seed + 40_000)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    metrics_path = out_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    rng = random.Random(args.seed)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        policy.train()
        rng.shuffle(pairs)
        losses: list[float] = []
        progress = tqdm(
            range(0, len(pairs), args.batch_size),
            desc=f"DPO epoch {epoch}/{args.epochs}",
        )
        for start in progress:
            batch = pairs[start : start + args.batch_size]
            loss_terms: list[torch.Tensor] = []
            for prompt, chosen, rejected in batch:
                pi_chosen = sequence_logprob(policy, tokenizer, prompt, chosen, device)
                pi_rejected = sequence_logprob(policy, tokenizer, prompt, rejected, device)
                with torch.no_grad():
                    ref_chosen = sequence_logprob(
                        reference, tokenizer, prompt, chosen, device
                    )
                    ref_rejected = sequence_logprob(
                        reference, tokenizer, prompt, rejected, device
                    )

                # DPO increases the relative probability of chosen over rejected
                # compared with the frozen reference model.
                advantage = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
                loss_terms.append(-F.logsigmoid(args.beta * advantage))

            loss = torch.stack(loss_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.3f}")

        eval_metrics, _ = evaluate_examples(
            policy, tokenizer, val_examples[: args.eval_size], device=device, greedy=True
        )
        row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": sum(losses) / max(1, len(losses)),
            **eval_metrics,
        }
        append_jsonl(metrics_path, row)
        print(row)

    checkpoint_path = out_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        policy,
        tokenizer,
        extra={
            "stage": "dpo",
            "base_checkpoint": args.base_checkpoint,
            "args": vars(args),
        },
    )
    final_metrics, rows = evaluate_examples(
        policy, tokenizer, val_examples[: args.eval_size], device=device, greedy=True
    )
    write_json(out_dir / "final_metrics.json", final_metrics)
    write_json(out_dir / "example_trace.json", rows[:5])
    return final_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--out_dir", default="outputs/dpo")
    parser.add_argument("--train_size", type=int, default=300)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--eval_size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = train_dpo(args)
    print(metrics)


if __name__ == "__main__":
    main()
