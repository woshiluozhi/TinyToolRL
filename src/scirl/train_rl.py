"""Small-scale REINFORCE/RL post-training for tool-call generation."""

from __future__ import annotations

import argparse
import random

import torch
from tqdm import tqdm

from .agent import sample_tool_call_with_logprob, sequence_logprob
from .evaluate import evaluate_examples
from .reward import score_tool_call
from .tasks import TOOL_PREFIX, make_dataset
from .utils import (
    append_jsonl,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_json,
)


def train_rl(args: argparse.Namespace) -> dict[str, float]:
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

    train_examples = make_dataset(args.train_size, seed=args.seed + 50_000)
    val_examples = make_dataset(args.val_size, seed=args.seed + 60_000)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    metrics_path = out_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    rng = random.Random(args.seed)
    baseline = 0.0
    progress = tqdm(range(1, args.updates + 1), desc="RL updates")
    for update in progress:
        batch = rng.sample(train_examples, k=min(args.batch_size, len(train_examples)))
        loss_terms: list[torch.Tensor] = []
        raw_rewards: list[float] = []
        shaped_rewards: list[float] = []
        kl_terms: list[float] = []

        for example in batch:
            completion_text, logprob, entropy = sample_tool_call_with_logprob(
                policy,
                tokenizer,
                example.policy_prompt(),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
            call_text = TOOL_PREFIX + completion_text
            reward = score_tool_call(call_text, example).reward
            with torch.no_grad():
                ref_logprob = sequence_logprob(
                    reference, tokenizer, example.policy_prompt(), completion_text, device
                )

            # Sample-based KL estimate. It discourages the policy from moving
            # too far away from the SFT model in one small experiment.
            sample_kl = float((logprob.detach() - ref_logprob).item())
            shaped_reward = reward - args.kl_coef * sample_kl
            baseline = args.baseline_momentum * baseline + (
                1 - args.baseline_momentum
            ) * shaped_reward
            advantage = shaped_reward - baseline

            loss_terms.append(-advantage * logprob - args.entropy_coef * entropy)
            raw_rewards.append(reward)
            shaped_rewards.append(shaped_reward)
            kl_terms.append(sample_kl)

        loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        row = {
            "update": update,
            "loss": float(loss.item()),
            "reward": sum(raw_rewards) / max(1, len(raw_rewards)),
            "shaped_reward": sum(shaped_rewards) / max(1, len(shaped_rewards)),
            "sample_kl": sum(kl_terms) / max(1, len(kl_terms)),
            "baseline": baseline,
        }
        append_jsonl(metrics_path, row)
        progress.set_postfix(
            reward=f"{row['reward']:.3f}", kl=f"{row['sample_kl']:.3f}"
        )

        if update % args.eval_every == 0 or update == args.updates:
            eval_metrics, _ = evaluate_examples(
                policy,
                tokenizer,
                val_examples[: args.eval_size],
                device=device,
                greedy=True,
            )
            append_jsonl(out_dir / "eval_metrics.jsonl", {"update": update, **eval_metrics})
            print({"update": update, **eval_metrics})

    checkpoint_path = out_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        policy,
        tokenizer,
        extra={
            "stage": "rl",
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
    parser.add_argument("--out_dir", default="outputs/rl")
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--eval_size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--baseline_momentum", type=float, default=0.90)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = train_rl(args)
    print(metrics)


if __name__ == "__main__":
    main()
