"""Supervised fine-tuning (SFT) for the tiny tool-call policy."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ToolCallSFTDataset, collate_lm
from .evaluate import evaluate_examples
from .model import ModelConfig, TinyTransformerLM, count_parameters
from .tasks import build_text_corpus, make_dataset
from .tokenizer import CharTokenizer
from .utils import (
    append_jsonl,
    ensure_dir,
    get_device,
    save_checkpoint,
    set_seed,
    write_json,
)


def train_sft(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    train_examples = make_dataset(args.train_size, seed=args.seed)
    val_examples = make_dataset(args.val_size, seed=args.seed + 10_000)
    tokenizer = CharTokenizer.build(build_text_corpus(train_examples + val_examples))
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = TinyTransformerLM(config).to(device)

    train_ds = ToolCallSFTDataset(train_examples, tokenizer, args.max_seq_len)
    if len(train_ds) == 0:
        raise ValueError("No training examples fit max_seq_len.")
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_lm, pad_id=tokenizer.pad_id),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    metrics_path = out_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    print(f"device={device} parameters={count_parameters(model):,} vocab={tokenizer.vocab_size}")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(loader, desc=f"SFT epoch {epoch}/{args.epochs}")
        losses: list[float] = []
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(input_ids, labels=labels)
            assert loss is not None

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.3f}")

        eval_subset = val_examples[: args.eval_size]
        eval_metrics, _ = evaluate_examples(
            model, tokenizer, eval_subset, device=device, greedy=True
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
        model,
        tokenizer,
        extra={
            "stage": "sft",
            "args": vars(args),
            "checkpoint": str(checkpoint_path),
        },
    )

    final_metrics, rows = evaluate_examples(
        model, tokenizer, val_examples[: args.eval_size], device=device, greedy=True
    )
    write_json(out_dir / "final_metrics.json", final_metrics)
    write_json(out_dir / "example_trace.json", rows[:5])
    return final_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/sft")
    parser.add_argument("--train_size", type=int, default=800)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--eval_size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = train_sft(args)
    print(metrics)


if __name__ == "__main__":
    main()
