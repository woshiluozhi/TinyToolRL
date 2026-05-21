"""PyTorch datasets and collation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .tasks import TaskExample
from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class LMItem:
    input_ids: torch.Tensor
    labels: torch.Tensor


class ToolCallSFTDataset(Dataset[LMItem]):
    """Dataset for supervised fine-tuning.

    We train a causal LM on prompt + ideal_call, but mask the prompt tokens in
    the loss. In other words, the model is graded only on the answer it should
    generate, not on reproducing the question.
    """

    def __init__(
        self,
        examples: list[TaskExample],
        tokenizer: CharTokenizer,
        max_seq_len: int,
    ):
        self.items: list[LMItem] = []
        for example in examples:
            prompt_ids = tokenizer.encode(example.policy_prompt(), add_eos=False)
            completion_ids = tokenizer.encode(example.policy_completion(), add_eos=True)
            full_ids = prompt_ids + completion_ids
            if len(full_ids) > max_seq_len:
                continue

            input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            labels = torch.tensor(full_ids[1:], dtype=torch.long)

            # labels[j] predicts full_ids[j+1]. If j+1 is still inside the
            # prompt, mask it out with -100.
            for j in range(len(labels)):
                if j + 1 < len(prompt_ids):
                    labels[j] = -100

            self.items.append(LMItem(input_ids=input_ids, labels=labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> LMItem:
        return self.items[idx]


def collate_lm(items: list[LMItem], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in items)
    input_batch = torch.full((len(items), max_len), pad_id, dtype=torch.long)
    label_batch = torch.full((len(items), max_len), -100, dtype=torch.long)

    for row, item in enumerate(items):
        seq_len = len(item.input_ids)
        input_batch[row, :seq_len] = item.input_ids
        label_batch[row, :seq_len] = item.labels

    return {"input_ids": input_batch, "labels": label_batch}
