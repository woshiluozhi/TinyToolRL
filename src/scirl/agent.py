"""Generation and agent-environment interaction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .reward import extract_expression
from .tasks import TOOL_PREFIX, TaskExample
from .tokenizer import CharTokenizer
from .tools import CalculatorError, safe_calculate


@dataclass
class AgentTrace:
    prompt: str
    call_text: str
    expression: str | None
    observation: float | None
    final_text: str
    error: str | None


@torch.no_grad()
def generate_tool_call(
    model,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 20,
    greedy: bool = False,
    device: torch.device | str = "cpu",
) -> str:
    """Generate the model-controlled part of one tool call.

    In the default agent loop the prompt already ends with ``CALL: CALC[``.
    The model only needs to generate something like ``3*5]\n``.
    """

    model.eval()
    device = torch.device(device)
    ids = tokenizer.encode(prompt, add_eos=False)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        context = ids[-model.config.max_seq_len :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(input_ids)
        next_logits = logits[0, -1]

        if greedy:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_id = _sample_from_logits(next_logits, temperature, top_k)

        ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        if tokenizer.decode([next_id], skip_special=False) == "\n":
            break

    return tokenizer.decode(generated)


def run_agent(
    model,
    tokenizer: CharTokenizer,
    example: TaskExample,
    device: torch.device | str = "cpu",
    greedy: bool = True,
    temperature: float = 0.8,
) -> AgentTrace:
    """Generate a tool call, execute it, and format a final answer."""

    completion_text = generate_tool_call(
        model,
        tokenizer,
        example.policy_prompt(),
        device=device,
        greedy=greedy,
        temperature=temperature,
    )
    if completion_text.lstrip().upper().startswith("CALL:"):
        call_text = completion_text
    else:
        call_text = TOOL_PREFIX + completion_text
    expression = extract_expression(call_text)
    observation: float | None = None
    error: str | None = None
    if expression is None:
        error = "missing CALL: CALC[...]"
    else:
        try:
            observation = safe_calculate(expression)
        except CalculatorError as exc:
            error = str(exc)

    if observation is None:
        final = f"{call_text}OBS: ERROR\nFINAL: ERROR {example.unit}\n"
    else:
        final = (
            f"{call_text}"
            f"OBS: {observation:.6g}\n"
            f"FINAL: {observation:.4g} {example.unit}\n"
        )

    return AgentTrace(
        prompt=example.prompt,
        call_text=call_text,
        expression=expression,
        observation=observation,
        final_text=final,
        error=error,
    )


def sequence_logprob(
    model,
    tokenizer: CharTokenizer,
    prompt: str,
    completion: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Log probability of completion tokens conditioned on prompt."""

    device = torch.device(device)
    prompt_ids = tokenizer.encode(prompt, add_eos=False)
    completion_ids = tokenizer.encode(completion, add_eos=True)
    full_ids = prompt_ids + completion_ids
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits[0], dim=-1)

    selected = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    mask = torch.zeros_like(selected, dtype=torch.bool)
    for j in range(len(labels)):
        if j + 1 >= len(prompt_ids):
            mask[j] = True
    return selected[mask].sum()


def sample_tool_call_with_logprob(
    model,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device | str,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Sample a completion and return (text, logprob_sum, entropy_sum)."""

    model.train()
    device = torch.device(device)
    ids = tokenizer.encode(prompt, add_eos=False)
    generated: list[int] = []
    logprob_terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        context = ids[-model.config.max_seq_len :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(input_ids)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        next_logits = _top_k_logits(next_logits, top_k)
        dist = torch.distributions.Categorical(logits=next_logits)
        token = dist.sample()
        token_id = int(token.item())

        logprob_terms.append(dist.log_prob(token))
        entropy_terms.append(dist.entropy())
        generated.append(token_id)
        ids.append(token_id)

        if token_id == tokenizer.eos_id:
            break
        if tokenizer.decode([token_id], skip_special=False) == "\n":
            break

    if not logprob_terms:
        zero = torch.tensor(0.0, device=device)
        return "", zero, zero
    text = tokenizer.decode(generated)
    return text, torch.stack(logprob_terms).sum(), torch.stack(entropy_terms).sum()


def _sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-6)
    logits = _top_k_logits(logits, top_k)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, _ = torch.topk(logits, top_k)
    threshold = values[-1]
    return torch.where(logits < threshold, torch.full_like(logits, -float("inf")), logits)
