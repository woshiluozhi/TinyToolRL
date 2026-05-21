"""Reward functions for tool-use post-training.

强化学习后训练的关键是：我们到底奖励什么？这里把奖励拆成几部分：

1. 格式奖励：模型是否写出了 CALL: CALC[...]
2. 工具奖励：CALC 里的表达式算出来是否接近真值
3. 简洁奖励：输出不要无限变长

这不是“唯一正确”的奖励，而是一个可读、可改、可做 ablation 的起点。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .tasks import TaskExample
from .tools import CalculatorError, safe_calculate


CALL_RE = re.compile(r"CALL:\s*CALC\[(?P<expr>[^\]]+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    format_reward: float
    tool_reward: float
    length_penalty: float
    expression: str | None
    tool_value: float | None
    error: str | None


def extract_expression(text: str) -> str | None:
    """Pull the expression out of CALL: CALC[...]."""

    match = CALL_RE.search(text)
    if not match:
        return None
    return match.group("expr").strip()


def relative_error(pred: float, target: float) -> float:
    """Stable relative error that also works when target is near zero."""

    return abs(pred - target) / max(1.0, abs(target))


def score_tool_call(text: str, example: TaskExample) -> RewardBreakdown:
    """Score one generated tool call against the task ground truth."""

    expression = extract_expression(text)
    format_reward = 0.20 if expression is not None else -0.20
    tool_reward = 0.0
    value: float | None = None
    error: str | None = None

    if expression is not None:
        try:
            value = safe_calculate(expression)
            err = relative_error(value, example.result)
            # Smooth reward: exact answers get 0.90, small errors still get a
            # little signal, terrible errors approach 0.
            tool_reward = 0.90 * math.exp(-12.0 * err)
        except CalculatorError as exc:
            error = str(exc)
            tool_reward = -0.25

    # Very long completions usually mean the model is drifting out of the
    # required one-line protocol. Keep this small so it does not dominate.
    length_penalty = -0.02 * max(0, len(text) - 80) / 20
    reward = max(-1.0, min(1.0, format_reward + tool_reward + length_penalty))

    return RewardBreakdown(
        reward=reward,
        format_reward=format_reward,
        tool_reward=tool_reward,
        length_penalty=length_penalty,
        expression=expression,
        tool_value=value,
        error=error,
    )


def is_correct(text: str, example: TaskExample, tolerance: float = 1e-6) -> bool:
    """Return True if the tool call computes the exact target within tolerance."""

    expression = extract_expression(text)
    if expression is None:
        return False
    try:
        value = safe_calculate(expression)
    except CalculatorError:
        return False
    return relative_error(value, example.result) <= tolerance

