"""Synthetic scientific tasks for the tiny tool-using agent.

这个文件负责“出题”。真实 LLM 后训练常常需要昂贵的人类数据或复杂
benchmark；这里我们用可控的合成科学计算题来学习完整研究流程。

每道题的目标不是让模型直接背答案，而是让模型学会输出一个 calculator
tool call，例如：

    CALL: CALC[3*5]

环境随后会执行 CALC[...] 里的表达式并生成最终答案。
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable


TOOL_PREFIX = "CALL: CALC["


def fmt_number(x: float) -> str:
    """Format a number for prompts and expressions.

    小模型很怕无意义的字符复杂度。这里尽量把 3.0 写成 3，把 3.50 写成
    3.5，让训练数据更干净。
    """

    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class TaskExample:
    """One synthetic problem.

    prompt:
        The input shown to the policy model.
    expression:
        The ideal arithmetic expression inside CALC[...].
    ideal_call:
        The exact one-line target used for supervised fine-tuning.
    result:
        Numeric ground truth produced by the expression.
    unit:
        Unit of the final answer.
    bad_expressions:
        Plausible but wrong expressions used to create rejected DPO samples.
    """

    kind: str
    prompt: str
    expression: str
    ideal_call: str
    result: float
    unit: str
    bad_expressions: tuple[str, ...]

    def final_answer(self) -> str:
        return f"{self.result:.4g} {self.unit}"

    def policy_prompt(self) -> str:
        """Prompt used by the LM policy.

        We constrain the fixed tool-call prefix and ask the model to generate
        only the arithmetic expression plus the closing bracket. This mirrors
        real structured decoding / function-calling systems.
        """

        return self.prompt + TOOL_PREFIX

    def policy_completion(self) -> str:
        return f"{self.expression}]\n"

    @staticmethod
    def completion_from_expression(expression: str) -> str:
        return f"{expression}]\n"

    def oracle_trace(self) -> str:
        """A full tool-use trace, useful for reports and debugging."""

        return (
            f"{self.ideal_call}"
            f"OBS: {self.result:.6g}\n"
            f"FINAL: {self.final_answer()}\n"
        )


def _prompt(kind: str, question: str, relation: str, unit: str) -> str:
    """Create a deliberately regular prompt.

    Regular prompts make the first version easier to train. Later you can make
    the wording more diverse to test real generalization.
    """

    return (
        f"Task: {kind}\n"
        f"Question: {question}\n"
        f"Formula hint: {relation}\n"
        "Tool: CALC[arithmetic_expression]\n"
        f"Return the {unit} answer by writing exactly one tool call line.\n"
    )


def _call(expression: str) -> str:
    return f"CALL: CALC[{expression}]\n"


def _ohm_law(rng: random.Random) -> TaskExample:
    current = rng.randint(1, 12)
    resistance = rng.randint(2, 30)
    expr = f"{fmt_number(current)}*{fmt_number(resistance)}"
    result = current * resistance
    prompt = _prompt(
        "ohm_law",
        f"A circuit has current {fmt_number(current)} A and resistance "
        f"{fmt_number(resistance)} ohm. What is the voltage?",
        "voltage = current * resistance",
        "V",
    )
    bad = (
        f"{fmt_number(current)}+{fmt_number(resistance)}",
        f"{fmt_number(resistance)}/{fmt_number(current)}",
        f"{fmt_number(current)}-{fmt_number(resistance)}",
    )
    return TaskExample("ohm_law", prompt, expr, _call(expr), result, "V", bad)


def _kinetic_energy(rng: random.Random) -> TaskExample:
    mass = rng.randint(1, 12)
    speed = rng.randint(2, 20)
    expr = f"0.5*{fmt_number(mass)}*{fmt_number(speed)}**2"
    result = 0.5 * mass * speed**2
    prompt = _prompt(
        "kinetic_energy",
        f"An object has mass {fmt_number(mass)} kg and speed "
        f"{fmt_number(speed)} m/s. What is its kinetic energy?",
        "kinetic energy = 0.5 * mass * speed^2",
        "J",
    )
    bad = (
        f"{fmt_number(mass)}*{fmt_number(speed)}",
        f"0.5*{fmt_number(mass)}*{fmt_number(speed)}",
        f"{fmt_number(mass)}*{fmt_number(speed)}**2",
    )
    return TaskExample("kinetic_energy", prompt, expr, _call(expr), result, "J", bad)


def _density(rng: random.Random) -> TaskExample:
    volume = rng.randint(2, 25)
    density = rng.randint(2, 15)
    mass = density * volume
    expr = f"{fmt_number(mass)}/{fmt_number(volume)}"
    result = mass / volume
    prompt = _prompt(
        "density",
        f"A sample has mass {fmt_number(mass)} g and volume "
        f"{fmt_number(volume)} cm^3. What is the density?",
        "density = mass / volume",
        "g/cm^3",
    )
    bad = (
        f"{fmt_number(mass)}*{fmt_number(volume)}",
        f"{fmt_number(volume)}/{fmt_number(mass)}",
        f"{fmt_number(mass)}+{fmt_number(volume)}",
    )
    return TaskExample("density", prompt, expr, _call(expr), result, "g/cm^3", bad)


def _pressure(rng: random.Random) -> TaskExample:
    area = rng.randint(2, 20)
    pressure = rng.randint(3, 18)
    force = pressure * area
    expr = f"{fmt_number(force)}/{fmt_number(area)}"
    result = force / area
    prompt = _prompt(
        "pressure",
        f"A force of {fmt_number(force)} N acts on area "
        f"{fmt_number(area)} m^2. What is the pressure?",
        "pressure = force / area",
        "Pa",
    )
    bad = (
        f"{fmt_number(force)}*{fmt_number(area)}",
        f"{fmt_number(area)}/{fmt_number(force)}",
        f"{fmt_number(force)}-{fmt_number(area)}",
    )
    return TaskExample("pressure", prompt, expr, _call(expr), result, "Pa", bad)


def _acceleration(rng: random.Random) -> TaskExample:
    mass = rng.randint(1, 15)
    acceleration = rng.randint(2, 16)
    force = mass * acceleration
    expr = f"{fmt_number(force)}/{fmt_number(mass)}"
    result = force / mass
    prompt = _prompt(
        "acceleration",
        f"A net force of {fmt_number(force)} N acts on mass "
        f"{fmt_number(mass)} kg. What is the acceleration?",
        "acceleration = force / mass",
        "m/s^2",
    )
    bad = (
        f"{fmt_number(force)}*{fmt_number(mass)}",
        f"{fmt_number(mass)}/{fmt_number(force)}",
        f"{fmt_number(force)}+{fmt_number(mass)}",
    )
    return TaskExample("acceleration", prompt, expr, _call(expr), result, "m/s^2", bad)


def _wave_speed(rng: random.Random) -> TaskExample:
    frequency = rng.randint(2, 20)
    wavelength = rng.randint(1, 12)
    expr = f"{fmt_number(frequency)}*{fmt_number(wavelength)}"
    result = frequency * wavelength
    prompt = _prompt(
        "wave_speed",
        f"A wave has frequency {fmt_number(frequency)} Hz and wavelength "
        f"{fmt_number(wavelength)} m. What is the wave speed?",
        "wave speed = frequency * wavelength",
        "m/s",
    )
    bad = (
        f"{fmt_number(frequency)}+{fmt_number(wavelength)}",
        f"{fmt_number(frequency)}/{fmt_number(wavelength)}",
        f"{fmt_number(wavelength)}/{fmt_number(frequency)}",
    )
    return TaskExample("wave_speed", prompt, expr, _call(expr), result, "m/s", bad)


GENERATORS: tuple[Callable[[random.Random], TaskExample], ...] = (
    _ohm_law,
    _kinetic_energy,
    _density,
    _pressure,
    _acceleration,
    _wave_speed,
)


def make_dataset(size: int, seed: int = 0) -> list[TaskExample]:
    """Generate a deterministic dataset.

    ``seed`` lets us reproduce the same train/validation split, which is
    essential for credible experiments.
    """

    rng = random.Random(seed)
    examples: list[TaskExample] = []
    for _ in range(size):
        generator = rng.choice(GENERATORS)
        examples.append(generator(rng))
    return examples


def build_text_corpus(examples: list[TaskExample]) -> list[str]:
    """Text used to build the character vocabulary."""

    corpus: list[str] = []
    for example in examples:
        corpus.append(example.prompt)
        corpus.append(example.policy_prompt())
        corpus.append(example.policy_completion())
        corpus.append(example.ideal_call)
        corpus.append(example.oracle_trace())
        corpus.extend(example.bad_expressions)
    return corpus


def make_preference_pairs(
    examples: list[TaskExample], seed: int = 0
) -> list[tuple[str, str, str]]:
    """Create (prompt, chosen, rejected) pairs for DPO.

    Chosen = correct tool call. Rejected = plausible but wrong formula.
    """

    rng = random.Random(seed)
    pairs: list[tuple[str, str, str]] = []
    for example in examples:
        rejected_expr = rng.choice(example.bad_expressions)
        pairs.append(
            (
                example.policy_prompt(),
                example.policy_completion(),
                example.completion_from_expression(rejected_expr),
            )
        )
    return pairs
