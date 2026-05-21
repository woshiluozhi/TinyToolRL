# A Reproducible Study of Small-Scale RL Post-Training for Tool-Using Scientific Agents

## Abstract

This project studies a minimal but complete post-training pipeline for tool-using scientific agents. A tiny character-level Transformer is trained to produce structured calculator calls for synthetic physics-style problems. We compare supervised fine-tuning (SFT), direct preference optimization (DPO), and REINFORCE-style reinforcement learning (RL) with verifiable rewards. The goal is not to build a competitive large language model, but to expose the core mechanics of tool-use post-training in a reproducible setting that can run on a single consumer machine.

## 1. Introduction

Tool use is a central capability of modern language agents. Instead of relying on parametric memory or mental arithmetic, an agent can call external tools such as calculators, Python interpreters, search engines, or scientific databases. This project focuses on a small scientific setting where each task has an unambiguous numeric answer and can be checked automatically.

The main research question is:

> Can a small language policy learn reliable scientific tool calls through supervised imitation, preference optimization, and verifiable RL rewards?

The study uses structured decoding. The environment provides the fixed prefix `CALL: CALC[`, and the model generates only the arithmetic expression and closing bracket. This mirrors practical constrained generation settings such as function calling, JSON mode, and grammar-constrained decoding.

## 2. Task Design

The benchmark consists of synthetic scientific calculation tasks:

- Ohm's law: `voltage = current * resistance`
- Kinetic energy: `energy = 0.5 * mass * speed^2`
- Density: `density = mass / volume`
- Pressure: `pressure = force / area`
- Acceleration: `acceleration = force / mass`
- Wave speed: `speed = frequency * wavelength`

Each example contains:

- A natural-language prompt with a formula hint.
- A ground-truth arithmetic expression.
- A numeric result.
- Plausible incorrect expressions for preference training.

Because the data are synthetic, evaluation is exact and reproducible. This makes the setup suitable for studying training dynamics and reward design.

## 3. Model and Tokenization

The model is a small causal Transformer implemented in PyTorch. It uses a character-level tokenizer to avoid external model downloads and make the pipeline fully self-contained. The Transformer predicts the next character conditioned on the prompt and previous generated characters.

This design intentionally trades realism for clarity. Character-level modeling is less efficient than subword tokenization, but it makes the entire system transparent and easy to inspect.

## 4. Methods

### 4.1 Supervised Fine-Tuning

SFT trains the policy to imitate oracle tool calls. The prompt is:

```text
<problem text>
CALL: CALC[
```

The completion is:

```text
<correct expression>]
```

The loss is masked so that only completion tokens contribute to the objective.

### 4.2 Direct Preference Optimization

DPO uses synthetic preference pairs. For each prompt, the chosen response is the correct expression, and the rejected response is a plausible but wrong formula. A frozen SFT model is used as the reference policy.

The DPO loss increases the relative log probability of the chosen completion over the rejected completion compared with the reference model.

### 4.3 RL Post-Training

RL uses sampled completions and a verifiable reward. The environment parses `CALL: CALC[...]`, executes the expression with a safe calculator, and compares the result with the ground truth.

The reward includes:

- Format reward for producing a parseable tool call.
- Tool reward based on relative numeric error.
- A small length penalty.
- A penalty for invalid or unsafe expressions.

The implementation uses REINFORCE with an exponential moving baseline and a sample-based KL penalty against the SFT reference model.

## 5. Evaluation

The main metric is exact tool-call accuracy:

```text
accuracy = 1 if safe_calculate(generated_expression) equals target_result
```

Additional metrics include:

- Format rate.
- Mean reward.
- Per-task accuracy.
- Qualitative traces.

The command below runs the full pipeline:

```powershell
python scripts\run_quickstart.py
```

Results are saved under:

```text
outputs/quickstart/
```

In the verified run recorded in `docs/RESULTS.md`, SFT reached 97% exact tool-call accuracy, DPO reached 97%, and RL reached 96% on 100 held-out synthetic examples.

## 6. Expected Findings

A short smoke test may mainly validate code execution and often has low exact accuracy. With the default training configuration, SFT should learn stable tool-call formatting and substantially improve exact calculator-call accuracy. DPO and RL are included as post-training methods for comparison and ablation; because the reward and preference data are synthetic, their effect depends on the strength of the SFT initialization and hyperparameters.

Important observations to inspect:

- SFT loss should decrease monotonically.
- Format rate should approach 1 before exact accuracy becomes high.
- More complex formulas, especially kinetic energy, may require more training because they include `0.5` and exponentiation.
- RL can improve sampled reward but may degrade deterministic greedy accuracy if the learning rate or KL penalty is poorly tuned.

## 7. Limitations

This is a controlled prototype, not a claim of state-of-the-art scientific reasoning.

Key limitations:

- Synthetic prompts are regular and include formula hints.
- The model is tiny and trained from scratch.
- Tool use is limited to one calculator call.
- The environment formats the final answer deterministically.
- DPO preferences are synthetic rather than human-labeled.

These limitations are acceptable for a learning project, but they should be stated clearly in any application or research discussion.

## 8. Future Work

The next research extensions are:

1. Add paraphrased prompts and evaluate out-of-distribution generalization.
2. Add unit conversion and multi-step scientific tasks.
3. Replace the tiny Transformer with an open-source small LLM and LoRA fine-tuning.
4. Add Python execution as a second tool.
5. Run reward ablations and compare SFT-only, SFT+DPO, and SFT+RL.
6. Write a workshop-style paper with tables, plots, and failure analysis.

## 9. Relevance for an AI PhD Transition

This project demonstrates several skills that are legible to AI supervisors:

- Implementing a complete post-training pipeline.
- Designing verifiable rewards.
- Building safe tool execution.
- Running reproducible experiments.
- Reporting limitations and future work clearly.

For a physics-trained researcher, this project also shows a credible bridge from mathematical/scientific problem solving to agentic AI and reinforcement learning.
