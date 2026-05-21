# Small-scale RL Post-training for Tool-Using Scientific Agents

这是一个可运行、可学习、可扩展的小型研究项目。它把“大模型后训练 + 工具调用 + 强化学习”的核心流程缩小到普通电脑能跑的尺度。

作品雏形：代码自包含、实验可复现、有 SFT/DPO/RL、有评测、有图表、有英文 technical report。

## 项目做了什么

模型面对一类科学计算题，例如：

```text
Task: kinetic_energy
Question: An object has mass 4 kg and speed 7 m/s. What is its kinetic energy?
Formula hint: kinetic energy = 0.5 * mass * speed^2
Tool: CALC[arithmetic_expression]
Return the J answer by writing exactly one tool call line.
CALL: CALC[
```

它需要补全工具表达式：

```text
0.5*4*7**2]
```

环境会拼成完整工具调用并执行：

```text
CALL: CALC[0.5*4*7**2]
OBS: 98
FINAL: 98 J
```

这模拟了真实 tool-calling agent 的关键问题：模型不直接心算，而是学习什么时候、怎样调用可靠工具。

## 为什么这样设计

真实 LLM 后训练通常需要下载开源模型、GPU、数据清洗和复杂训练框架。这个项目为了教学和可复现，做了三件简化：

- 使用一个从零训练的 tiny character-level Transformer，而不是下载大模型。
- 使用合成科学题，保证每道题都有可验证真值。
- 使用结构化解码：固定 `CALL: CALC[` 前缀，模型只生成表达式。这类似真实函数调用或 JSON mode。

小而完整，比大而跑不动更适合入门。

## 目录结构

```text
src/scirl/
  tasks.py         # 合成科学题：欧姆定律、动能、密度、压强、加速度、波速
  tools.py         # 安全计算器 CALC[...]，不用危险的 eval
  reward.py        # 奖励函数：格式、计算正确性、长度惩罚
  tokenizer.py     # 教学用 char tokenizer
  model.py         # tiny causal Transformer
  data.py          # SFT 数据集和 padding
  agent.py         # 生成工具调用、执行工具、形成 trace
  train_sft.py     # supervised fine-tuning
  train_dpo.py     # preference optimization
  train_rl.py      # REINFORCE-style RL post-training
  evaluate.py      # 评测准确率、奖励、分任务表现
  plot_results.py  # 画训练曲线
scripts/
  run_quickstart.py # 一键跑 SFT + DPO + RL + plots
  demo_agent.py     # 查看模型生成案例
docs/
  LEARNING_GUIDE.zh-CN.md
  TECHNICAL_REPORT.md
  RESULTS.md
tests/
  test_core.py
```

## 快速开始

先运行单元测试：

```powershell
python -m unittest discover -s tests
```

跑一个很短的烟测：

```powershell
python scripts\run_quickstart.py --fast
```

`--fast` 只证明代码能闭环，不代表最终性能。想看到像样的学习效果，跑默认配置：

```powershell
python scripts\run_quickstart.py
```

训练完成后看几个样例：

```powershell
python scripts\demo_agent.py --checkpoint outputs\quickstart\sft\checkpoint.pt --n 5
```

本仓库当前已经跑过一次默认实验，结果记录在 [docs/RESULTS.md](docs/RESULTS.md)。该次运行中 SFT 准确率为 97%，DPO 为 97%，RL 为 96%。

也可以单独跑某个阶段：

```powershell
$env:PYTHONPATH='src'
python -m scirl.train_sft --out_dir outputs\sft --epochs 16
python -m scirl.evaluate --checkpoint outputs\sft\checkpoint.pt --out_dir outputs\sft_eval
python -m scirl.train_dpo --base_checkpoint outputs\sft\checkpoint.pt --out_dir outputs\dpo
python -m scirl.train_rl --base_checkpoint outputs\sft\checkpoint.pt --out_dir outputs\rl
```

## 输出文件

默认实验会生成：

```text
outputs/quickstart/
  sft/checkpoint.pt
  sft/final_metrics.json
  sft/train_metrics.jsonl
  dpo/checkpoint.pt
  rl/checkpoint.pt
  figures/sft_loss.png
  figures/sft_accuracy.png
  figures/rl_reward.png
  summary.json
```

重点看：

- `accuracy`: 工具表达式算出来是否等于真值。
- `format_rate`: 是否成功形成可解析的 `CALL: CALC[...]`。
- `mean_reward`: 奖励函数平均值，不完全等于准确率。
- `samples.csv`: 每个样例的生成表达式、错误信息和奖励。

## 你应该怎么学习

建议顺序：

1. 先读 [src/scirl/tasks.py](src/scirl/tasks.py)，理解数据怎么生成。
2. 再读 [src/scirl/tools.py](src/scirl/tools.py)，理解为什么不能直接 `eval` 模型输出。
3. 读 [src/scirl/train_sft.py](src/scirl/train_sft.py)，看 SFT 如何把 prompt/completion 变成 next-token loss。
4. 读 [src/scirl/reward.py](src/scirl/reward.py)，思考奖励设计会怎样影响 RL。
5. 读 [src/scirl/train_rl.py](src/scirl/train_rl.py)，理解 policy gradient、baseline、KL penalty。
6. 最后读 [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)，学习如何把项目包装成研究报告。

## 可以继续扩展的方向

- 把 char tokenizer 换成 Hugging Face tokenizer 和小模型。
- 把工具从 calculator 扩展到 Python sandbox、检索、单位换算。
- 加入更自然的题目措辞，测试 out-of-distribution generalization。
- 做 reward ablation：去掉格式奖励、去掉 KL、改变容错阈值。
- 把科学题换成你自己的物理场景。
