# 中文学习指南：从物理博士到 AI 后训练项目

这份指南按“你是 AI 工程小白，但有科研训练”的视角写。你的优势不是已经会所有框架，而是能理解建模、误差、实验和机制。

## 1. 这个项目对应真实 AI 研究里的什么

真实大模型后训练常见流程：

1. 先用 SFT 让模型模仿高质量答案。
2. 再用偏好优化，比如 DPO，让模型更偏向好答案、远离坏答案。
3. 再用 RL，让模型根据可验证奖励继续探索。
4. 最后做严格评测，找失败案例。

本项目的缩小版：

1. SFT：模型模仿正确的 `CALC[...]` 表达式。
2. DPO：正确表达式是 chosen，错误公式是 rejected。
3. RL：模型采样一个表达式，计算器执行后给奖励。
4. 评测：检查表达式算出来是否等于真值。

你可以把它看成一个“显微镜”：不是直接看大模型工业系统，而是把机制放大给你看。

## 2. 数据：为什么用合成科学题

文件：[src/scirl/tasks.py](../src/scirl/tasks.py)

每道题都有：

- `prompt`: 给模型看的题目。
- `expression`: 正确的计算表达式。
- `result`: 真值。
- `bad_expressions`: 看起来合理但错的表达式，用于 DPO。

例如欧姆定律：

```text
voltage = current * resistance
```

如果电流是 `3 A`，电阻是 `5 ohm`，正确表达式是：

```text
3*5
```

这类数据的好处是可无限生成、可复现、可自动判分。缺点是语言分布太规整，所以不能声称解决了真实科学推理，只能说完成了一个可控原型。

## 3. 工具：为什么不用 eval

文件：[src/scirl/tools.py](../src/scirl/tools.py)

模型输出是不可信文本。如果直接：

```python
eval(model_output)
```

理论上模型可以输出删除文件、访问系统等危险内容。项目里使用 `ast.parse`，只允许数字、加减乘除、幂、少数数学函数。

这是 tool-using agent 的基本安全意识：工具能力越强，边界越重要。

## 4. 模型：tiny Transformer 是什么

文件：[src/scirl/model.py](../src/scirl/model.py)

这个模型是一个 causal language model。它做的事情很简单：

```text
给定前面的字符，预测下一个字符。
```

比如已经看到：

```text
CALL: CALC[3*
```

它要预测下一个字符可能是 `5`。

真实 LLM 也是 next-token prediction，只是 tokenizer 更复杂、参数更多、数据更大。

## 5. SFT：先学会模仿

文件：[src/scirl/train_sft.py](../src/scirl/train_sft.py)

SFT 的训练样本是：

```text
prompt = 题目 + "CALL: CALC["
completion = "3*5]\n"
```

模型只在 completion 上计算 loss，不要求它复述题目。这叫 prompt masking。

直觉上，SFT 是“老师带着做题”。它不鼓励探索，只鼓励模仿标准答案。

## 6. DPO：偏好优化

文件：[src/scirl/train_dpo.py](../src/scirl/train_dpo.py)

DPO 的一个训练样本：

```text
chosen:   3*5]
rejected: 3+5]
```

DPO 不需要显式 reward model。它直接让当前 policy 相对 reference model 更偏向 chosen。

简化理解：

```text
让 P(chosen) / P(rejected) 变大。
```

这对应 RLHF 里的偏好学习思想，但比 PPO 类方法更简单稳定。

## 7. RL：从奖励里学习

文件：[src/scirl/train_rl.py](../src/scirl/train_rl.py)

RL 阶段不再只看标准答案，而是让模型采样表达式：

```text
模型输出：3*5]
工具执行：15
奖励：接近真值，所以高
```

奖励来自 [src/scirl/reward.py](../src/scirl/reward.py)：

- 有可解析格式：加分。
- 表达式算对：大量加分。
- 输出太长：小惩罚。
- 表达式非法：惩罚。

训练算法是 REINFORCE：

```text
如果一个采样答案奖励高，就提高它的概率；
如果奖励低，就降低它的概率。
```

代码里还加了两个稳定技巧：

- baseline：降低梯度方差。
- KL penalty：别让 RL 后的模型偏离 SFT 模型太远。

## 8. 评测：不要只看 loss

文件：[src/scirl/evaluate.py](../src/scirl/evaluate.py)

语言模型 loss 降低不一定代表 agent 能做对题。这个项目的主指标是：

```text
accuracy = 工具表达式执行结果是否等于真值
```

同时也看：

- `format_rate`: 是否输出可解析工具调用。
- `mean_reward`: 奖励函数是否提升。
- `accuracy/<task>`: 哪类科学题最容易失败。

研究时要养成一个习惯：loss 是训练信号，accuracy 是任务表现，failure cases 是下一轮研究方向。

## 9. 你可以写进申请材料的研究问题

这个项目可以包装成：

```text
Can small language policies learn reliable scientific tool use through
supervised imitation, preference optimization, and verifiable RL rewards?
```

你可以讨论：

- 结构化解码能否显著提高小模型工具调用格式稳定性？
- SFT 学到的是公式，还是只记住模板？
- RL reward 会不会鼓励 reward hacking？
- DPO 在合成偏好上是否改善或损害泛化？
- 分任务 accuracy 的差异来自公式复杂度还是数字复制能力？

## 10. 下一步最有价值的改造

按申请价值排序：

1. 加自然语言扰动：同一个公式用 10 种问法。
2. 加单位换算：cm 到 m、g 到 kg，让工具调用更像科学推理。
3. 加 Python 工具：允许模型写一小段 Python，而不只是表达式。
4. 加真实开源模型：用 Qwen/Llama 小模型做 LoRA SFT。
5. 写 ablation 表：SFT-only、SFT+DPO、SFT+RL、不同 reward。

做到第 4、5 步，就更接近可以放进英文邮件和 technical report 的 AI 研究项目。

