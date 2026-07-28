---
layout: ../../layouts/ProjectLayout.astro
title: Rejection Sampling
description: LLM/VLM对齐中的拒绝采样技术详解
tags: ["LLM", "RLHF", "Alignment"]
---

# Rejection Sampling 拒绝采样

在LLM和VLM的训练流程中，**拒绝采样**（Rejection Sampling，常被称为Best-of-N）是一种高效的对齐技术。

核心理念非常朴素：**与其费力教模型如何写出好答案，不如先让模型生成一堆答案，然后筛选出最好的那个让它模仿。** 这一方法在LLaMA 2和Qwen的训练中发挥了关键作用。

## 1. 核心逻辑：优中选优

大模型在生成内容时存在随机性。对于同一个Prompt，模型可能生成质量参差不齐的多个回复。拒绝采样的目标是将模型生成的**数据分布**向**高质量区间**移动。

**标准流程**

1. **采样**（Sampling）：给定提示词，让策略模型以较高的Temperature生成$N$个候选回复
2. **打分**（Scoring）：使用奖励模型对这$N$个回复进行质量打分
3. **筛选**（Selection）：保留得分最高的那一个回复，拒绝其余回复
4. **微调**（Training）：将`<Prompt, Best Response>`作为新的训练数据，对模型进行SFT

## 2. 代码实现

以下代码模拟了一个标准的Best-of-N采样流程。在推理阶段利用高温度系数鼓励模型探索，在筛选阶段利用奖励模型收敛质量。

```python
import torch

class RejectionSampler:
    def __init__(self, policy_model, reward_model, tokenizer):
        self.policy = policy_model  # 生成模型（如LLaMA）
        self.rm = reward_model      # 奖励模型（打分器）
        self.tokenizer = tokenizer

    def generate_best_sample(self, prompt, n=8, temperature=1.0):
        """
        执行Best-of-N采样
        
        Args:
            prompt: 输入提示词
            n: 采样数量，通常为8到128
            temperature: 采样温度，高温度有助于增加多样性
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.policy.device)
        
        # 1. 批量生成（Exploration）
        # 使用高温度鼓励模型生成多样的回答
        # output_sequences: [n, seq_len]
        with torch.no_grad():
            output_sequences = self.policy.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=n,
                temperature=temperature,
                do_sample=True,  # 必须开启采样
                top_p=0.9
            )

        # 解码生成的文本
        candidates = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        # 2. 奖励打分（Exploitation）
        # 让RM评估每一个候选回答的质量
        # scores: [n]
        with torch.no_grad():
            scores = self.rm.score(prompt, candidates)

        # 3. 拒绝与接受
        # 选择分数最高的索引
        best_idx = torch.argmax(scores).item()
        
        return {
            "prompt": prompt,
            "response": candidates[best_idx],
            "score": scores[best_idx]
        }
```

**工程关键点**

- **Temperature设置**：必须设置`temperature > 0.7`。如果温度过低，模型生成的$N$个结果几乎一样，筛选就失去了意义
- **迭代式训练**：Meta在LLaMA 2论文中提到，这个过程可以迭代进行。用V1模型采样出的最佳数据训练出V2模型，再用V2模型采样训练V3，模型能力会螺旋上升

## 3. VLM领域的特殊应用

在视觉多模态模型中，拒绝采样常用于解决**幻觉问题**（Hallucination）。

VLM经常会看错图片细节，例如数错物体数量。在此场景下，我们不需要训练复杂的神经网络奖励模型，可以直接使用**规则检测器**作为打分器。

**案例：思维链推理增强**

| 步骤 | 内容 |
|:---|:---|
| **Prompt** | 请一步步推理，图中红色的苹果有几个？ |
| **Ground Truth** | 3个 |
| **采样** | 让模型生成10条不同的推理路径 |
| **筛选** | 解析每条路径的最终答案。答案=3则保留，否则拒绝 |
| **结果** | 即使模型原本只有30%的概率答对，通过拒绝采样，我们能构造出100%正确的推理数据集用于微调 |

## 4. 拒绝采样 vs PPO

拒绝采样和近端策略优化（PPO）是RLHF阶段的两种主要路线。

| 维度 | 拒绝采样（Best-of-N） | PPO（强化学习） |
|:---|:---|:---|
| **本质** | **离线选择**：先选好数据，再做监督学习 | **在线更新**：边生成边根据反馈调整梯度 |
| **稳定性** | **极高**：本质是SFT，不易训练崩溃 | **低**：对超参数极度敏感，容易不收敛 |
| **计算资源** | **推理密集型**：需要大量推理算力生成样本 | **训练密集型**：需要同时加载Actor, Critic, Ref, RM四个模型 |
| **能力上限** | 受限于当前模型的最佳表现 | 理论上可探索出模型从未见过的更优解 |

## 5. 总结

拒绝采样是一种**用推理换性能**的策略。

它将复杂的强化学习问题转化为了稳定的监督学习问题。在工程实践中，如果你没有足够的算力或经验去调试不稳定的PPO，拒绝采样是提升模型对齐效果的首选方案。

## 6. 相关词条

- [Attention机制](/glossary/attention)
- [Dropout正则化](/glossary/dropout)
