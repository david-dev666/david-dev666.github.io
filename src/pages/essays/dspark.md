---
layout: ../../layouts/EssayLayout.astro
title: DSpark：用自回归头补上并行起草的短板
date: 2026-07-17
description: 并行骨干负责速度和首 token 质量，串行头用极低开销补上后续位置的条件依赖
---

## 1. 投机解码的效率公式

LLM 自回归解码速度慢，主要有两个原因：首先是推理算法无法并行，第 $t$ 个 token 的预测依赖于前 $t-1$ 个 token 的 KV 向量，靠堆算力无法加速；其次是显存访问瓶颈，每生成一个 token，需要将完整的模型权重从 HBM 读取到计算单元一次，计算量并不大，大部分时间在等待数据搬运。batch 推理能摊薄一部分访存开销，但由于推理请求是随机到达的，batch size 波动较大，并行度也有限。

为了加速 LLM 推理，有研究者提出了投机解码（Speculative Decoding）方法，其核心思路是"先起草，再验证"：用一个小模型（Draft Model）快速生成 N 个候选 token，再交给大模型（Target Model）并行验证，一次接受多个 token，从而减少大模型逐 token 解码的开销。

投机解码的效率公式：

<div class="math-block">

$$L = \frac{T_{draft} + T_{verify}}{\tau}$$

</div>

$L$ 是生成一个 token 的平均延迟，$T_{draft}$ 是起草耗时，$T_{verify}$ 是验证耗时，$\tau$ 是每轮平均被接受的 token 数。要降低 $L$，可以从三个方向入手：加速起草，加速验证，让起草更准确。


## 2. 两种起草方法

在 DSpark 之前，投机解码的起草模型分为两条路线。

**自回归起草**以 Eagle 系列为代表，其 draft 网络是 1～2 层的轻量 Transformer Decoder，输入是目标模型最后一层的 hidden state 序列和对应 token 的 embedding 序列拼接，输出是下一个位置的 hidden state，再由目标模型的 LM head 解码为 token。draft 网络每生成一个 token，就将其 append 到序列末尾再走一次前向，其运算方式与大模型很像，并且也维护了 KV cache 来避免重算。这种逐 token 的预测建模了 token 之间的条件依赖，起草质量较高。但为了起草速度，网络做得较浅，相当于用网络容量换推理速度。

Eagle 系列预测的是 hidden state 而非 token，好处在于：hidden state 是完整的语义表征，相比于 token 携带了更丰富的信息，因此 hidden state 层面的预测保留了更完整的信息，预测准确率更高；同时，hidden state 是连续空间，draft model 预测有误差时，LM head 仍有可能映射到正确的 token，其容错率比直接预测 token 要高。

**并行起草**以 DFlash 为代表，它是一个轻量级的块扩散模型（Block Diffusion），通过单步去噪生成 $\gamma$ 长度块的 hidden state，再并行采样出所有候选 token。

具体实现为：将待生成的 $\gamma$ 个位置全部填为固定的特殊 mask token，经目标模型的 embedding 层转换为 noise embedding；同时取目标模型多个中间层当前序列的 hidden states，拼接后投影到 draft 网络的 hidden dim，得到上下文特征 $x_{\text{ctx}}$。draft 网络通过 cross-attention 注入 $x_{\text{ctx}}$ 作为条件，单次前向完成去噪，输出 $\gamma$ 个位置的 hidden state，最后通过目标模型的 LM head 计算 logits，并采样得到候选 token。

DFlash 并行解码的优势在于：$T_{draft}$ 与 block 长度几乎无关——无论起草 8 个还是 16 个 token，都只需要一次前向，因此 draft 网络可以做得更深。**DFlash 的局限性在于：位置 $k$ 看到的是位置 $k-1$ 采样前的 hidden state，而非最终采样出的具体 token——当 $x_{k-1}$ 存在多个合理选项时，位置 $k$ 的预测和真实的条件分布有偏差，这就是论文所说的多模态碰撞（Multi-Modal Collision）。位置越靠后累积偏差越大，$\tau$ 随 block 长度衰减。** 为了保证起草的准确率，$\gamma$ 一般不大，Qwen3 取 16，LLaMA 取 10。


| 维度 | 自回归起草（Eagle） | 并行起草（DFlash） |
|------|---------------------|-------------------|
| 起草方式 | 特征层面逐 token 自回归 | 块扩散，单步去噪并行输出 |
| 网络结构 | 1–2 层轻量 Transformer Decoder | 多层 Transformer Decoder + cross-attention |
| $T_{draft}$ 与 $\gamma$ 关系 | $T_{draft} \propto \gamma$，块越长越慢 | 几乎无关，一次前向出整个块 |
| 典型 $\gamma$ | 8（MoE 取 4） | 16（LLaMA 取 10） |
| 首位置预测 | 弱（网络浅） | 强（网络深） |
| 后续位置预测 | 强（建模条件依赖） | 弱（看不到前一个 token） |
| $\tau$ 随位置变化 | 维持 | 迅速衰减 |

并行起草凭借其网络深度首个 token 预测更准，但缺乏 token 间的条件依赖，后续位置接受率衰减；自回归起草网络较浅，首位置预测偏弱，但后续位置能维持准确率。

## 3. 半自回归架构：重并行骨干 + 轻量串行头

并行起草速度略快，首位置 token 质量高，自回归起草的整体准确率较好。为了结合这两类方法的优势，DSpark 采用：并行骨干负责首位置质量和起草速度，轻量串行头负责补上后续位置的条件依赖。具体分两阶段。

**并行阶段**：一个较深的并行骨干网络（基于 DFlash），单次前向输出 $\gamma$ 个位置的隐藏状态 $h_1, \dots, h_\gamma$，再由目标模型 LM head 得到基础 logits $U_1, \dots, U_\gamma$。这一步保留了并行模型"首 token 质量高、$T_{draft}$ 与 $\gamma$ 无关"的优点。

**串行阶段**：并行骨干输出的 logits $U_k$ 是各位置独立预测的，位置 $k$ 不知道前面位置最终采样出了什么 token，串行阶段要补上这个条件依赖。具体做法是在基础 logits 上加一个偏置项 $B_k$：

<div class="math-block">

$$p_k(v) = \text{softmax}\big(U_k(v) + B_k(x_0, x_{<k}, v)\big)$$

</div>

对基础 logits 加上偏置修正后再走 softmax，每个位置的预测就变成了"给定前面 token"后的条件分布。$B_k$ 默认只用一阶马尔可夫依赖，即只取决于前一个位置真实采样出的 token $x_{k-1}$（$x_{k-1}$ 是起草过程中位置 $k-1$ 经采样确定的具体 token），并做低秩分解以控制参数量：

```python
# Markov 头：偏置只依赖上一个 token，做低秩分解降低参数量
# W1: (V, r)  W2: (r, V)  r 通常取 256，远小于词表大小 V
def markov_bias(x_prev, W1, W2):
    # x_prev: 上一步真实采样出的 token id
    row = W1[x_prev]          # (r,)  查表得到低秩表示
    bias = row @ W2           # (V,)  投影回整个词表维度
    return bias               # 作为 logits 的加法修正项
```

论文实验结果：

在网络深度上，2 层并行骨干 + 串行头的 DSpark，在所有任务上超过了 5 层纯并行骨干的 DFlash，说明串行头的效果好于堆深并行骨干。串行头的额外延迟很小，$\gamma=4$ 时占整轮 0.2%，$\gamma=16$ 时占 1.3%。加了串行头后，接受长度最高提升 30%，且 $\gamma$ 越大提升越大。在 Qwen3-4B/8B/14B 上，DSpark 的平均接受长度比 DFlash 高 16%–18%，比 Eagle3 高 27%–31%。

## 4. 总结

DSpark 结合了串行和并行两类投机解码方法的优势，并行骨干负责起草速度和首 token 质量，串行头用极低的开销补上后续位置的条件依赖。在 DeepSeek-V4 的线上流量中，相比之前的生产基线 MTP-1（每轮固定验证 2 个 token），DSpark 在同等吞吐下把单用户生成速度提升了 60%–85%。
