---
layout: ../../layouts/ProjectLayout.astro
title: RoPE旋转位置编码
description: 从直觉到源码理解旋转位置编码
tags: ["LLM", "Transformer", "Positional Embedding"]
---

# RoPE: 旋转位置编码

**RoPE**（Rotary Positional Embedding）是目前主流大模型（LLaMA、Qwen、Mistral等）采用的位置编码方案。

**核心思想**：用向量的旋转角度来编码位置信息。位置$m$的向量旋转$m \cdot \theta$度，位置$n$的向量旋转$n \cdot \theta$度，两者点积时相对角度差为$(m-n) \cdot \theta$——只与相对距离有关，与绝对位置无关。

## 1. 位置编码的演进

| 阶段 | 代表模型 | 方法 | 核心缺陷 |
|:---|:---|:---|:---|
| **可学习绝对编码** | BERT, GPT-2 | 直接学习位置矩阵 | 外推性为零，超出训练长度即失效 |
| **加性Sin/Cos编码** | Original Transformer | 位置向量**加**到词向量 | 混淆语义与位置信息 |
| **旋转位置编码** | LLaMA, Qwen | 位置信息**乘**入向量 | ✓ 解决上述问题 |

## 2. 为什么选择RoPE

RoPE的三个核心优势：

- **相对位置编码**：通过旋转角度的差值表示相对距离，符合"词与词的关系只取决于距离"的直觉
- **保留语义信息**：旋转操作不改变向量模长，最大程度保留原始语义
- **理论可外推**：基于周期性三角函数，理论上可处理任意长度序列

## 3. 数学直觉：复平面上的旋转

将词向量的每两个维度看作一个复数$z = x + iy$，位置编码就是让这个复数旋转。

### 3.1 旋转角度的计算

当一个词出现在句子的第$m$个位置时，RoPE会对这个词向量内部的每一个"时钟"进行拨动：

$$
\text{旋转角度} = \text{当前位置 } m \times \text{该维度的固有速度 } \theta
$$

- **Token A（位置1）**：它的所有时钟都转动了$1$个单位的时间
- **Token B（位置2）**：它的所有时钟都转动了$2$个单位的时间
- **Token C（位置100）**：它的所有时钟都转动了$100$个单位的时间

### 3.2 为什么能表示相对位置？

Transformer关注的是词与词之间的关系（Attention），这通过**向量点积**计算。几何上，点积取决于两个向量的**夹角差**。

假设计算Token B（$m=2$）和Token C（$n=100$）的关系：

| Token | 旋转角度 |
|:---|:---|
| B | $2 \cdot \theta$ |
| C | $100 \cdot \theta$ |
| **相对角度差** | $(100-2) \cdot \theta = 98\theta$ |

其中$\theta$是该维度对应的基础频率，不同维度的$\theta$值不同，形成多尺度的位置感知。

**结论**：模型不需要知道B和C的绝对位置，只需要看它们的角度差，就能算出中间隔了98个步长。这就是**相对位置编码**的本质。

用复数形式表示，设Query在位置$m$，Key在位置$n$：

$$
\text{Re}[(z_q \cdot e^{im\theta}) \cdot \overline{(z_k \cdot e^{in\theta})}] = \text{Re}[z_q \cdot \bar{z}_k \cdot e^{i(m-n)\theta}]
$$

结果只依赖相对距离$(m-n)$，与绝对位置无关。

## 4. 代码实现（LLaMA风格）

以下代码参考Meta LLaMA官方实现，使用复数运算替代旋转矩阵乘法以利用GPU加速。

### 4.1 预计算频率表

```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转角度（复数形式）
    
    Args:
        dim: 每个注意力头的维度 (head_dim)
        end: 最大序列长度 (max_seq_len)
        theta: 基础频率，LLaMA默认10000.0
    Returns:
        freqs_cis: [end, dim//2] 复数张量
    """
    # 1. 计算每个维度对的频率: theta^(-2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. 生成位置序列: [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    
    # 3. 外积得到每个(位置, 维度)的旋转角度
    freqs = torch.outer(t, freqs).float()
    
    # 4. 转为复数: e^(i*angle) = cos(angle) + i*sin(angle)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
```

**关键点解析**：

- **频率设计**：低维度用高频（变化快），高维度用低频（变化慢），使模型能同时捕获局部和全局位置关系
- **复数表示**：`torch.polar(r, θ)`生成$r \cdot e^{i\theta}$，这里$r=1$，即单位圆上的点

### 4.2 应用旋转编码

```python
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    将RoPE应用到Query和Key
    
    Args:
        xq: Query [batch, seq_len, n_heads, head_dim]
        xk: Key   [batch, seq_len, n_heads, head_dim]
        freqs_cis: 预计算的复数频率表
    """
    # 1. 将最后一维两两分组，转为复数
    # [B, L, H, D] -> [B, L, H, D/2, 2] -> [B, L, H, D/2] (complex)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. 调整freqs_cis形状以支持广播
    # [L, D/2] -> [1, L, 1, D/2]
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(-1))
    
    # 3. 核心操作：复数乘法 = 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**关键点解析**：

- **复数乘法即旋转**：$(a+bi)(c+di)$在几何上等价于向量旋转，避免了显式构造旋转矩阵
- **只作用于Q和K**：Value不需要位置编码，因为位置信息已通过Q-K点积传递

## 5. 总结

| 要点 | 说明 |
|:---|:---|
| **核心操作** | 将词向量按维度两两分组，在复平面上旋转 |
| **位置编码** | 位置$m$ → 旋转角度$m \cdot \theta$ |
| **相对性来源** | 点积时角度相减，只保留相对距离$(m-n)$ |
| **工程实现** | 复数乘法替代矩阵旋转，GPU友好 |

## 6. 相关词条

- [Attention机制](/glossary/attention)
- [GQA分组查询注意力](/glossary/GQA)

## 7. 推荐阅读

- **苏剑林（苏神）**：[Transformer升级之路：博采众长的旋转式位置编码](https://kexue.fm/archives/8265)——RoPE提出者的原始博客，中文社区最权威的数学推导
