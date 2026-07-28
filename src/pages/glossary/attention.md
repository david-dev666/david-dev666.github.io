---
layout: ../../layouts/ProjectLayout.astro
title: Attention机制
description: 深度学习中的注意力机制详解
tags: ["Transformer", "NLP", "Deep Learning"]
---

# Attention机制

注意力机制（Attention Mechanism）的核心在于让模型学会"聚焦"：在处理数据时动态分配权重，关注输入中最关键的信息。

在2017年之前，NLP领域由RNN和LSTM主导。随着Google发表论文《Attention Is All You Need》，Transformer架构正式诞生。它证明了我们不需要复杂的循环神经网络，仅通过Attention机制就能高效地捕获序列中的长距离依赖，彻底改变了深度学习的处理范式。

## 1. 核心思想

Attention机制的本质是一个**基于内容的寻址（Content-based Addressing）**过程。它模拟了人类视觉的聚焦行为，通过计算相关性权重，从输入中提取关键信息。

在工程实现上，这相当于一个软性的数据库查询系统：

- **Query(Q)—查询向量**  
  代表当前的查询意图。即"当前关注点是什么？"（类似于SQL中的WHERE条件）

- **Key(K)—键向量**  
  代表数据的索引特征。用于与Q进行匹配（点积运算），计算两者之间的相似度或权重

- **Value(V)—值向量**  
  代表数据的实际内容。模型根据Q和K计算出的权重，对V进行加权求和，从而聚合出最相关的信息

## 2. 数学表达

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

| 符号 | 维度形状 | 物理含义 |
|------|----------|----------|
| $Q$ | $(n, d_k)$ | Query矩阵（查询变量），代表目标序列信息 |
| $K$ | $(m, d_k)$ | Key矩阵（索引变量），代表源序列信息 |
| $V$ | $(m, d_v)$ | Value矩阵（内容变量），承载源序列的特征表示 |
| $n$ | Scalar | Query序列长度（输出/目标序列长度） |
| $m$ | Scalar | Key/Value序列长度（输入/源序列长度） |
| $\sqrt{d_k}$ | Scalar | 缩放因子，用于平滑Softmax梯度，防止梯度消失 |

**维度设计关键说明**：

**1. 序列长度关系（$\boldsymbol{n}$ vs $\boldsymbol{m}$）**：

- **Self-Attention（自注意力）**：$Q, K, V$均来自同一输入序列，故$n = m$

- **Cross-Attention（交叉注意力）**：$Q$来自解码器（目标序列），$K, V$来自编码器（源序列），此时$n \neq m$（例如机器翻译中，源句长10，译文长8）

**2. 特征维度（$\boldsymbol{d_k}$ vs $\boldsymbol{d_v}$）**：

$\boldsymbol{d_k}$决定点积相似度的计算空间，$\boldsymbol{d_v}$决定输出向量的特征维度

工程实践中，为便于多头（Multi-Head）拼接与残差连接，通常设$d_k = d_v$

## 3. 代码实现（PyTorch）

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch_size, n, d_k]
    K: [batch_size, m, d_k]
    V: [batch_size, m, d_v]
    """
    d_k = Q.size(-1)

    # 1. 计算分数（Score）
    # [batch, n, d_k] x [batch, d_k, m] -> [batch, n, m]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 应用掩码（Masking）
    if mask is not None:
        # 将mask为0的位置设为极小值，Softmax后变为0
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. 计算权重（Weights）
    # 对最后一维（m）进行归一化
    attn_weights = F.softmax(scores, dim=-1)

    # 4. 加权求和（Context Vector）
    # [batch, n, m] x [batch, m, d_v] -> [batch, n, d_v]
    output = torch.matmul(attn_weights, V)

    return output, attn_weights
```

**关键逻辑解析**：

**维度变换**：

scores的形状为[batch, n, m]，构建了Query（长度$\boldsymbol{n}$）与所有Key（长度$\boldsymbol{m}$）的关系矩阵。

transpose(-2, -1)是为了将$K$转置，使其满足矩阵乘法维度要求（$n \times d_k$乘$d_k \times m$）。

**缩放因子（$\sqrt{d_k}$）**：

作用：稳定点积的方差，防止数值过大。

原理：假设$Q$和$K$的每个元素都是均值为0、方差为1的独立随机变量，则点积$Q \cdot K$的方差为$d_k$。当$d_k$很大时，点积结果的方差也很大，导致Softmax输出接近one-hot分布（几乎全是0和1），梯度接近0，模型无法有效学习。除以$\sqrt{d_k}$可将方差重新归一化为1，保持数值稳定。

**Mask机制**：

masked_fill(mask == 0, -1e9)：将指定位置的注意力分数设为极小值（-1e9），经过Softmax后变为0，实现信息屏蔽。主要用于两种场景：

①因果遮蔽（Causal Masking）- 在Decoder中防止模型看到未来Token；

②填充遮蔽（Padding Mask）- 忽略序列中的填充位置。

**输出含义**：

output：融合了上下文信息的特征表示，形状与$Q$保持一致[batch, n, d_v]。

attn_weights：是可选输出，可用于可视化注意力分布热力图，展示Query对每个Key-Value位置的关注权重。

## 4. Attention的三种核心变体

理解Attention变体的关键，在于理清Query、Key和Value的数据来源。

| 类型 | 核心逻辑 | 应用场景 |
|------|----------|----------|
| Self-Attention | 序列内部做关联 | 文本编码（Encoder） |
| Masked Self-Attention | 序列内部做关联，但屏蔽未来信息 | 文本生成（Decoder） |
| Cross-Attention | 两个不同序列之间做关联 | 序列转换（Seq2Seq） |

### 核心区别：数据的来源

**Self-Attention（自注意力）**：Q、K、V全部来自同一个序列。即$Q=X, K=X, V=X$。作用：理解输入序列内部的词与词之间的关系。

**Cross-Attention（交叉注意力）**：Q来自目标序列（Decoder），而K、V来自源序列（Encoder）。即$Q=Target, K=Source, V=Source$。作用：两个序列进行信息交互。

### Cross-Attention代码交互逻辑

在Encoder-Decoder架构（如Transformer机器翻译）中，Cross-Attention连接了编码器和解码器。

```python
# 假设我们已经有了编码器输出和解码器当前的输入
# memory: 编码器输出（源序列信息），形状[Batch, m, d_model]
# target: 解码器输入（当前生成内容），形状[Batch, n, d_model]

# 初始化多头注意力模块
cross_attn_layer = MultiHeadAttention(d_model=512, n_heads=8)

# !!! 关键步骤 !!!
# 1. Query来自Decoder（Target）-> 我现在想查什么？
# 2. Key来自Encoder（Memory）-> 字典里有哪些索引？
# 3. Value来自Encoder（Memory）-> 字典里的具体内容？
output = cross_attn_layer(Q=target, K=memory, V=memory)
```

**查询方向**：代码显式地将target作为Query传入。这意味着模型是站在"输出端"的视角，去"输入端"查找相关信息。

**KV同源**：Key和Value始终保持一致，均来自memory（编码器输出）。这保证了索引键和实际内容的一一对应关系。

**维度对齐**：由于Q和K来自不同序列，它们的序列长度$n$和$m$通常不相等，但特征维度$d_k$必须一致才能进行矩阵乘法运算。

## 5. Multi-Head Attention（多头注意力）

单一的Attention容易过早聚焦于某种特定的局部特征。Multi-Head Attention机制类似于使用多组不同的滤镜，将输入特征投影到多个独立的子空间中并行计算。这使模型能够同时捕捉语法、语义、韵律等不同层面的信息。

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        # 定义线性投影层
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. 线性投影 + 分头（Split Heads）
        # 形状变换: [Batch, Length, d_model] -> [Batch, Length, n_heads, d_k]
        # 转置变换: -> [Batch, n_heads, Length, d_k] 以便并行计算
        q_s = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 并行计算所有头的Attention
        context, attn_weights = scaled_dot_product_attention(q_s, k_s, v_s, mask)

        # 3. 拼接（Concat）+ 融合
        # 形状变换: [Batch, n_heads, Length, d_k] -> [Batch, Length, n_heads * d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        
        # 4. 输出投影
        output = self.W_o(context)
        return output
```

**子空间拆分**：代码不直接创建多个Layer，而是将大维度d_model拆分为n_heads个d_k维度。这保证了多头机制的总计算量与单头机制基本持平。

**维度变换技巧**：transpose(1, 2)将n_heads维度移到序列长度之前。在PyTorch中，矩阵乘法默认在最后两维进行，这样可以让所有Head的计算在一个矩阵运算步骤中并行完成。

**内存连续性**：contiguous()是必须的步骤。因为transpose操作只改变了stride而未改变内存布局，view操作需要内存连续的Tensor才能正确合并维度。

**信息融合**：最后的W_o线性层至关重要，它负责将各个Head捕获的不同侧面信息整合成最终的特征表示。

## 6. 性能瓶颈与复杂度分析

为什么ChatGPT这类模型都有最大上下文长度限制？根本原因在于Attention机制的计算复杂度。

我们将序列长度记为$n$，特征维度记为$d$。

| 变体类型 | 时间复杂度 | 空间复杂度 | 核心特征 |
|----------|------------|------------|----------|
| Standard Self-Attention | $O(n^2 \cdot d)$ | $O(n^2)$ | 平方级增长，序列翻倍计算量翻四倍，显存杀手 |
| Linear Attention | $O(n \cdot d^2)$ | $O(n \cdot d)$ | 线性增长，通过改变计算顺序降低开销但牺牲精度 |
| Sparse Attention | $O(n \sqrt{n} \cdot d)$ | $O(n \sqrt{n})$ | 局部关注，只计算部分关键位置的注意力 |

### 标准Self-Attention的痛点

在标准计算中，模型需要计算任意两个Token之间的相关性。生成一个$n \times n$的注意力矩阵是最大的瓶颈：

**计算量**：对于$n=1000$，需要计算$10^6$次；对于$n=10000$，需要计算$10^8$次。

**内存**：这个巨大的矩阵必须存储在显存中，导致显存需求随序列长度呈平方级暴涨。

### 优化方案

为了让模型能处理几万甚至几十万字的文本，业界提出了两种思路：

**Linear Attention**：利用矩阵乘法的结合律$(QK^T)V \approx Q(K^TV)$。先将$K$和$V$结合成一个$d \times d$的小矩阵，再与$Q$相乘。避开了$n \times n$的大矩阵计算，将复杂度降为$O(n)$。适合超长序列，但可能丢失细节信息。

**Sparse Attention**：不再让每个词都看所有词，而是只看附近的词或特定的关键词。将$n \times n$的全连接矩阵变成稀疏矩阵，大部分位置填0不计算。大幅降低计算量，在长文档理解中应用广泛。

## 7. 相关词条

- Transformer
- Self-Attention
- Positional Encoding
- Layer Normalization