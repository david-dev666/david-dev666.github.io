---
layout: ../../layouts/ProjectLayout.astro
title: FFN前馈网络
description: Transformer中的前馈神经网络详解
tags: ["Transformer", "NLP", "Deep Learning"]
---

# FFN前馈网络

前馈网络FFN是Transformer架构中的核心组件。它占据了模型约**2/3**的参数量，虽结构简单，却承担着**知识存储**与**特征变换**的关键职责，常被形容为模型的"参数黑洞"。

理解FFN**先升维、后降维**的设计，是掌握Transformer记忆机制与表征能力的关键。

## 1. 核心思想

FFN本质是一个**逐位置的非线性变换器（Position-wise Nonlinear Transformer）**。它位于Multi-Head Attention之后，独立处理序列中的每一个Token。

与Attention的**空间混合**不同，FFN执行的是**通道混合**：

* **Attention**：不同位置Token之间的交流，聚合上下文信息。
* **FFN**：单个位置特征内部的非线性变换，提取高层语义。

## 2. 数学表达

### 经典FFN

标准Transformer采用如下公式：

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2
$$

| 符号 | 形状 | 物理含义 |
|:---|:---|:---|
| $x$ | $n \times d_{model}$ | 输入特征，$n$为序列长度 |
| $W_1$ | $d_{model} \times d_{ff}$ | **升维矩阵**，通常$d_{ff} = 4d_{model}$ |
| $W_2$ | $d_{ff} \times d_{model}$ | **降维矩阵**，将特征压缩回原始空间 |
| $\text{GELU}$ | - | 激活函数，引入非线性并筛选特征 |

**维度流向**：$d_{model} \to 4d_{model} \to d_{model}$

### 现代变体SwiGLU

LLaMA及PaLM等现代大模型采用SwiGLU变体：

$$
\text{SwiGLU}(x) = W_{down} \cdot \left( \text{Swish}(W_{gate} \cdot x) \odot (W_{up} \cdot x) \right)
$$

| 符号 | 形状 | 物理含义 |
|:---|:---|:---|
| $W_{gate}$ | $d_{model} \times d_{ff}$ | **门控矩阵**，决定特征激活程度 |
| $W_{up}$ | $d_{model} \times d_{ff}$ | **特征矩阵**，提取候选特征 |
| $W_{down}$ | $d_{ff} \times d_{model}$ | **降维矩阵**，融合输出 |
| $\odot$ | - | Hadamard积，即逐元素乘法 |

SwiGLU引入了门控机制，通过$\text{Swish}(W_{gate} \cdot x)$动态控制信息流。实验证明其能提升训练稳定性与模型性能。

## 3. 代码实现

以下基于PyTorch实现，包含经典版与LLaMA版。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    经典FFN实现，对应BERT/GPT架构
    结构：Linear -> GELU -> Dropout -> Linear
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4  # 默认4倍宽
        
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Seq_Len, Dim]
        x = self.w_up(x)        # 1. 升维投影
        x = F.gelu(x)           # 2. 激活函数(GELU)
        x = self.dropout(x)     # 3. 正则化
        x = self.w_down(x)      # 4. 降维
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU实现，对应LLaMA架构
    特点：引入Gate路径，通常调整d_ff比例以平衡参数量
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        # LLaMA通常将倍率从4调整为8/3，以保持参数量与经典FFN相近
        d_ff = d_ff or int(d_model * 8 / 3)
        
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))  # 门控路径：决定"通过多少信息"
        feat = self.w_up(x)            # 特征路径：提取"是什么信息"
        x = gate * feat                # 逐元素相乘
        x = self.dropout(x)
        x = self.w_down(x)             # 降维
        return x
```

**工程实现细节**：

* **参数平衡**：SwiGLU多了一个$W_{gate}$矩阵。为保持总参数量与经典FFN一致，通常将隐藏层宽度$d_{ff}$缩小为原来的$\frac{2}{3}$。
* **独立性**：`nn.Linear`仅在最后一维特征维度上进行计算，序列维度上的每个Token互不干扰。

## 4. 设计原理：为何"先升后降"？

### 4.1 升维：特征解耦

类似于SVM中的**核技巧（Kernel Trick）**。

* **问题**：在低维空间$d_{model}$中，复杂的语义特征往往相互纠缠，难以通过线性边界区分。
* **解法**：通过$W_1$投影到高维空间$d_{ff}$，将特征流形"展开"。
* **效果**：配合激活函数，模型在高维空间能更轻易地切分和筛选语义特征。

> 形象理解：一张揉皱的纸很难画直线分开上面的红蓝点。但若将纸展开铺平（即升维），分离就变得简单。

### 4.2 记忆：Key-Value存储器

论文《Transformer Feed-Forward Layers Are Key-Value Memories》提出了主流解释：FFN是大规模的**Key-Value记忆网络**。

| 权重部分 | 角色 | 作用 |
|:---|:---|:---|
| $W_1$权重行 | **Key** | **模式探测器**。检测输入是否存在特定模式，如"法国"与"首都"同时出现。 |
| 激活函数 | **Gate** | **筛选器**。匹配度高则激活，反之归零抑制。 |
| $W_2$权重列 | **Value** | **内容提取器**。若模式被激活，则输出对应的语义增量，如指向"巴黎"。 |

**升维意义**：$d_{ff}$决定了记忆槽位的数量。宽度越大，模型能存储的事实性知识越多。

### 4.3 冗余：信息的无损重组

ReLU或GELU等激活函数具有截断性，会造成信息损失（死神经元）。

* **升维**：提供信息冗余。即使激活函数过滤了一半神经元，剩余的维度仍足以重构完整信息。
* **降维**：将筛选后的精华特征压缩回紧凑的潜空间，供下一层继续计算。

## 5. 组件对比

| 组件 | 变换类型 | 核心作用 | 器官类比 |
|:---|:---|:---|:---|
| **Attention** | **空间混合（Mix Spatial）** | 上下文聚合，解决指代消歧 | **眼睛**：看清相互关系 |
| **FFN** | **通道混合（Mix Channel）** | 知识检索推理，固化逻辑规则 | **大脑**：思考与回忆 |

## 6. 复杂度分析

FFN是计算密集型模块。

| 指标 | 公式 | 说明 |
|:---|:---|:---|
| **参数量** | $2 \times d_{model} \times d_{ff}$ | 两个大矩阵的权重之和 |
| **计算量** | $O(n \times d_{model} \times d_{ff})$ | 与序列长度$n$呈线性关系 |

以**GPT-3 175B**为例：$d_{model}=12288$，$d_{ff}=49152$。单层FFN参数量约为**1.2B**，占据了单个Transformer Block参数量的绝大部分。

## 7. 相关词条

- Attention机制
- Transformer
- GELU激活函数
- Layer Normalization
