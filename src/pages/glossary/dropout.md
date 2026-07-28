---
layout: ../../layouts/ProjectLayout.astro
title: Dropout
description: 深度学习中的 Dropout 正则化技术详解
tags: ["Regularization", "Deep Learning", "Training"]
---

# Dropout

Dropout 是一种经典的正则化技术，通过在训练时随机"丢弃"部分神经元来防止过拟合。

## 1. 核心思想

神经网络容易对训练数据产生过度依赖，形成"共适应"现象——某些神经元只有在特定神经元存在时才能正常工作。Dropout 打破这种依赖：

- **训练时**：以概率 $p$ 随机将神经元输出置零，并将保留的神经元输出除以 $(1-p)$ 进行缩放
- **推理时**：使用全部神经元，无需任何缩放操作

> 这种做法称为**Inverted Dropout**，是现代框架（PyTorch、TensorFlow）的标准实现。相比传统 Dropout 在推理时缩放，Inverted Dropout 将缩放操作前移到训练阶段，使推理时零开销。

## 2. 数学公式

### 前向传播

$$
y = \frac{1}{1-p} \cdot x \odot m, \quad m_i \sim \text{Bernoulli}(1-p)
$$

| 符号 | 含义 |
|------|------|
| $x$ | 输入向量 |
| $m$ | 二值掩码，每个元素以 $1-p$ 概率为 1 |
| $p$ | 丢弃概率，常用 0.1 ~ 0.5 |
| $\odot$ | 逐元素乘法 |

### PyTorch 实现

```python
import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # 生成掩码：以 (1-p) 概率保留
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        
        # 应用掩码并缩放，保持期望值不变
        return x * mask / (1 - self.p)
```

- **训练/推理切换**：`self.training` 标志决定是否启用 dropout，推理时直接返回原输入
- **期望不变性**：假设输入期望为 $\mathbb{E}[x]$，dropout 后期望为 $\frac{(1-p) \cdot x}{1-p} = x$，保持一致

## 3. Dropout 变体

| 变体 | 作用对象 | 应用场景 |
|------|----------|----------|
| Dropout | 神经元 | 全连接层 |
| Dropout2d | 整个通道 | CNN 特征图 |
| DropPath | 整条路径 | ResNet、ViT |
| DropBlock | 连续区域 | CNN 空间特征 |

### Dropout2d 实现

```python
def dropout2d(x, p=0.5, training=True):
    """
    x: [B, C, H, W] - CNN 特征图
    随机丢弃整个通道，而非单个像素
    """
    if p == 0. or not training:
        return x
    
    # mask 形状 [B, C, 1, 1]，同一通道内所有位置共享同一 mask
    mask = torch.bernoulli(torch.full((x.size(0), x.size(1), 1, 1), 1 - p, device=x.device))
    
    return x * mask / (1 - p)
```

- **通道级丢弃**：相邻像素高度相关，逐像素 dropout 效果有限；丢弃整个通道能更有效打破特征依赖
- **广播机制**：mask 的 `[B, C, 1, 1]` 形状会自动广播到 `[B, C, H, W]`

### DropPath 实现

```python
def drop_path(x, drop_prob=0.1, training=True):
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # 按 batch 维度随机丢弃整条路径
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
    
    return x * mask / keep_prob
```

- **作用位置**：用于残差连接的分支，即 `output = x + drop_path(block(x))`
- **整体丢弃**：mask 形状为 `[B, 1, 1, ...]`，同一样本的整个 block 输出要么全保留要么全丢弃
- **随机深度效果**：训练时网络深度随机变化，相当于训练了多个不同深度的子网络
- **典型配置**：ViT 中 drop_prob 通常从 0 线性增长到 0.1~0.3，越深的层丢弃概率越高

## 4. 为什么有效？

- **集成学习视角**：每次前向传播相当于采样一个子网络，最终模型是指数级子网络的隐式集成
- **特征冗余**：强迫每个神经元独立学习有用特征，而非依赖其他神经元
- **噪声注入**：训练时引入随机性，提升模型对扰动的鲁棒性

## 5. 大模型为何弃用 Dropout？

在 LLaMA、PaLM 等大语言模型的源码中，几乎找不到 Dropout 的身影。原因如下：

### 训练稳定性

深层网络对噪声极度敏感。Dropout 引入的随机性会随层数叠加被放大，导致：

- 梯度方差剧烈波动
- FP16/BF16 混合精度下易引发数值溢出，Loss 震荡甚至发散

### 显存与效率

```python
# Dropout 的隐性开销
def dropout_forward(x, p=0.1):
    # 需要额外存储 mask 供反向传播使用
    mask = torch.bernoulli(torch.full_like(x, 1 - p))  # 显存开销
    output = x * mask / (1 - p)                        # 计算开销
    return output, mask
```

以 70B 模型为例，单层 Dropout 的 Mask 存储开销乘以 80 层，显存压力巨大。

### 内生正则化

现代 LLM 已具备足够的泛化能力：

| 机制 | 作用 |
|------|------|
| 海量数据 | Trillion token 级别训练数据，难以过拟合 |
| RMSNorm | 提供训练稳定性，替代 LayerNorm |
| 残差连接 | 保证梯度有效传播 |
| FlashAttention | 为追求速度，默认剔除 Dropout |

### 实验结论

研究表明，Dropout 收益与模型规模呈负相关：

- **小模型**：数据少，Dropout 是防过拟合神器
- **大模型**：参数量超过阈值后，Dropout 反而拖慢收敛，降低最终效果

## 总结

Dropout 是小模型时代的经典正则化手段，但在大模型时代，海量数据和精细的架构设计已提供足够的泛化能力。工程实践中，用更多数据替代显式正则化、用精简算子换取更高吞吐量，是当前的主流选择。

## 相关词条

- [Batch Normalization](/glossary/batch-normalization)
- [Layer Normalization](/glossary/layer-normalization)
- [Weight Decay](/glossary/weight-decay)
- [Attention 机制](/glossary/attention)
