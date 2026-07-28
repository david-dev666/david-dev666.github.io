---
layout: ../../layouts/ProjectLayout.astro
title: Adam优化器
description: 深度学习中的Adam优化器详解与显存分析
tags: ["Optimizer", "Deep Learning", "Training"]
---

# Adam优化器：自适应矩估计

Adam是由Diederik Kingma和Jimmy Ba于2014年提出的随机优化算法。作为深度学习领域的默认优化器，它结合了Momentum算法的动量加速特性与RMSprop算法的自适应学习率特性，旨在解决稀疏梯度和噪声问题。

## 1. 核心思想

传统随机梯度下降SGD存在三个主要缺陷：

1. **全局学习率**：所有参数共享同一个学习率，无法应对稀疏特征与稠密特征更新频率不一致的问题
2. **局部极值**：在鞍点或平坦区域梯度接近于零，导致参数更新停滞
3. **震荡收敛**：在峡谷状的损失曲面中，SGD容易在谷壁间震荡而无法快速下降到谷底

Adam通过维护梯度的**一阶矩估计**和**二阶矩估计**来解决上述问题：

- **一阶矩$\boldsymbol{m}$**：梯度的指数移动平均值，模拟物理动量，利用惯性冲过局部最优点
- **二阶矩$\boldsymbol{v}$**：梯度平方的指数移动平均值，用于标准化梯度幅度，实现参数级的自适应学习率

## 2. 数学形式

给定模型参数$\theta$、损失函数$L$、时间步$t$和全局学习率$\alpha$。

### 2.1 更新公式

$$
\begin{aligned}
g_t &= \nabla_\theta L(\theta_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

| 符号 | 推荐值 | 物理含义 |
|------|--------|----------|
| $\beta_1$ | 0.9 | 一阶矩衰减率，控制动量的记忆长度 |
| $\beta_2$ | 0.999 | 二阶矩衰减率，控制梯度方差的平滑程度 |
| $\epsilon$ | $10^{-8}$ | 数值稳定项，防止分母为零 |

### 2.2 机制解析

**偏差修正Bias Correction**：$m$和$v$初始化为全零向量。在训练初期，指数移动平均值会严重向零偏差。除以$1-\beta^t$项用于放大初期的估计值，随着$t$增加，该项逐渐趋向于1，修正作用随之减弱。

**自适应缩放**：更新项中的分母$\sqrt{\hat{v}_t}$起到了归一化作用。若某参数历史梯度较大，其$v$增大，导致有效学习率降低；反之则有效学习率增加。这种机制使得Adam对梯度的缩放具有不变性。

## 3. 代码实现

以下代码展示了Adam算法的PyTorch风格实现，重点关注状态管理与原位操作。

```python
import torch

class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        # 初始化动量与自适应项，形状与参数完全一致
        # 这就是 Adam 显存占用的主要来源
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            g = p.grad
            
            # 1. 更新动量 (Momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # 2. 更新方差 (RMSprop)
            # 注意：g**2 为逐元素平方
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            
            # 3. 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 4. 参数更新
            # 除法实现了对不同参数不同步长的自适应调整
            update = m_hat / (torch.sqrt(v_hat) + self.eps)
            p.data -= self.lr * update
            
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

## 4. 显存占用分析

在大模型训练中，优化器状态Optimizer States往往是显存占用的最大来源。

### 4.1 理论占用推导

Adam需要为每个模型参数$\theta$维护两个额外的状态值$m$和$v$。

若使用FP32存储状态，对于参数量为$N$的模型，Adam状态占用的显存为$2 \times N \times 4$字节。

### 4.2 70B模型实例

以LLaMA-70B为例，分析FP16混合精度训练下的显存分布：

| 组件 | 数据类型 | 计算公式 | 显存占用 |
|------|----------|----------|----------|
| **模型参数** | FP16 | $70B \times 2$ | 140GB |
| **梯度** | FP16 | $70B \times 2$ | 140GB |
| **一阶矩$m$** | FP32 | $70B \times 4$ | 280GB |
| **二阶矩$v$** | FP32 | $70B \times 4$ | 280GB |
| **总计** | — | — | **840GB** |

在混合精度训练中，为了保证数值稳定性，优化器状态$m$和$v$通常保持为FP32精度，这使得优化器显存占用远超模型本身。

## 5. 变体与优化

针对Adam显存占用高的问题，工业界衍生出了多种变体。

| 变体 | 改进策略 | 显存优势 | 适用场景 |
|------|----------|----------|----------|
| **AdamW** | 解耦权重衰减 | 无 | 任何需要L2正则化的场景 |
| **AdaFactor** | 矩阵分解$v$ | 降低约50% | 超大规模Transformer预训练 |
| **8-bit Adam** | 状态量化 | 降低约75% | 显存受限的微调任务 |

### AdamW解析

AdamW修正了Adam在处理L2正则化时的理论错误。

在标准Adam中，L2正则化项被添加到梯度$g$中，这意味着权重衰减也会被$m$和$v$平滑处理。AdamW将权重衰减项直接作用于参数$\theta$，使其与梯度更新解耦。

```python
# AdamW 的核心差异
# 1. 计算不含 Weight Decay 的自适应更新量
adaptive_update = m_hat / (torch.sqrt(v_hat) + eps)

# 2. 将 Weight Decay 独立施加于参数
p.data -= lr * (adaptive_update + weight_decay * p.data)
```

## 6. 最佳实践

**超参数选择**：

- **NLP/Transformer**：推荐$\alpha = 10^{-4}$至$10^{-5}$
- **CV/GAN**：$\beta_1$常设为0.5以稳定训练
- **学习率预热**：Adam在训练初期因方差估计不稳定，容易导致参数发散，必须配合Warmup策略使用

**数值稳定性**：当训练出现NaN Loss时，除了检查数据，应尝试增大$\epsilon$值或开启梯度裁剪Gradient Clipping。

## 7. 相关词条

- **SGD with Momentum**
- **RMSprop**
- **Weight Decay**
- **Mixed Precision Training**
