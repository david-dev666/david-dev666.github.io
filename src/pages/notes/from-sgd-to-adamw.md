---
layout: ../../layouts/ProjectLayout.astro
title: 从SGD到AdamW
description: 优化器十五年演进史，每一步解决了上一代的什么问题
tags: ["Optimizer", "Deep Learning", "Training"]
---

# 从SGD到AdamW

深度学习优化器从1950年代的梯度下降走到2017年的AdamW，并不是一场理论突进，而是一连串"上一代留下了什么坑、下一代怎么填"的工程迭代。本文按时间线讲清这条主线。

> Adam的完整数学推导、PyTorch实现与显存分析见词条 [Adam优化器](/glossary/adam)，本文只讲它在演进链上的位置与动机。

## 1. 演进时间线

<svg viewBox="0 0 880 180" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: auto; max-width: 880px; margin: 1.5rem 0;">
  <line x1="40" y1="90" x2="840" y2="90" stroke="#d1d5db" stroke-width="2"/>
  <g font-family="Inter, sans-serif" font-size="13" text-anchor="middle">
    <g>
      <circle cx="60" cy="90" r="6" fill="#9ca3af"/>
      <text x="60" y="70" fill="#374151" font-weight="600">GD</text>
      <text x="60" y="115" fill="#6b7280" font-size="11">1847</text>
      <text x="60" y="135" fill="#9ca3af" font-size="11">全量梯度</text>
    </g>
    <g>
      <circle cx="180" cy="90" r="6" fill="#9ca3af"/>
      <text x="180" y="70" fill="#374151" font-weight="600">SGD</text>
      <text x="180" y="115" fill="#6b7280" font-size="11">1951</text>
      <text x="180" y="135" fill="#9ca3af" font-size="11">小批量采样</text>
    </g>
    <g>
      <circle cx="300" cy="90" r="6" fill="#3b82f6"/>
      <text x="300" y="70" fill="#1d4ed8" font-weight="600">Momentum</text>
      <text x="300" y="115" fill="#6b7280" font-size="11">1964</text>
      <text x="300" y="135" fill="#9ca3af" font-size="11">引入惯性</text>
    </g>
    <g>
      <circle cx="430" cy="90" r="6" fill="#3b82f6"/>
      <text x="430" y="70" fill="#1d4ed8" font-weight="600">AdaGrad</text>
      <text x="430" y="115" fill="#6b7280" font-size="11">2011</text>
      <text x="430" y="135" fill="#9ca3af" font-size="11">参数级LR</text>
    </g>
    <g>
      <circle cx="550" cy="90" r="6" fill="#3b82f6"/>
      <text x="550" y="70" fill="#1d4ed8" font-weight="600">RMSprop</text>
      <text x="550" y="115" fill="#6b7280" font-size="11">2012</text>
      <text x="550" y="135" fill="#9ca3af" font-size="11">滑动平均</text>
    </g>
    <g>
      <circle cx="680" cy="90" r="7" fill="#7c3aed"/>
      <text x="680" y="70" fill="#6d28d9" font-weight="700">Adam</text>
      <text x="680" y="115" fill="#6b7280" font-size="11">2014</text>
      <text x="680" y="135" fill="#9ca3af" font-size="11">动量+自适应</text>
    </g>
    <g>
      <circle cx="810" cy="90" r="7" fill="#7c3aed"/>
      <text x="810" y="70" fill="#6d28d9" font-weight="700">AdamW</text>
      <text x="810" y="115" fill="#6b7280" font-size="11">2017</text>
      <text x="810" y="135" fill="#9ca3af" font-size="11">解耦权重衰减</text>
    </g>
  </g>
</svg>

## 2. 起点：梯度下降GD

参数沿损失函数负梯度方向移动：

$$
\boldsymbol{\theta}_{t} = \boldsymbol{\theta}_{t-1} - \alpha \nabla L(\boldsymbol{\theta}_{t-1})
$$

**问题**：每一步都需要遍历整个数据集计算梯度。在ImageNet量级的数据上，单步更新需数小时，工业上不可行。

## 3. SGD：用噪声换速度

每步只用一个或一小批样本估计梯度：

$$
\boldsymbol{\theta}_{t} = \boldsymbol{\theta}_{t-1} - \alpha \nabla L_{\text{batch}}(\boldsymbol{\theta}_{t-1})
$$

**收益**：每步$O(B)$而非$O(N)$，更新频率提高几个量级。

**新问题**：小批量梯度是真实梯度的有噪估计。在峡谷状的损失曲面上，梯度方向在两侧谷壁间反复横跳，参数沿"之"字路径前进，垂直分量大量浪费。

![Optimizer Trajectories](/images/optimizers/fig_trajectory.png)

上图Beale函数的等高线对比，红色SGD轨迹明显在长峡谷中震荡，前进效率低。

## 4. Momentum：给梯度装上惯性

物理类比：把参数看成在损失曲面上滚动的小球，速度由历史梯度的指数滑动平均决定：

$$
\begin{aligned}
\boldsymbol{m}_t &= \beta \boldsymbol{m}_{t-1} + (1-\beta) \boldsymbol{g}_t \\
\boldsymbol{\theta}_t &= \boldsymbol{\theta}_{t-1} - \alpha \boldsymbol{m}_t
\end{aligned}
$$

$\beta = 0.9$时，$\boldsymbol{m}_t$近似于过去10步梯度的加权平均。

**关键效果**：在峡谷场景中，左右震荡的水平分量正负相消被抑制，沿谷底的稳定分量则被累积放大。上图蓝色轨迹直冲谷底，对照SGD红色震荡路径，差异一目了然。

> Momentum是第一个把"历史信息"引入更新规则的优化器。在此之前，SGD每一步都是无记忆的。

## 5. AdaGrad：每个参数有自己的学习率

Momentum解决了**方向**问题，但**步长**问题没碰：所有参数共享同一个全局学习率$\alpha$。

这在稀疏特征场景下严重低效。例如一个词嵌入矩阵中，常见词每步都被更新、不需要大步长；罕见词偶尔出现一次、需要大步长。共用一个$\alpha$只能折中。

AdaGrad的方案：累积每个参数的历史梯度平方和，分母里开方作为该参数的步长缩放因子：

$$
\begin{aligned}
\boldsymbol{r}_t &= \boldsymbol{r}_{t-1} + \boldsymbol{g}_t^2 \\
\boldsymbol{\theta}_t &= \boldsymbol{\theta}_{t-1} - \frac{\alpha}{\sqrt{\boldsymbol{r}_t} + \epsilon} \boldsymbol{g}_t
\end{aligned}
$$

历史梯度大的参数$\boldsymbol{r}_t$大，有效学习率被压低；历史梯度小的参数则保留较大有效步长。

**新问题**：$\boldsymbol{r}_t$是单调累加的，永远只增不减。训练后期分母无限增大，所有参数有效学习率都趋近于零，模型停止学习。

## 6. RMSprop：把累加换成滑动平均

Hinton在课堂上提出的修正，没有正式论文，但成了Adam的关键组件。

把AdaGrad的累加改成指数滑动平均：

$$
\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2) \boldsymbol{g}_t^2
$$

效果：$\boldsymbol{v}_t$不再无限膨胀，它反映的是"近期"梯度方差。学习率衰减问题被根治，参数级自适应特性得以保留。

## 7. Adam：动量与自适应合体

到这里所有零件已经备齐。Adam做的事很直接，把Momentum的一阶矩$\boldsymbol{m}$和RMSprop的二阶矩$\boldsymbol{v}$合并到一个更新规则里：

$$
\begin{aligned}
\boldsymbol{m}_t &= \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1) \boldsymbol{g}_t \\
\boldsymbol{v}_t &= \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2) \boldsymbol{g}_t^2 \\
\boldsymbol{\theta}_t &= \boldsymbol{\theta}_{t-1} - \alpha \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}
\end{aligned}
$$

其中$\hat{\boldsymbol{m}}, \hat{\boldsymbol{v}}$是偏差修正项，用于纠正$\boldsymbol{m}_0 = \boldsymbol{v}_0 = \boldsymbol{0}$带来的初期估计偏低。

![SGD vs Adam Effective Update](/images/optimizers/fig_adaptive_lr.png)

上图直观展示Adam的两个特性：左图稠密梯度参数，紫色Adam步长更平滑、抗噪；右图稀疏梯度参数，SGD只在尖峰时刻更新一下就归零，Adam通过动量把这次更新"延续"了好几步，让稀疏信号能持续推动参数。

> Adam一度成为深度学习的默认选择。直到Loshchilov在2017年指出，它在处理权重衰减时有一个微妙但严重的bug。

详细推导、PyTorch实现、混合精度下的显存计算见 → [Adam优化器](/glossary/adam)。

## 8. AdamW：解耦权重衰减

权重衰减是防止过拟合的标准手段。在SGD里，它等价于在损失函数加一个L2正则项：

$$
L'(\boldsymbol{\theta}) = L(\boldsymbol{\theta}) + \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2
\;\Rightarrow\;
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \alpha(\boldsymbol{g}_t + \lambda \boldsymbol{\theta}_{t-1})
$$

也就是说，对SGD而言"在梯度里加$\lambda \boldsymbol{\theta}$"和"参数自己缩水$\lambda \boldsymbol{\theta}$"是一回事。

**Adam里这个等价不再成立**。如果按L2正则的传统做法把$\lambda \boldsymbol{\theta}$加进梯度$\boldsymbol{g}$，那么这个衰减项会被$\boldsymbol{m}, \boldsymbol{v}$平滑、被$\sqrt{\hat{\boldsymbol{v}}}$缩放：

$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \alpha \frac{\hat{\boldsymbol{m}}_t(\text{含 } \lambda\boldsymbol{\theta})}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}
$$

> 反直觉的后果：梯度大的参数$\boldsymbol{v}$大、分母大，"权重衰减"项被同步缩小——历史梯度越剧烈的参数，正则化反而越弱。这与正则化的本意完全相反。

AdamW的修正很简单，把权重衰减从梯度里抽出来，直接作用在参数上：

$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \alpha \left( \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon} + \lambda \boldsymbol{\theta}_{t-1} \right)
$$

这一行改动让Transformer类模型的训练稳定性和泛化能力显著提升，AdamW从此取代Adam成为大模型预训练的事实标准。GPT、LLaMA、Qwen系列均使用AdamW。

```python
# AdamW 与 Adam 的唯一差异
# Adam:  把 weight_decay 加到梯度里
g = g + weight_decay * p.data           # ✗ 会被 v 缩放

# AdamW: weight_decay 直接作用在参数上
p.data -= lr * (m_hat / (v_hat.sqrt() + eps))
p.data -= lr * weight_decay * p.data    # ✓ 与自适应分母解耦
```

## 9. 一张表看懂演进

| 优化器 | 解决了上一代什么问题 | 引入了什么新机制 | 留下的问题 |
|:---|:---|:---|:---|
| **GD** | — | 负梯度更新 | 全量数据，慢 |
| **SGD** | 全量过慢 | 小批量采样 | 梯度噪声，峡谷震荡 |
| **Momentum** | 峡谷震荡 | 一阶矩，惯性平滑 | 全局学习率 |
| **AdaGrad** | 全局学习率 | 参数级自适应步长 | 学习率单调衰减到零 |
| **RMSprop** | 学习率衰减 | 二阶矩用滑动平均 | 缺少动量 |
| **Adam** | 缺少动量 | 一阶矩+二阶矩+偏差修正 | 权重衰减与自适应耦合 |
| **AdamW** | 权重衰减失真 | 解耦weight decay | 显存占用未优化 |

## 10. 实战决策

什么时候用什么，本质上还是在权衡**收敛速度**、**最终精度**和**显存预算**。

| 场景 | 推荐 | 原因 |
|:---|:---|:---|
| **大模型预训练**（LLM、VLM） | AdamW | 收敛快、稳定，权重衰减必需 |
| **CV分类任务从零训练** | SGD + Momentum | 长期实验显示泛化精度略高于Adam系 |
| **大模型微调** | AdamW（小lr、低weight decay） | 与预训练一致，避免分布偏移 |
| **显存极度受限** | 8-bit AdamW / AdaFactor | 优化器状态占模型参数2倍，需要量化或分解 |
| **强化学习** | Adam | 梯度噪声极大，自适应分母关键 |

**两条铁律**：

1. **Adam系必须配Warmup**。训练初期$\boldsymbol{v}_t$估计极不稳定，没有warmup很容易第一千步就发散。
2. **AdamW的weight_decay是独立超参，不要照搬SGD的设置**。常见取值$0.01 \sim 0.1$，与learning rate解耦调参。

## 11. 相关资料

- 词条深挖：[Adam优化器](/glossary/adam) — 数学推导、PyTorch实现、70B模型显存分析
- 论文：Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014)
- 论文：Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2017)
