---
layout: ../../layouts/ProjectLayout.astro
title: CSMapping
description: 基于扩散模型与拓扑推理的众包高精地图构建系统
tags: ["Autonomous Driving", "Mapping", "Diffusion Model"]
---

# CSMapping

众包建图中的语义重构与拓扑推理系统。

<p class="related-content" style="margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
<strong>论文信息</strong>：<a href="https://arxiv.org/abs/2512.03510" style="color: #2563eb;">arXiv:2512.03510</a> | Zhijian Qiao et al. | HKUST | Dec 2025
</p>

CSMapping是一个针对众包数据的高精地图构建系统。它巧妙地结合了传统的矢量化方法与现代生成式模型，旨在解决众包数据中常见的**噪声大**、**覆盖不全**和**拓扑断裂**等核心难题。

## 1. 核心能力

- **高鲁棒性语义重构**：利用潜在扩散模型学习地图分布先验，即使观测数据充满噪声，也能"脑补"出清晰准确的道路结构
- **盲区智能补全**：对于传感器未扫描到的区域，系统能基于已有的上下文信息，生成合乎逻辑的道路延伸
- **拟人化拓扑推理**：摒弃死板的规则，直接从真实驾驶轨迹中聚类提取中心线，生成的路径符合人类驾驶习惯且满足车辆动力学约束
- **端到端可扩展性**：系统性能随数据规模呈现Scaling Law特性，众包数据越多，地图质量越高

## 2. 系统架构

CSMapping由三个互补的子模块构成，分别负责初始提取、生成式优化和拓扑连接。

| 模块名称 | 核心职责 | 关键算法/技术 |
|:---|:---|:---|
| **矢量化建图模块** | 处理原始数据，生成初始的粗糙地图 | 体素连接关联算法(VCA)、切比雪夫多项式拟合 |
| **生成式建图模块** | 修复噪声，补全盲区，生成高质量栅格 | 潜在空间扩散模型(LDM)、受约束的MAP估计 |
| **拓扑建图模块** | 从轨迹流中提取可行驶中心线 | 连续动态时间规整(CDTW)、动力学约束下的k-medoids聚类 |

## 3. 生成式建图流水线详解

这是CSMapping最具创新性的部分。系统并未直接训练一个端到端的回归网络，而是采用了一种<strong>"先验学习+逆向优化"</strong>的策略。

<p style="text-align: center; margin: 1.5rem 0;">
<img src="/images/csmapping_pipeline.png" alt="CSMapping生成式建图流程图" style="max-width: 60%; display: block; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
</p>
<p style="text-align: center; color: #666; font-size: 0.85em; margin-top: -0.5rem;">Figure 6: CSMapping从先验学习到推理优化的完整Pipeline</p>

### 3.1 训练阶段：构建"地图大脑"（图6a）

这一阶段的目标是让模型学习**什么是合理的地图**。系统使用大量高质量的高精地图（HD Map）进行训练。

<strong>VAE压缩（蓝色路径）</strong>：使用变分自编码器将256×256的稀疏栅格地图压缩为16×16×16的紧凑潜在特征。目的是降低后续扩散模型的计算复杂度，提取地图的高层语义特征。

<strong>扩散模型训练（粉色路径）</strong>：在潜在空间中训练一个Diffusion Transformer (DiT)，模型学习如何从纯高斯噪声中一步步还原出潜在特征。通过adaLN（自适应层归一化）注入时间步信息，并融合标准地图（SD Map）作为全局拓扑指引。

**SD Map条件注入机制**：SD Map的融合借鉴了ControlNet的核心思想，通过<strong>零初始化卷积层（Zero-initialized Convolution）</strong>实现与DiT的深度集成，整个注入过程分为三步：

1. **潜在特征编码**：将栅格化的SD道路骨架图$\boldsymbol{c}_{sd}$输入预训练的VAE编码器$\mathcal{E}$，得到与HD地图潜在变量$\boldsymbol{x}_t$维度一致的特征矩阵$\boldsymbol{z}_{sd} \in \mathbb{R}^{16 \times 16 \times 16}$

2. **零初始化投影**：编码后的特征通过零初始化卷积层$\mathcal{Z}_\theta$进行投影：$\boldsymbol{f}_{cond} = \mathcal{Z}_\theta(\boldsymbol{z}_{sd})$。训练初期该层权重和偏置均为0，输出为全零矩阵，模型等同于无条件生成；随着训练进行，该层逐渐学习如何将SD的拓扑特征变换为去噪的引导信号

3. **多尺度残差融合**：处理后的特征$\boldsymbol{f}_{cond}$被注入到DiT的每个Block中，通过残差连接与隐藏状态相加：$\boldsymbol{h}_{i+1} = \text{DiTBlock}_i(\boldsymbol{h}_i, t) + \mathcal{Z}_{\theta,i}(\boldsymbol{z}_{sd})$

这种设计的优势在于：零初始化保证了注入过程不破坏扩散模型预训练好的先验分布；当观测数据与SD存在冲突时，模型可以通过调节$\mathcal{Z}_\theta$的权重来选择性地忽略错误的SD指引。

**训练目标函数**：

$$
\mathcal{L} = \mathbb{E}_{\boldsymbol{x}_0, \boldsymbol{\epsilon}, t} [\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t, \boldsymbol{c}_{sd})\|^2_2]
$$

### 3.2 推理阶段：基于观测的逆向寻优（图6b）

在实际应用中，我们没有真值地图，只有充满噪声的众包观测。系统不再进行随机生成，而是进行**有约束的搜索**。

**扩散反演初始化**：首先利用矢量化模块生成一份粗糙的初始地图，通过扩散过程将其反向映射回潜在空间的噪声点$\boldsymbol{x}_T$。相比随机噪声初始化，这提供了一个极佳的"热启动"点，避免优化陷入局部最优。

**梯度反传优化**：冻结预训练好的DiT网络参数，寻找最佳的初始噪声$\boldsymbol{x}_T$，使得经过DiT生成的地图能最大程度地匹配当前的众包观测数据。

**高斯流形约束**：为了防止生成的地图"崩坏"，代码将$\boldsymbol{x}_T$参数化为正交高斯基底的线性组合：

$$
\boldsymbol{x}_T = \sum_{i=0}^K w_i \boldsymbol{b}_i, \quad \text{s.t. } \|\boldsymbol{w}\|_2 = 1
$$

这意味着优化过程实际上是在单位超球面上搜索最优的权重向量$\boldsymbol{w}$。

**掩码观测损失**：系统构建了一个可靠性掩码，仅在观测次数多、置信度高的区域计算损失。在掩码之外的区域（盲区），没有观测约束，模型完全依靠训练阶段学到的先验知识自动补全道路结构。

## 4. 关键算法与代码实现

### 4.1 矢量化关联算法(VCA)

在处理众包数据时，同一个车道线往往被切碎成无数个小段。VCA算法解决了如何将这些碎片连接成完整语义实例的问题。

- **哈希体素化**：将所有观测到的线段栅格化到一个哈希表中
- **连通性判断**：计算两个端点间的前背景计数比例，如果两个端点之间有足够多的连续观测点，则认为它们属于同一实例
- **数学表示**：最终提取出的语义要素使用切比雪夫多项式进行参数化，相比贝塞尔曲线，它在拟合复杂道路几何时更加稳定：

$$
\boldsymbol{p}_j = \sum_{i=0}^n \boldsymbol{a}_i \cos(i \arccos(t_j))
$$

### 4.2 拓扑推理算法

不同于传统的几何规则拼接，本系统通过聚类真实轨迹来生成中心线。

- **CDTW距离度量**：使用连续动态时间规整计算两条轨迹的相似度，相比普通DTW，CDTW能在亚像素级别处理相位偏差，对轨迹的局部拉伸和压缩不敏感
- **置信度加权K-medoids**：在聚类时，不仅考虑形状相似度，还引入了轨迹的置信度权重
- **动力学平滑**：聚类得到的中心线虽然形状正确，但可能不平滑。系统最后使用AL-iLQR算法对其进行优化，强制其满足车辆的运动学约束（如最大曲率限制）

### 4.3 SD Map条件注入

以下伪代码展示了零初始化卷积层如何实现SD Map的条件控制：

```python
class DiTBlockWithSD(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dit_block = DiTBlock(hidden_size)  # 原始DiT块
        # 零初始化卷积层：控制SD特征注入
        self.zero_conv = nn.Conv2d(latent_dim, hidden_size, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x, t, sd_latent):
        # 1. 正常的DiT前向传播（包含时间步t的调制）
        h = self.dit_block(x, t)
        
        # 2. 将SD特征投影并融合
        # sd_latent是由VAE Encoder对SD Map编码后的矩阵
        sd_feat = self.zero_conv(sd_latent)
        
        # 3. 通过残差相加实现条件控制
        return h + sd_feat
```

### 4.4 潜在空间优化

以下伪代码展示了推理阶段的核心优化逻辑，如何在潜在空间中寻找最佳地图：

```python
class LatentOptimizer(nn.Module):
    def __init__(self, basis_num=512):
        # 初始化正交高斯基底，用于约束搜索空间
        self.basis = torch.randn(basis_num, latent_dim).qr()[0] 
        # 优化变量w，初始化为均匀分布
        self.w = nn.Parameter(torch.ones(basis_num) / math.sqrt(basis_num))

    def forward(self, observation_map, reliability_mask):
        # 1. 重参数化：合成潜在变量x_T
        # 归一化w确保x_T始终位于高斯流形上
        w_norm = self.w / self.w.norm()
        x_T = w_norm @ self.basis 

        # 2. 通过冻结的DiT生成地图
        # 使用确定性采样器保证梯度可导
        pred_map = self.frozen_diffusion_model.generate(x_T)

        # 3. 计算掩码观测损失
        # 仅在观测置信度高的区域(mask=1)计算MSE Loss
        # mask=0的区域完全由Diffusion先验负责补全
        loss = ((pred_map - observation_map) ** 2 * reliability_mask).sum()
        
        return loss
```

## 5. 总结

CSMapping的核心哲学是<strong>"数据驱动先验，观测约束生成"</strong>。

它不再试图硬编码复杂的几何规则来修复破碎的众包数据，而是相信扩散模型已经"看过"足够多的完美地图。工程师的任务变成了设计巧妙的约束条件，引导模型在潜在空间中找到那个既符合观测事实，又符合地图逻辑的最优解。
