---
layout: ../../layouts/ProjectLayout.astro
title: Image Attention
description: 图像处理中的注意力机制详解
tags: ["Attention", "CNN", "Computer Vision"]
---

# Image Attention

**核心思想**：图像注意力的本质是为特征图的每个位置/通道生成一个0~1的权重，然后用这个权重去乘原始特征——权重大的被放大，权重小的被抑制。

图像注意力机制旨在模拟人类视觉系统的选择性聚焦能力。不同方法的区别仅在于**权重是怎么算出来的**、**权重作用在哪个维度**：

| 类型 | 权重维度 | 核心问题 |
|:---|:---|:---|
| **通道注意力** | $1 \times C \times 1 \times 1$ | 哪些特征通道更重要？ |
| **空间注意力** | $1 \times 1 \times H \times W$ | 图像哪些位置更重要？ |
| **坐标注意力** | $1 \times C \times H \times 1$ + $1 \times C \times 1 \times W$ | 哪一行、哪一列更重要？ |

通过**通道\(Channel\)**、**坐标\(Coordinate\)** 及**空间\(Spatial\)** 级的权重分配，让模型学会忽略背景噪声，聚焦核心信息。

## 1. 通道注意力\(Channel Attention\)

关注特征的**内容\(What\)**。卷积神经网络中每层输出的特征图包含多个通道，但并非每个通道都承载重要信息。通道注意力通过学习每个通道的权重，赋予关键特征更高的响应值。

**代表模型**：
- **SE-Net**：Squeeze-and-Excitation Networks，通过全连接层显式建模通道间的相互依赖关系
- **ECA-Net**：Efficient Channel Attention，使用一维卷积替代全连接层，避免降维并有效捕获跨通道交互

### SE-Block实现

算法核心分为**压缩\(Squeeze\)** 和**激励\(Excitation\)** 两步：

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 1. Squeeze: 全局平均池化
        # 将空间维度 H×W 压缩为 1×1，获得全局感受野
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Excitation: 瓶颈结构
        # 先降维再升维，降低参数量并引入非线性
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 维度变化: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self.avg_pool(x).view(b, c)
        
        # 生成通道权重: [B, C] -> [B, C, 1, 1]
        # Sigmoid将权重限制在0~1之间
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale: 将权重作用回原特征图
        return x * y.expand_as(x)
```

**关键点解析**：
- **全局信息聚合**：Global Average Pooling将$H \times W$的空间信息压缩为单个标量，突破卷积操作感受野的局限
- **非线性依赖建模**：两层FC配合ReLU激活，拟合通道间复杂的非线性关系
- **极低算力消耗**：参数量仅增加$2 \times C \times C/r$，对推理速度影响可忽略

## 2. 坐标注意力\(Coordinate Attention\)

关注特征的**位置\(Where\)**。SE-Net的全局平均池化虽然有效，但丢失了位置信息。坐标注意力通过将垂直和水平方向的池化分离，在捕获长距离依赖的同时保留精确的空间坐标。

### Coordinate Attention实现

核心思路是分别沿X轴和Y轴进行池化，生成一对方向感知的特征图：

```python
class CoordAtt(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAtt, self).__init__()
        # 两个方向的自适应池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)
        
        # 共享卷积层，减少参数量
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        
        # 1. 方向分离池化
        # x_h: [N, C, H, 1], x_w: [N, C, 1, W]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 变换为[N, C, W, 1]以便拼接

        # 2. 拼接与共享变换
        # 在空间维度拼接，利用1×1卷积同时处理两个方向的信息
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        # 3. 切分与独立投影
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复形状

        # 生成两个方向的注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return x * a_h * a_w
```

**关键点解析**：
- **方向感知**：不同于SE-Net的"压扁"操作，这里分别生成高度方向向量$\boldsymbol{a}_h \in \mathbb{R}^{H \times 1}$和宽度方向向量$\boldsymbol{a}_w \in \mathbb{R}^{1 \times W}$
- **参数共享**：中间层使用1×1卷积对X和Y的特征进行统一变换，既降低计算量，又利用了维度间的相关性
- **精准定位**：最终输出同时对行和列加权，使网络能精确定位目标对象的坐标

## 3. 混合注意力\(CBAM\)

**CBAM\(Convolutional Block Attention Module\)** 串联了通道注意力和空间注意力。设计逻辑非常直观：先判断**是什么**，再判断**在哪里**。

**流程**：`Input → Channel Attention → Spatial Attention → Output`

### 空间注意力实现

在通道维度上进行压缩，聚合空间特征：

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 保持输出尺寸不变
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 通道维度的统计特征
        # avg_out: [B, 1, H, W], max_out: [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 2. 拼接后通道数为2，经过大核卷积生成空间权重图
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))
```

**关键点解析**：
- **通道压缩**：Mean和Max操作将通道数压缩为2，提取特征图在空间上的"热力"分布
- **大感受野**：7×7卷积核确保注意力图能感知到较宽阔的邻域上下文
- **互补性**：通道模块筛选特征类别，空间模块筛选特征位置，串联通常优于并联

## 总结

| 机制类型 | 核心关注点 | 关键操作 | 适用场景 |
|:---|:---|:---|:---|
| **通道(Channel)** | 特征的内容 | Global Pooling → MLP | 提升特征判别性，通用性强 |
| **坐标(Coordinate)** | 特征的位置 | Split Pooling → Conv | 需要精细定位的任务(检测、分割) |
| **混合(CBAM)** | 内容 + 位置 | Channel Attn → Spatial Attn | 算力允许时追求更高精度 |

在工程落地时，SE-Block因其极低的算力损耗，常被直接插入ResNet或MobileNet的残差结构中；而Coordinate Attention则更适合轻量级网络在目标检测任务中的应用。

## 相关词条

- [Attention 机制](/glossary/attention)
