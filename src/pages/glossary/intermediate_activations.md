---
layout: ../../layouts/ProjectLayout.astro
title: 中间激活值
description: 深度学习训练中的显存管理机制详解
tags: ["Deep Learning", "Memory Management", "Training"]
---

# 中间激活值

中间激活值（Intermediate Activations）是深度学习训练过程中的核心概念：神经网络在前向传播时生成的中间层输出，必须保存在显存中以供反向传播计算梯度使用。

对于初学者而言，最大的困惑往往是：为什么前向传播已经结束，显存却迟迟不释放？答案在于训练的本质——它是一个闭环系统，而非单向流水线。

## 1. 训练生命周期

神经网络训练包含三个严格的顺序步骤：

**前向传播（Forward Pass）**：数据输入模型，层层计算得到预测结果与Loss

**反向传播（Backward Pass）**：根据Loss，利用链式法则从后向前计算每个参数的梯度

**参数更新（Optimizer Step）**：利用梯度更新权重

**显存驻留的根本原因**：反向传播计算梯度时，必须使用前向传播产生的中间结果。如果这些结果被丢弃，梯度链条就会断裂。

## 2. 计算链条解析

以Transformer中最消耗显存的Self-Attention层为例，拆解其计算与依赖关系。

### 前向传播：构建计算图

在PyTorch等框架中，执行前向代码时，系统会自动构建计算图并保存关键变量。

```python
def forward(Q, K, V):
    # 1. 计算相关性分数
    # Q, K形状: [N, D] -> scores形状: [N, N]
    scores = Q @ K.T / math.sqrt(D)
    
    # 2. 归一化（关键显存占用点）
    # Softmax的输出会被系统自动锁定在显存中
    attention_weights = F.softmax(scores, dim=-1)
    
    # 3. 加权求和
    output = attention_weights @ V
    
    return output, attention_weights
# 此时attention_weights驻留在显存中等待反向传播
```

### 反向传播：梯度的逆流

为什么必须锁死`attention_weights`？观察$\boldsymbol{V}$的梯度计算公式：

$$
\frac{\partial \text{Loss}}{\partial V} = \text{AttentionWeights}^T \cdot \frac{\partial \text{Loss}}{\partial \text{Output}}
$$

代码视角的梯度计算：

```python
def backward(grad_output, attention_weights):
    # 计算V的梯度必须要有attention_weights参与
    grad_V = attention_weights.T @ grad_output
    
    # 计算scores的梯度也需要attention_weights参与
    # 因为Softmax的导数依赖于其输出值本身
    grad_scores = softmax_backward(grad_output, attention_weights)
    
    return grad_V, grad_scores
```

**结论**：`attention_weights`是计算梯度的必要参数，是连接前向与反向的桥梁，不可或缺。

## 3. 时空权衡策略

针对显存占用问题，工程上存在两种处理策略。这本质上是计算时间与存储空间的博弈。

| 策略 | 机制 | 代价 | 收益 |
|------|------|------|------|
| 缓存机制（默认） | 前向计算时保存所有中间激活值 | 高显存占用，序列长度$n=10000$时可能占用数GB显存 | 反向传播速度极快，直接读取显存数据 |
| 重计算机制 | 前向后立即释放，反向时重新计算 | 计算时间增加，相当于额外进行一次前向计算 | 极低的显存占用 |

**工程选择**：由于GPU算力昂贵且训练时间宝贵，PyTorch等主流框架默认选择缓存机制。用显存换取约30%的训练提速通常是划算的。

## 4. 自动求导原理

PyTorch通过动态计算图管理内存生命周期。其内部实现逻辑如下：

```python
# PyTorch内部伪代码
class Softmax(Function):
    @staticmethod
    def forward(ctx, input):
        output = compute_softmax(input)
        # ctx.save_for_backward将张量写入显存持久化区域
        ctx.save_for_backward(output) 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 从显存中读取之前保存的output
        output = ctx.saved_tensors[0]
        # 使用读取的值计算梯度
        return compute_softmax_grad(output, grad_output)
```

当调用`loss.backward()`时，程序沿着计算图回溯，依次取出`saved_tensors`。只有当该节点的梯度计算完成，这部分显存才会被释放。

## 5. 梯度检查点优化

当遇到OOM（显存溢出）错误时，可以使用梯度检查点（Gradient Checkpointing）技术。它是上述两种策略的折衷方案。

**原理**：不保存网络内部每一层的中间激活值，只保存少数几个关键节点的输入。

**过程**：反向传播经过某一层时，利用保存的输入临时重新计算该层的中间值。

**效果**：显存占用降低50%-70%，但训练时间增加30%左右。

```python
import torch.utils.checkpoint as checkpoint

# 使用checkpoint换取显存空间
def forward_with_checkpoint(Q, K, V):
    # 前向传播时不保存中间值，反向时重算
    return checkpoint.checkpoint(attention_layer, Q, K, V)
```

## 6. 核心要点

关于中间激活值，需要理解以下三个关键结论：

**数学必需**：反向传播的链式法则明确要求用到前向传播的中间结果

**效率至上**：默认保留中间值是为了避免重复计算，节省训练时间

**框架设计**：PyTorch等框架的自动求导机制依赖于中间值的存储

## 7. 相关词条

- Attention
- Backpropagation
- Gradient Checkpointing
- Memory Management