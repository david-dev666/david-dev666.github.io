---
layout: ../../layouts/ProjectLayout.astro
title: GQA分组查询注意力
description: Grouped Query Attention机制详解
tags: ["Transformer", "LLM", "Inference Optimization"]
---

# GQA: Grouped Query Attention

GQA（Grouped Query Attention）是Google在2023年提出的注意力机制优化方案，在Multi-Head Attention和Multi-Query Attention之间找到了平衡点，被LLaMA2、Mistral等主流大模型广泛采用。

## 1. 核心思想

GQA的本质是**KV头共享策略的折中方案**。它将Q头分成G组，每组共享一套K、V，从而在推理效率和模型质量之间取得平衡。

| 方案 | KV头数 | KV Cache压缩比 | 特点 |
|:---|:---|:---|:---|
| **MHA** | H（与Q相同） | 1x | 表达能力最强，显存开销最大 |
| **MQA** | 1 | H x | 显存最省，质量有损失 |
| **GQA** | G（介于1和H之间） | H/G x | 平衡效率与质量 |

当G=H时退化为MHA，当G=1时退化为MQA。

## 2. 从MHA到GQA的演进

### 2.1 Multi-Head Attention（MHA）

标准多头注意力中，每个头都有独立的Q、K、V投影矩阵：

```
Head 1: Q₁, K₁, V₁
Head 2: Q₂, K₂, V₂
  ...
Head H: Qₕ, Kₕ, Vₕ
```

**KV Cache的显存瓶颈**

在自回归解码中，模型每生成一个token都需要与所有历史token计算注意力。若每次重新计算历史K和V，复杂度为$O(n^2)$。工程实践采用KV Cache机制缓存已计算的K、V向量，将增量计算降至$O(n)$。

$$
\text{KV Cache} = 2 \times B \times L \times H_{kv} \times d_h \times N_{layers} \times \text{bytes}
$$

| 符号 | 含义 |
|:---|:---|
| $B$ | batch size |
| $L$ | 序列长度 |
| $H_{kv}$ | KV头数量 |
| $d_h$ | 每个头的维度 |
| $N_{layers}$ | 层数 |

以LLaMA2-70B为例（Q头64，KV头8，head_dim=128，80层，FP16）：

| 配置 | 计算 | KV Cache大小 |
|:---|:---|:---|
| MHA | 2 × 4096 × 64 × 128 × 80 × 2B | ≈ 10.5GB |
| GQA-8 | 2 × 4096 × 8 × 128 × 80 × 2B | ≈ 1.3GB |

GQA将KV Cache压缩至MHA的**1/8**。

### 2.2 Multi-Query Attention（MQA）

MQA由Google在2019年提出，核心思想是**所有Q头共享同一组K和V**：

```
Head 1: Q₁ ─┐
Head 2: Q₂ ─┼─→ 共享 K, V
  ...       │
Head H: Qₕ ─┘
```

**为何共享KV而非共享Q？**

| 角度 | 分析 |
|:---|:---|
| **语义角色** | Q定义检索模式，是多头表达能力的核心；KV是被检索的上下文，冗余度较高 |
| **显存收益** | KV Cache仅缓存K和V，共享KV直接削减缓存，共享Q无此收益 |
| **质量损失** | 研究表明不同头的KV表示高度冗余，共享后性能损失可控 |

**对训练的影响**

MQA在推理阶段收益显著（KV Cache需持续驻留显存），但在训练阶段影响不同：训练时整个序列并行计算，K和V无需缓存，仅作为中间张量参与前向/反向传播。MQA减少了KV投影层参数量，一定程度上限制了模型学习容量，从头训练的MQA模型在困惑度上通常略逊于MHA。

### 2.3 Grouped Query Attention（GQA）

GQA将Q头分成G组，每组共享一套K、V：

```
Group 1: Q₁, Q₂     ─→ K₁, V₁
Group 2: Q₃, Q₄     ─→ K₂, V₂
  ...
Group G: Qₕ₋₁, Qₕ   ─→ Kᵍ, Vᵍ
```

## 3. 为什么选择GQA

| 优势 | 说明 |
|:---|:---|
| **推理高效** | KV头从H减少到G，降低KV Cache显存占用和内存带宽压力 |
| **质量保持** | 相比MQA保留部分KV多样性，GQA-8质量接近MHA，速度接近MQA |
| **工程友好** | 支持从MHA模型通过UpTraining转换，无需从头训练 |

## 4. 代码实现（PyTorch）

### 4.1 核心操作：KV头扩展

GQA的关键是将少量KV头复制扩展以匹配Q头数量：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将KV头重复扩展以匹配Q头数量
    x: [batch, seq_len, num_kv_heads, head_dim]
    返回: [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    batch, seq_len, num_kv_heads, head_dim = x.shape
    # 插入新维度并扩展: [B, L, H_kv, 1, d] -> [B, L, H_kv, n_rep, d]
    x = x[:, :, :, None, :].expand(batch, seq_len, num_kv_heads, n_rep, head_dim)
    # 合并维度: [B, L, H_kv * n_rep, d]
    return x.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)
```

**关键逻辑解析**：

`[:, :, :, None, :]`在head维度后插入新维度，`expand`沿该维度复制n_rep次，`reshape`将复制维度合并回head维度。这是无拷贝操作（expand仅改变stride）。

### 4.2 完整GQA实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads  # 每个KV头对应的Q头数
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q投影: num_heads个头; KV投影: 仅num_kv_heads个头
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # 1. 线性投影
        q = self.q_proj(x)  # [B, L, num_heads * head_dim]
        k = self.k_proj(x)  # [B, L, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [B, L, num_kv_heads * head_dim]
        
        # 2. 重塑为多头形式
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # 3. 扩展KV头 (GQA核心步骤)
        k = repeat_kv(k, self.n_rep)  # [B, L, num_heads, head_dim]
        v = repeat_kv(v, self.n_rep)
        
        # 4. 转置为 [B, num_heads, L, head_dim] 以便并行计算
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        # 5. 计算注意力 (与标准MHA相同)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        # 6. 加权求和并合并多头
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        
        return self.o_proj(output)
```

**维度变换关键点**：

| 步骤 | Q维度 | K/V维度（扩展前） | K/V维度（扩展后） |
|:---|:---|:---|:---|
| 投影后 | [B, L, H×d] | [B, L, G×d] | - |
| 重塑后 | [B, L, H, d] | [B, L, G, d] | [B, L, H, d] |
| 转置后 | [B, H, L, d] | - | [B, H, L, d] |

### 4.3 使用PyTorch 2.0 SDPA加速

```python
class GQAWithSDPA(nn.Module):
    """使用Flash Attention / Memory-Efficient Attention加速"""
    
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        # PyTorch 2.0自动选择最优后端 (Flash Attention / xFormers)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, seq_len, -1))
```

## 5. 主流模型配置

| 模型 | Q头数 | KV头数 | 压缩比 | 说明 |
|:---|:---|:---|:---|:---|
| LLaMA2-70B | 64 | 8 | 8x | 首个大规模采用GQA的开源模型 |
| Mistral-7B | 32 | 8 | 4x | 高效7B模型的代表 |
| Qwen2-72B | 64 | 8 | 8x | 国产大模型采用GQA |
| Gemma-7B | 16 | 1 | 16x | 实际为MQA配置 |

## 6. 总结

GQA通过分组共享KV头，在保持模型质量的前提下显著提升推理效率：

| 要点 | 说明 |
|:---|:---|
| **核心操作** | `repeat_kv`将G个KV头扩展为H个，与Q头对齐 |
| **显存收益** | KV Cache压缩H/G倍，LLaMA2-70B从10.5GB降至1.3GB |
| **质量权衡** | GQA-8质量接近MHA，速度接近MQA |
| **工程实践** | 支持从MHA模型UpTraining转换，PyTorch 2.0原生支持SDPA加速 |

## 7. 相关词条

- Multi-Head Attention
- KV Cache
- Flash Attention
- Inference Optimization
