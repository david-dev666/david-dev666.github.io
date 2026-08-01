---
layout: ../../layouts/EssayLayout.astro
title: DSpark：给投机解码装一个"置信度调度器"
date: 2026-07-17
description: 半自回归起草 + 置信度调度验证，如何同时解决生成质量与系统吞吐两个正交问题
---

## 1. 投机解码的效率公式

LLM 自回归解码速度慢，主要有两个原因：首先是推理算法无法并行，第 t 个 token 的预测依赖于前 t-1 个 token 的 KV 向量，靠堆算力无法加速；其次是显存访问瓶颈，每生成一个 token，需要将完整的模型权重从 HBM 读取到计算单元一次，计算量并不大，大部分时间在等待数据搬运。batch 推理能摊薄一部分访存开销，但由于推理请求是随机到达的，batch size 波动较大，并行度也有限。

为了加速 LLM 推理，有研究者提出了投机解码（Speculative Decoding）方法，其核心思路是"先起草，再验证"：用一个小模型（Draft Model）快速生成 N 个候选 token，再交给大模型（Target Model）并行验证，一次接受多个 token，从而减少大模型逐 token 解码的开销。

投机解码的效率公式：

<div class="math-block">

$$L = \frac{T_{draft} + T_{verify}}{\tau}$$

</div>

$L$ 是生成一个 token 的平均延迟，$T_{draft}$ 是起草耗时，$T_{verify}$ 是验证耗时，$\tau$ 是每轮平均被接受的 token 数。要降低 $L$，可以从三个方向入手：加速起草，加速验证，让起草更准确。


## 2. 两种起草模型的两难

在 DSpark 之前，起草模型分成两条路线，各有一个绕不开的缺陷。

| 类型 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| 自回归起草（Autoregressive） | Eagle 系列 | 逐 token 建模条件依赖，起草质量高 | $T_{draft} \propto \gamma$，块越长起草越慢，被迫用浅层网络+短块 |
| 并行起草（Parallel） | DFlash、Medusa | 一次前向输出整个 $\gamma$ 长度的块，$T_{draft}$ 几乎与块长无关 | 各位置独立预测，$\tau$ 随位置迅速衰减 |

<div class="info-block">
<div class="info-block-title">讨论：并行起草为什么会"衰减"？</div>
<div class="info-block-content">

并行起草模型在一次前向中同时预测 $x_1, x_2, \dots, x_\gamma$，但预测 $x_2$ 时并不知道 $x_1$ 究竟采样成了什么——它看到的只是 $x_1$ 的所有可能性叠加在一起的隐藏状态。当 $x_1$ 存在多个合理选项时（比如续写"我喜欢"后面可以接"吃"或"睡"），模型对 $x_2$ 的预测会是这些可能性的一种"平均"，而不是"给定 $x_1=\text{吃}$ 之后"该接什么词。论文把这种现象称为**多模态碰撞（Multi-Modal Collision）**：位置越靠后，累积的分支越多，预测和真实目标分布的偏差就越大，接受率也就跌得越快。

自回归模型则不存在这个问题——它逐 token 生成，每一步都严格条件于上一步*真实采样出*的 token，所以后面位置的条件接受率反而能维持住甚至略微上升。这也是论文观察到的一个现象：自回归模型的首 token 接受率不如并行模型（首 token 只能靠浅网络硬顶），但越往后走，自回归反超并行。
</div>
</div>

也就是说：并行模型赢在"起跑快、首步准"，输在"越往后越不准"；自回归模型正好相反。DSpark 的第一个洞察，就是把两者的优点缝在一起。

## 3. 半自回归架构：重并行骨干 + 轻量串行头

DSpark 的起草模型分两阶段工作。

**并行阶段**：一个较深的并行骨干网络（基于 DFlash），单次前向输出 $\gamma$ 个位置的隐藏状态 $h_1, \dots, h_\gamma$ 和基础 logits $U_1, \dots, U_\gamma$。这一步保留了并行模型"首 token 质量高、$T_{draft}$ 几乎与 $\gamma$ 无关"的优点。

**串行阶段**：在基础 logits 之上叠加一个极轻量的偏置项 $B_k$，把"给定前面已经采样出的 token"这个条件信息注入进来：

<div class="math-block">

$$P(X|x_0) = \prod_{k=1}^{\gamma} p_k(x_k \mid x_0, x_{<k}), \quad p_k(v \mid x_0, x_{<k}) = \text{softmax}\big(U_k(v) + B_k(x_0, x_{<k}, v)\big)$$

</div>

默认版本只用一阶马尔可夫依赖，即 $B_k$ 只取决于前一个 token $x_{k-1}$，并做低秩分解以控制开销：

```python
# Markov 头：偏置只依赖上一个 token，做低秩分解降低参数量
# W1: (V, r)  W2: (r, V)  r 通常取 256，远小于词表大小 V
def markov_bias(x_prev, W1, W2):
    # x_prev: 上一步真实采样出的 token id
    row = W1[x_prev]          # (r,)  查表得到低秩表示
    bias = row @ W2           # (V,)  投影回整个词表维度
    return bias               # 作为 logits 的加法修正项
```

**关键设计**：这个修正是局部加法，不需要对整个序列重新做全局归一化，所以每个位置的 $p_k$ 仍然是一个精确的 softmax 分布——这意味着可以直接套用标准的拒绝采样做验证，而不必像 CRF 类非自回归方法那样为了精确验证付出额外代价。论文的实验也验证了这一点：2 层的 DSpark 串行头就能超过 5 层的纯并行 DFlash 基线，且块长从 4 扩展到 16，额外延迟只增加 0.2%–1.3%，换来的是最高 30% 的接受长度提升。**一点点自回归，就能走很远。**

## 4. 置信度调度验证：把验证长度变成一个吞吐优化问题

起草质量提升之后，第二个问题是：验证多长的草稿块最划算？固定长度验证是"一刀切"——负载轻时验证太短是浪费，负载重时验证太长又会挤占宝贵的批处理容量。DSpark 把这个问题拆成两步。

**第一步：估计每个位置的存活概率。** 训练一个置信度头，输出标量 $c_k \in (0,1)$，表示"前面全部被接受的条件下，这个 token 也被接受"的概率：

<div class="math-block">

$$c_k = \sigma\big(w^T [h_k; W_1[x_{k-1}]]\big)$$

</div>

监督信号用的是解析解——草稿分布和目标分布的全变差距离：$c_k^* = 1 - \frac{1}{2}\lVert p_k^d - p_k^t \rVert_1$。

<div class="info-block">
<div class="info-block-title">讨论：神经网络给出的置信度为什么不能直接用？</div>
<div class="info-block-content">

论文发现，置信度头本身判别力很强（ROC-AUC 能到 0.81–0.90），但存在明显的过度自信（Overconfident）问题——校准误差（ECE）高达 3%–8%。原因不难理解：分类任务的置信度头习惯于把预测"推向"0或1两端来降低交叉熵，但这里我们要的是概率的准确刻度，不是排序。

DSpark 用了一个很轻的修正——**序列温度缩放（Sequential Temperature Scaling）**：因为整条链的联合存活概率是逐位置概率的连乘 $\prod_i c_i$，误差会沿链累积放大，所以从左到右逐位置做一维网格搜索，找一个最优温度标量去缩放每一位的置信度，使 ECE 最小。这是一个保序变换，不会打乱原来的排序，只是把"过分自信"的数值往合理区间拉回去。做完之后 ECE 降到了约1%。
</div>
</div>

**第二步：把验证长度选择变成全局吞吐最大化。** 有了每个请求每个位置的存活概率 $a_{r,j} = \prod_{i \le j} c_{r,i}$，就可以定义批大小 $B = \sum_r (1+\ell_r)$、期望接受数 $\tau = \sum_r (1 + \sum_j a_{r,j})$，以及目标吞吐量：

<div class="math-block">

$$\Theta = \tau \cdot \text{SPS}(B)$$

</div>

其中 $\text{SPS}(B)$（Steps Per Second）是引擎在批大小 $B$ 下的实测吞吐，来自离线剖析表。调度算法很直接：把所有请求的候选位置按存活概率 $a_{r,j}$ 降序排列，贪心地一个个纳入验证集合，同时更新 $B$、$\tau$、$\Theta$，一旦 $\Theta$ 不再增长就立刻停止。

```python
# 硬件感知前缀调度器：贪心 + 早停
# 由于 a_{r,j} 关于 j 单调不增，贪心策略等价于全局最优分配
candidates = sorted(all_positions, key=lambda p: p.survival_prob, reverse=True)

best_theta, verify_set = 0.0, []
tau, batch_size = base_tau, base_batch_size
for pos in candidates:
    tau_next = tau + pos.survival_prob
    batch_next = batch_size + delta_batch(pos)   # 纳入该位置对批容量的增量
    theta_next = tau_next * sps_table[batch_next] # 查表得到新吞吐

    if theta_next <= best_theta:
        break  # 吞吐不再增长，立即早停，保证决策不依赖未采样的未来token
    best_theta = theta_next
    tau, batch_size = tau_next, batch_next
    verify_set.append(pos)
```

这里的早停不只是效率考量，更是正确性要求：如果继续往后看已经采样出的、但尚未纳入决策的 token，就相当于让调度器"偷看未来"，论文附录专门证明了这会引入选择偏差。轻负载时，调度器会主动把验证预算从固定的 2 token 扩展到 4–6 token；重负载饱和时，则平滑收缩验证长度，优先砍掉低置信度的位置，把批处理容量让给更值得验证的请求。

## 5. 训练目标

草稿模型的训练损失由三项组成，且都按位置加权 $w_k = e^{-(k-1)/\gamma}$（更看重靠前的位置，因为前面错了后面全部作废）：

<div class="math-block">

$$\mathcal{L} = 0.1 \cdot \mathcal{L}_{ce} + 0.9 \cdot \mathcal{L}_{tv} + 1.0 \cdot \mathcal{L}_{conf}$$

</div>

$\mathcal{L}_{ce}$ 是常规交叉熵，$\mathcal{L}_{tv}$ 是草稿分布与目标分布的全变差损失（直接优化接受率而不是绕道交叉熵），$\mathcal{L}_{conf}$ 是置信度头的二分类损失。训练时目标模型全程冻结，草稿模型也共享并冻结了 embedding 层和 LM head，只更新骨干网络、串行头和置信度头——这保证了新增的训练成本很小，且不会干扰目标模型原有的能力。

## 6. 一个反直觉的发现

论文里最有意思的一组对比,是关于"为什么并行生成的首 token 反而比自回归更准"。直觉上自回归模型应该处处占优,但实验显示,在位置 1(首 token),并行模型的接受率明显高于自回归模型(比如聊天场景 0.72 对 0.53)。

原因在于资源分配的取舍:自回归起草模型的延迟随块长线性增长,为了控制 $T_{draft}$,只能被迫使用很浅的网络;并行模型不受这个约束,可以把参数预算全部押注在网络深度上,首 token 的预测自然更准。而首 token 又是整条链的"地基"——一旦它被拒绝,后面全部白起草——所以这里的杠杆效应最大。DSpark 的架构选择,本质上是想让骨干网络继续吃到"深层网络预测首 token 更准"的红利,再用一个几乎不增加延迟的轻量串行头,去弥补并行模型在中后段位置的短板。

## 7. 总结

DSpark 没有在"自回归"和"并行"两条路线之间选边站,而是把两者拆开来看:并行骨干负责起草速度和首 token 质量,轻量串行头负责补上 token 间依赖,置信度调度器负责在系统层面把验证预算花在刀刃上。三者分别对应效率公式里的 $T_{draft}$、$\tau$、有效 $T_{verify}$,是同一个优化问题的三个正交切面,分别用架构设计和调度算法逐一击破。

在 DeepSeek-V4 的真实线上流量中,相比固定验证长度的生产基线 MTP-1,DSpark 在同等吞吐下把单用户生成速度提到了 60%–85%。这也印证了论文的一个核心立场:投机解码的性能瓶颈,从来不只是"起草模型准不准"这一个变量,系统侧"验证策略聪不聪明"同样重要——把两者放在一起联合优化,才能真正把 Pareto 前沿往外推。
