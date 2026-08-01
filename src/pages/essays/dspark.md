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


## 2. 两种起草方法

在 DSpark 之前，投机解码的起草模型分为两条路线。

**自回归起草**以 Eagle 系列为代表，其 draft 网络是 1～2 层的轻量 Transformer Decoder，输入是目标模型最后一层的 hidden state 序列和对应 token 的 embedding 序列拼接，输出是下一个位置的 hidden state，再由目标模型的 LM head 解码为 token。draft 网络每生成一个 token，就将其 append 到序列末尾再走一次前向，其运算方式与大模型很像，并且也维护了 KV cache 来避免重算。这种逐 token 的预测建模了 token 之间的条件依赖，起草质量较高。但为了起草速度，网络做得较浅，相当于用网络容量换推理速度。

Eagle 系列预测的是 hidden state 而非 token，好处在于：hidden state 是完整的语义表征，相比于 token 携带了更丰富的信息，因此 hidden state 层面的预测保留了更完整的信息，预测准确率更高；同时，hidden state 是连续空间，draft model 预测有误差时，LM head 仍有可能映射到正确的 token，其容错率比直接预测 token 要高。

**并行起草**以 DFlash 为代表，它是一个轻量级的块扩散模型（Block Diffusion），通过单步去噪生成 $\gamma$ 长度块的 hidden state，再并行采样出所有候选 token。

具体实现为：将待生成的 $\gamma$ 个位置全部填为固定的特殊 mask token，经目标模型的 embedding 层转换为 noise embedding；同时取目标模型多个中间层当前序列的 hidden states，拼接后投影到 draft 网络的 hidden dim，得到上下文特征 $x_{\text{ctx}}$。draft 网络通过 cross-attention 注入 $x_{\text{ctx}}$ 作为条件，单次前向完成去噪，输出 $\gamma$ 个位置的 hidden state，最后通过目标模型的 LM head 计算 logits，并采样得到候选 token。

DFlash 并行解码的优势在于：$T_{draft}$ 与 block 长度几乎无关——无论起草 8 个还是 32 个 token，都只需要一次前向，因此 draft 网络可以做得更深。DFlash 的局限性在于：各位置并行预测，看不到前一个位置具体采样得到的 token，只能基于隐藏层状态进行预测，因此位置越靠后累积偏差越大，$\tau$ 随 block 长度衰减。为了保证起草的准确率，$\gamma$ 一般不大，Qwen3 取 16，LLaMA 取 10。


| 维度 | 自回归起草（Eagle） | 并行起草（DFlash） |
|------|---------------------|-------------------|
| 起草方式 | 特征层面逐 token 自回归 | 块扩散，单步去噪并行输出 |
| 网络结构 | 1–2 层轻量 Transformer Decoder | 多层 Transformer Decoder + cross-attention |
| $T_{draft}$ 与 $\gamma$ 关系 | $T_{draft} \propto \gamma$，块越长越慢 | 几乎无关，一次前向出整个块 |
| 典型 $\gamma$ | 8（MoE 取 4） | 16（LLaMA 取 10） |
| 首位置预测 | 弱（网络浅） | 强（网络深） |
| 后续位置预测 | 强（建模条件依赖） | 弱（看不到前一个 token） |
| $\tau$ 随位置变化 | 维持 | 迅速衰减 |

并行起草凭借其网络深度首个 token 预测更准，但缺乏 token 间的条件依赖，后续位置接受率衰减；自回归起草网络较浅，首位置预测偏弱，但后续位置能维持准确率。

## 3. 半自回归架构：重并行骨干 + 轻量串行头

并行起草速度略快，首位置 token 质量高，自回归起草的整体准确率较好。为了结合这两类方法的优势，DSpark 采用：并行骨干负责首位置质量和起草速度，轻量串行头负责补上后续位置的条件依赖。具体分两阶段。

**并行阶段**：一个较深的并行骨干网络（基于 DFlash），单次前向输出 $\gamma$ 个位置的隐藏状态 $h_1, \dots, h_\gamma$，再由目标模型 LM head 得到基础 logits $U_1, \dots, U_\gamma$。这一步保留了并行模型"首 token 质量高、$T_{draft}$ 与 $\gamma$ 无关"的优点。

**串行阶段**：并行骨干输出的 logits $U_k$ 是各位置独立预测的，位置 $k$ 不知道前面位置最终采样出了什么 token，串行阶段要补上这个条件依赖。具体做法是在基础 logits 上加一个偏置项 $B_k$：

<div class="math-block">

$$p_k(v) = \text{softmax}\big(U_k(v) + B_k(x_0, x_{<k}, v)\big)$$

</div>

对基础 logits 加上偏置修正后再走 softmax，每个位置的预测就变成了"给定前面 token"后的条件分布。$B_k$ 默认只用一阶马尔可夫依赖，即只取决于前一个位置真实采样出的 token $x_{k-1}$（$x_{k-1}$ 是起草过程中位置 $k-1$ 经采样确定的具体 token），并做低秩分解以控制参数量：

```python
# Markov 头：偏置只依赖上一个 token，做低秩分解降低参数量
# W1: (V, r)  W2: (r, V)  r 通常取 256，远小于词表大小 V
def markov_bias(x_prev, W1, W2):
    # x_prev: 上一步真实采样出的 token id
    row = W1[x_prev]          # (r,)  查表得到低秩表示
    bias = row @ W2           # (V,)  投影回整个词表维度
    return bias               # 作为 logits 的加法修正项
```

论文实验结果：

在网络深度上，2 层并行骨干 + 串行头的 DSpark，在所有任务上超过了 5 层纯并行骨干的 DFlash，说明串行头的效果好于堆深并行骨干。串行头的额外延迟很小，$\gamma=4$ 时占整轮 0.2%，$\gamma=16$ 时占 1.3%。加了串行头后，接受长度最高提升 30%，且 $\gamma$ 越大提升越大。在 Qwen3-4B/8B/14B 上，DSpark 的平均接受长度比 DFlash 高 16%–18%，比 Eagle3 高 27%–31%。

## 4. 置信度调度验证：把验证长度变成一个吞吐优化问题

前面讲到，draft 模型生成的候选 token 交由 target 模型验证，验证方法是拒绝采样——逐个比较 draft 和 target 的概率比值，决定接受或拒绝。这要求 draft 模型每个位置的概率 $p_k^d$ 能直接算出来，DSpark 的局部 softmax 恰好满足这个条件。

验证的基本逻辑是：从前往后逐个判断，一旦某个位置被拒绝，后面的候选 token 全部作废，由 target 模型从该位置重新采样。所以验证长度越长，可能接受的 token 越多，但也意味着 target 模型要处理更多位置。问题是：验证多长的草稿块最划算？固定长度验证是"一刀切"——负载轻时验证太短是浪费，负载重时验证太长又会挤占宝贵的批处理容量。DSpark 把这个问题拆成两步。

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
