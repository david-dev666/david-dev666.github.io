---
layout: ../../layouts/EssayLayout.astro
title: DPO：从带 KL 约束的 RLHF 目标出发
date: 2026-07-25
description: 
---

## 1. 起点：RLHF 的原始优化目标

### 1.1 标准 RLHF 三段式

RLHF（Reinforcement Learning from Human Feedback）的标准流程分三步：

1. **SFT（Supervised Fine-Tuning）**：用高质量的指令-回答数据微调基座模型，得到一个能正常对话的模型。这个模型之后被称为参考模型 $\pi_{ref}$。
2. **奖励建模（Reward Modeling）**：收集人类偏好数据（同一个 prompt 下比较两个回答哪个更好），训练一个奖励模型 $r_\phi(x, y)$ 来打分。
3. **强化学习（RL）**：用 PPO 等算法，在奖励模型的引导下继续训练策略 $\pi_\theta$，最大化奖励。

### 1.2 Reward Hacking 和 KL 约束

第三步如果只做纯粹的奖励最大化：

<div class="math-block">

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(y \mid x)}\big[r_\phi(x, y)\big]$$

</div>

模型会找到奖励模型的漏洞——生成得分很高，但是无意义的文本，这叫**奖励作弊（Reward Hacking）**。

为了解决这个问题，RLHF 在优化目标里加了一项 KL 散度惩罚，要求新策略 $\pi_\theta$ 不能离参考模型 $\pi_{ref}$ 太远：

<div class="math-block">

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(y \mid x)}\big[r(x, y)\big] - \beta \, D_{KL}\big(\pi_\theta(y \mid x) \;\big\|\; \pi_{ref}(y \mid x)\big) \tag{1}
$$

</div>

其中 $D_{KL}(P \| Q) = \sum_y P(y) \log\frac{P(y)}{Q(y)}$。$\beta$ 控制约束强度：$\beta$ 越大，策略越不能偏离参考模型，$\beta \to 0$ 则退化为纯奖励最大化。优化目标可以概括为：**在保证模型原有语言能力的前提下，向人类偏好的方向微调。**


## 2. 约束优化问题的闭式解

式 $(1)$ 看起来需要跑强化学习才能求解，但实际上，对于任意固定的奖励函数 $r$，它是一个**带约束的凸优化问题**，可以解析求解。

### 2.1 写出拉格朗日函数

首先把目标写成最小化形式（加负号），并加上概率归一化的约束 $\sum_y \pi(y \mid x) = 1$：

<div class="math-block">

$$\min_{\pi} \; \mathbb{E}_{y \sim \pi}\big[-r(x, y)\big] + \beta \sum_y \pi(y \mid x) \log\frac{\pi(y \mid x)}{\pi_{ref}(y \mid x)}$$

$$\text{s.t.} \quad \sum_y \pi(y \mid x) = 1$$

</div>

引入拉格朗日乘子 $\lambda(x)$（注意 $\lambda$ 只依赖 $x$，因为约束是对每个 $x$ 独立施加的），写出拉格朗日函数：

<div class="math-block">

$$\mathcal{L}(\pi, \lambda) = \sum_y \pi(y \mid x)\big[-r(x, y)\big] + \beta \sum_y \pi(y \mid x) \log\frac{\pi(y \mid x)}{\pi_{ref}(y \mid x)} + \lambda(x)\Big(\sum_y \pi(y \mid x) - 1\Big)$$

</div>

### 2.2 对 $\pi(y \mid x)$ 求导

把求和号里的项按单个 $y$ 拆开，对 $\pi(y \mid x)$ 求偏导：

<div class="math-block">

$$\frac{\partial \mathcal{L}}{\partial \pi(y \mid x)} = -r(x, y) + \beta\Big[\log\frac{\pi(y \mid x)}{\pi_{ref}(y \mid x)} + 1\Big] + \lambda(x)$$

</div>

这里用到了 $\frac{\partial}{\partial \pi}\big[\pi \log\frac{\pi}{\pi_{ref}}\big] = \log\frac{\pi}{\pi_{ref}} + 1$。

令偏导为 $0$：

<div class="math-block">

$$-r(x, y) + \beta\Big[\log\frac{\pi(y \mid x)}{\pi_{ref}(y \mid x)} + 1\Big] + \lambda(x) = 0$$

</div>

### 2.3 解出 $\pi^*(y \mid x)$

移项整理：

<div class="math-block">

$$\beta \log\frac{\pi(y \mid x)}{\pi_{ref}(y \mid x)} = r(x, y) - \lambda(x) - \beta$$

$$\log\frac{\pi(y \mid x)}{\pi_{ref}(y \mid x)} = \frac{1}{\beta} r(x, y) - \frac{\lambda(x) + \beta}{\beta}$$

</div>

两边取指数：

<div class="math-block">

$$\frac{\pi(y \mid x)}{\pi_{ref}(y \mid x)} = \exp\!\left(\frac{1}{\beta} r(x, y)\right) \cdot \exp\!\left(-\frac{\lambda(x) + \beta}{\beta}\right)$$

</div>

令 $C(x) = \exp(-\frac{\lambda(x) + \beta}{\beta})$，这是一个只依赖 $x$ 的常数：

<div class="math-block">

$$\pi(y \mid x) = C(x) \cdot \pi_{ref}(y \mid x) \cdot \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

</div>

由归一化条件 $\sum_y \pi(y \mid x) = 1$ 可以确定 $C(x)$：

<div class="math-block">

$$C(x) \cdot \sum_y \pi_{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right) = 1 \quad\Rightarrow\quad C(x) = \frac{1}{\sum_y \pi_{ref}(y \mid x) \exp(r(x, y) / \beta)}$$

</div>

记 $Z(x) = \sum_y \pi_{ref}(y \mid x) \exp(r(x, y) / \beta)$，最终得到闭式解：

<div class="math-block">

$$\boxed{\pi_r^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{ref}(y \mid x) \, \exp\!\left(\frac{1}{\beta} r(x, y)\right)} \tag{2}$$

</div>

<div class="info-block">
<div class="info-block-title">直觉：这个式子在说什么？</div>
<div class="info-block-content">

这个式子讲了一件很朴素的事：**最优策略就是参考模型分布按奖励重新"称重"后的结果。**

- 奖励高的 $y$，概率质量被放大 $\exp(r / \beta)$ 倍
- 奖励低的 $y$，概率质量被压低
- 然后重新归一化

$\beta$ 越大，指数项越平缓，重加权的力度越小，最优策略就越接近原始的 $\pi_{ref}$。

这是 DPO 最关键洞察的起点：**最优策略天生就是参考模型的一个重加权版本，而不是从零学出来的独立分布。**

</div>
</div>

---

## 3. 反解奖励：用策略表示奖励

式 $(2)$ 建立了「给定奖励 → 最优策略」的映射。把它反过来解，就能把奖励函数用策略表示出来。

对式 $(2)$ 两边取对数：

<div class="math-block">

$$\log \pi_r^*(y \mid x) = \log \pi_{ref}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

</div>

移项：

<div class="math-block">

$$\frac{1}{\beta} r(x, y) = \log \pi_r^*(y \mid x) - \log \pi_{ref}(y \mid x) + \log Z(x)$$

</div>

两边乘 $\beta$：

<div class="math-block">

$$\boxed{r(x, y) = \beta \log\frac{\pi_r^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)} \tag{3}$$

</div>

<div class="info-block">
<div class="info-block-title">讨论：为什么要和参考模型做差？——第一层原因</div>
<div class="info-block-content">

式 $(3)$ 给出了一个深刻的等价关系：**策略和参考模型的对数概率比，本身就等价于奖励**（只差一个只和 $x$ 有关的常数 $\beta \log Z(x)$）。

换句话说，$\log\pi_\theta(y \mid x) - \log\pi_{ref}(y \mid x)$ 天生带着「这是奖励」的物理含义——它不是 DPO 作者拍脑袋加进去的正则项，而是从 KL 约束 RL 问题的闭式解里反解出来的必然结果。

奖励不是某个独立的神经网络的输出，而是「当前策略相对于参考模型**多**偏好这个 $y$ 多少」——奖励衡量的是**概率的相对提升**，不是绝对概率本身。

</div>
</div>

---

## 4. 代入 Bradley-Terry 偏好模型

### 4.1 Bradley-Terry 模型

人类偏好数据是成对比较的形式：给定 prompt $x$，标注员说「$y_w$ 比 $y_l$ 好」。这类数据通常用 Bradley-Terry 模型建模：

<div class="math-block">

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big) \tag{4}$$

</div>

直觉：两个回答的「实力」差距越大，chosen 胜出的概率越接近 $1$；实力相当时概率约为 $0.5$。$\sigma$ 把任意实数映射到 $(0, 1)$ 区间，和逻辑回归是同一个数学结构。

传统 RLHF 用这个式子的负对数似然来训练奖励模型 $r_\phi$。DPO 的做法是把式 $(3)$ 中反解出来的奖励表达式直接代进去。

### 4.2 代入隐式奖励

将式 $(3)$ 代入式 $(4)$：

<div class="math-block">

$$
\begin{aligned}
P(y_w \succ y_l \mid x) &= \sigma\!\left(\Big[\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} + \beta\log Z(x)\Big] - \Big[\beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)} + \beta\log Z(x)\Big]\right) \\[8pt]
&= \sigma\!\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right) \tag{5}
\end{aligned}
$$

</div>

<div class="info-block">
<div class="info-block-title">讨论：为什么要做差？——第二层原因</div>
<div class="info-block-content">

这是整个推导里最关键的一步代数技巧：$\beta \log Z(x)$ 只依赖 $x$，和 $y_w$、$y_l$ 都无关，两个奖励相减时被原样消掉。

$Z(x) = \sum_y \pi_{ref}(y \mid x) \exp(r / \beta)$ 是对整个输出空间穷举求和的配分函数——对于语言模型来说，输出空间是所有可能的 token 序列，这个求和根本不可计算。如果只有单个 $y$，绕不开它；但只要两个 $y$ 共享同一个 $x$（进而共享同一个 $Z(x)$），做差就能让它精确抵消。

这恰好利用了 Bradley-Terry 模型「只看差值、不看绝对值」的性质——DPO 巧妙地借这个性质甩掉了原本棘手的归一化常数。

</div>
</div>

### 4.3 化简结果

消去 $Z(x)$ 后，偏好概率只由策略和参考模型的对数概率比之差决定：

<div class="math-block">

$$\boxed{P(y_w \succ y_l \mid x) = \sigma\!\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)} \tag{6}$$

</div>

---

## 5. DPO 损失函数

对式 $(6)$ 取负对数似然，就得到了 DPO 的最终损失函数：

<div class="math-block">

$$\boxed{\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\,\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)\right]} \tag{7}$$

</div>

<div class="info-block">
<div class="info-block-title">讨论：sigmoid 和 -log 是套娃多此一举吗？</div>
<div class="info-block-content">

记 $z = \beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}$（即隐式奖励差）。两层操作分工完全不同：

**sigmoid 不是可选项，它就是 Bradley-Terry 模型本身。** 把无界的分数差 $z$ 映射成一个合法的概率 $P(y_w \succ y_l) = \sigma(z)$。没有它，「$y_w$ 比 $y_l$ 好的概率是多少」这句话就无法用一个数来表达。

**-log 才是真正决定「怎么训练」的一步。** 原因藏在梯度里：

<div class="math-block">

$$\frac{\partial}{\partial z}\big[-\log\sigma(z)\big] = \sigma(z) - 1 = -\sigma(-z)$$

</div>

$\sigma(-z)$ 正是「模型当前判错这一对的概率」。这个梯度是自适应权重：
- 模型判得越离谱（$z \to -\infty$，即把好的判成差的），梯度越接近 $-1$，狠狠纠正
- 模型已经判对且自信（$z \to +\infty$），梯度自动趋于 $0$，不再浪费更新

如果去掉 -log，直接最大化 $\sigma(z)$，梯度会变成 $\sigma(z)(1 - \sigma(z))$——这个式子在**两端都会消失**：自信判对时消失是合理的，但离谱判错时（$z \to -\infty$，$\sigma(z) \to 0$）也消失，恰恰是最该纠错的地方没有信号。取负对数似然正是为了把「该强则强」的不对称梯度找回来。

工程实现上通常用 `logsigmoid` 一步算完，避免分开计算带来的数值下溢。

</div>
</div>

---

## 6. 代码实现

把式 $(7)$ 翻译成代码，这是 DPO 最常被引用的一段：

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # 层 1：每个模型内部，chosen 和 rejected 的对数概率差
    # 对应 Bradley-Terry 的「分数差」
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # 层 2：策略的对数概率差 减去 参考模型的同一差值
    # 等价于隐式奖励差（式 (6) 中 σ 的输入）
    logits = beta * (pi_logratios - ref_logratios)

    # 负对数似然：等价于二分类交叉熵
    loss = -F.logsigmoid(logits).mean()
    return loss
```

代码里有两层减法，对应推导中的两个关键步骤：

1. **`pi_logratios` 和 `ref_logratios`**：同一个模型内部 chosen 减 rejected，对应 Bradley-Terry 要求的「奖励差」（式 $(4)$）。
2. **`pi_logratios - ref_logratios`**：策略和参考模型之间做差，对应式 $(3)$ 中「奖励 = 对数概率比」的反解关系。

两者叠在一起，就是式 $(6)$ 中 $\sigma$ 的输入。

---

## 7. 直觉：如果不减参考模型会怎样？

<div class="info-block">
<div class="info-block-title">讨论：去掉参考模型项，直接用策略的绝对概率做损失，为什么不行？</div>
<div class="info-block-content">

假设直接用 $\sigma(\beta\log\pi_\theta(y_w \mid x) - \beta\log\pi_\theta(y_l \mid x))$ 做损失，看起来也能让 chosen 的概率高于 rejected，为什么不行？

**问题一：没有锚点，容易整体坍缩。** 优化这个目标最简单的办法，是无限提升 $y_w$ 的绝对概率、压低 $y_l$ 的绝对概率，而不需要维持模型的语言能力——因为损失只关心两者的相对大小。参考模型项相当于一个隐式的 KL 约束：如果 $\pi_\theta(y \mid x)$ 涨得比 $\pi_{ref}(y \mid x)$ 还快，说明策略在这个 $y$ 上「用力过猛」，会被扣分。这正是式 $(1)$ 里 $\beta D_{KL}(\pi_\theta \| \pi_{ref})$ 想做的事——DPO 没有丢掉这个约束，只是把它揉进了同一个 loss。

**问题二：不同 $y$ 的「先天概率」不一样。** 常见短语（比如「我不知道」）在任何语言模型下先天概率都偏高，专业、具体的回答先天概率天然偏低——这和回答质量无关，只和 token 的常见程度有关。直接比较 $\pi_\theta(y_w)$ 和 $\pi_\theta(y_l)$ 的绝对大小，学到的更多是「哪句话更常见」而不是「哪句话更好」。

而 $\log\frac{\pi_\theta(y \mid x)}{\pi_{ref}(y \mid x)}$ 衡量的是「当前策略比参考模型**多**偏好这个 $y$ 多少」——用参考模型的先天概率做了归一化，剩下的才是训练过程中新学到的、真正和偏好方向相关的信号。这正是式 $(3)$ 的物理含义：**奖励是相对参考模型的概率提升量，不是绝对概率本身。**

</div>
</div>

---

## 8. 总结：推导链条一览

DPO 要解决的本质问题和标准 RLHF 一样：在一个 KL 约束下最大化奖励。区别在于 DPO 没有先训练奖励模型再跑 PPO，而是走了一条纯解析的路：

| 步骤 | 做了什么 | 关键公式 |
|------|---------|---------|
| 1. 设定目标 | 带 KL 约束的奖励最大化 | 式 $(1)$ |
| 2. 解析求解 | 用变分法求出闭式最优策略 | 式 $(2)$：$\pi^* \propto \pi_{ref} \cdot \exp(r / \beta)$ |
| 3. 反解奖励 | 把 $r$ 用 $\pi$ 和 $\pi_{ref}$ 表示 | 式 $(3)$：$r = \beta \log(\pi / \pi_{ref}) + \beta \log Z$ |
| 4. 代入 BT 模型 | 奖励代入 Bradley-Terry，做差消去 $Z(x)$ | 式 $(6)$ |
| 5. 取负对数似然 | 得到 DPO loss | 式 $(7)$ |

最终效果：一个原本需要「奖励模型 + PPO」两阶段强化学习才能解决的问题，被压缩成了一行二分类交叉熵损失。参考模型作为隐式的 KL 锚点保留在损失里，配分函数在做差时自然抵消——两个关键设计都不是巧合，而是从数学推导中自然涌现出来的。
