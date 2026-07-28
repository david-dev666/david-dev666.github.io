---
layout: ../../layouts/EssayLayout.astro
title: DPO：从带KL约束的RLHF目标出发
date: 2026-07-22
description: 
---

## 1. RLHF 的原始目标：带 KL 约束的强化学习

RLHF（Reinforcement Learning from Human Feedback）的标准流程分三步：首先做SFT让模型学会回答问题，再用人类偏好数据训一个奖励模型（RM）$r_\phi(x,y)$， 再用强化学习去最大化RM给出的奖励。 训练的时候发现，如果不加约束，模型会找到奖励的漏洞，产出高奖励但是低质量的文本，也叫**奖励作弊（Reward Hacking）**。

为了防止跑偏，RLHF 在奖励项上加了一个 KL 惯性约束，要求新策略 $\pi_\theta$ 不能离参考模型（通常是 SFT 之后的模型）$\pi_{ref}$ 太远：

<div class="math-block">

$$\max_{\pi_\theta} \; \mathbb{E}_{x\sim D,\, y\sim \pi_\theta(y|x)}\big[r_\phi(x,y)\big] - \beta \, D_{KL}\big(\pi_\theta(y|x) \,\|\, \pi_{ref}(y|x)\big)$$

</div>

$\beta$ 控制约束强度：$\beta$ 越大，越不允许模型偏离参考模型；$\beta \to 0$，则退化为纯粹的奖励最大化，容易跑偏。

这一步，参考模型的出场是"显式"的——它作为一个正则项约束训练过程，直觉上很好理解：不要让模型为了迎合奖励模型，忘了自己本来说人话的能力。

## 2. 这个约束优化问题有解析解

上面这个目标看起来要跑 RL 才能求解，但对于任意固定的奖励函数 $r$，这个 KL 约束优化问题其实有闭式解（用变分法可以直接推出）：

<div class="math-block">

$$\pi_r^*(y|x) = \frac{1}{Z(x)} \, \pi_{ref}(y|x) \, \exp\!\left(\frac{1}{\beta} r(x,y)\right)$$

</div>

其中 $Z(x) = \sum_y \pi_{ref}(y|x)\exp(r(x,y)/\beta)$ 是归一化常数（配分函数），只依赖 $x$，和具体的 $y$ 无关。

<div class="info-block">
<div class="info-block-title">讨论：这个式子在说什么？</div>
<div class="info-block-content">

这个式子讲了一件很朴素的事：最优策略，就是在参考模型的分布上，**按奖励重新"称重"**——奖励高的 $y$，概率质量被放大 $\exp(r/\beta)$ 倍；奖励低的被压低；然后重新归一化。$\beta$ 越大，指数项越平缓，重新称重的力度越小，最优策略就越接近原始的 $\pi_{ref}$。

这是 DPO 最关键洞察的起点：**最优策略天生就是参考模型的一个重加权版本，而不是从零学出来的独立分布**。

</div>
</div>

把这个式子反过来解，就能把奖励函数用策略表示出来：

<div class="math-block">

$$r(x,y) = \beta \log\frac{\pi_r^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

</div>

**这就是"为什么要和参考模型做差"的第一层原因**：这不是 DPO 作者事后加上去的正则项，而是从 KL 约束 RL 问题里反解出来的——策略和参考模型的对数概率比，本身就等价于奖励（只差一个只和 $x$ 有关的常数）。换句话说，$\log\pi_\theta(y|x) - \log\pi_{ref}(y|x)$ 天生带着"这是奖励"的物理含义，不是人为设计出来去凑损失函数的。

## 3. Bradley-Terry 偏好模型：做差消去配分函数 $Z(x)$

人类偏好数据一般是成对比较：给定 prompt $x$，标注员说"$y_w$（chosen）比 $y_l$（rejected）好"。这类偏好数据通常用 Bradley-Terry 模型建模：

<div class="math-block">

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$

</div>

$\sigma$ 是 sigmoid 函数。传统 RLHF 训练奖励模型 $r_\phi$ 时用的正是这个式子的负对数似然——DPO 的分类损失外壳（sigmoid + 负对数）直接沿用了这里，并没有重新发明：偏好数据本来就是"谁赢了"的二元标签，Bradley-Terry 模型用它拟合这个二元结果的概率，统计学上这就是逻辑回归。DPO 唯一做的，是把奖励模型 $r_\phi(x,y)$ 换成了策略与参考模型的隐式奖励，损失的形式原样保留。

DPO 的做法是：把第 2 节反解出来的奖励表达式，直接代进 Bradley-Terry 公式：

<div class="math-block">

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\Big[\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} + \beta\log Z(x)\Big] - \Big[\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} + \beta\log Z(x)\Big]\right)$$

</div>

<div class="info-block">
<div class="info-block-title">讨论：为什么恰好是这一步把参考模型"用活"了？</div>
<div class="info-block-content">

这是"为什么要做差"的第二层原因，也是整个推导里最关键的一步代数技巧：$\log Z(x)$ 只依赖 $x$，跟 $y_w$、$y_l$ 都没关系，两个奖励相减时会被原样减掉。

$Z(x)$ 是对整个输出空间 $y$ 穷举求和的配分函数（回忆一下：这和生成式模型里"图片词表" $256^{12000}$ 的问题是同一类困境），本身不可计算。如果只有单个 $y$，没法绕开它；但只要两个 $y$ 共享同一个 $x$（进而共享同一个 $Z(x)$），做差就能让它精确抵消。这正是 Bradley-Terry 模型"只看差值、不看绝对值"的性质，被 DPO 巧妙借用来甩掉了原本棘手的归一化常数。

</div>
</div>

消去 $Z(x)$ 后：

<div class="math-block">

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

</div>

## 4. DPO 损失函数

对这个概率取负对数似然，就得到了 DPO 的最终损失：

<div class="math-block">

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\,\mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

</div>

<div class="info-block">
<div class="info-block-title">讨论：sigmoid 和 -log，是不是套娃多此一举？</div>
<div class="info-block-content">

记 $z = \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$（即隐式奖励差），两层操作分工完全不同。

**sigmoid 不是可选项，它就是 Bradley-Terry 模型本身**：把无界的分数差 $z$ 映射成一个合法的概率 $P(y_w\succ y_l)=\sigma(z)$。没有它，"$y_w$ 比 $y_l$ 好的概率是多少"这句话就无法用一个数来表达——这一步在第 3 节推导时就已经确定了，不是损失函数设计时才加进去的。

**-log 才是真正决定"怎么训练"的一步**，原因藏在梯度里：

$$\frac{\partial}{\partial z}\big[-\log\sigma(z)\big] = \sigma(z) - 1 = -\sigma(-z)$$

$\sigma(-z)$ 正是"模型当前判错这一对的概率"。这个梯度是自适应权重：模型判得越离谱（$z\to-\infty$），梯度越接近 $-1$，狠狠纠正；模型已经判对且自信（$z\to+\infty$），梯度自动趋于 $0$，不再浪费更新。

如果去掉 -log，直接最大化 $\sigma(z)$，梯度会变成 $\sigma(z)\big(1-\sigma(z)\big)$——这个式子在**两端都会消失**：模型自信判对时梯度消失是合理的，但模型离谱判错时（$z\to-\infty$，$\sigma(z)\to0$）梯度也会消失，恰恰是最该纠错的地方却没有信号。取负对数似然（等价于二分类交叉熵）正是为了避开这个坑，把"该强则强"的不对称梯度找回来。所以 sigmoid 管建模、-log 管梯度形状，两者叠加正好是标准的 logistic loss，工程实现上通常用 `logsigmoid` 一步算完，避免分开计算带来的数值下溢。

</div>
</div>

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 当前正在训练的策略模型 |
| $\pi_{ref}$ | 参考模型，一般是 SFT 之后、偏好训练开始前的 checkpoint，训练中冻结不更新 |
| $y_w, y_l$ | 同一个 $x$ 下人类标注的偏好对，$y_w$ 优选，$y_l$ 次选 |
| $\beta$ | 隐式 KL 约束强度，越大越贴近参考模型 |

代码实现（这是 DPO 最常被引用的一段）：

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # 隐式奖励 = beta * (当前策略 - 参考模型) 的对数概率差
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # 两层"做差"：策略内部 chosen 减 rejected，再减掉参考模型的同一个差值
    logits = beta * (pi_logratios - ref_logratios)

    loss = -F.logsigmoid(logits).mean()
    return loss
```

这里有两层减法容易搞混：

1. **同一个模型内部，chosen 减 rejected**（`pi_logratios`、`ref_logratios`）——这是 Bradley-Terry 要求的"奖励差"。
2. **策略和参考模型之间做差**（`pi_logratios - ref_logratios`）——这是第 2 节"策略是参考模型的隐式奖励重加权"这个关系反解出来的。

## 5. 直觉：如果不减参考模型会怎样？

<div class="info-block">
<div class="info-block-title">讨论：去掉参考模型这一项，会发生什么？</div>
<div class="info-block-content">

假设直接用 $\sigma(\beta\log\pi_\theta(y_w|x) - \beta\log\pi_\theta(y_l|x))$ 做损失，看起来也能让 chosen 的概率高于 rejected，为什么不行？

**问题一：没有锚点，容易整体坍缩。** 优化这个目标最简单的办法，是无限提升 $y_w$ 的绝对概率、压低 $y_l$ 的绝对概率，而不需要维持模型的语言能力——因为损失只关心两者的相对大小，不关心模型是否还能生成通顺、多样的文本。参考模型项相当于一个隐式的 KL 约束：如果 $\pi_\theta(y|x)$ 涨得比 $\pi_{ref}(y|x)$ 还快，说明策略在这个 $y$ 上"用力过猛"，会被扣分；抑制 $y_l$ 也不能无限往下压，压得比参考模型还低才算真正拉开差距。这正是第 1 节里 $\beta D_{KL}(\pi_\theta \| \pi_{ref})$ 想做的事——DPO 没有丢掉这个约束，只是把它揉进了同一个 loss，而不是拆成"奖励模型 + KL 惩罚"两项分别优化。

**问题二：不同 $y$ 的"先天概率"不一样，绝对概率无法直接比较。** 常见短语（比如"我不知道"）在任何语言模型下先天概率都偏高，专业、具体的回答先天概率天然偏低——这和回答质量无关，只和 token 的常见程度有关。直接比较 $\pi_\theta(y_w)$ 和 $\pi_\theta(y_l)$ 的绝对大小，学到的更多是"哪句话更常见"而不是"哪句话更好"。而 $\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ 这个比值，衡量的是"当前策略比参考模型（训练开始时的模型）**多**偏好这个 $y$ 多少"——用参考模型的先天概率做了归一化，剩下的才是训练过程中新学到的、真正和偏好方向相关的信号。这正是第 2 节 $r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}$ 的物理含义：奖励是"相对参考模型的概率提升量"，不是绝对概率本身。

</div>
</div>

## 6. 总结

DPO 要解决的本质问题和标准 RLHF 一样：在一个 KL 约束下最大化奖励。区别在于 DPO 没有先训练奖励模型再跑强化学习，而是发现了这个约束优化问题的解析解——最优策略必然是参考模型按奖励重加权后的结果——于是把奖励反解成策略和参考模型的对数概率比，再代入 Bradley-Terry 偏好模型。这一步替换带来两个连带的好处：一是配分函数 $Z(x)$ 在做差后自然抵消，绕开了不可计算的归一化；二是参考模型作为隐式的 KL 锚点保留在了损失里，防止优化坍缩成只看相对大小、不顾语言能力的退化解。最终，一个原本需要"奖励模型 + PPO"两阶段强化学习才能解决的问题，被压缩成了一行二分类交叉熵损失。
</content>
