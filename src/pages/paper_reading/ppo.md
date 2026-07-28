---
layout: ../../layouts/ProjectLayout.astro
title: PPO
description: 深度强化学习与大语言模型对齐的基石算法
tags: ["Reinforcement Learning", "RLHF", "OpenAI"]
---

# PPO: Proximal Policy Optimization

实现一阶优化算法在稳定性与数据效率上的完美平衡，现代大语言模型RLHF训练事实上的标准方案。

<p class="related-content" style="margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
<strong>论文信息</strong>：<a href="https://arxiv.org/abs/1707.06347" style="color: #2563eb;">arXiv:1707.06347</a> | John Schulman et al. | OpenAI | July 2017
</p>

Proximal Policy Optimization是一类旨在平衡实现复杂度、采样效率与调参鲁棒性的策略梯度算法。它通过引入裁剪机制来限制策略更新的幅度，有效解决了传统策略梯度算法中步长难以确定导致的训练崩溃问题，同时避免了TRPO等二阶优化算法的高昂计算成本。

## 1. 核心能力

- **数值稳定性**：通过裁剪概率比率，强制将策略更新限制在安全区域内，防止因单次过大的参数更新导致策略性能崩塌且无法恢复
- **样本高效性**：区别于传统策略梯度算法只能对采样数据进行一次梯度更新，PPO允许在同一批采样数据上进行多轮小批量随机梯度下降优化，显著提升了数据利用率
- **实现简洁性**：仅依赖一阶梯度信息，无需计算海森矩阵或进行复杂的共轭梯度求解，易于集成到现有的深度学习框架中



## 2. 算法原理：裁剪代理目标函数

PPO的核心在于重新定义了优化目标。它构造了一个保守的代理目标函数，该函数构成了策略性能的悲观下界。

### 2.1 概率比率

定义新旧策略在动作$a$上的概率比率$r_t(\theta)$：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

该比率量化了新策略相对于旧策略的行为变化幅度。

### 2.2 裁剪机制

为了防止$r_t(\theta)$偏离1过远，PPO引入了裁剪函数`clip`。最终的目标函数$L^{CLIP}$取未裁剪目标与裁剪目标的最小值：

$$
L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]
$$

其中$\epsilon$为裁剪阈值，论文中推荐值为0.2，$\hat{A}_t$为优势函数估计。

这一设计蕴含了**截断收益，保留惩罚**的博弈思想：

- **当优势函数为正**：表示当前动作优于平均水平。我们希望提升该动作概率，即增大$r_t$。但为了稳健，一旦$r_t$超过$1+\epsilon$，收益将被截断，防止策略过度自信更新
- **当优势函数为负**：表示当前动作劣于平均水平。我们希望降低该动作概率，即减小$r_t$。若$r_t$低于$1-\epsilon$，惩罚项依旧保留（因为取最小值），确保模型能充分吸取错误教训



## 3. 系统架构：Actor-Critic

在实际工程应用及LLM训练中，PPO通常采用Actor-Critic共享参数架构或分离架构，总损失函数包含策略梯度损失、价值函数损失与熵正则项。

| 组件 | 数学符号 | 职责 | 优化目标 |
|:---|:---|:---|:---|
| **Actor** | $\pi_\theta$ | 策略网络，负责输出动作概率分布 | 最大化$L^{CLIP}$以提升期望回报 |
| **Critic** | $V_\phi$ | 价值网络，负责评估当前状态的价值 | 最小化$L^{VF}$即均方误差$(V_\phi(s_t) - V_t^{target})^2$ |
| **Entropy** | $H[\pi]$ | 熵正则项，衡量策略的随机性 | 最大化熵以鼓励探索，防止过早收敛 |

## 4. 关键算法与代码实现

### 4.1 PPO 损失函数实现

这是算法最核心的逻辑，直接决定了梯度的反向传播路径。

```python
import torch

def compute_ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    """
    计算 PPO 裁剪代理损失
    
    Args:
        old_log_probs: 旧策略下动作的对数概率，需要 detach
        new_log_probs: 当前策略网络计算出的对数概率，保留梯度
        advantages: 优势函数值，通常经过 GAE 计算与归一化
        epsilon: 裁剪超参数
    """
    # 计算概率比率 r_t = exp(log_new - log_old)
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 策略梯度第一部分：未裁剪的原始目标
    surr1 = ratio * advantages
    
    # 策略梯度第二部分：裁剪后的目标
    # 将 ratio 强制限制在 [1-eps, 1+eps] 区间内
    ratio_clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    surr2 = ratio_clipped * advantages
    
    # 取两者的最小值实现悲观下界
    # 注意：由于优化器执行的是最小化操作，故取负号实现最大化
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss

```

### 4.2 广义优势估计(GAE)

PPO依赖高质量的优势函数估计。论文推荐使用GAE算法来平衡偏差与方差：

$$
\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

其中$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$是TD误差。参数$\lambda$控制了偏差-方差权衡：$\lambda=1$时等价于蒙特卡洛估计，$\lambda=0$时等价于单步TD估计。

### 4.3 训练主循环

PPO采用**交互采样**与**多轮优化**交替进行的模式。

1. **收集数据**：运行旧策略$\pi_{\theta_{old}}$收集$T$个时间步的轨迹数据，包括状态、动作、奖励及旧对数概率
2. **计算优势**：利用Critic网络计算价值估计，结合奖励序列通过GAE算法计算优势函数
3. **多轮更新**：
   - 将收集到的轨迹数据打散，划分为多个小批量Minibatch
   - 遍历$K$个Epoch（通常为3-10次）
   - 在每个Batch上，重新计算新策略的概率分布与比率，执行梯度下降更新Actor和Critic参数



```python
# 伪代码演示训练流程
for iteration in range(total_iterations):
    # 1. 采样阶段 (Rollout)
    trajectories = run_policy(env, actor_old)
    
    # 2. 优势计算
    values = critic(trajectories.states)
    advantages = compute_gae(trajectories.rewards, values)
    
    # 3. 优化阶段 (Optimization)
    for epoch in range(k_epochs):
        for batch in get_batches(trajectories, advantages):
            # 计算新策略分布
            new_log_probs = actor_new(batch.states, batch.actions)
            
            # 计算各项损失
            loss_policy = compute_ppo_loss(batch.old_log_probs, new_log_probs, batch.advantages)
            loss_value = mse_loss(critic(batch.states), batch.returns)
            loss_entropy = -torch.mean(entropy)
            
            total_loss = loss_policy + c1 * loss_value - c2 * loss_entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
    # 同步旧策略参数
    actor_old.load_state_dict(actor_new.state_dict())

```

## 5. 总结

PPO的核心哲学在于**有节制的更新**。

它摒弃了复杂的信任区域约束求解，转而使用极其简单的裁剪操作实现了类似的效果。这种工程上的优雅使其能够适应从机器人控制到Atari游戏，再到如今大语言模型微调等各种复杂的强化学习场景，成为连接监督学习与强化学习的重要桥梁。