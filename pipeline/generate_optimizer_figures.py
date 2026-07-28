"""生成 notes/from-sgd-to-adamw.md 所需的两张配图。

输出目录：public/images/optimizers/
- fig_trajectory.png   各优化器在病态椭圆碗上的轨迹对比（教学版）
- fig_adaptive_lr.png  稀疏 vs 稠密参数的有效学习率曲线对比

依赖：numpy, matplotlib
运行：python pipeline/generate_optimizer_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

# ---------- 全局样式 ----------
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.labelsize"] = 11
rcParams["axes.titlesize"] = 13
rcParams["legend.frameon"] = False

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "optimizers",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- Fig 1: 优化器轨迹对比 ----------
# 选用病态椭圆碗 f(x, y) = 0.5 * (a*x^2 + b*y^2)，a >> b
# 直观形成"狭长峡谷"，是演示一阶优化器行为的教科书案例
A_COEF = 20.0   # x 方向陡峭
B_COEF = 1.0    # y 方向平坦


def loss(x, y):
    return 0.5 * (A_COEF * x ** 2 + B_COEF * y ** 2)


def grad(x, y):
    return np.array([A_COEF * x, B_COEF * y])


def run_sgd(start, lr, steps):
    p = np.array(start, dtype=float)
    traj = [p.copy()]
    for _ in range(steps):
        p = p - lr * grad(*p)
        traj.append(p.copy())
    return np.array(traj)


def run_momentum(start, lr, beta, steps):
    p = np.array(start, dtype=float)
    v = np.zeros_like(p)
    traj = [p.copy()]
    for _ in range(steps):
        v = beta * v + grad(*p)
        p = p - lr * v
        traj.append(p.copy())
    return np.array(traj)


def run_adam(start, lr, b1, b2, eps, steps):
    p = np.array(start, dtype=float)
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    traj = [p.copy()]
    for t in range(1, steps + 1):
        g = grad(*p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(p.copy())
    return np.array(traj)


def _draw_contour(ax):
    x = np.linspace(-1.6, 1.6, 400)
    y = np.linspace(-3.5, 3.5, 400)
    X, Y = np.meshgrid(x, y)
    Z = loss(X, Y)
    levels = np.linspace(0.5, 25, 12)
    ax.contourf(X, Y, Z, levels=levels, cmap="Blues", alpha=0.35)
    ax.contour(X, Y, Z, levels=levels, colors="#94a3b8", linewidths=0.5, alpha=0.8)


def _decorate(ax, title):
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"$\theta_1$  (steep direction)")
    ax.set_ylabel(r"$\theta_2$  (flat direction)")
    ax.set_title(title)
    ax.scatter(0, 0, color="#10b981", s=120, marker="*",
               zorder=6, edgecolor="white", linewidth=1.0, label="Optimum")


def fig_trajectory():
    start = (1.2, 3.0)

    # 各自调到能体现自身"特征"的超参，步数都设成"刚到谷底"，避免在原点反复打转
    # - SGD：lr 大到刚好在 steep 方向震荡，但还能向下推进（典型 zigzag）
    # - Momentum：lr 小一点 + 高动量，让横向震荡相互抵消，沿峡谷平滑下滑
    # - Adam：自适应步长，沿两轴几乎等速直奔最优
    sgd_traj = run_sgd(start, lr=0.09, steps=50)
    mom_traj = run_momentum(start, lr=0.005, beta=0.9, steps=70)
    # Adam 跑 80 步够收敛，但末段会在原点周围绕小圈（二次型上 Adam 的典型过冲），
    # 教学图里只保留前 50 步：这时 Adam 已贴近原点、形状干净，不展示二阶过冲细节。
    adam_traj = run_adam(start, lr=0.08, b1=0.9, b2=0.999, eps=1e-8, steps=80)[:50]

    # 2x2 布局：左侧三合一对比，右侧三个独立子图
    fig = plt.figure(figsize=(13, 8), dpi=140)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1, 1], hspace=0.35, wspace=0.3)

    # === 左大图：三合一 ===
    ax_main = fig.add_subplot(gs[:, 0])
    _draw_contour(ax_main)
    ax_main.plot(*sgd_traj.T, color="#ef4444", lw=1.4,
                 label="SGD (zigzag)", alpha=0.9, marker="o", markersize=2.5)
    ax_main.plot(*mom_traj.T, color="#3b82f6", lw=1.6,
                 label="Momentum (smooth)", alpha=0.9, marker="o", markersize=2.5)
    ax_main.plot(*adam_traj.T, color="#7c3aed", lw=1.8,
                 label="Adam (adaptive)", marker="o", markersize=2.5)
    # 终点标记：空心圆标记每条轨迹的最后位置
    for traj, color in [(sgd_traj, "#ef4444"), (mom_traj, "#3b82f6"), (adam_traj, "#7c3aed")]:
        ax_main.scatter(*traj[-1], facecolors="white", edgecolors=color,
                        s=70, linewidth=2.0, zorder=6)
    ax_main.scatter(*start, color="black", s=80, zorder=5,
                    edgecolor="white", linewidth=1.2, label="Start")
    _decorate(ax_main, "All three overlaid")
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.legend(loc="lower right", fontsize=10,
                   framealpha=0.9, facecolor="white")

    # === 右上：SGD 单独图 ===
    ax1 = fig.add_subplot(gs[0, 1])
    _draw_contour(ax1)
    ax1.plot(*sgd_traj.T, color="#ef4444", lw=1.3,
             marker="o", markersize=2.5, alpha=0.95)
    ax1.scatter(*sgd_traj[-1], facecolors="white", edgecolors="#ef4444",
                s=70, linewidth=2.0, zorder=6)
    ax1.scatter(*start, color="black", s=50, zorder=5,
                edgecolor="white", linewidth=1.0)
    _decorate(ax1, "SGD: zig-zag in the canyon")
    # 标注震荡区：放右下空白处，避开红色 zigzag
    ax1.text(0.35, -2.8, "oscillates across\nthe steep walls",
             fontsize=9, color="#b91c1c")

    # === 右上中：Momentum ===
    ax2 = fig.add_subplot(gs[0, 2])
    _draw_contour(ax2)
    ax2.plot(*mom_traj.T, color="#3b82f6", lw=1.5,
             marker="o", markersize=2.5, alpha=0.95)
    ax2.scatter(*mom_traj[-1], facecolors="white", edgecolors="#3b82f6",
                s=70, linewidth=2.0, zorder=6)
    ax2.scatter(*start, color="black", s=50, zorder=5,
                edgecolor="white", linewidth=1.0)
    _decorate(ax2, "Momentum: glides down the valley")
    # 放左下空白，远离起点
    ax2.text(-1.4, -2.8, "oscillation\ncancelled out",
             fontsize=9, color="#1d4ed8")

    # === 右下：Adam ===
    ax3 = fig.add_subplot(gs[1, 1:])
    _draw_contour(ax3)
    ax3.plot(*adam_traj.T, color="#7c3aed", lw=1.7,
             marker="o", markersize=2.8, alpha=0.95)
    ax3.scatter(*adam_traj[-1], facecolors="white", edgecolors="#7c3aed",
                s=80, linewidth=2.0, zorder=6)
    ax3.scatter(*start, color="black", s=50, zorder=5,
                edgecolor="white", linewidth=1.0)
    _decorate(ax3, "Adam: per-axis step size, near-straight path")
    ax3.text(-1.5, -2.5, "steep axis: small step\nflat axis: large step",
             fontsize=9, color="#5b21b6")

    fig.suptitle(
        r"Optimizers on $f(\theta) = \frac{1}{2}(20\,\theta_1^2 + \theta_2^2)$ — "
        "a narrow valley exposes their differences",
        fontsize=13, y=0.995,
    )

    out = os.path.join(OUTPUT_DIR, "fig_trajectory.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")


# ---------- Fig 2: 自适应学习率示意 ----------
def fig_adaptive_lr():
    """
    模拟两类参数：
      - 稠密参数：每步都有较大梯度
      - 稀疏参数：偶发大梯度，多数步梯度为 0
    展示 SGD vs Adam 下的有效步长对比。
    """
    np.random.seed(0)
    steps = 200

    g_dense = np.random.randn(steps) * 1.0
    g_sparse = np.zeros(steps)
    spikes = np.random.choice(steps, size=20, replace=False)
    g_sparse[spikes] = np.random.randn(20) * 1.0

    def adam_effective_lr(grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        m = v = 0.0
        eff = []
        for t, g in enumerate(grads, start=1):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1 ** t)
            v_hat = v / (1 - b2 ** t)
            step = lr * m_hat / (np.sqrt(v_hat) + eps)
            eff.append(step)
        return np.array(eff)

    sgd_dense = 1e-3 * g_dense
    sgd_sparse = 1e-3 * g_sparse
    adam_dense = adam_effective_lr(g_dense)
    adam_sparse = adam_effective_lr(g_sparse)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140, sharey=False)

    ax = axes[0]
    ax.plot(sgd_dense, color="#ef4444", lw=1.0, label="SGD", alpha=0.85)
    ax.plot(adam_dense, color="#7c3aed", lw=1.4, label="Adam")
    ax.set_title("Dense Gradient Parameter")
    ax.set_xlabel("step")
    ax.set_ylabel("update size")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend()

    ax = axes[1]
    ax.plot(sgd_sparse, color="#ef4444", lw=1.0, label="SGD", alpha=0.85)
    ax.plot(adam_sparse, color="#7c3aed", lw=1.4, label="Adam")
    ax.set_title("Sparse Gradient Parameter")
    ax.set_xlabel("step")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend()

    fig.suptitle("Effective Update: SGD vs Adam", y=1.02)
    out = os.path.join(OUTPUT_DIR, "fig_adaptive_lr.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    fig_trajectory()
    fig_adaptive_lr()
