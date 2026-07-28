---
layout: ../../layouts/ProjectLayout.astro
title: ControlNet
description: 为文生图扩散模型添加精确的空间条件控制
tags: ["Diffusion Model", "Image Generation", "ControlNet"]
---

# ControlNet

为文生图扩散模型添加精确的空间条件控制。

<p class="related-content" style="margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
<strong>仓库地址</strong>：<a href="https://github.com/lllyasviel/ControlNet" style="color: #2563eb;">github.com/lllyasviel/ControlNet</a>
</p>

## 1. 核心算法

ControlNet 是一种专为文生图扩散模型（如 Stable Diffusion）设计的神经网络架构，通过引入显式的**空间条件控制（Spatial Conditioning）**，将边缘检测、深度图、人体姿态等结构化信息作为额外输入，解决了传统扩散模型难以精确把控图像几何结构与布局的问题。

### 1.1 传统方法的局限性

在 ControlNet 提出之前，可控图像生成面临三大核心挑战：

| 挑战 | 问题描述 |
|:--|:--|
| **空间语义缺失** | CLIP 文本编码器擅长理解"颜色"、"风格"等语义特征，但缺乏对几何形状和空间位置的细粒度感知。即便是详尽的文本描述，也无法约束生成对象的轮廓或透视关系 |
| **灾难性遗忘** | 直接微调扩散模型不仅需要高昂算力，更会破坏预训练权重分布，导致模型丧失原有的高质量生成能力 |
| **小样本过拟合** | 传统微调对数据规模极其敏感，缺乏大规模成对数据时极易过拟合，使得精确可控生成长期成为高门槛技术 |

### 1.2 ControlNet 的技术方案

<div class="info-block">
<div class="info-block-title">前置知识：Stable Diffusion 架构</div>
<div class="info-block-content">

理解 ControlNet 需要先了解其基座模型 Stable Diffusion 的核心组件：

| 组件 | 功能 |
|:--|:--|
| **VAE** | 将图像压缩到潜在空间（512×512×3 → 64×64×4），降低计算成本 |
| **UNet** | 在潜在空间执行迭代去噪，从纯噪声逐步生成清晰图像 |
| **CLIP 编码器** | 将文本提示转换为语义向量，引导生成方向 |

<p style="text-align: center; margin: 1rem 0;">
<img src="/images/unet1.jpeg" alt="UNet架构示意图" style="max-width: 75%; display: block; margin: 0 auto; border-radius: 6px;">
</p>

**UNet** 采用对称的编码器-解码器架构，形成特征的"U形"流动。左侧编码器通过卷积和下采样逐层压缩特征（13层，分辨率从64×64降至8×8），提取从边缘纹理到高级语义的多尺度表示；底部中间块处理最抽象的8×8特征；右侧解码器通过上采样逐层恢复分辨率，<strong>跳跃连接（Skip Connection）</strong>将编码器各层的细节特征直接传递给对应的解码器层，避免细节丢失。

在**Stable Diffusion**系统中，每个去噪时间步，UNet接收噪声图像$z_t$、时间步$t$和文本条件$c$，预测噪声$\epsilon_\theta(z_t, t, c)$用于计算去噪结果。

</div>
</div>

ControlNet 核心思路：**不修改 UNet 解码器，而是复制编码器作为可训练副本，通过 13 个零卷积层向跳跃连接注入空间控制信号**。这种设计被称为<strong>"锁定-可训练"双副本架构（Locked-Trainable Dual Copy）</strong>，实现了低成本、零破坏的微调范式。具体实现包含四个核心机制：

| 机制 | 说明 |
|:--|:--|
| **特征对齐** | `input_hint_block`卷积层将像素级控制条件（如Canny边缘图、OpenPose骨架）映射到与潜在空间维度一致的特征向量 |
| **双副本机制** | 锁定副本冻结原始权重，保留生成先验；可训练副本通过反向传播学习解析控制条件 |
| **零卷积冷启动** | 零初始化卷积层（$W=0, b=0$）确保训练初期输出为零，避免破坏预训练模型；权重从零逐渐更新，平滑注入控制信号 |
| **效率优势** | 锁定副本无需计算梯度，大幅降低显存；仅用5万张图像的Fill50K数据集，约4000步即可收敛 |

## 2. 仓库结构

```
ControlNet/
├── annotator/          # 条件提取器（Canny、HED、Midas深度、OpenPose等）
├── cldm/               # 核心实现
│   ├── cldm.py         # ControlNet + ControlledUnetModel + ControlLDM
│   └── ddim_hacked.py  # 修改后的DDIM采样器
├── ldm/                # Stable Diffusion基础模块
├── models/             # 配置文件（cldm_v15.yaml, cldm_v21.yaml）
├── gradio_*.py         # Gradio演示
└── tutorial_*.py       # 训练脚本
```

### 2.1 工作流程

用户输入原始图像和文本提示，`annotator`中的条件提取器将图像转换为控制条件（Canny边缘图、深度图、姿态图等）。控制条件通过`input_hint_block`编码为多尺度特征，文本提示通过CLIP编码器生成上下文向量。ControlNet的可训练编码器融合两者，通过零初始化卷积层输出13个控制信号，注入UNet解码路径的跳跃连接，在每个时间步引导去噪方向。最终DDIM采样器迭代去噪，VAE解码器输出图像。

### 2.2 核心模块

| 类名 | 位置 | 职责 |
|:--|:--|:--|
| `ControlNet` | `cldm/cldm.py` | 可训练编码器副本，`forward`输出13个多尺度控制信号 |
| `ControlledUnetModel` | `cldm/cldm.py` | 继承UNetModel，锁定副本不计算梯度，解码路径注入控制信号 |
| `ControlLDM` | `cldm/cldm.py` | 继承LatentDiffusion，`apply_model`通过`control_scale`调节控制强度 |
| `zero_module` | `ldm/.../util.py` | 将模块参数置零，`make_zero_conv`创建零初始化1×1卷积 |

### 2.3 条件提取器

`annotator`目录下各提取器实现简洁。以Canny为例，`CannyDetector`仅需一行`cv2.Canny`调用；`annotator/util.py`提供`HWC3`（确保图像格式）和`resize_image`（调整尺寸为64倍数）等工具函数。

## 3. 核心架构

### 3.1 整体架构与输入流

**ControlNet如何介入现有的UNet？**

ControlNet并非直接修改主模型的输入层，而是作为一个并行分支存在。我们可以对比一下两者的输入差异：

| 组件 | 输入数据 | 作用说明 |
|:---|:---|:---|
| **主UNet (Locked)** | 噪声潜码`x_noisy` + 文本`context` | 负责基础的去噪生成，参数锁定，不直接接收控制图 |
| **ControlNet (Trainable)** | 噪声潜码`x_noisy` + 文本`context` + 控制信号`hint` | 专门学习如何将`hint`（如边缘图）映射为去噪特征 |

> **核心理解**：ControlNet不改变主UNet的输入结构（无需重新训练主模型），而是将计算出的控制特征，通过**跳跃连接（Skip Connection）**以"做加法"的形式注入到主UNet的解码器中。

ControlNet架构本身是通用的。它不决定任务是txt2img还是img2img——这取决于初始潜码是纯噪声还是由原图加噪而来。ControlNet只是在去噪过程中提供额外的空间约束（Spatial Condition）。

<div class="info-block">
<div class="info-block-title">3.2 ControlNet核心架构</div>
<div class="info-block-content">

<p style="text-align: center; margin: 1rem 0;">
<img src="/images/controlnet_architecture.png" alt="ControlNet架构图" style="max-width: 60%; display: block; margin: 0 auto; border-radius: 6px;">
</p>

上图展示了ControlNet如何附加到Stable Diffusion的U-Net结构上。

* **左侧**：原始SD模型（Locked，参数锁定）。
* **右侧**：ControlNet结构（Trainable，可训练副本）。

**核心数据流：**

1. **输入层**：
* 时间步$t$和文本提示$\boldsymbol{c}_t$同时注入双路网络。
* 图像潜码$\boldsymbol{z}_t$(Input)输入到主SD Encoder，同时也输入到ControlNet。
* **控制条件$\boldsymbol{c}_f$**(如Canny边缘图)经过一个**Zero Convolution**层后，与$\boldsymbol{z}_t$的特征相加，进入ControlNet的编码块。

2. **ControlNet处理**：
* ControlNet复制了SD编码器的完整结构（12个编码块 + 1个中间块）。
* 结构分级：从Block A($64 \times 64$)到Block D($8 \times 8$)，再到Middle Block。
* 这些层是**可训练的(Trainable Copy)**，用于学习如何将条件$\boldsymbol{c}_f$转化为控制特征。

3. **特征注入(Skip Connections)**：
* ControlNet的输出经过**Zero Convolution**层处理。
* 处理后的信号直接**相加**到SD Decoder对应的跳跃连接(Skip-connection)上。
* 最终输出为预测噪声$\boldsymbol{\epsilon}_\theta$。

<hr style="border: 0; border-top: 1px dashed #ccc; margin: 15px 0;">

**关键技术点解析**

| 组件 | 作用 |
|:---|:---|
| **SD Encoder (Locked)** | 保持原SD的生成能力，提取$\boldsymbol{z}_t$的语义特征，**不进行梯度更新**。 |
| **ControlNet (Trainable)** | SD编码器的深拷贝。专门用于提取控制条件$\boldsymbol{c}_f$的特征，并与主特征融合。 |
| **Zero Convolution** | 初始化权重和偏置均为0的$1 \times 1$卷积。确保训练初始阶段ControlNet输出为0，即$y = x$，完全保留SD原有能力，实现"零破坏"启动。 |

**Q&A深入理解**

* **Q: 所谓的"13个连接点"具体指哪里？**
* **A:** 对应图中的蓝色箭头。SD的Encoder包含12个Block(4个分辨率 × 3个重复层)加上1个Middle Block。
* ControlNet输出的特征会注入到SD **Decoder**对应的12个Block和1个Middle Block中。
* 公式：$12 + 1 = 13$个控制信号注入点。

* **Q: 为什么图中有$64 \times 64$到$8 \times 8$的变化？**
* **A:** 这是U-Net的下采样过程。Block A是浅层特征（高分辨率$64 \times 64$），包含更多几何/边缘细节；Block D是深层特征（低分辨率$8 \times 8$），包含更多语义信息。ControlNet在所有尺度上都施加了控制，从而保证了从整体轮廓到局部细节的一致性。

* **Q: 这种设计对显存有什么影响？**
* **A:** 非常高效。因为左侧SD模型是锁定的（Locked），不需要计算梯度，只有右侧的ControlNet需要更新参数。
* 根据论文数据，在NVIDIA A100上微调时，相比单纯训练SD，ControlNet仅增加了约**23%的GPU显存占用**。

</div>
</div>

### 3.3 设计优势总结

| 设计 | 实现机制 | 优势 |
|:---|:---|:---|
| **零破坏(Zero-Padding)** | Zero Convolution($1 \times 1$卷积) | 训练初期完全等价于预训练SD，避免"灾难性遗忘"，微调收敛极快。 |
| **计算高效** | 锁定原模型参数(Frozen) | 大幅减少显存需求(仅增~23%)，无需重新训练庞大的生成模型。 |
| **多尺度控制** | 注入所有Decoder层 | 无论是宏观构图($8 \times 8$)还是微观纹理($64 \times 64$)都能被精确控制。 |
| **组合扩展性** | 特征直接相加 | 多个ControlNet的输出可以直接叠加（如：姿态控制 + 深度控制），无需修改模型架构。 |

## 4. 代码实现

### 4.1 推理流程

**入口函数**（`cldm/cldm.py`）：

```python
def apply_model(self, x_noisy, t, cond, *args, **kwargs):
    # Step 1: ControlNet 生成 13 个多尺度控制信号
    control = self.control_model(x=x_noisy, hint=cond['c_concat'], 
                                  timesteps=t, context=cond['c_crossattn'])
    # Step 2: 控制强度缩放（推理时可调节）
    control = [c * scale for c, scale in zip(control, self.control_scales)]
    # Step 3: 锁定的 UNet 使用控制信号去噪
    return diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control)
```

`x_noisy`是当前时间步的带噪图像，`hint`是控制条件图像如边缘图或姿态图，`context`是文本嵌入向量。

`control_model`输出一个包含13个张量的列表，对应UNet的12个编码层和1个中间层。每个张量的空间分辨率与对应UNet层匹配。

`control_scales`是一个长度为13的缩放系数列表，默认全为1.0。推理时可以调整这些值来控制不同层的影响强度，例如将前几层设为0可以减弱低层细节控制。

**ControlNet 前向传播**：

```python
def forward(self, x, hint, timesteps, context):
    emb = self.time_embed(timesteps)           # 时间步编码
    guided_hint = self.input_hint_block(hint)  # 控制条件编码: 512×512→64×64
    
    outs = []
    h = x
    for module, zero_conv in zip(self.input_blocks, self.zero_convs):
        h = module(h, emb, context)
        if guided_hint is not None:            # 仅第一层注入控制条件
            h += guided_hint
            guided_hint = None
        outs.append(zero_conv(h))              # 每层输出经零卷积
    
    h = self.middle_block(h, emb, context)
    outs.append(self.middle_block_out(h))
    return outs  # 返回 13 个控制信号
```

`input_hint_block`是一个轻量级卷积网络，将控制条件从原始分辨率下采样到64×64，与UNet第一层的特征图尺寸对齐。

`guided_hint`仅在第一层通过元素加法注入。这是ControlNet的关键设计：控制条件只在入口处注入一次，后续层通过网络自然传播这些信息。`guided_hint = None`确保后续循环不会重复注入。

`zero_conv`是权重初始化为零的1×1卷积。训练初期输出全零，随着训练逐渐学习到有意义的控制信号。这保证了训练开始时ControlNet不会干扰原始UNet的行为。

**UNet 注入控制信号**：

```python
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps, context, control=None):
        hs = []
        with torch.no_grad():                  # 编码器锁定，不计算梯度
            h = x
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
        
        if control is not None:
            h += control.pop()                 # middle_block 注入控制
        
        for module in self.output_blocks:
            # 核心：跳跃连接 = 编码器特征 + 控制信号
            h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)
```

`torch.no_grad()`包裹编码器部分，冻结原始UNet权重不参与梯度计算。这是ControlNet能够快速微调的关键：只有ControlNet分支和解码器的注入点需要训练。

`hs`列表存储编码器每层的输出，用于UNet的跳跃连接。`control.pop()`按后进先出顺序取出控制信号，确保高分辨率的控制信号对应高分辨率的解码层。

`hs.pop() + control.pop()`是核心注入方式：控制信号与编码器特征相加后，再与解码器当前特征拼接。这种加法注入而非替换保留了原始UNet的语义信息，控制信号仅起到调制作用。

### 4.2 零卷积(Zero Convolution)的设计美学

ControlNet最精妙的设计之一，就是如何让一个全新的网络"无感"地接入到已经训练完美的SD模型中。这里使用了**零卷积**来实现Zero-Shot Cold Start（冷启动）。

**核心代码实现**

在PyTorch中，实现非常简单，就是显式地将权重和偏置初始化为0：

```python
class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.zero_module(self.conv)  # 执行零初始化

    def zero_module(self, module):
        """将模块的权重和偏置置为0"""
        for p in module.parameters():
            p.detach().zero_()  # detach()断开计算图，zero_()原地置零
        return module

    def forward(self, x):
        return self.conv(x)
```

`detach()`的作用是绕过Autograd计算图，直接对参数tensor进行原地修改。如果不调用`detach()`，PyTorch会尝试追踪这个置零操作并构建梯度，导致报错或产生不必要的计算开销。

**为什么必须是全零初始化？**

如果按照常规思路使用Gaussian(Xavier/Kaiming)初始化，初始权重$W$会服从正态分布。这就意味着在Step 0时：

$$y_{total} = y_{SD} + \text{noise}$$

这会瞬间破坏预训练SD原本精细的特征分布。模型在训练初期不得不花费大量步数去"抵消"这些随机噪声，甚至引发"灾难性遗忘"。

零卷积的作用是让初始状态下的ControlNet分支输出恒为$\vec{0}$：

$$y_{total} = y_{SD} + 0$$

这样，微调的第一步完全等价于原版SD，保证了模型站在巨人的肩膀上起步，而不是从坑里爬出来。

**权重是0，梯度怎么传？**

这里有一个反直觉的地方：**初始化为0，并不意味着梯度也是0**。

我们可以通过简单的链式法则验证一下梯度流。前向传播很简单：$y = W \cdot x + b$，因为$W=0$，所以$y=0$。但在反向传播时，我们要计算的是Loss对$W$的偏导：

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial W}$$

因为$\frac{\partial y}{\partial W} = x$（这里的$x$是ControlNet编码器的特征输出，它是由输入图片决定的，$x \neq 0$），所以：

$$\frac{\partial \mathcal{L}}{\partial W} = \delta_{out} \cdot x \neq 0$$

只要SD的Decoder传回了误差信号($\delta_{out}$)，且ControlNet提取到了特征($x$)，这个0权重的层就能瞬间获得梯度并开始更新。它就像一个完美的"软开关"，从0开始平滑开启。

**训练动态观察**

这种初始化策略下的训练过程非常线性且可控：

| 训练阶段 | 权重范数$\|W\|$ | 现象 |
|:---|:---|:---|
| Step 0 | 0.00 | 模型表现与原版SD完全一致，无缝衔接 |
| Early Stage | ~0.1 | 权重开始打破零点，控制信号（如边缘引导）开始微弱介入 |
| Converged | >0.5 | 权重显著增长，ControlNet已经学会在特定区域强行"接管"生成过程 |

### 4.3 控制条件编码

控制条件是512×512×3的像素图像，而UNet工作在64×64×320的特征空间。`input_hint_block`实现维度对齐：

```python
self.input_hint_block = TimestepEmbedSequential(
    # 512×512×3 → 256×256×16 → 128×128×32 → 64×64×96 → 64×64×320
    conv_nd(dims, 3, 16, 3, padding=1), nn.SiLU(),     # 保持尺寸
    conv_nd(dims, 16, 32, 3, padding=1, stride=2),     # ↓2x
    conv_nd(dims, 32, 96, 3, padding=1, stride=2),     # ↓2x  
    conv_nd(dims, 96, 256, 3, padding=1, stride=2),    # ↓2x
    zero_module(conv_nd(dims, 256, 320, 3, padding=1)) # 末层零卷积
)
```

设计要点：3次下采样对齐UNet第一层尺寸（64×64）；渐进式通道扩张匹配UNet通道数（320）；末层零卷积保证训练初期无影响。

### 4.4 网络结构总览

ControlNet复制UNet编码器结构，13层输出通过零卷积注入主UNet的跳跃连接：

```
ControlNet（可训练）                    主UNet（锁定）                   
┌─────────────────────┐              ┌──────────────────────────┐
│ input_blocks[0-2]   │──zero_conv──→│ decoder[10-12] 64×64×320 │
│ input_blocks[3-5] ↓ │──zero_conv──→│ decoder[7-9]   32×32×640 │
│ input_blocks[6-8] ↓ │──zero_conv──→│ decoder[4-6]  16×16×1280 │
│ input_blocks[9-11]↓ │──zero_conv──→│ decoder[1-3]   8×8×1280  │
│ middle_block        │──zero_conv──→│ middle_block   8×8×1280  │
└─────────────────────┘              └──────────────────────────┘
```

控制信号在13个尺度同时影响生成：低分辨率层控制全局结构，高分辨率层控制局部细节。

## 5. 使用示例

```python
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.canny import CannyDetector
import numpy as np

# 1. 加载模型
model = create_model('./models/cldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth'))
ddim_sampler = DDIMSampler(model)

# 2. 准备控制条件
apply_canny = CannyDetector()
input_image = np.array(Image.open("input.jpg"))
detected_map = apply_canny(input_image, 100, 200)

# 3. 预处理控制图
control = torch.from_numpy(detected_map).float().cuda() / 255.0
control = control.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

# 4. 设置文本提示
prompt = "a beautiful landscape, high quality, detailed"
cond = {
    "c_concat": [control],
    "c_crossattn": [model.get_learned_conditioning([prompt])]
}

# 5. 采样生成
shape = (4, 64, 64)  # 潜在空间尺寸
samples, _ = ddim_sampler.sample(
    ddim_steps=20,
    conditioning=cond,
    batch_size=1,
    shape=shape
)

# 6. 解码输出
x_samples = model.decode_first_stage(samples)
```

## 6. 参考资料

- https://arxiv.org/abs/2302.05543
- https://github.com/lllyasviel/ControlNet
