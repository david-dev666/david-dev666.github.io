---
layout: ../../layouts/ProjectLayout.astro
title: CLIP
description: 图文跨模态检索与零样本图像分类
tags: ["VLM", "CLIP"]
---

# CLIP

图文跨模态检索与零样本图像分类。

<p class="related-content" style="margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
<strong>相关内容</strong>：<a href="/projects/clip_models" style="color: #2563eb;">CLIP 预训练模型</a>
</p>

## 1. 核心能力

- **零样本图像分类**：无需在特定数据集上微调，即可利用文本描述直接对图片进行分类
- **跨模态检索**：实现以文搜图或以图搜文
- **多模态特征提取**：为各类下游任务提供统一且强大的视觉和文本特征表示

## 2. 仓库结构

```
clip/
├── __init__.py          # 包入口，导出clip.py接口
├── clip.py              # 主API：load(), tokenize(), available_models()
├── model.py             # 模型架构：CLIP, VisionTransformer, ModifiedResNet, Transformer
├── simple_tokenizer.py  # BPE文本分词器
└── bpe_simple_vocab_16e6.txt.gz  # BPE词表
```

| 模块 | 功能 |
|------|------|
| `clip.load(name)` | 加载预训练模型 |
| `clip.tokenize(texts)` | 文本转token张量|
| `model.encode_image()` | 图像→特征向量 |
| `model.encode_text()` | 文本→特征向量 |

## 3. Notebooks脚本

仓库提供了两个Notebook，用于演示方法。

### 3.1 Interacting_with_CLIP.ipynb

**入门教程**：演示CLIP的图文相似度计算与零样本图像分类

**整体流程**：加载模型 → 准备数据 → 提取特征 → 计算相似度 → 零样本分类

```python
# ========== 1. 加载模型 ==========
import clip
model, preprocess = clip.load("ViT-B/32")

# ========== 2. 准备输入 ==========
# 图像预处理：Resize → CenterCrop → Normalize
image_input = preprocess(Image.open("dog.jpg")).unsqueeze(0).cuda()
# 文本分词：添加[SOT]和[EOT]，padding到77
text_tokens = clip.tokenize(["a dog", "a cat", "a bird"]).cuda()

# ========== 3. 提取特征 ==========
with torch.no_grad():
    image_features = model.encode_image(image_input).float()  # [1, 512]
    text_features = model.encode_text(text_tokens).float()    # [3, 512]

# ========== 4. 计算相似度 ==========
# 归一化后点积 = 余弦相似度
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = image_features @ text_features.T  # [1, 3]

# ========== 5. 零样本分类 ==========
# 乘以100（logit_scale）后softmax得到概率
probs = (100.0 * similarity).softmax(dim=-1)
# 输出: tensor([[0.95, 0.03, 0.02]]) → 分类为"a dog"
```

| 步骤 | 说明 |
|------|------|
| **加载模型** | `clip.load()`返回`model`（图像+文本编码器）和`preprocess`（图像预处理函数） |
| **准备输入** | 图像经`preprocess`变为`[3, 224, 224]`张量；文本经`tokenize()`变为`[N, 77]`的token张量 |
| **提取特征** | `encode_image()`和`encode_text()`分别将图像和文本映射到512维向量空间 |
| **计算相似度** | L2归一化后，向量点积即为余弦相似度，结果shape为`[1, 3]` |
| **零样本分类** | 乘以温度系数100放大差异，`softmax`转为概率分布，最高概率对应的文本即为分类结果 |

图像`preprocess`包括：

| 操作 | 含义 | 原因 |
|------|------|------|
| Resize | 将图像缩放到指定尺寸 | 模型输入要求固定尺寸（224×224），原始图像尺寸各异 |
| CenterCrop | 从图像中心裁剪出正方形区域 | 去除边缘无关内容，保留主体 |
| Normalize | 将像素值标准化（减均值、除标准差） | 使数据分布与训练时一致，提升效果 |

**核心公式**：

余弦相似度：

$$
\text{similarity}(I, T) = \frac{\boldsymbol{f}_I \cdot \boldsymbol{f}_T}{|\boldsymbol{f}_I| \times |\boldsymbol{f}_T|}
$$

其中$\boldsymbol{f}_I = \text{encode\_image}(I)$，$\boldsymbol{f}_T = \text{encode\_text}(T)$。

零样本分类概率：

$$
P(\text{class}_i | I) = \frac{e^{100 \times \text{sim}(I, T_i)}}{\sum_j e^{100 \times \text{sim}(I, T_j)}}
$$

### 3.2 Prompt_Engineering_for_ImageNet.ipynb

**进阶技巧**：通过Prompt Engineering提升ImageNet零样本分类效果

```python
# ========== 80种文本模板（Prompt Engineering核心）==========
imagenet_templates = [
    'a bad photo of a {}.',           # 模糊/低质量场景
    'a photo of many {}.',            # 多个物体
    'a sculpture of a {}.',           # 雕塑形式
    'a photo of the hard to see {}.', # 难以辨认
    'a low resolution photo of the {}.',
    'a rendering of a {}.',           # 渲染图
    'graffiti of a {}.',              # 涂鸦
    'a tattoo of a {}.',              # 纹身
    'the embroidered {}.',            # 刺绣
    'a photo of a large {}.',         # 大尺寸
    'a photo of a small {}.',         # 小尺寸
    'a photo of the {}.',
    'a photo of a {}.',
    # ... 共80种模板，覆盖各种场景和描述方式
]

# ========== 构建零样本分类器权重 ==========
def zeroshot_classifier(classnames, templates):
    """
    为每个类别生成文本特征（多模板ensemble）
    
    返回: [embed_dim, num_classes]的权重矩阵
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            # 1. 用所有模板生成该类别的描述
            texts = [template.format(classname) for template in templates]
            # 例如classname="dog" → ["a bad photo of a dog.", "a photo of many dog.", ...]
            
            # 2. 编码所有描述
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)  # [80, 512]
            
            # 3. 归一化后取平均（ensemble）
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)  # [512]
            class_embedding /= class_embedding.norm()
            
            zeroshot_weights.append(class_embedding)
        
        # 堆叠成矩阵[512, 1000]
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

# 构建ImageNet 1000类的分类器
zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)

# ========== 零样本预测 ==========
with torch.no_grad():
    for images, target in loader:
        # 1. 提取图像特征
        image_features = model.encode_image(images.cuda())
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 2. 计算与所有类别的相似度
        logits = 100. * image_features @ zeroshot_weights  # [B, 1000]
        
        # 3. 计算准确率
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
```

- **80种文本模板的设计思路**：
  - 覆盖不同**图像质量**：`a bad photo`、`a low resolution photo`
  - 覆盖不同**艺术形式**：`a sculpture`、`graffiti`、`a tattoo`、`embroidered`
  - 覆盖不同**尺寸描述**：`a large`、`a small`
  - 覆盖不同**数量描述**：`many`
  - 目的是让文本特征更鲁棒，适应各种图像变体

- **`zeroshot_classifier()`函数流程**：
  - **Step 1**：对每个类别（如"dog"），用80个模板生成80个描述
  - **Step 2**：将80个描述编码为80个512维向量
  - **Step 3**：归一化后取平均，得到该类别的"代表性"特征向量
  - **Step 4**：对平均后的向量再次归一化，确保单位长度
  - **最终输出**：`[512, 1000]`的权重矩阵，每列是一个类别的特征（1000是ImageNet的类别数）

- **零样本预测流程**：
  - 图像特征`[B, 512]`与权重矩阵`[512, 1000]`相乘
  - 得到`[B, 1000]`的logits，每行是该图像与1000个类别的相似度
  - `topk(5)`取相似度最高的5个类别作为预测结果

- **为什么ensemble有效**：
  - 单一模板（如`"a photo of a dog"`）可能无法覆盖所有图像场景
  - 多模板平均后的特征更加"居中"，对各种变体都有较好的响应
  - 实验表明，80模板ensemble比单模板提升约5%准确率

**启示**：文本prompt的设计对CLIP性能影响很大，这也是后来Prompt Learning（如CoOp、CLIP-Adapter）研究方向的起点。

## 4. Core Code

### 4.1 CLIP主模型类(model.py)

该类实现了CLIP的核心双塔架构，包含图像编码器和文本编码器，并定义了前向传播中的相似度计算逻辑。

```python
class CLIP(nn.Module):
    """CLIP主模型：集成图像与文本双塔编码器"""
    
    def __init__(self,
                 embed_dim: int,           # 图文特征对齐后的统一维度
                 image_resolution: int,    # 输入图像分辨率
                 vision_layers,            # 视觉编码器层数配置
                 vision_width: int,        # 视觉编码器通道宽度
                 vision_patch_size: int,   # ViT架构的Patch大小
                 context_length: int,      # 文本序列最大长度，固定为77
                 vocab_size: int,          # BPE词表大小
                 transformer_width: int,   # 文本Transformer宽度
                 transformer_heads: int,   # 文本Transformer注意力头数
                 transformer_layers: int   # 文本Transformer层数
                 ):
        super().__init__()
        self.context_length = context_length

        # 1. 初始化图像编码器：根据vision_layers类型动态选择架构
        if isinstance(vision_layers, (tuple, list)):
            # 传入列表时使用ResNet架构
            self.visual = ModifiedResNet(...)
        else:
            # 传入整数时使用ViT架构
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_width // 64,
                output_dim=embed_dim
            )

        # 2. 初始化文本编码器：构建GPT风格的Transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # 构建因果掩码以防止看到未来Token
        )

        # 文本嵌入层与位置编码
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        # 文本投影层：将文本特征映射至与图像特征相同的维度
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # 温度参数：控制Softmax分布陡峭程度，初始化为log(1/0.07)且可学习
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        """图像编码流程：输入图像并输出特征向量"""
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """文本编码流程：输入Token IDs并输出特征向量"""
        # Token嵌入与位置编码叠加
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)

        # Transformer编码，需调整维度以符合PyTorch标准
        x = x.permute(1, 0, 2)  # 转换为 [Seq_Len, Batch, Width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # 还原为 [Batch, Seq_Len, Width]
        x = self.ln_final(x)

        # 提取EOT Token特征作为全句表示
        # text.argmax(dim=-1) 可定位到每条数据中EOT Token的索引位置
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        """前向传播：计算图文特征的余弦相似度矩阵"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 执行L2归一化，使得后续点积等价于余弦相似度
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算缩放后的相似度 logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
```

- **双塔独立编码机制**：`self.visual`负责图像，`self.transformer`负责文本。两者结构独立，仅在最后的共享嵌入空间进行交互
- **视觉骨干动态选择**：代码通过判断`vision_layers`的数据类型来决定实例化ResNet还是VisionTransformer，这种工厂模式设计极大增强了代码的复用性
- **EOT Token聚合策略**：与BERT使用`[CLS]`不同，CLIP使用序列结束标记`EOT`的输出特征作为整个文本序列的语义表示，代码利用`argmax`快速定位该Token
- **可学习的温度参数**：`logit_scale`是一个标量参数，在训练中自动调整，用于缩放点积结果。较小的温度系数会使Softmax分布更加尖锐，从而强化正负样本的区分度

### 4.2 Vision Transformer图像编码器(model.py)

这是基于ViT的图像特征提取器，核心逻辑是将图像分块并序列化，完全摒弃了传统的CNN归纳偏置。

```python
class VisionTransformer(nn.Module):
    """ViT图像编码器：图像分块与Transformer编码"""
    
    def __init__(self, input_resolution: int, patch_size: int, width: int, 
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        # Patch Embedding：利用大核卷积实现图像分块与线性投影
        # 输入 [B, 3, H, W] -> 输出 [B, width, Grid, Grid]
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        # CLS Token：用于聚合全局图像信息的特殊向量
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        
        # 绝对位置编码：包含所有Patch位置与1个CLS位置
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        
        # 特征投影层：映射至最终输出维度
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # 1. 图像分块与投影
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 展平为 [B, width, Grid*Grid]
        x = x.permute(0, 2, 1)                     # 转置为 [B, Seq_Len, width]

        # 2. 拼接 CLS Token
        cls_token = self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)
        x = torch.cat([cls_token, x], dim=1)

        # 3. 叠加位置编码
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        # 4. Transformer 编码
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        # 5. 提取 CLS 输出并投影
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj

        return x
```

- **卷积实现Patch Embedding**：代码巧妙地使用`Conv2d`代替循环切片。设置卷积核大小与步长均等于`patch_size`，即可在一步操作中完成分块和线性映射，大幅提升计算效率
- **CLS Token机制**：这是一个可学习的向量，始终拼接在序列首位。利用Self-Attention的全局感受野，该Token能够自动聚合全图信息，最终作为图像的特征表示
- **绝对位置编码**：CLIP使用可学习的参数作为位置编码，直接叠加在输入序列上。序列总长度为Patch数量加上1个CLS Token

### 4.3 文本分词器(simple_tokenizer.py)

CLIP使用Byte Pair Encoding (BPE)算法，这是一种介于字符级和单词级之间的子词分词方法。

```python
class SimpleTokenizer:
    """BPE分词器：将文本转换为Token ID序列"""
    
    def __init__(self, bpe_path: str = default_bpe()):
        # 初始化字节到Unicode的映射
        self.byte_encoder = bytes_to_unicode()
        
        # 加载BPE合并规则
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')

        # 构建基础词表：由基础字符、合并后的子词及特殊Token组成
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab] # 添加词尾标识
        
        for merge in merges:
            vocab.append(''.join(merge))
            
        # 添加起始与结束特殊Token
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        
        # 构建双向映射字典
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}

    def encode(self, text):
        """执行分词：文本 -> Token IDs"""
        bpe_tokens = []
        # 预处理：清洗文本、转小写并压缩空白
        text = whitespace_clean(basic_clean(text)).lower()
        
        # 正则切分后进行BPE编码
        for token in re.findall(self.pat, text):
            # 将UTF-8字节映射为字符
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 应用BPE合并规则并查表
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            
        return bpe_tokens
```

- **字节级BPE**：算法从UTF-8字节级别开始合并，这确保了模型可以处理任何unicode字符串，甚至包括emoji或未知词汇，彻底解决了OOV (Out of Vocabulary)问题
- **词尾标记**：`</w>`标记用于区分单词内部的子词和单词结尾的子词，它保留了单词的边界信息，使分词过程可逆，即可以从Token序列完美还原原始文本
- **分词流水线**：文本依次经过清洗、正则切分、字节映射、BPE合并、查表转换五个步骤，最终变为机器可读的ID列表

### 4.4 tokenize函数(clip.py)

该函数负责将原始文本转换为模型所需的定长张量格式。

```python
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False):
    """
    预处理函数：文本 -> 模型输入张量
    输出 Shape: [Batch_Size, 77]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    
    # 构造序列：[SOT] + 文本内容 + [EOT]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    # 初始化全0张量
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                # 截断策略：保留前76个Token，强制将最后一位设为EOT
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"文本超长: {texts[i]}")
        
        # 填充张量
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
```

- **固定上下文长度**：CLIP强制将所有文本输入统一为77个Token，这是由位置编码矩阵的维度决定的硬性约束
- **三段式结构**：有效的输入序列必须包含起始符`SOT`和结束符`EOT`。`SOT`标记文本开始，而`EOT`不仅标记结束，更是特征提取的关键锚点
- **截断保护机制**：当文本过长时，简单的截断会丢失`EOT`。代码强制在截断后的序列末尾覆盖写入`EOT`，保证模型能提取到合法的句向量

### 4.5 对比损失函数

这是CLIP训练的核心，即InfoNCE Loss的对称版本实现。

```python
import torch.nn.functional as F

def clip_loss(image_features, text_features, temperature=0.07):
    """
    计算双向对比损失
    image_features: [Batch, Dim]
    text_features:  [Batch, Dim]
    """
    # 特征归一化
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # 计算相似度矩阵并应用温度系数
    # logits shape: [Batch, Batch]
    logits = (image_features @ text_features.T) / temperature

    # 生成Ground Truth标签
    # 对角线位置 (i, i) 为正样本，其余为负样本
    labels = torch.arange(len(logits), device=logits.device)

    # 计算双向交叉熵损失
    loss_i2t = F.cross_entropy(logits, labels)      # 图像 -> 文本
    loss_t2i = F.cross_entropy(logits.T, labels)    # 文本 -> 图像

    # 返回平均损失
    return (loss_i2t + loss_t2i) / 2
```

- **对角线正样本原则**：在一个Batch中，第$i$张图和第$i$个文本是原始配对数据。因此，相似度矩阵的对角线元素即为正样本，其余位置均为负样本
- **InfoNCE简化实现**：使用`cross_entropy`计算对比损失是一个数学上的巧妙转化，它等价于最大化正样本概率，同时最小化该行/列中所有负样本的概率
- **双向对称优化**：代码同时计算了"图找文"`loss_i2t`和"文找图"`loss_t2i`两个方向的损失，这种对称设计迫使图文特征空间实现严格的一一对应和双向对齐
- **大Batch Size的必要性**：负样本的数量等于`Batch Size - 1`。Batch越大，负样本越多，对比学习的任务难度越高，模型学到的特征就越鲁棒

## 5. 参考资料

- https://arxiv.org/abs/2103.00020
- https://github.com/openai/CLIP
