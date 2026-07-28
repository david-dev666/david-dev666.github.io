---
layout: ../../layouts/ProjectLayout.astro
title: MapTrace
description: 基于合成数据的地图路径追踪能力训练
tags: ["VLM", "Spatial Reasoning", "Data Synthesis"]
---

# MapTrace: Scalable Data Generation for Route Tracing on Maps

<p class="related-content" style="margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
<strong>论文信息</strong>：<a href="https://arxiv.org/abs/2512.19609" style="color: #2563eb;">arXiv:2512.19609</a> | Google | Dec 2025
</p>

本文针对多模态大语言模型在空间推理能力上的短板，提出了一套合成地图数据生产流水线。由于获取精确真实的地图数据成本极高且受权限限制，作者设计了全自动的数据生成方案。

## 1. 核心问题与挑战

当前的多模态大语言模型虽然在语义理解上表现出色，但在精细空间推理方面仍面临巨大挑战：

| 挑战 | 说明 |
|:---|:---|
| **数据稀缺** | 获取大规模、精确的地图路径标注成本极高 |
| **权限限制** | 商业地图数据往往受限于许可协议 |
| **空间感知缺失** | 预训练模型往往无法遵守基本的路径约束，如避开建筑物或保持在可通行区域内 |

## 2. 核心算法：合成地图数据流水线

MapTrace的核心贡献在于建立了一个全自动地图路径数据生成流水线：

![MapTrace Pipeline](/images/maptrace.png)

上图展示了完整的数据生成流程。Description Generator生成地图文本描述，Image Generator渲染为风格化地图。随后通过K-Means聚类提取多组颜色掩码，Mask Critic筛选出有效的可通行区域。合并后的掩码被转换为Pixel Graph用于Dijkstra寻路，最终由Path Critic审核路径质量。

该系统共4次使用模型，均通过Prompting方法调优，无需额外训练：

| 阶段 | 模型 | 功能 |
|:---|:---|:---|
| 地图描述生成 | Gemini-2.5-Pro | 生成多样化的地图文本描述 |
| 图像渲染 | Imagen 4.0 | 将文本描述渲染为风格化地图 |
| Mask审核 | Gemini-2.5-Pro | 审核可通行区域掩码 |
| 路径审核 | Gemini-2.5-Pro | 审核生成的路径质量 |

### 2.1 地图图像生成

利用LLM生成多样的地图描述，涵盖动物园、博物馆、购物中心等12类场景，随后使用文生图模型Imagen-4.0渲染成具有特定风格、包含简化图标和设施的2D俯视图。

与Nano Banana Pro相比，Imagen 4侧重于纯粹的图像生成，其艺术创作、超写实能力较强，而精准控制和图片编辑能力较弱。

### 2.2 路径掩码提取

该阶段旨在从彩色图像中识别出可通行区域。

**K-Means聚类算法**：将RGB地图图像中的每个像素视为一个3维向量，利用K-Means算法将像素划分为k个颜色簇。每个簇代表一类区域，作为候选的可通行区域。

**Mask Critic**：由Gemini-2.5-Pro担任，是保证数据质量的关键环节。工作流程如下：

1. **双输入对比**：模型同时接收原始地图图像和某一颜色簇的二值掩码
2. **多维度分析**：
   - 成分分析：估计掩码的组成成分，如70%人行道、25%车行道、5%草地
   - 错误识别：列出被错误包含的非目标区域
3. **最终裁定**：
   - Good：正确区域占比 >60%
   - Fair：目标与非目标区域混合
   - Poor：正确区域占比 <40%
4. **合并结果**：只有被判定为有效的掩码才会被合并，形成最终的通行掩码M

作者直接利用Gemini-2.5-Pro的零样本理解能力，无需重新训练。经人工抽检，Mask Critic的准确率达到83%，误报率仅为9%，证明了利用VLM进行自动化数据筛选的可行性。

### 2.3 像素图构建

这一步将可通行掩码转换为支持算法寻路的图结构$G=(V, E, w)$。

**节点与边**：掩码被划分为$b \times b$的块，包含通行像素的块即为节点。中心点距离在$\delta_{max}$以内的节点相连形成边。

**带惩罚的权重算法**：边的权重不仅考虑几何距离，还引入节点密度惩罚$\lambda$：

$$
w(v_i, v_j) = \|c_i - c_j\|_2 \cdot (1 + \lambda((1 - \rho_i) + (1 - \rho_j)))
$$

其中$\rho$代表像素密度。该算法确保路径更倾向于通过高密度通行区，而非贴近障碍物边缘。

### 2.4 路径生成与审核

这一步在建好的图中进行寻路计算和质量检验。

**核心算法**：

| 算法 | 作用 |
|:---|:---|
| **Dijkstra** | 在像素图中计算起止点之间的全局最短路径 |
| **RDP** | 对生成的坐标序列进行精简，在减少数据量的同时保持路径几何形状 |

**Path Critic**：同样由Gemini-2.5-Pro担任，负责终审：

- **边界检查**：验证路径是否完全位于地图地理区域内，严禁触碰标题、图例等非地图元素
- **通行性检查**：严格审查路径是否穿过建筑、不可通行陆地或水域
- **审核结果**：给出GOOD或BAD的裁定

经人工抽检，Path Critic的准确率为76%。

## 3. Mask Critic Prompt设计分析

原文Prompt：

```
Your goal is to evaluate a segmentation mask's accuracy in identifying paths based on a source image.

You will be given two images: 
* `[Image 1: Source]` is the original map image. 
* `[Image 2: Mask]` is the segmentation mask where path/target areas should be marked in white.

**Evaluation Criteria:**
* Target Areas (Considered 'Correct'): Any type of map paths: paved sidewalks, marked crosswalks, pedestrian-only paths, public plazas, indoor walkways etc.
* Non-Target Areas (Considered 'Incorrect'): Vehicle lanes of roads, grass, dirt, buildings, cars, and any other non-pedestrian surface.
* If the majority of the image is white, it is likely of poor quality.

**Analysis Task:**
Instead of a simple Yes/No, please provide the following structured analysis of `[Image 2: Mask]`:

1. **Composition Analysis**: Estimate the composition of the total white area in the mask. Break it down by the type of ground it covers, with approximate percentages. 
   * Example: The mask covers approximately 70% sidewalks, 25% vehicle roadway, and 5% grass.
2. **Major Errors**: List the most significant `Non-Target Areas` that were incorrectly included in the mask.
3. **Final Judgment**: Based on your analysis, provide a final one-word judgment on the mask's precision: `Good`, `Fair`, or `Poor`. 
   * `Good`: The mask is almost entirely composed of Target Areas (>60%). 
   * `Fair`: The mask contains a mix of Target and Non-Target areas.
   * `Poor`: The mask is predominantly composed of Non-Target areas (<40% correct).
```

**Prompt设计要点**：

| 技巧 | 说明 |
|:---|:---|
| **多模态对齐校验** | 采用原图+结果同步输入模式，利用MLLM的视觉对齐能力，强制模型在原图语境下执行空间匹配校验 |
| **显式边界定义** | 详细列举目标与非目标区域，极大降低了模型在处理复杂地图时的语义歧义 |
| **思维链约束** | 模型被要求先进行成分分析和错误列表的推导，最后才给出评级，这种先推理后决策的结构能显著提高最终判断的准确性 |
| **量化分级标准** | 通过设定具体的百分比阈值，将模糊的定性判断转化为标准化的定量判断 |
| **启发式过滤** | 加入对全白/全黑图像的负面提醒，防止模型在遇到聚类算法崩坏产生的无效数据时给出错误评价 |

## 4. 评价指标与实验结果

**评价指标**：

| 指标 | 说明 |
|:---|:---|
| **成功率SR** | 衡量模型生成可解析且有效坐标序列的能力。若模型未输出坐标、格式无法解析或NDTW无定义，则判定为失败 |
| **NDTW** | 归一化动态时间规整，评估预测路径与真值之间的几何相似度，能灵活对齐长度不等、分辨率不同的坐标序列，数值越低代表路径对齐精度越高 |

**实验结果**：在MapTrace数据集上微调后，Gemini-2.5-Flash和Gemma3-27B在开源数据集MapBench上的寻路成功率提升，路径误差显著降低。

**启示**：精细空间推理能力可以通过高质量的合成监督数据显式教授给模型。MapTrace通过聚类将地图照片转换为图结构，然后用几何算法生成路径，生成一批高质量地图寻路数据，以教会VLM模型在复杂地图中寻找最优路径。

## 5. NDTW算法

NDTW解决了轨迹序列比对中的非线性对齐问题，流程为：

1. **空间归一化**：将序列中的所有坐标点除以地图宽高，映射至[0, 1]区间，消除地图分辨率差异
2. **动态规划寻优**：构建累积代价矩阵，通过非线性规整计算两个序列点对之间的最小累积欧几里得距离，解决采样点疏密不同导致的时间轴错位问题
3. **路径长度归一化**：将最小累积距离除以最优对齐路径的实际步数，将总距离误差转换为平均步长误差

**Python实现**：

```python
import numpy as np
from scipy.spatial.distance import cdist

def calculate_ndtw(prediction, ground_truth, map_width, map_height):
    """
    计算归一化动态时间规整NDTW
    
    Args:
        prediction: 预测坐标序列 [(x1, y1), ...]
        ground_truth: 真实坐标序列 [(x1, y1), ...]
        map_width, map_height: 地图尺寸
    Returns:
        归一化DTW距离
    """
    if not prediction or not ground_truth:
        return float('inf')

    # 1. 坐标归一化至[0, 1]
    preds = np.array(prediction, dtype=float) / [map_width, map_height]
    gt = np.array(ground_truth, dtype=float) / [map_width, map_height]

    # 2. 计算欧式距离矩阵
    dist_matrix = cdist(preds, gt, metric='euclidean')

    # 3. 动态规划计算累积距离代价
    rows, cols = dist_matrix.shape
    cost = np.full((rows + 1, cols + 1), fill_value=np.inf)
    cost[0, 0] = 0

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            choices = [cost[i-1, j], cost[i, j-1], cost[i-1, j-1]]
            cost[i, j] = dist_matrix[i-1, j-1] + min(choices)

    # 4. 按路径长度归一化
    dtw_distance = cost[rows, cols]
    normalized_distance = dtw_distance / max(rows, cols)
    
    return normalized_distance
```

## 6. MapBench数据集

MapBench是由Xing等人于2025年推出的多模态大模型地图寻路推理基准测试数据集。

<p class="related-content" style="margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
<strong>论文信息</strong>：<a href="https://arxiv.org/abs/2503.14607" style="color: #2563eb;">arXiv:2503.14607</a>
</p>

| 特性 | 说明 |
|:---|:---|
| **数据规模** | 96张真实商业导视图，涵盖商场、动物园、博物馆、校园等多样化场景，共1,573条人工标注路径查询 |
| **任务目标** | 要求模型根据文本指令或起止点，在真实地图图像上识别地标序列或生成导航路径 |

本文作者基于MapBench数据集验证模型从合成数据中学到的空间拓扑知识能否泛化至复杂的真实物理世界场景。
