---
layout: ../../layouts/ProjectLayout.astro
title: mmap
description: 操作系统层面的内存映射技术与大模型训练
tags: ["OS", "Memory", "LLM Training"]
---

# 内存映射mmap：大模型海量数据的克星

内存映射是操作系统的核心技术，它将磁盘文件直接映射到进程的虚拟地址空间。在LLM时代，mmap是处理TB级训练数据和百GB级模型权重的关键技术，它使得16GB内存的普通机器也能处理远超其容量的数据集。

## 1. 核心思想

传统文件读写效率低下的根源在于数据冗余拷贝。数据必须先从磁盘进入内核缓冲区，再从内核缓冲区拷贝到用户进程内存。这种方式不仅消耗CPU周期，还要求预先分配足额内存。

mmap的核心思想是**虚拟地址映射**。操作系统将文件的磁盘地址直接映射到进程的虚拟内存空间，建立一种**数据已在内存中**的假象，实际数据并未立即加载。

| 对比维度 | 传统I/O | mmap |
|----------|---------|------|
| **数据路径** | 磁盘 → 内核缓冲区 → 用户内存 | 磁盘 → 页面缓存 → 虚拟地址 |
| **拷贝次数** | 2次 | 1次即零拷贝 |
| **内存占用** | 必须预分配完整空间 | 按需加载访问的页面 |
| **访问方式** | 需要seek定位 | 类似数组下标直接访问 |

所谓的**零拷贝**指的是数据不需要在内核空间和用户空间之间来回复制。数据直接通过DMA传输到内核的Page Cache，用户进程直接通过指针访问这块物理内存。

### 页面缓存Page Cache

Page Cache是内核维护的一块内存区域，用于缓存最近访问的磁盘数据。当多个进程读取同一文件时，它们共享Page Cache中的同一份物理副本。mmap充分利用了这一机制：

1. 程序访问映射的虚拟地址
2. 若数据不在内存，触发缺页中断
3. 操作系统直接将磁盘数据读入Page Cache
4. 操作系统修改页表，将用户虚拟地址指向Page Cache中对应的物理页面

## 2. 数学视角

从抽象代数的角度看，mmap建立了一个从虚拟地址空间到文件内容的双射函数：

$$
f: [base, base + len) \rightarrow [offset, offset + len)
$$

其中$base$为映射在虚拟内存中的起始地址，$offset$为文件内的起始偏移量，$len$为映射长度。

对于任意增量$\Delta$，访问虚拟地址$base + \Delta$在逻辑上等价于直接读取文件位置$offset + \Delta$的数据。这种线性映射保证了$O(1)$的访问时间复杂度。

## 3. 代码实现

我们将重点放在NumPy的基础用法以及PyTorch训练数据的构建上。

### 3.1 NumPy memmap基础

NumPy提供了非常便捷的接口`np.memmap`，它允许我们将磁盘上的二进制文件直接视为一个大型数组。

```python
import numpy as np

# 创建一个映射到磁盘文件的数组对象
# mode='r' 表示只读模式，保护磁盘数据不被修改
# dtype 指定数据类型，决定了读取时的字节步长
data = np.memmap('train.bin', dtype=np.uint16, mode='r')

# 此时数据并未加载到内存
# data 对象仅存储了文件的元数据和映射关系
print(f"数组形状: {data.shape}")

# 关键点：按需加载
# 只有执行切片读取时，OS 才会触发缺页中断
# 仅读取这 10 个元素对应的物理页面到内存
batch = data[1000:1010]
```

### 3.2 大模型训练中的Pipeline实现

在实际训练中，我们需要自定义Dataset来配合DataLoader使用。这是nanoGPT等主流项目的标准写法。

```python
import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    def __init__(self, data_path, block_size):
        # 初始化阶段仅建立映射，不消耗物理内存
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
    
    def __len__(self):
        # 数据集总长度需减去一个 block_size 
        # 保证最后一个样本有足够的上下文
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # 核心逻辑：
        # 1. 像访问数组一样切片
        # 2. 操作系统自动处理缺页，从磁盘加载该块数据
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # 将 numpy 数组转换为 torch tensor
        # 注意：这里 .astype(np.int64) 会产生一次内存拷贝
        # 这是为了满足 PyTorch Embedding 层对 LongTensor 的要求
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y
```

`__getitem__`是mmap真正发挥作用的地方。无论数据集是10GB还是10TB，每次调用该方法时，内存中仅会存在`block_size`大小的数据片段。

### 3.3 模型权重的零拷贝加载

在加载70B甚至更大参数的模型时，传统`torch.load`会导致CPU内存溢出。使用mmap配合safetensors格式可以解决此问题。

```python
import torch
from safetensors.torch import load_file

# 使用 safetensors 进行零拷贝加载
# 框架底层使用 mmap 将权重文件映射
# 只有当权重被移动到 GPU 时，才真正占用物理内存
state_dict = load_file("model.safetensors", device="cpu")

# 逐层加载到 GPU
for key, tensor in state_dict.items():
    # 数据流向：磁盘 -> Page Cache -> GPU 显存
    # 避免了在 CPU RAM 中创建完整的模型副本
    gpu_tensor = tensor.to("cuda")
```

## 4. 底层机制与性能分析

理解底层机制有助于我们规避性能陷阱。

### 4.1 缺页中断Page Fault

当程序访问一个尚未加载到物理内存的虚拟地址时，CPU会触发异常，控制权移交内核。

1. **MMU查找**：内存管理单元发现页表项无效
2. **内核接管**：内核检查虚拟地址合法性
3. **磁盘I/O**：内核发起DMA请求，将4KB页面从磁盘读入Page Cache
4. **页表更新**：建立虚拟页到物理页的映射
5. **恢复执行**：CPU重新执行刚才的指令

### 4.2 性能权衡

| 场景 | 表现 | 原因分析 |
|------|------|----------|
| **顺序读取** | 极优 | 操作系统会触发预读Read-ahead机制，提前加载后续页面 |
| **随机小块读取** | 较差 | 频繁触发缺页中断，导致用户态与内核态频繁切换 |
| **多进程共享** | 极优 | 多个DataLoader worker进程共享同一份物理内存，节省资源 |

在大模型训练中，尽量保证数据存储的连续性。如果是完全随机访问SSD上的小文件，mmap的优势会下降，此时应考虑预取策略。

## 5. 总结

mmap并非仅仅是一个API，它代表了一种**按需分配、惰性加载**的系统设计哲学。

对于AI工程师而言，掌握mmap意味着：

1. 在有限的硬件资源上训练无限的数据集
2. 极大地加快模型加载和推理启动速度
3. 深入理解PyTorch等框架底层的IO优化原理

## 6. 相关词条

- **Virtual Memory**：虚拟内存
- **Page Fault**：缺页中断
- **DMA**：直接内存访问
- **Zero Copy**：零拷贝
- **Safetensors**：HuggingFace推出的安全且高效的张量存储格式
