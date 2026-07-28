---
layout: ../../layouts/ProjectLayout.astro
title: Rerank重排序
description: RAG系统中的精排策略与LLM实现方案
tags: ["RAG", "LLM", "Information Retrieval"]
---

# Rerank: 检索增强生成中的精排机制

在RAG（Retrieval-Augmented Generation）系统中，Rerank是连接粗召回与最终生成的关键环节。它对初步检索的候选文档进行精细化排序，确保最相关的内容进入LLM的上下文窗口。

## 1. 为什么需要Rerank

典型的RAG流程为：`Query → Retriever → Reranker → Generator`

**Retriever的局限性**

向量检索（如FAISS、Milvus）依赖Embedding的余弦相似度进行召回，存在以下问题：

| 问题 | 说明 |
|:---|:---|
| **语义漂移** | Embedding可能将表面相似但语义无关的文档排在前面 |
| **粒度粗糙** | 向量相似度难以捕捉细粒度的语义匹配 |
| **召回量大** | 通常返回Top-K（K=50~100），直接输入LLM会超出上下文限制 |

**Reranker的价值**

Reranker使用更强的模型（如Cross-Encoder或LLM）对Query-Document对进行逐一评分，实现精细化排序：

- 将召回的100篇文档精排到Top-5~10
- 过滤语义不匹配的噪声文档
- 提升最终生成质量

## 2. 基于LLM的Rerank方法

根据《Large Language Models for Information Retrieval: A Survey》，LLM做Rerank主要有三类方法：

### 2.1 微调LLM做重排

将Rerank建模为序列分类或打分任务，对LLM进行有监督微调。

**核心思路**：输入`[Query, Document]`对，输出相关性分数。

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class FinetunedReranker:
    """基于微调的Cross-Encoder重排器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def score(self, query: str, documents: list[str]) -> list[float]:
        """计算Query与每个Document的相关性分数"""
        scores = []
        for doc in documents:
            # 将Query和Document拼接为一个序列
            inputs = self.tokenizer(
                query, doc,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            with torch.no_grad():
                # 输出logits，取相关性得分
                logits = self.model(**inputs).logits
                score = logits.squeeze().item()
            scores.append(score)
        return scores
    
    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[str]:
        """重排并返回Top-K文档"""
        scores = self.score(query, documents)
        # 按分数降序排列
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [documents[i] for i in ranked_indices[:top_k]]
```

**关键点解析**：

- **Cross-Encoder架构**：Query和Document同时输入，通过Attention进行深度交互，比Bi-Encoder（分别编码再算相似度）表达能力更强
- **代表模型**：BGE-Reranker、Cohere Rerank、Jina Reranker
- **适用场景**：对延迟要求不苛刻、追求高精度的场景

### 2.2 Prompt驱动的重排

利用LLM的指令遵循能力，通过精心设计的Prompt让模型判断文档相关性。

**方法一：Pointwise打分**

逐一评估每个文档的相关性：

```python
from openai import OpenAI

def pointwise_rerank(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """Pointwise方式：对每个文档单独打分"""
    client = OpenAI()
    
    prompt_template = """请评估以下文档与查询的相关性，返回0-10的分数。

查询：{query}
文档：{document}

只返回数字分数，不要其他内容。"""
    
    scores = []
    for doc in documents:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_template.format(query=query, document=doc)}],
            temperature=0
        )
        try:
            score = float(response.choices[0].message.content.strip())
        except ValueError:
            score = 0
        scores.append(score)
    
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [documents[i] for i in ranked_indices[:top_k]]
```

**方法二：Listwise排序**

一次性输入多个文档，让LLM直接输出排序结果：

```python
def listwise_rerank(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """Listwise方式：一次性对所有文档排序"""
    client = OpenAI()
    
    # 构建文档列表
    doc_list = "\n".join([f"[{i+1}] {doc[:200]}..." for i, doc in enumerate(documents)])
    
    prompt = f"""请根据与查询的相关性，对以下文档进行排序。

查询：{query}

文档列表：
{doc_list}

请直接返回排序后的文档编号，用逗号分隔，最相关的排在最前面。
例如：3,1,5,2,4"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    # 解析返回的排序结果
    try:
        ranking = [int(x.strip()) - 1 for x in response.choices[0].message.content.split(",")]
        return [documents[i] for i in ranking[:top_k] if i < len(documents)]
    except:
        return documents[:top_k]
```

**方法三：Pairwise比较**

两两比较文档的相对相关性：

```python
def pairwise_compare(query: str, doc_a: str, doc_b: str) -> str:
    """比较两个文档哪个更相关"""
    client = OpenAI()
    
    prompt = f"""给定查询和两个文档，判断哪个文档与查询更相关。

查询：{query}
文档A：{doc_a[:300]}
文档B：{doc_b[:300]}

只返回"A"或"B"。"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
```

| 方法 | 调用次数 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **Pointwise** | $O(N)$ | 实现简单，可并行 | 分数校准困难 |
| **Listwise** | $O(1)$ | 调用少，考虑全局 | 受上下文长度限制 |
| **Pairwise** | $O(N^2)$ | 比较精确 | 调用次数多，成本高 |

### 2.3 LLM数据增强

利用LLM生成训练数据，增强专用Reranker模型的能力。

**核心流程**：

```
原始Query → LLM生成相关/不相关文档 → 训练专用Reranker
```

```python
def generate_training_pairs(queries: list[str], num_negatives: int = 3) -> list[dict]:
    """使用LLM生成Reranker训练数据"""
    client = OpenAI()
    training_data = []
    
    for query in queries:
        # 生成正例
        pos_prompt = f"为以下查询生成一段高度相关的文档段落：\n查询：{query}"
        pos_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": pos_prompt}]
        )
        positive_doc = pos_response.choices[0].message.content
        
        # 生成负例（表面相关但实际不相关）
        neg_prompt = f"""为以下查询生成{num_negatives}段看起来相关但实际不回答问题的文档。
查询：{query}
用---分隔每段。"""
        neg_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": neg_prompt}]
        )
        negative_docs = neg_response.choices[0].message.content.split("---")
        
        training_data.append({
            "query": query,
            "positive": positive_doc,
            "negatives": [d.strip() for d in negative_docs if d.strip()]
        })
    
    return training_data
```

**应用价值**：

- 解决人工标注成本高的问题
- 生成hard negatives（难负例），提升模型区分能力
- 扩充小语种或专业领域的训练数据

## 3. 专用Reranker vs LLM Reranker

| 维度 | 专用Reranker | LLM Reranker |
|:---|:---|:---|
| **延迟** | 快（~10ms/doc） | 慢（~100ms/doc） |
| **成本** | 低（本地推理） | 高（API调用） |
| **精度** | 在特定领域可能更优 | 上下文丰富时表现出色 |
| **部署** | 需要GPU资源 | 调用API即可 |
| **代表** | BGE-Reranker, Cohere | GPT-4, Claude |

**工程建议**：

- **追求速度**：使用轻量级Cross-Encoder（如bge-reranker-base）
- **追求精度**：Prompt驱动的LLM Rerank或级联方案
- **平衡方案**：先用专用Reranker筛选Top-20，再用LLM精排到Top-5

## 4. 完整RAG流程示例

```python
class RAGPipeline:
    """带Rerank的RAG完整流程"""
    
    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever    # 向量检索器
        self.reranker = reranker      # 重排器
        self.generator = generator    # LLM生成器
    
    def __call__(self, query: str, top_k_retrieve: int = 50, top_k_rerank: int = 5) -> str:
        # 1. 粗召回：向量检索Top-50
        candidates = self.retriever.search(query, top_k=top_k_retrieve)
        
        # 2. 精排：Rerank到Top-5
        reranked_docs = self.reranker.rerank(query, candidates, top_k=top_k_rerank)
        
        # 3. 生成：将精排结果注入Prompt
        context = "\n\n".join(reranked_docs)
        prompt = f"""基于以下参考资料回答问题。

参考资料：
{context}

问题：{query}
回答："""
        
        return self.generator.generate(prompt)
```

## 5. 总结

| 要点 | 说明 |
|:---|:---|
| **定位** | Rerank是RAG中粗召回到精生成的桥梁 |
| **LLM方法** | 微调（高精度）、Prompt驱动（灵活）、数据增强（扩展） |
| **权衡** | LLM Rerank精度高但慢，专用Reranker快但需训练 |
| **实践** | 级联方案（专用Reranker + LLM）兼顾效率与精度 |

## 6. 相关词条

- [Attention机制](/glossary/attention)
- RAG
- Embedding
- Cross-Encoder
