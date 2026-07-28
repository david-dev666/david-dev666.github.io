---
layout: ../../layouts/ProjectLayout.astro
title: 意图识别
description: 基于 BERT 的意图识别
tags: ["NLU", "BERT", "Classification"]
---

# 意图识别

意图识别是 NLU 的核心任务：判定用户输入的目的类别。

基于 BERT 的方案：取 `[CLS]` 向量，接全连接层进行多分类。

## 1. 模型结构

```python
import torch
import torch.nn as nn
from transformers import BertModel

class IntentClassifier(nn.Module):
    def __init__(self, bert_path, num_intents, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # 取 [CLS] 向量作为句子表征
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
```

- **[CLS] 向量**：BERT 在预训练时，`[CLS]` 位置被设计为聚合整句语义，适合做句子级分类
- **Dropout**：防止在小数据集上过拟合，通常设为 0.1
- **输出维度**：`num_intents` 对应意图类别数，如 "查询"、"预订"、"取消" 等

### 1.1 为什么用 [CLS]？

BERT 的 `[CLS]` 经过 12/24 层 Transformer 编码后，已融合了整个序列的上下文信息。相比对所有 token 做 pooling，直接用 `[CLS]` 更简洁且效果相当。

## 2. 数据格式

```json
{"text": "帮我查一下明天北京的天气", "intent": "weather_query"}
{"text": "订一张去上海的机票", "intent": "flight_booking"}
{"text": "取消我的酒店预订", "intent": "hotel_cancel"}
```

**预处理流程**：
- 构建 `intent2id` 映射字典
- 使用 BERT tokenizer 将文本转为 `input_ids`
- 生成 `attention_mask` 标记有效 token

## 3. 训练代码

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def train(model, dataloader, epochs=3, lr=2e-5, max_grad_norm=1.0):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

- **AdamW**：带权重衰减的 Adam，BERT 微调标准优化器
- **Warmup**：学习率从 0 线性增长到目标值，稳定初期训练
- **梯度裁剪**：`clip_grad_norm_` 将梯度 L2 范数限制在 1.0，防止深层 Transformer 梯度爆炸

### 3.1 超参数配置

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| 学习率 | 2e-5 ~ 5e-5 | BERT 微调的标准范围 |
| Batch Size | 16 ~ 32 | 受显存限制 |
| max_grad_norm | 1.0 | 梯度裁剪阈值 |
| Warmup | 10% steps | 学习率预热比例 |
| Epoch | 3 ~ 5 | 通常不需要太多轮次 |

## 4. 推理

```python
def predict(model, tokenizer, text, intent_labels):
    model.eval()
    
    inputs = tokenizer(
        text, 
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        logits = model(
            inputs['input_ids'].to(device),
            inputs['attention_mask'].to(device)
        )
    
    probs = torch.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_id].item()
    
    return intent_labels[pred_id], confidence
```

- **eval 模式**：关闭 Dropout，使用确定性推理
- **no_grad**：禁用梯度计算，节省显存和加速推理
- **softmax**：将 logits 转为概率分布，方便设置置信度阈值

## 相关词条

- [Attention 机制](/glossary/attention)
- [Dropout](/glossary/dropout)
