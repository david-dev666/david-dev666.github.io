---
layout: ../../layouts/EssayLayout.astro
title: Jev：不生成文本的决策模型
date: 2026-09-20
description: System One 模型，输入 state 与一组问题，输出带类型和概率的决策
---

## 1. Jev 是什么

- 2026 年 9 月 15 日，旧金山初创公司 TypeSafe AI 发布 Jev。创始人 Diogo Almeida 是前 OpenAI 研究员，参与过 InstructGPT 与 RLHF。
- 官方定义：

  > Jev evaluates typed questions against a state and returns structured results directly. No text generation, no parsing.

- System One 与 LLM 相对：LLM 更像 system two，谨慎思考，速度偏慢；Jev 负责快速判断。
- 命名取自杰文斯悖论——单次决策变便宜之后，调用频率反而越来越高。

## 2. 怎么用

先声明答案的形状，一共三种原语：

| 原语 | 答案形态 | 返回 |
|------|---------|------|
| `choice` | 可枚举选项 | 所选选项、每个选项的概率、单独的 confidence |
| `score` | 描述性等级 | 概率加权后的连续值、档位说明、单独的 confidence |
| `noul` | 是非判断 | 0～1 的数，本身就是概率，没有单独的 confidence |

- `choice` 给的 confidence 直接支持置信度路由：高于阈值就行动，低于阈值就回退到更大的模型。
- `score` 就是打分：你给一串有顺序的档位，比如「不影响使用 / 影响部分功能 / 完全不可用」，Jev 输出一个分数。分数取的是各档位概率的加权平均，而不是档位序号，所以可以落在两档之间，不必取整值。
- `noul` 只回答是或否。0.5 表示两个方向势均力敌，不是「程度中等」；要测程度用 `score`。

问题并行运行，Jev 只摄取一次 state，同时评估请求中的每个问题。

它只负责判断：TypeSafe 的指导是把控制流留在你的代码里，让 Jev 做**原子化、结构化的决策**，而不是一个庞大的判断。

## 3. 相比 LLM，Jev 适合做什么

- **浏览器与计算机使用**。Browser Use 和 TypeSafe 合作的 Jev Ultrafast 就是范例：不截图、不上视觉模型，直接读 DOM，每一步做一次快速判断。
- **编码智能体的上下文压缩**。把「这段上下文留还是删」拆成逐个判断，而不是按时间一刀切。
- **智能体护栏**。在每条命令或工具调用前加一道判断，先询问，拦截危险的那些。校准概率加上自己代码里的阈值，优于需要解读的文本答案。
- **模型路由**。在便宜模型和贵模型之间加一层调度，先用 Jev 分类请求，再决定交给谁。
- **批量分类**。state 固定，一个请求里问许多狭窄的问题，然后读概率。这是最可能在生产里活下来的模式。

共同点：这些环节原本要么写成正则和硬编码，要么顺手调一次 LLM，缺的正是一个便宜、够快、能直接进代码的判断层。TypeSafe 自己的说法是 **smart if-statement**。

## 4. Jev 真的比低成本 LLM 更便宜吗

按输入 token 算并不是。Jev 每 1M 输入 token 收费 \$0.042。APIMaster 上，gpt-5.6-luna 促销价低至 \$0.022，约为 Jev 输入费率的一半；glm-5.3-flash 为 \$0.105，deepseek-flash 为 \$0.15。最便宜的托管生成模型，输入价格已经低于 Jev。

Jev 的成本论点建立在另外三件事上：

- **输出免费**。定价只按输入 token 计，输出不计费。生成模型每次判断都要输出文本，而输出定价通常是输入的数倍。
- **延迟**。官方公布的端到端响应为 70ms 到 500ms，与前沿模型的对比描述为快 40 倍到 200 倍。这个量级才允许同步调用，否则只能进异步队列。
- **边际问题几乎免费**。因为问题在一个 state 上并行展开，第十个问题的成本只是第十次 API 调用的一小部分。

是否划算取决于工作负载。判断次数少、篇幅长、能从推理中获益，带结构化输出的低成本 LLM 是更好的选择；反过来，判断次数多、每次都很短，或者必须在用户察觉停顿之前给出答案，这笔账就不一样了。

## 5. 附：一个最小请求

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
Content-Type: application/json
```

```json
{
  "state": "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
  "model": "jev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated the customer appears",
      "criteria": [
        "Calm, just stating facts",
        "Frustrated but civil",
        "Very angry, strong language"
      ]
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "The message conveys urgency or time-sensitivity"
    }
  }
}
```

返回，取自官方文档：

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "confidence": 0.78,
      "probabilities": { "technical": 0.85, "sales": 0.0, "billing": 0.15 }
    },
    "frustration": {
      "type": "score",
      "score": 1.0,
      "confidence": 1.0,
      "legend": {
        "0": "Calm, just stating facts",
        "1": "Frustrated but civil",
        "2": "Very angry, strong language"
      },
      "probabilities": { "0": 0.0, "1": 1.0, "2": 0.0 }
    },
    "is_urgent": {
      "type": "noul",
      "noul": 1.0
    }
  },
  "usage": { "input_tokens": 392, "output_tokens": 65 }
}
```

几个要点：

- `state` 是所有问题共享的上下文，只放真正需要的信息，多了会拉低准确率。
- 答案被约束在你给的选项或档位内，不会超出范围。
