// 选择题题库
// 题型说明：
//   type: "single" 单选（只有一个正确选项）
//   type: "multiple" 多选（有多个正确选项）
//   options: 选项列表，{ text: 选项内容, correct: 是否为正确选项 }
//   explanation: 答案解析，答完后展示
//   tags: 题目所属知识点标签（用于展示）
//
// 添加新题：直接在 questions 数组末尾追加一个对象即可。

export interface QuizOption {
  text: string;
  correct: boolean;
}

export interface QuizQuestion {
  id: number;
  type: "single" | "multiple";
  question: string;
  options: QuizOption[];
  explanation: string;
  tags: string[];
}

export const questions: QuizQuestion[] = [
  {
    id: 1,
    type: "single",
    question: "AdamW 相比 Adam 的核心改动是什么？",
    options: [
      { text: "引入动量项平滑梯度", correct: false },
      { text: "把权重衰减从梯度中解耦，直接作用在参数上", correct: true },
      { text: "用滑动平均替代梯度的单调累加", correct: false },
      { text: "为每个参数分配独立的学习率", correct: false },
    ],
    explanation:
      "Adam 把权重衰减（weight decay，防过拟合的手段）混进梯度一起算，导致它被自适应缩放干扰——梯度越大的参数，权重衰减反而越弱。AdamW 把权重衰减单独从参数上扣，与自适应机制解耦，训练更稳定。",
    tags: ["Optimizer", "Training"],
  },
  {
    id: 2,
    type: "single",
    question: "在峡谷状的损失曲面上，Momentum 相比 SGD 的主要收益是什么？",
    options: [
      { text: "每步计算量更小", correct: false },
      { text: "抑制左右震荡、累积沿谷底的稳定分量", correct: true },
      { text: "为每个参数设置不同的学习率", correct: false },
      { text: "避免了学习率的单调衰减", correct: false },
    ],
    explanation:
      "峡谷地形下 SGD 的梯度会在两侧谷壁间来回震荡。Momentum 引入「惯性」——用历史梯度的加权平均代替当前梯度，让左右震荡的分量互相抵消、沿谷底的分量不断累积，从而直冲谷底。",
    tags: ["Optimizer"],
  },
  {
    id: 3,
    type: "single",
    question: "bf16 与 fp16 同为 16 位浮点，二者的本质区别是什么？",
    options: [
      { text: "bf16 指数量更多、范围大但精度差；fp16 尾数更多、范围小但精度好", correct: true },
      { text: "bf16 是定点数，fp16 是浮点数", correct: false },
      { text: "bf16 只能用于训练，fp16 只能用于推理", correct: false },
      { text: "二者数值范围与精度完全相同，仅命名不同", correct: false },
    ],
    explanation:
      "两者都是 16 位浮点，区别在位数分配：bf16 是 8 指数位 + 7 尾数位（范围大、精度差，误差约 0.78%）；fp16 是 5 指数位 + 10 尾数位（范围小、精度好，误差约 0.098%）。一句话：bf16 用精度换范围，fp16 用范围换精度。",
    tags: ["Quantization", "VLM"],
  },
  {
    id: 4,
    type: "multiple",
    question: "关于多模态模型的量化，下列哪些做法是正确的？（多选）",
    options: [
      { text: "视觉塔保高精度，只量化 LLM 部分", correct: true },
      { text: "量化 Projector 模块以最大化显存收益", correct: false },
      { text: "QK 和 softmax 始终保持 bf16 / fp32 精度", correct: true },
      { text: "校准集必须包含图文混合样本", correct: true },
    ],
    explanation:
      "视觉塔存在大量 outlier channel（个别通道数值飙到平均值的几十倍），量化会压坏小数值，且它占比小、保精度划算；Projector（连接图像与语言的中间层）是信息瓶颈，误差会被 LLM 逐层放大，不能量化；QK 和 softmax 对量化极度敏感，必须保高精度；校准集（量化前统计分布的样本集）若只有文本，图像 token 的分布模型没见过，会出错。",
    tags: ["Quantization", "VLM", "Inference"],
  },
  {
    id: 5,
    type: "single",
    question: "SFT 的学习目标本质上是什么？",
    options: [
      { text: "最大化奖励函数值", correct: false },
      { text: "最小化与标注答案的 token 级差异（模仿学习）", correct: true },
      { text: "通过探索扩展解空间", correct: false },
      { text: "最小化验证集损失", correct: false },
    ],
    explanation:
      "SFT 是模仿学习：让模型逐个 token（最小文本单位）模仿标准答案，评判标准是「和答案有多像」。因此它的天花板就是标注数据的平均水平。RL 则是引入评判者打分，让模型探索标注之外的更好解法。",
    tags: ["LLM", "Post-Training"],
  },
  {
    id: 6,
    type: "multiple",
    question: "下列哪些现象说明 SFT 已接近其能力边界，可考虑切换到 RL？（多选）",
    options: [
      { text: "验证集 loss 持续下降，但 benchmark 分数停滞", correct: true },
      { text: "模型已能可靠遵循指令格式", correct: true },
      { text: "Pass@k 显著高于 Pass@1", correct: true },
      { text: "训练 loss 发散", correct: false },
    ],
    explanation:
      "Pass@k 指同一道题让模型生成 k 次、取最好一次算通过。Pass@k 远高于 Pass@1，说明模型其实「会」这道题、只是发挥不稳定，这个差距正是 RL 能挖掘的空间——把「偶尔对」变成「稳定对」。另外两条：loss 还在降但真实分数停滞，说明在背数据规律而非提升能力；格式能力已稳定，说明 SFT 的基础任务已完成，边际收益很低。",
    tags: ["LLM", "RLHF", "Post-Training"],
  },
  {
    id: 7,
    type: "single",
    question: "在一次 VLM 全模型转 bf16 推理导致召回率与准确率双双下降约 10% 的案例中，最根本的错误是什么？",
    options: [
      { text: "bf16 精度天然不足，不该用于任何推理", correct: false },
      { text: "同时改了多个变量（dtype、KV cache、token 上限等）却只归因给 bf16", correct: true },
      { text: "没有使用 int8 量化", correct: false },
      { text: "视觉塔权重本身有问题", correct: false },
    ],
    explanation:
      "只设了 dtype=bf16，但推理框架（如 vLLM）背后会连带改动视觉塔、KV cache（缓存历史 token 的键值矩阵）、视觉 token 上限、预处理等多重变量。同时变多个因素却只归因给 bf16，是实验方法错误。正确做法是单变量实验：一次只改一个因素，其余冻结，逐个隔离。",
    tags: ["Quantization", "Methodology"],
  },
  {
    id: 8,
    type: "single",
    question: "FlashAttention V2 相对 V1 的关键改进是什么？",
    options: [
      { text: "使用更小的分块大小", correct: false },
      { text: "调换循环顺序为外层遍历 Q 块、内层遍历 K/V 块", correct: true },
      { text: "完全移除 Softmax", correct: false },
      { text: "将注意力矩阵存回 HBM", correct: false },
    ],
    explanation:
      "V1 外层遍历 K/V 块、内层遍历 Q 块，导致每个 Q 块被反复从 HBM（GPU 慢速主内存）加载。V2 调换顺序后，每个 Q 块只加载一次、在片上缓存一次性算完，大幅减少慢速内存访问，吞吐量提升约 2 倍。",
    tags: ["Attention", "Inference"],
  },
];
