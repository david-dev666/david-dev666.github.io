// 题库 1
// 题型说明：
//   type: "single" 单选（只有一个正确选项）
//   type: "multiple" 多选（有多个正确选项）
//   options: 选项列表，{ text: 选项内容, correct: 是否为正确选项 }
//   explanation: 答案解析，答完后展示
//   tags: 题目所属知识点标签（用于展示）
//
// 添加新题：直接在 questions 数组末尾追加一个对象即可。

import type { QuizQuestion } from "./types";

// 当前题库名称（页面会显示在每题上方）
export const bankName = "题库 1";

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
  // ==================== AI 基础知识 ====================
  {
    id: 9,
    type: "single",
    question: "机器学习中「过拟合」指的是什么？",
    options: [
      { text: "模型在训练集上表现好，但在新数据上泛化差", correct: true },
      { text: "模型在训练集和新数据上都表现差", correct: false },
      { text: "模型训练速度过快", correct: false },
      { text: "模型参数量太少", correct: false },
    ],
    explanation:
      "过拟合指模型把训练集的噪声和细节也背下来了，导致对新数据（没见过的样本）泛化能力差。应对手段：加正则、早停、更多数据、Dropout 等。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 10,
    type: "single",
    question: "欠拟合（underfitting）最典型的表现是？",
    options: [
      { text: "训练集和验证集误差都偏高", correct: true },
      { text: "训练集误差低、验证集误差高", correct: false },
      { text: "验证集误差低于训练集", correct: false },
      { text: "模型参数量过大", correct: false },
    ],
    explanation:
      "欠拟合指模型能力不足以学到数据规律，训练集和验证集都表现差。常见原因：模型太简单、训练不足、特征不足。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 11,
    type: "single",
    question: "监督学习与无监督学习的核心区别是？",
    options: [
      { text: "监督学习有标签，无监督学习没有", correct: true },
      { text: "监督学习不需要数据", correct: false },
      { text: "无监督学习必须用神经网络", correct: false },
      { text: "二者没有区别", correct: false },
    ],
    explanation:
      "监督学习用「输入-标签」对训练，目标是学输入到输出的映射；无监督学习没有标签，目标是发现数据内在结构，如聚类、降维。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 12,
    type: "single",
    question: "下列哪个是典型的无监督学习任务？",
    options: [
      { text: "K-means 聚类", correct: true },
      { text: "图像分类", correct: false },
      { text: "垃圾邮件过滤", correct: false },
      { text: "房价预测", correct: false },
    ],
    explanation:
      "K-means 在没有标签的情况下把相似样本归为一类，是无监督聚类；其余几个都用标签，属监督学习。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 13,
    type: "single",
    question: "「梯度消失」问题通常发生在？",
    options: [
      { text: "深层网络的反向传播中，梯度趋近于零", correct: true },
      { text: "浅层网络的参数更新", correct: false },
      { text: "数据标准化之后", correct: false },
      { text: "卷积操作时", correct: false },
    ],
    explanation:
      "层数很深时，梯度逐层连乘，值可能趋于 0（消失）或爆炸。消失会让浅层参数几乎不更新。缓解手段：残差连接、ReLU、归一化、梯度裁剪。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 14,
    type: "single",
    question: "ReLU 激活函数的优点不包括？",
    options: [
      { text: "缓解梯度消失", correct: false },
      { text: "计算简单、非线性强", correct: false },
      { text: "输出没有上界，可能数值溢出", correct: true },
      { text: "稀疏激活", correct: false },
    ],
    explanation:
      "ReLU 的优点：正区间梯度恒为 1、缓解梯度消失、计算快、稀疏激活。它的缺点是输出无上界，可能造成数值不稳定，以及负数输入梯度为 0（神经元死亡）。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 15,
    type: "single",
    question: "为什么需要给网络引入激活函数？",
    options: [
      { text: "没有激活函数，多层线性层等价于一层", correct: true },
      { text: "激活函数能减少参数量", correct: false },
      { text: "激活函数能加速训练", correct: false },
      { text: "激活函数用于数据增强", correct: false },
    ],
    explanation:
      "线性层叠加还是线性（等价单层），无法逼近非线性函数。激活函数引入非线性，让网络能学习复杂映射。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 16,
    type: "single",
    question: "Dropout 的作用是？",
    options: [
      { text: "训练时随机丢弃部分神经元，抑制过拟合", correct: true },
      { text: "减少模型推理时的计算量", correct: false },
      { text: "把浮点精度降到更低", correct: false },
      { text: "对输入做归一化", correct: false },
    ],
    explanation:
      "Dropout 在训练时按概率随机让部分神经元失活，迫使网络不依赖单一神经元，起到正则化、抑制过拟合的作用。推理时通常关闭。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 17,
    type: "single",
    question: "学习率（learning rate）设置过大可能导致？",
    options: [
      { text: "损失震荡甚至发散", correct: true },
      { text: "收敛过快", correct: false },
      { text: "绝对没有影响", correct: false },
      { text: "模型立即过拟合", correct: false },
    ],
    explanation:
      "学习率过大会让参数跨越最优点、来回震荡，甚至损失爆炸发散。过小则收敛极慢。常用 warmup + 余弦退火等方式调度。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 18,
    type: "single",
    question: "批量归一化（BatchNorm）主要解决什么问题？",
    options: [
      { text: "中间层分布漂移（internal covariate shift）", correct: true },
      { text: "模型参数量过大", correct: false },
      { text: "数据集不平衡", correct: false },
      { text: "推理速度慢", correct: false },
    ],
    explanation:
      "BatchNorm 对每一批数据的激活做归一化，使中间层输入分布稳定，从而允许更大的学习率、加速收敛并有一定正则效果。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 19,
    type: "single",
    question: "交叉熵损失常用于哪类任务？",
    options: [
      { text: "分类任务", correct: true },
      { text: "回归任务", correct: false },
      { text: "聚类任务", correct: false },
      { text: "数据压缩", correct: false },
    ],
    explanation:
      "交叉熵衡量两个概率分布的差异，配合 Softmax 用于分类，比均方误差更适配概率输出。回归任务一般用 MSE。",
    tags: ["AI Basics", "Loss"],
  },
  {
    id: 20,
    type: "single",
    question: "均方误差（MSE）损失最常用于？",
    options: [
      { text: "回归任务", correct: true },
      { text: "二分类任务", correct: false },
      { text: "多分类任务", correct: false },
      { text: "序列生成", correct: false },
    ],
    explanation:
      "MSE 度量预测值与真实值的平方差，适合连续值的回归任务。分类通常用交叉熵。",
    tags: ["AI Basics", "Loss"],
  },
  {
    id: 21,
    type: "single",
    question: "「早停」（early stopping）策略是？",
    options: [
      { text: "验证集性能不再提升时停止训练", correct: true },
      { text: "训练刚开始就停止", correct: false },
      { text: "固定训练固定步数", correct: false },
      { text: "只在数据量少时使用", correct: false },
    ],
    explanation:
      "早停监控验证集指标，一旦不再改善（可能开始过拟合）就停止训练，既防过拟合又省算力。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 22,
    type: "single",
    question: "「数据增强」的主要目的是？",
    options: [
      { text: "通过对原始数据做变换，扩充数据并提升泛化", correct: true },
      { text: "减少模型参数量", correct: false },
      { text: "提升推理速度", correct: false },
      { text: "替换掉有噪声的数据", correct: false },
    ],
    explanation:
      "数据增强对样本做旋转、翻转、加噪等变换生成更多变体，增加数据多样性，降低过拟合、提升泛化能力。",
    tags: ["AI Basics", "Data"],
  },
  {
    id: 23,
    type: "single",
    question: "混淆矩阵中，TP 表示？",
    options: [
      { text: "真实为正且预测为正", correct: true },
      { text: "真实为正但预测为负", correct: false },
      { text: "真实为负但预测为正", correct: false },
      { text: "真实为负且预测为负", correct: false },
    ],
    explanation:
      "TP（True Positive）是「真实正、预测正」；FP 是误报（真实负、预测正）；FN 是漏报（真实正、预测负）；TN 是正确预测的负样本。",
    tags: ["AI Basics", "Metrics"],
  },
  {
    id: 24,
    type: "single",
    question: "精确率（Precision）的定义是？",
    options: [
      { text: "预测为正的样本中，真正为正的比例", correct: true },
      { text: "真实为正的样本中，被预测为正的比例", correct: false },
      { text: "所有样本中预测正确的比例", correct: false },
      { text: "真实为负且预测为负的比例", correct: false },
    ],
    explanation:
      "Precision = TP / (TP + FP)，衡量「预测正里面有多少是对的」；Recall = TP / (TP + FN)，衡量「正样本里有多少被找出来」。",
    tags: ["AI Basics", "Metrics"],
  },
  {
    id: 25,
    type: "single",
    question: "召回率（Recall）的定义是？",
    options: [
      { text: "真实为正的样本中，被预测为正的比例", correct: true },
      { text: "预测为正的样本中，真正为正的比例", correct: false },
      { text: "所有样本中预测正确的比例", correct: false },
      { text: "预测为负的样本中真正的比例", correct: false },
    ],
    explanation:
      "Recall = TP / (TP + FN)，衡量「正样本里被找出多少」。Precision 高但 Recall 低，说明「宁可少报也不误报」。",
    tags: ["AI Basics", "Metrics"],
  },
  {
    id: 26,
    type: "single",
    question: "F1 分数是精确率和召回率的什么？",
    options: [
      { text: "调和平均", correct: true },
      { text: "算术平均", correct: false },
      { text: "几何平均", correct: false },
      { text: "加权求和", correct: false },
    ],
    explanation:
      "F1 = 2·P·R / (P + R)，是精确率和召回率的调和平均，在两者都不高时会给较低值，用于平衡 P 和 R。",
    tags: ["AI Basics", "Metrics"],
  },
  {
    id: 27,
    type: "single",
    question: "「训练集 / 验证集 / 测试集」各自的用途？",
    options: [
      { text: "训练集学参数、验证集调超参、测试集评估最终泛化", correct: true },
      { text: "三个数据集用途完全相同", correct: false },
      { text: "验证集用来训练", correct: false },
      { text: "测试集用来调超参", correct: false },
    ],
    explanation:
      "训练集用来更新参数；验证集用于调超参和早停（不参与梯度）；测试集只在最终评估一次，反映真实泛化能力，避免过拟合到验证集。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 28,
    type: "single",
    question: "K 折交叉验证的做法是？",
    options: [
      { text: "把数据分 K 份，轮流取一份做验证、其余训练", correct: true },
      { text: "只训练一次", correct: false },
      { text: "只保留一份数据训练", correct: false },
      { text: "把所有数据同时用于训练和验证", correct: false },
    ],
    explanation:
      "K 折交叉验证把数据均分 K 份，每轮取 1 份做验证、其余 K-1 份训练，循环 K 次，结果取平均，更充分地利用有限数据评估。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 29,
    type: "single",
    question: "梯度下降中，mini-batch 的作用是？",
    options: [
      { text: "用一小批样本估计梯度，兼顾效率与稳定", correct: true },
      { text: "必须用全部数据计算梯度", correct: false },
      { text: "只用一个样本估计梯度", correct: false },
      { text: "不参与梯度计算", correct: false },
    ],
    explanation:
      "mini-batch 用一批样本的平均梯度估计全局梯度，比全量更高效、比单样本更稳定。批大小是训练重要超参。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 30,
    type: "single",
    question: "特征归一化（标准化）的好处是？",
    options: [
      { text: "让各特征量纲一致，加速收敛", correct: true },
      { text: "增加特征数量", correct: false },
      { text: "去除样本标签", correct: false },
      { text: "把数据变成离散值", correct: false },
    ],
    explanation:
      "特征尺度差异大时，梯度更新在部分维度上会很慢。标准化让特征分布到相近范围，梯度更均衡，收敛更快更稳。",
    tags: ["AI Basics", "Data"],
  },
  {
    id: 31,
    type: "multiple",
    question: "下列哪些属于缓解过拟合的手段？（多选）",
    options: [
      { text: "L2 权重正则", correct: true },
      { text: "Dropout", correct: true },
      { text: "数据增强", correct: true },
      { text: "无脑增大模型参数量", correct: false },
    ],
    explanation:
      "L2 正则、Dropout、数据增强都能提升泛化、抑制过拟合。盲目增大参数量反而更容易过拟合。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 32,
    type: "multiple",
    question: "关于激活函数，下列正确的有？（多选）",
    options: [
      { text: "ReLU 正区间梯度恒为 1，缓解梯度消失", correct: true },
      { text: "Sigmoid 在输入很大时梯度趋近 0", correct: true },
      { text: "Tanh 输出范围是 [-1, 1]", correct: true },
      { text: "ReLU 对所有负数输入都有非零梯度", correct: false },
    ],
    explanation:
      "ReLU 对负输入梯度为 0（可能神经元死亡），不是非零。Sigmoid 两端饱和、梯度趋 0；Tanh 输出 [-1,1]。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 33,
    type: "multiple",
    question: "训练时损失不下降，可能的原因有？（多选）",
    options: [
      { text: "学习率过小", correct: true },
      { text: "数据没有归一化", correct: true },
      { text: "梯度消失", correct: true },
      { text: "参数量过大导致立即过拟合", correct: false },
    ],
    explanation:
      "学习率过小收敛慢、数据未归一化导致梯度不平衡、梯度消失让深层不更新，都可能导致 loss 不降。过拟合是「降得快但泛化差」，不是不降。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 34,
    type: "multiple",
    question: "分类任务的常用评估指标包括？（多选）",
    options: [
      { text: "Accuracy（准确率）", correct: true },
      { text: "Precision（精确率）", correct: true },
      { text: "Recall（召回率）", correct: true },
      { text: "BLEU（译文评估）", correct: false },
    ],
    explanation:
      "分类常用 Accuracy/Precision/Recall/F1、AUC 等；BLEU 是机器翻译/生成任务的指标，不属于分类。",
    tags: ["AI Basics", "Metrics"],
  },
  {
    id: 35,
    type: "multiple",
    question: "数据不平衡时，下列做法合理的有？（多选）",
    options: [
      { text: "对少数类过采样", correct: true },
      { text: "使用加权损失", correct: true },
      { text: "只用 Accuracy 作为唯一指标", correct: false },
      { text: "对多数类降采样", correct: true },
    ],
    explanation:
      "过采样少数类、降采样多数类、加权损失都能缓解不平衡。此时 Accuracy 会失真（全猜多数类也可能很高），应看 Precision/Recall/AUC。",
    tags: ["AI Basics", "Data"],
  },
  {
    id: 36,
    type: "multiple",
    question: "关于批大小（batch size），下列正确的有？（多选）",
    options: [
      { text: "批越大，梯度越稳定", correct: true },
      { text: "批越大，显存占用越高", correct: true },
      { text: "批越小，梯度噪声越大", correct: true },
      { text: "批大小对训练没有任何影响", correct: false },
    ],
    explanation:
      "批越大梯度越平滑但占显存多、更新次数少；批越小噪声大但更新频繁。批大小显著影响训练动态。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 37,
    type: "multiple",
    question: "超参数调优的常见方法包括？（多选）",
    options: [
      { text: "网格搜索", correct: true },
      { text: "随机搜索", correct: true },
      { text: "贝叶斯优化", correct: true },
      { text: "在测试集上反复调优", correct: false },
    ],
    explanation:
      "网格/随机/贝叶斯搜索都用于调超参。在测试集上反复调参等于把测试集当验证集用，会让评估结果失真。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 38,
    type: "multiple",
    question: "模型泛化能力好的表现是？（多选）",
    options: [
      { text: "在未见过的数据上表现稳定", correct: true },
      { text: "训练集和测试集误差差距小", correct: true },
      { text: "只在训练集上记住答案", correct: false },
      { text: "对数据扰动不过度敏感", correct: true },
    ],
    explanation:
      "泛化好指模型学到可迁移的规律，对未见数据稳定、训练与测试差距小、对轻微扰动不敏感。只在训练集上记答案是过拟合。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 39,
    type: "multiple",
    question: "下列哪些是防止梯度爆炸的手段？（多选）",
    options: [
      { text: "梯度裁剪（gradient clipping）", correct: true },
      { text: "权重初始化控制", correct: true },
      { text: "批归一化", correct: true },
      { text: "无限增大学习率", correct: false },
    ],
    explanation:
      "梯度裁剪直接限制梯度范数、好的初始化避免前期爆炸、归一化稳定分布，都能防梯度爆炸。增大学习率反而加剧爆炸。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 40,
    type: "single",
    question: "Embedding 层的作用是？",
    options: [
      { text: "把离散符号映射为稠密向量", correct: true },
      { text: "对向量做归一化", correct: false },
      { text: "压缩图像", correct: false },
      { text: "做卷积运算", correct: false },
    ],
    explanation:
      "Embedding 把词/离散 token 查表映射为稠密向量，是神经网络处理离散输入的标准方式，后续层再学习向量间的语义关系。",
    tags: ["AI Basics", "Representation"],
  },
  {
    id: 41,
    type: "single",
    question: "Softmax 函数的作用是？",
    options: [
      { text: "把一组实数映射成和为 1 的概率分布", correct: true },
      { text: "把向量压缩到 [-1, 1]", correct: false },
      { text: "把矩阵转置", correct: false },
      { text: "把数值离散化", correct: false },
    ],
    explanation:
      "Softmax 将一组 logits 指数化并归一化，得到每个类别的概率（总和为 1），常与交叉熵配合用于多分类输出。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 42,
    type: "single",
    question: "「正则化」（regularization）的核心思想是？",
    options: [
      { text: "给损失加约束，限制模型复杂度", correct: true },
      { text: "增加训练数据量", correct: false },
      { text: "只训练部分层", correct: false },
      { text: "把模型换成更大的", correct: false },
    ],
    explanation:
      "正则化通过在损失函数中加入约束项（如权重惩罚），限制模型复杂度，抑制过拟合、提升泛化。",
    tags: ["AI Basics", "DL"],
  },
  {
    id: 43,
    type: "single",
    question: "「批量梯度下降」与「随机梯度下降」的本质区别是？",
    options: [
      { text: "前者用全量数据算梯度，后者用单个样本", correct: true },
      { text: "前者不用梯度", correct: false },
      { text: "后者没有损失函数", correct: false },
      { text: "二者结果一定相同", correct: false },
    ],
    explanation:
      "批量梯度下降（BGD）每次用全部数据算梯度，准确但慢；随机梯度下降（SGD）每次只用一个样本，快但有噪声。mini-batch 是两者的折中。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 44,
    type: "single",
    question: "特征工程的目标是？",
    options: [
      { text: "构造或选择对模型更有效的输入特征", correct: true },
      { text: "增加模型层数", correct: false },
      { text: "减少模型推理延迟", correct: false },
      { text: "压缩标签", correct: false },
    ],
    explanation:
      "特征工程通过构造、选择、变换特征，让模型更容易学到有效模式，对传统 ML 尤其重要。",
    tags: ["AI Basics", "Data"],
  },
  {
    id: 45,
    type: "single",
    question: "「模型蒸馏」（knowledge distillation）的思路是？",
    options: [
      { text: "用大模型（teacher）的输出去训练小模型（student）", correct: true },
      { text: "把大模型拆成两个", correct: false },
      { text: "对大模型做量化", correct: false },
      { text: "减少训练数据", correct: false },
    ],
    explanation:
      "蒸馏用大而强的 teacher 模型的软输出（概率分布）作为监督，训练小 student 模型，让小模型在更小的规模上逼近大模型能力。",
    tags: ["AI Basics", "Model"],
  },
  {
    id: 46,
    type: "single",
    question: "「偏差」（bias）与「方差」（variance）权衡中，高方差通常意味着？",
    options: [
      { text: "模型对训练集过拟合、对新数据波动大", correct: true },
      { text: "模型过于简单", correct: false },
      { text: "模型完全没有学习能力", correct: false },
      { text: "数据标签噪声大", correct: false },
    ],
    explanation:
      "高方差：模型太灵活，拟合了训练集噪声，换数据表现波动大（即过拟合）。高偏差：模型太简单，学不到规律（欠拟合）。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 47,
    type: "single",
    question: "「学习率调度」（learning rate schedule）的常见做法是？",
    options: [
      { text: "训练中按策略逐步调整学习率", correct: true },
      { text: "学习率全程固定不变", correct: false },
      { text: "随机生成学习率", correct: false },
      { text: "学习率越大越好", correct: false },
    ],
    explanation:
      "常用 warmup（先小后升）避免初期发散，再余弦退火/阶梯下降逐步减小，兼顾收敛速度与最终精度。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 48,
    type: "single",
    question: "「one-hot 编码」的特点？",
    options: [
      { text: "把类别表示成只有一个 1 其余全 0 的向量", correct: true },
      { text: "把类别映射成连续实数", correct: false },
      { text: "把数值归一化", correct: false },
      { text: "只适用于数值型特征", correct: false },
    ],
    explanation:
      "one-hot 用向量中某一位为 1 表示对应类别，简单但维度随类别数增长，且无法体现类别间语义关系（这正是 Embedding 要解决的）。",
    tags: ["AI Basics", "Data"],
  },
  {
    id: 49,
    type: "single",
    question: "「过拟合」与「欠拟合」同时缓解的方法是？",
    options: [
      { text: "过拟合加正则/增数据，欠拟合增强模型/调参", correct: true },
      { text: "两种都用同一手段", correct: false },
      { text: "两者无法解决", correct: false },
      { text: "只靠增加训练时长", correct: false },
    ],
    explanation:
      "两者方向相反：过拟合要降低复杂度/增数据/加正则；欠拟合要增大容量/调优/加特征。需对症下药。",
    tags: ["AI Basics", "ML"],
  },
  {
    id: 50,
    type: "single",
    question: "「初始化」对深度网络训练的重要性在于？",
    options: [
      { text: "好的初始化能避免梯度消失/爆炸、加速收敛", correct: true },
      { text: "初始化完全不影响训练", correct: false },
      { text: "初始化决定最终精度上限", correct: false },
      { text: "初始化必须全为 0", correct: false },
    ],
    explanation:
      "对称初始化（如全 0）会破坏学习；过大过小会引发爆炸/消失。He/Xavier 等按层大小缩放，让信号稳定传播。",
    tags: ["AI Basics", "Training"],
  },
  {
    id: 51,
    type: "single",
    question: "「微调」（fine-tuning）指？",
    options: [
      { text: "在预训练模型基础上用新任务数据继续训练", correct: true },
      { text: "从零开始训练一个新模型", correct: false },
      { text: "只调整数据不调模型", correct: false },
      { text: "删除部分模型层", correct: false },
    ],
    explanation:
      "微调利用预训练模型已学到的通用表征，在目标任务数据上继续训练（常配合较小学习率），比从零训练更高效。",
    tags: ["AI Basics", "Transfer"],
  },
  {
    id: 52,
    type: "multiple",
    question: "下列哪些属于生成式模型？（多选）",
    options: [
      { text: "扩散模型（Diffusion）", correct: true },
      { text: "自回归语言模型（GPT）", correct: true },
      { text: "线性回归", correct: false },
      { text: "VAE", correct: true },
    ],
    explanation:
      "扩散模型、自回归 LLM、VAE 都能从数据分布中生成新样本，属生成式；线性回归是判别/回归模型。",
    tags: ["AI Basics", "Generative"],
  },
  {
    id: 53,
    type: "multiple",
    question: "关于损失函数与梯度，下列正确的有？（多选）",
    options: [
      { text: "反向传播通过链式法则计算梯度", correct: true },
      { text: "梯度方向是损失上升最快的方向", correct: true },
      { text: "参数沿负梯度方向更新", correct: true },
      { text: "梯度与损失函数无关", correct: false },
    ],
    explanation:
      "反向传播用链式法则求梯度；梯度指向损失上升最快的方向，因此沿负梯度更新参数以最小化损失。",
    tags: ["AI Basics", "Optimization"],
  },
  // ==================== Transformer ====================
  {
    id: 54,
    type: "single",
    question: "Transformer 相比 RNN 的核心优势是？",
    options: [
      { text: "可并行处理整段序列，不受顺序限制", correct: true },
      { text: "参数量必然更少", correct: false },
      { text: "不需要位置信息", correct: false },
      { text: "只处理短序列", correct: false },
    ],
    explanation:
      "RNN 需逐步顺序处理、难以并行且长距离依赖弱。Transformer 用注意力一次性关联所有位置，可并行、长程依赖好。",
    tags: ["Transformer", "Architecture"],
  },
  {
    id: 55,
    type: "single",
    question: "Transformer 中的「自注意力」（self-attention）是？",
    options: [
      { text: "让每个 token 与序列内其他 token 交互", correct: true },
      { text: "让模型只关注第一个 token", correct: false },
      { text: "用卷积计算 token 关系", correct: false },
      { text: "仅关注相邻两个 token", correct: false },
    ],
    explanation:
      "自注意力让序列中每个位置与所有其他位置计算相关性（注意力权重），加权聚合信息，捕捉全局依赖。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 56,
    type: "single",
    question: "Transformer 的输入为什么需要位置编码（positional encoding）？",
    options: [
      { text: "注意力本身没有顺序概念，需注入位置信息", correct: true },
      { text: "增加模型参数量", correct: false },
      { text: "为了让输出更好看", correct: false },
      { text: "位置编码只在训练时用", correct: false },
    ],
    explanation:
      "注意力是对集合操作，打乱 token 顺序结果不变，缺少顺序感知。位置编码把位置信息加进输入，让模型知道词序。",
    tags: ["Transformer", "Position"],
  },
  {
    id: 57,
    type: "single",
    question: "多头注意力（multi-head attention）的作用是？",
    options: [
      { text: "让不同头关注不同子空间的关系", correct: true },
      { text: "减少计算量", correct: false },
      { text: "把注意力替换成卷积", correct: false },
      { text: "消除位置编码", correct: false },
    ],
    explanation:
      "多个头分别在不同投影子空间计算注意力，能捕获不同类型的依赖（如语法、指代等），再拼接融合，提升表达能力。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 58,
    type: "single",
    question: "Attention 中 Q、K、V 分别代表什么？",
    options: [
      { text: "Q 查询、K 键、V 值", correct: true },
      { text: "Q 键、K 值、V 查询", correct: false },
      { text: "三者等价", correct: false },
      { text: "Q 值、K 查询、V 键", correct: false },
    ],
    explanation:
      "Q（query）找要关注谁，K（key）被匹配的对象，V（value）实际要聚合的内容。Q 与 K 点积得权重，加权 V。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 59,
    type: "single",
    question: "Scaled Dot-Product Attention 中除以 √d 的作用是？",
    options: [
      { text: "防止点积过大导致 softmax 梯度消失", correct: true },
      { text: "减少参数量", correct: false },
      { text: "增加模型深度", correct: false },
      { text: "加快训练但不改变数值", correct: false },
    ],
    explanation:
      "点积随维度 d 增大而变大，softmax 会进入饱和区、梯度趋 0。除以 √d 把数值缩回合适范围，稳定训练。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 60,
    type: "single",
    question: "Transformer 编码器的前馈网络（FFN）在做什么？",
    options: [
      { text: "对每个位置独立做非线性变换", correct: true },
      { text: "在 token 之间交换信息", correct: false },
      { text: "计算注意力权重", correct: false },
      { text: "执行卷积", correct: false },
    ],
    explanation:
      "FFN 逐位置地对每个 token 做两次线性+激活的变换，不跨 token 通信，负责在每个位置做特征非线性变换。",
    tags: ["Transformer", "FFN"],
  },
  {
    id: 61,
    type: "single",
    question: "残差连接（residual connection）在 Transformer 中的作用？",
    options: [
      { text: "缓解深层梯度消失、稳定训练", correct: true },
      { text: "减少注意力计算", correct: false },
      { text: "替代位置编码", correct: false },
      { text: "降低参数量", correct: false },
    ],
    explanation:
      "残差连接把输入加到子层输出上（x + sublayer(x)），为梯度提供短路路径，缓解梯度消失，使深层网络可训练。",
    tags: ["Transformer", "Architecture"],
  },
  {
    id: 62,
    type: "single",
    question: "LayerNorm 与 BatchNorm 的主要区别是？",
    options: [
      { text: "LayerNorm 沿特征维度归一化，BatchNorm 沿 batch 维度", correct: true },
      { text: "两者完全等价", correct: false },
      { text: "LayerNorm 依赖 batch 大小", correct: false },
      { text: "BatchNorm 不参与训练", correct: false },
    ],
    explanation:
      "LayerNorm 对每个样本的特征做归一化，不依赖 batch 大小，适合变长序列；BatchNorm 对每个特征跨 batch 归一化。Transformer 用 LayerNorm。",
    tags: ["Transformer", "Norm"],
  },
  {
    id: 63,
    type: "single",
    question: "为什么 Transformer 用 LayerNorm 而不是 BatchNorm？",
    options: [
      { text: "序列长度可变、batch 小，BatchNorm 不稳定", correct: true },
      { text: "LayerNorm 参数量更少", correct: false },
      { text: "BatchNorm 无法反向传播", correct: false },
      { text: "两者都能用，无区别", correct: false },
    ],
    explanation:
      "训练时序列长度和 batch 都可能变化，BatchNorm 依赖 batch 统计易不稳定；LayerNorm 按样本独立归一化，更适配变长序列。",
    tags: ["Transformer", "Norm"],
  },
  {
    id: 64,
    type: "single",
    question: "Transformer 的自回归解码时如何保证只看到过去？",
    options: [
      { text: "用因果掩码（causal mask）屏蔽未来位置", correct: true },
      { text: "把未来 token 置零", correct: false },
      { text: "打乱序列", correct: false },
      { text: "无需处理，天然可见", correct: false },
    ],
    explanation:
      "解码时对注意力矩阵施加因果掩码，让第 i 个位置只能 attend 到 ≤ i 的位置，防止看到未来 token 造成泄漏。",
    tags: ["Transformer", "Decoding"],
  },
  {
    id: 65,
    type: "single",
    question: "「KV cache」在推理时保存的是？",
    options: [
      { text: "历史 token 的 Key 和 Value 矩阵", correct: true },
      { text: "所有模型权重", correct: false },
      { text: "历史损失值", correct: false },
      { text: "注意力分数", correct: false },
    ],
    explanation:
      "自回归生成每个新 token 时只需用它的 Q 与历史 K/V 计算注意力。缓存历史 K/V 可避免重复计算，显著加速生成。",
    tags: ["Transformer", "Inference"],
  },
  {
    id: 66,
    type: "single",
    question: "KV cache 会带来什么问题？",
    options: [
      { text: "随序列长度线性增长，占用显存", correct: true },
      { text: "让模型精度下降", correct: false },
      { text: "使训练变慢", correct: false },
      { text: "导致梯度消失", correct: false },
    ],
    explanation:
      "每个新 token 都要追加 K/V，缓存随上下文长度线性增长，长上下文或大批量时显存压力大（长序列推理的主要瓶颈）。",
    tags: ["Transformer", "Memory"],
  },
  {
    id: 67,
    type: "single",
    question: "GQA（分组查询注意力）相比 MHA 的改动是？",
    options: [
      { text: "多个 Q 头共享少数 K/V 头，减少 KV cache 和计算", correct: true },
      { text: "去掉 Q 头", correct: false },
      { text: "用卷积替代注意力", correct: false },
      { text: "增加更多 K/V 头", correct: false },
    ],
    explanation:
      "GQA 让多组 Q 头共享同一组 K/V 头，在基本不损失质量的前提下大幅减少 KV cache 和显存，是当前大模型的常用做法。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 68,
    type: "single",
    question: "MQA（多查询注意力）与 GQA 的关系是？",
    options: [
      { text: "MQA 是所有 Q 头共享一个 K/V 头的 GQA 特例", correct: true },
      { text: "两者完全不同", correct: false },
      { text: "MQA 比 MHA 更多参数", correct: false },
      { text: "MQA 只在训练用", correct: false },
    ],
    explanation:
      "MQA 是所有 Q 头共用一个 K/V 头，是 GQA（共享一组 K/V 头）中共享组数为 1 的极端特例，KV cache 最小但质量略降。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 69,
    type: "single",
    question: "Transformer 中「pre-norm」与「post-norm」指？",
    options: [
      { text: "归一化放在残差分支前还是后", correct: true },
      { text: "归一化放在输入前还是输出后", correct: true },
      { text: "归一化只在第一层", correct: false },
      { text: "归一化与残差无关", correct: false },
    ],
    explanation:
      "post-norm 先子层后归一化（原始版），pre-norm 先归一化再子层。现代大模型多用 pre-norm，更稳定、允许更深网络。",
    tags: ["Transformer", "Norm"],
  },
  {
    id: 70,
    type: "single",
    question: "「旋转位置编码」（RoPE）的特点？",
    options: [
      { text: "用旋转矩阵编码相对位置，利于外推", correct: true },
      { text: "只用绝对位置", correct: false },
      { text: "无法并行", correct: false },
      { text: "只用于 CNN", correct: false },
    ],
    explanation:
      "RoPE 把位置信息通过旋转矩阵作用于 Q/K，隐式编码相对位置，对长文本外推更友好，是 LLaMA/Qwen 等的标配。",
    tags: ["Transformer", "Position"],
  },
  {
    id: 71,
    type: "single",
    question: "「softmax 温度」（temperature）的作用是？",
    options: [
      { text: "调节输出分布的平滑/尖锐程度", correct: true },
      { text: "改变模型参数", correct: false },
      { text: "用于归一化梯度", correct: false },
      { text: "只影响训练", correct: false },
    ],
    explanation:
      "temperature < 1 使分布更尖锐（更确定性），> 1 更均匀（更随机）。用于采样时控制生成的随机性与多样性。",
    tags: ["Transformer", "Sampling"],
  },
  {
    id: 72,
    type: "single",
    question: "「top-k 采样」是？",
    options: [
      { text: "只从概率最高的 k 个 token 中采样", correct: true },
      { text: "从全部 token 均匀采样", correct: false },
      { text: "固定输出 k 个 token", correct: false },
      { text: "取概率最低的 k 个", correct: false },
    ],
    explanation:
      "top-k 把候选缩小到概率最高的 k 个 token，再按比例采样，避免选到低概率的离谱词，同时保留一定随机性。",
    tags: ["Transformer", "Sampling"],
  },
  {
    id: 73,
    type: "single",
    question: "「top-p」（nucleus）采样的思想是？",
    options: [
      { text: "动态选取累积概率达到 p 的最小子集采样", correct: true },
      { text: "固定取前 p 个 token", correct: false },
      { text: "把所有 token 概率设为 p", correct: false },
      { text: "与 top-k 完全相同", correct: false },
    ],
    explanation:
      "top-p 动态地把累积概率超过阈值 p 的那批 token 纳入候选，候选数量随分布变化，比固定 k 更灵活。",
    tags: ["Transformer", "Sampling"],
  },
  {
    id: 74,
    type: "single",
    question: "Transformer 的时间复杂度随序列长度如何变化？",
    options: [
      { text: "自注意力是 O(n²)，随序列变长开销剧增", correct: true },
      { text: "O(n)，线性增长", correct: false },
      { text: "O(log n)", correct: false },
      { text: "与长度无关", correct: false },
    ],
    explanation:
      "自注意力要算 n×n 的注意力矩阵，复杂度 O(n²)。长序列是瓶颈，催生了 FlashAttention、线性注意力、稀疏注意力等优化。",
    tags: ["Transformer", "Complexity"],
  },
  {
    id: 75,
    type: "single",
    question: "「张量并行」（tensor parallelism）的做法是？",
    options: [
      { text: "把单个大权重矩阵切分到多张卡上共同计算", correct: true },
      { text: "把数据分到多张卡训练", correct: false },
      { text: "让多张卡各自训练完整模型", correct: false },
      { text: "把模型复制到多张卡", correct: false },
    ],
    explanation:
      "张量并行把一层内的权重矩阵按行/列切分到多张 GPU，单层计算由多卡协同完成，用于单卡放不下大模型。",
    tags: ["Transformer", "Parallel"],
  },
  {
    id: 76,
    type: "single",
    question: "「流水线并行」（pipeline parallelism）是？",
    options: [
      { text: "把模型按层切分到多张卡，串行流水计算", correct: true },
      { text: "把数据并行分到各卡", correct: false },
      { text: "每张卡存完整模型", correct: false },
      { text: "只在推理时用", correct: false },
    ],
    explanation:
      "流水线并行把模型的不同层分到不同 GPU，前向/反向像流水线一样顺序执行，减少单卡显存压力，但有气泡开销。",
    tags: ["Transformer", "Parallel"],
  },
  {
    id: 77,
    type: "single",
    question: "「数据并行」是？",
    options: [
      { text: "多张卡各持完整模型副本，分不同数据批并行训练", correct: true },
      { text: "把模型切分到多卡", correct: false },
      { text: "只用一张卡", correct: false },
      { text: "把数据压缩", correct: false },
    ],
    explanation:
      "数据并行让每张卡持有完整模型副本，处理不同数据批，通过梯度同步（如 all-reduce）保持各卡一致。",
    tags: ["Transformer", "Parallel"],
  },
  {
    id: 78,
    type: "single",
    question: "「混合精度训练」通常指？",
    options: [
      { text: "用 fp16/bf16 计算 + fp32 主权重，兼顾速度与精度", correct: true },
      { text: "全程只用 fp32", correct: false },
      { text: "全程只用 int8", correct: false },
      { text: "随机选精度", correct: false },
    ],
    explanation:
      "混合精度用低精度（fp16/bf16）做前反向和存储激活、加快速度省显存，同时保留 fp32 主权重副本保证更新精度。",
    tags: ["Transformer", "Training"],
  },
  {
    id: 79,
    type: "single",
    question: "大模型预训练通常用什么优化器？",
    options: [
      { text: "AdamW", correct: true },
      { text: "普通 SGD", correct: false },
      { text: "Adagrad", correct: false },
      { text: "随机初始化", correct: false },
    ],
    explanation:
      "AdamW 收敛快、稳定且权重衰减与自适应解耦，是大模型预训练/微调的事实标准。",
    tags: ["Transformer", "Optimizer"],
  },
  {
    id: 80,
    type: "single",
    question: "「梯度累积」（gradient accumulation）解决什么？",
    options: [
      { text: "显存不够时等效增大 batch size", correct: true },
      { text: "让模型收敛更快", correct: false },
      { text: "消除梯度消失", correct: false },
      { text: "减小模型参数量", correct: false },
    ],
    explanation:
      "显存放不下大 batch 时，先分多个小 batch 计算梯度并累积，攒够再更新一次参数，等效于更大的有效 batch。",
    tags: ["Transformer", "Training"],
  },
  {
    id: 81,
    type: "single",
    question: "「稀疏注意力」的目的是？",
    options: [
      { text: "只计算部分位置对，降低长序列复杂度", correct: true },
      { text: "增加注意力头数", correct: false },
      { text: "让注意力更精确", correct: false },
      { text: "减少模型层数", correct: false },
    ],
    explanation:
      "稀疏注意力只让每个位置 attend 到局部或间隔的关键位置，把 O(n²) 降到接近 O(n)，用于超长序列（如长文档、长视频）。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 82,
    type: "single",
    question: "Transformer 编码器-解码器结构中，交叉注意力（cross-attention）的 Q 来自？",
    options: [
      { text: "解码器，K/V 来自编码器输出", correct: true },
      { text: "编码器，K/V 来自解码器", correct: false },
      { text: "两者都用自身", correct: false },
      { text: "不使用注意力", correct: false },
    ],
    explanation:
      "交叉注意力里，解码器的查询（Q）去编码器输出的键值（K/V）里找信息，实现「源序列」到「目标序列」的对齐。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 83,
    type: "single",
    question: "为什么 decoder-only 模型成为主流大模型架构？",
    options: [
      { text: "统一自回归建模，简单可扩展且通用", correct: true },
      { text: "参数量必然更小", correct: false },
      { text: "不需要位置编码", correct: false },
      { text: "不能做生成", correct: false },
    ],
    explanation:
      "decoder-only 用因果自注意力统一做自回归生成，结构简单、易于规模化，预训练+指令微调后通用性强，成为 GPT/LLaMA 等主流选择。",
    tags: ["Transformer", "Architecture"],
  },
  {
    id: 84,
    type: "single",
    question: "「词表」（vocabulary）指的是？",
    options: [
      { text: "模型能处理的所有 token 集合", correct: true },
      { text: "训练数据的数量", correct: false },
      { text: "模型的层数", correct: false },
      { text: "注意力头数", correct: false },
    ],
    explanation:
      "词表是文本被切成的所有基本单元（token）的集合，模型输出层维度通常等于词表大小，用于预测下一个 token。",
    tags: ["Transformer", "Tokenizer"],
  },
  {
    id: 85,
    type: "multiple",
    question: "关于自注意力，下列正确的有？（多选）",
    options: [
      { text: "能捕捉长距离依赖", correct: true },
      { text: "可并行计算", correct: true },
      { text: "复杂度随序列长度平方增长", correct: true },
      { text: "无法捕捉任何依赖", correct: false },
    ],
    explanation:
      "自注意力全局建模、可并行，但 O(n²) 复杂度是长序列瓶颈。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 86,
    type: "multiple",
    question: "下列哪些属于 FlashAttention 的优化手段？（多选）",
    options: [
      { text: "分块计算，避免存完整 n×n 矩阵", correct: true },
      { text: "利用 GPU 高速片上内存（SRAM）", correct: true },
      { text: "在线计算 softmax 归一化因子", correct: true },
      { text: "使用更大精度避免误差", correct: false },
    ],
    explanation:
      "FlashAttention 通过分块 + SRAM + 在线 softmax，避免把完整注意力矩阵写入慢速显存，省显存且提速，不是用高精度。",
    tags: ["Transformer", "FlashAttention"],
  },
  {
    id: 87,
    type: "multiple",
    question: "关于 RoPE，下列正确的有？（多选）",
    options: [
      { text: "通过旋转矩阵编码位置", correct: true },
      { text: "编码的是相对位置", correct: true },
      { text: "对长文本外推更友好", correct: true },
      { text: "无法参与注意力计算", correct: false },
    ],
    explanation:
      "RoPE 用旋转矩阵作用于 Q/K，隐式携带相对位置信息，利于长度外推，并参与注意力分数计算。",
    tags: ["Transformer", "Position"],
  },
  {
    id: 88,
    type: "multiple",
    question: "KV cache 优化手段包括？（多选）",
    options: [
      { text: "GQA/MQA 减少 K/V 头", correct: true },
      { text: "KV cache 量化压缩", correct: true },
      { text: "缓存逐 token 滑动窗口", correct: true },
      { text: "每次重新计算全部注意力", correct: false },
    ],
    explanation:
      "GQA/MQA 减头、量化压缩、滑动窗口缓存都能降低 KV cache 显存；重新计算违背缓存目的。",
    tags: ["Transformer", "Memory"],
  },
  {
    id: 89,
    type: "multiple",
    question: "大模型推理加速的常用方法有？（多选）",
    options: [
      { text: "量化（INT8/FP8）", correct: true },
      { text: "投机解码（speculative decoding）", correct: true },
      { text: "KV cache 复用", correct: true },
      { text: "每次重新前向整段序列", correct: false },
    ],
    explanation:
      "量化减运算、投机解码并行验证多个 token、KV cache 复用避免重复计算都能加速；重算整段序列更慢。",
    tags: ["Transformer", "Inference"],
  },
  {
    id: 90,
    type: "multiple",
    question: "关于 layer normalization，下列正确的有？（多选）",
    options: [
      { text: "按样本特征维度归一化", correct: true },
      { text: "不依赖 batch 大小", correct: true },
      { text: "适合变长序列", correct: true },
      { text: "需要在 batch 维度统计均值方差", correct: false },
    ],
    explanation:
      "LayerNorm 对单个样本的特征维度算均值方差归一化，与 batch 无关，天然适配变长序列。在 batch 维度统计的是 BatchNorm。",
    tags: ["Transformer", "Norm"],
  },
  {
    id: 91,
    type: "multiple",
    question: "关于位置编码，下列正确的有？（多选）",
    options: [
      { text: "绝对位置编码注入绝对位置", correct: true },
      { text: "RoPE 编码相对位置", correct: true },
      { text: "ALiBi 用偏置项表达相对距离", correct: true },
      { text: "位置编码可以完全省略", correct: false },
    ],
    explanation:
      "绝对编码（如正弦/可学习）给绝对位置；RoPE 编码相对位置；ALiBi 在注意力分数上加与距离相关的偏置。Transformer 都需某种位置信息。",
    tags: ["Transformer", "Position"],
  },
  {
    id: 92,
    type: "multiple",
    question: "关于注意力掩码，下列正确的有？（多选）",
    options: [
      { text: "因果掩码屏蔽未来位置", correct: true },
      { text: "padding mask 屏蔽填充位", correct: true },
      { text: "掩码通过把对应位置设为 -∞ 实现", correct: true },
      { text: "掩码只在推理时使用", correct: false },
    ],
    explanation:
      "因果掩码防泄漏、padding mask 忽略填充，都通过把注意力分数设 -∞ 使 softmax 归零；训练和推理都会用到。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 93,
    type: "multiple",
    question: "长上下文模型处理超长输入的挑战包括？（多选）",
    options: [
      { text: "注意力 O(n²) 计算开销", correct: true },
      { text: "KV cache 显存膨胀", correct: true },
      { text: "长距离位置外推能力", correct: true },
      { text: "模型参数量必然翻倍", correct: false },
    ],
    explanation:
      "超长序列的主要挑战是注意力复杂度、KV cache 显存和超出训练长度的外推能力，与参数量无直接关系。",
    tags: ["Transformer", "Context"],
  },
  {
    id: 94,
    type: "single",
    question: "「激活重计算」（activation checkpointing）是？",
    options: [
      { text: "反向传播时丢弃激活、需要时重新前向计算", correct: true },
      { text: "保留所有激活不占显存", correct: false },
      { text: "减少模型层数", correct: false },
      { text: "只用于推理", correct: false },
    ],
    explanation:
      "激活重计算用「计算换显存」：不保存每层激活，反向传播需梯度时再前向重算，能显著省显存、代价是更多计算。",
    tags: ["Transformer", "Memory"],
  },
  {
    id: 95,
    type: "single",
    question: "「MoE」（混合专家）模型的核心思想是？",
    options: [
      { text: "路由让每个 token 只激活部分专家子网络", correct: true },
      { text: "所有专家全部激活", correct: false },
      { text: "只有一个模型", correct: false },
      { text: "把模型量化", correct: false },
    ],
    explanation:
      "MoE 有多个「专家」FFN，路由器为每个 token 选择激活少量专家，从而在总参数量很大时保持激活参数量和计算可控。",
    tags: ["Transformer", "MoE"],
  },
  {
    id: 96,
    type: "single",
    question: "MoE 的「稀疏激活」好处是？",
    options: [
      { text: "用更少的激活参数/计算量获得更大容量", correct: true },
      { text: "让推理完全不需要计算", correct: false },
      { text: "减少模型文件大小", correct: false },
      { text: "消除注意力", correct: false },
    ],
    explanation:
      "MoE 总参数量大但每个 token 只走部分专家，推理计算量近似于小模型，而知识容量接近大模型，是扩展模型规模的高性价比方式。",
    tags: ["Transformer", "MoE"],
  },
  {
    id: 97,
    type: "single",
    question: "「长短期记忆」（LSTM）相对普通 RNN 的改进是？",
    options: [
      { text: "引入门控控制信息保留与遗忘", correct: true },
      { text: "去掉递归结构", correct: false },
      { text: "使用卷积", correct: false },
      { text: "不使用记忆", correct: false },
    ],
    explanation:
      "LSTM 通过输入门、遗忘门、输出门控制信息流，缓解了普通 RNN 的长距离依赖和梯度消失问题，但仍是顺序处理。",
    tags: ["Transformer", "RNN"],
  },
  {
    id: 98,
    type: "single",
    question: "「注意力」最早的核心思想「Query-Key-Value」类比理解是？",
    options: [
      { text: "用查询去匹配键，取对应值", correct: true },
      { text: "查询决定输出长度", correct: false },
      { text: "键是最终输出", correct: false },
      { text: "值决定查询", correct: false },
    ],
    explanation:
      "注意力像检索：Q 是「要找什么」，K 是「候选的标签」，V 是「候选的内容」，按 Q 与 K 的匹配度加权取 V。",
    tags: ["Transformer", "Attention"],
  },
  {
    id: 99,
    type: "single",
    question: "「上下文长度」对 KV cache 的影响是？",
    options: [
      { text: "上下文越长，KV cache 占用越大", correct: true },
      { text: "上下文长度与 KV cache 无关", correct: false },
      { text: "上下文越长 KV cache 越小", correct: false },
      { text: "只在训练时有影响", correct: false },
    ],
    explanation:
      "KV cache 随已生成的 token 数增长，上下文/生成长度越大，缓存的 K/V 越多，显存占用越大。",
    tags: ["Transformer", "Memory"],
  },
  {
    id: 100,
    type: "single",
    question: "「温度 + top-k + top-p」组合采样的目的是？",
    options: [
      { text: "同时控制分布的平滑度和候选范围", correct: true },
      { text: "固定输出单一结果", correct: false },
      { text: "提高模型精度", correct: false },
      { text: "减少推理时间", correct: false },
    ],
    explanation:
      "temperature 先调分布形状，top-k/top-p 再缩小候选集合，组合使用能在多样性与合理性之间精细调节生成。",
    tags: ["Transformer", "Sampling"],
  },
  {
    id: 101,
    type: "single",
    question: "「beam search」生成策略的特点是？",
    options: [
      { text: "每步保留若干候选序列，追求整体概率最高", correct: true },
      { text: "每步只保留一个候选", correct: false },
      { text: "完全随机生成", correct: false },
      { text: "与贪婪搜索完全相同", correct: false },
    ],
    explanation:
      "beam search 每步保留 beam width 个最优候选，兼顾全局优化，适合追求确定性的任务，但可能缺乏多样性。",
    tags: ["Transformer", "Decoding"],
  },
  {
    id: 102,
    type: "single",
    question: "「贪心解码」（greedy decoding）是？",
    options: [
      { text: "每步选概率最高的 token", correct: true },
      { text: "随机选 token", correct: false },
      { text: "回溯整句", correct: false },
      { text: "同时生成多个候选", correct: false },
    ],
    explanation:
      "贪心每步取当前最高概率 token，快但容易陷入局部最优、重复或缺乏创造性。",
    tags: ["Transformer", "Decoding"],
  },
  {
    id: 103,
    type: "multiple",
    question: "关于推理时 KV cache 与 attention 计算，正确的有？（多选）",
    options: [
      { text: "生成新 token 时复用历史 K/V，不重算", correct: true },
      { text: "KV cache 让每个新 token 计算量近似常量", correct: true },
      { text: "KV cache 是推理加速的关键之一", correct: true },
      { text: "KV cache 会明显增加每次请求的重复计算", correct: false },
    ],
    explanation:
      "复用历史 K/V 避免每次重算整段注意力，使新 token 的计算量相对固定，是生成式推理加速的关键。",
    tags: ["Transformer", "Inference"],
  },


  {
    id: 104,
    type: "multiple",
    question: "卷积操作相比全连接层，参数数量大幅下降的关键原因有哪些？",
    options: [
      { text: "卷积核所有位置共享权重（参数共享）", correct: true },
      { text: "每个输出只连接输入的局部邻域（局部连接）", correct: true },
      { text: "卷积不需要梯度反向传播", correct: false },
      { text: "卷积核尺寸一定比图像小", correct: false },
    ],
    explanation:
      "参数共享让同一卷积核滑动时复用同一套权重，局部连接又让每个输出只看局部邻域，二者把参数量从全连接的 O(N*M) 压到 O(k*k*Cin)，是 CNN 高效的核心。",
    tags: ["CNN", "卷积原理"],
  },
  {
    id: 105,
    type: "single",
    question: "对 32×32 的输入用 5×5 卷积核做 valid 卷积（padding=0、stride=1），输出特征图尺寸是？",
    options: [
      { text: "28×28", correct: true },
      { text: "32×32", correct: false },
      { text: "36×36", correct: false },
      { text: "30×30", correct: false },
    ],
    explanation:
      "公式：输出 = (输入 - 核 + 2*pad) / stride + 1 = (32-5+0)/1+1 = 28。valid 卷积会把边缘裁掉一圈。",
    tags: ["CNN", "特征图尺寸"],
  },
  {
    id: 106,
    type: "single",
    question: "padding 的主要作用是什么？",
    options: [
      { text: "抵消卷积造成的尺寸缩小，保住边缘信息", correct: true },
      { text: "让卷积核变成正方形", correct: false },
      { text: "减小特征图的通道数", correct: false },
      { text: "提高卷积的计算速度", correct: false },
    ],
    explanation:
      "padding 在边缘补零，避免每层卷积都缩尺寸（可以保持 H、W 不变），也让原本只被扫到一次的边缘像素有更多参与机会。",
    tags: ["CNN", "padding"],
  },
  {
    id: 107,
    type: "single",
    question: "输入 7×7、卷积核 3×3、padding=1、stride=2，输出特征图尺寸是？",
    options: [
      { text: "4×4", correct: true },
      { text: "7×7", correct: false },
      { text: "3×3", correct: false },
      { text: "5×5", correct: false },
    ],
    explanation:
      "输出 = (7-3+2*1)/2 + 1 = 6/2+1 = 4。stride=2 会进行下采样，把尺寸约减半。",
    tags: ["CNN", "stride"],
  },
  {
    id: 108,
    type: "single",
    question: "感受野（receptive field）指的是什么？",
    options: [
      { text: "特征图某个像素能“看到”的输入图区域大小", correct: true },
      { text: "卷积核权重的总数", correct: false },
      { text: "padding 补零的宽度", correct: false },
      { text: "一层的输出通道数量", correct: false },
    ],
    explanation:
      "感受野是高层特征图上一个值对应回输入图像的范围。堆叠多层小卷积核能快速扩大感受野，让深层能看到更大的上下文。",
    tags: ["CNN", "感受野"],
  },
  {
    id: 109,
    type: "single",
    question: "连续堆叠 3 个 3×3 卷积（stride=1）得到的感受野，约等于一个多大的卷积？",
    options: [
      { text: "7×7", correct: true },
      { text: "9×9", correct: false },
      { text: "3×3", correct: false },
      { text: "6×6", correct: false },
    ],
    explanation:
      "每层 3×3 贡献 2 的扩张，三层约 1+2+2+2=7。这正是 VGG 用一堆小卷积核替代大核的做法——同感受野但参数量更省、非线性更强。",
    tags: ["CNN", "VGG"],
  },
  {
    id: 110,
    type: "single",
    question: "最大池化（max pooling）的主要作用是什么？",
    options: [
      { text: "取窗口最大值做下采样，保留最强响应、提供平移不变性", correct: true },
      { text: "对每个像素取平均值消除噪声", correct: false },
      { text: "增加特征图的分辨率", correct: false },
      { text: "在通道维度上压缩特征", correct: false },
    ],
    explanation:
      "max pooling 取局部区域最大响应，缩小特征图尺寸、减少计算，同时让网络对轻微平移更鲁棒。",
    tags: ["CNN", "池化"],
  },
  {
    id: 111,
    type: "single",
    question: "平均池化（average pooling）相比最大池化，更适合哪种场景？",
    options: [
      { text: "需要平滑、保留整体统计信息的场景（如全局平均池化做分类）", correct: true },
      { text: "需要保留边缘和纹理强响应的场景", correct: false },
      { text: "需要扩大感受野的下采样", correct: false },
      { text: "需要提升梯度的稀疏性", correct: false },
    ],
    explanation:
      "平均池化保留的是区域内的整体强度，而非极值。GAP（全局平均池化）常用在分类头前，把特征图压成向量、天然无全连接参数。",
    tags: ["CNN", "池化"],
  },
  {
    id: 112,
    type: "multiple",
    question: "关于 1×1 卷积，下列说法正确的是？",
    options: [
      { text: "可以在不改变空间尺寸的情况下调整通道数", correct: true },
      { text: "本质上是作用在每个空间位置上的全连接/逐点线性变换", correct: true },
      { text: "只能增加通道数，不能减少", correct: false },
      { text: "能替代所有池化操作", correct: false },
    ],
    explanation:
      "1×1 卷积只跨通道做线性组合、不动空间布局，所以既能升维也能降维，常用来做通道压缩（bottleneck）和跨通道融合。",
    tags: ["CNN", "1×1卷积"],
  },
  {
    id: 113,
    type: "single",
    question: "在 ResNet 的 bottleneck 结构里，先 1×1 降维再卷积再 1×1 升维的目的是？",
    options: [
      { text: "减少中间 3×3 卷积的计算量", correct: true },
      { text: "让网络变得更深但不做任何压缩", correct: false },
      { text: "强行提高输出的通道数", correct: false },
      { text: "代替 batch norm 做归一化", correct: false },
    ],
    explanation:
      "先用 1×1 把通道压小，3×3 在低维空间算，最后 1×1 再升回去，大幅省算力——这正是 bottleneck 相对原版残差块的效率来源。",
    tags: ["CNN", "ResNet"],
  },
  {
    id: 114,
    type: "single",
    question: "CNN 中“局部连接”指什么？",
    options: [
      { text: "每个输出只连接输入的一个局部邻域，而非全部输入", correct: true },
      { text: "只连接同一行的像素", correct: false },
      { text: "每层只连接相邻层", correct: false },
      { text: "只训练一部分权重", correct: false },
    ],
    explanation:
      "局部连接意味着输出单元只看到输入的一小块区域，配合参数共享，极大减少参数并天然契合图像的局部结构。",
    tags: ["CNN", "卷积原理"],
  },
  {
    id: 115,
    type: "single",
    question: "处理彩色 RGB 三通道输入时，第一个卷积层如何工作？",
    options: [
      { text: "卷积核深度等于输入通道数（此处为 3），对每个通道卷积后求和", correct: true },
      { text: "把三个通道拼接成灰度图再卷积", correct: false },
      { text: "每个通道分别训练独立的网络", correct: false },
      { text: "只用其中一个通道的信息", correct: false },
    ],
    explanation:
      "卷积核的第三个维度必须等于输入通道数（如 3），先逐通道卷积再相加得到单通道输出，多个核产生多个输出通道。",
    tags: ["CNN", "多通道卷积"],
  },
  {
    id: 116,
    type: "single",
    question: "输入 C_in 个通道、使用 K 个 3×3 卷积核，输出通道数是多少？",
    options: [
      { text: "K", correct: true },
      { text: "C_in", correct: false },
      { text: "C_in × K", correct: false },
      { text: "3 × K", correct: false },
    ],
    explanation:
      "每个卷积核负责产生一个输出通道，所以输出通道数等于卷积核个数 K。每个核内部深度为 C_in。",
    tags: ["CNN", "多通道卷积"],
  },
  {
    id: 117,
    type: "single",
    question: "转置卷积（transposed convolution）的主要用途是？",
    options: [
      { text: "把低分辨率特征图上采样到高分辨率", correct: true },
      { text: "对图像做降采样压缩", correct: false },
      { text: "替代池化以减少计算", correct: false },
      { text: "只用于卷积核权重可视化", correct: false },
    ],
    explanation:
      "转置卷积是卷积的反向操作，把特征图空间尺寸放大，常用于语义分割、生成模型（如 GAN）等需要上采样的场景。",
    tags: ["CNN", "转置卷积"],
  },
  {
    id: 118,
    type: "multiple",
    question: "关于空洞卷积（dilated convolution），下列说法正确的是？",
    options: [
      { text: "通过在核元素间插入空洞扩大感受野，但不增加参数量", correct: true },
      { text: "常用于分割等需要大感受野又不想牺牲分辨率的任务", correct: true },
      { text: "会降低输出特征图的空间分辨率", correct: false },
      { text: "空洞率越大，参数量就越多", correct: false },
    ],
    explanation:
      "空洞卷积把核元素拉开间隔，感受野随空洞率线性增长而参数量不变；被 DeepLab 等分割网络大量采用，保留分辨率的同时看更大范围。",
    tags: ["CNN", "空洞卷积"],
  },
  {
    id: 119,
    type: "single",
    question: "深度可分离卷积（depthwise separable convolution）的组成是？",
    options: [
      { text: "逐通道卷积 + 逐点（1×1）卷积", correct: true },
      { text: "两个普通卷积叠加", correct: false },
      { text: "卷积 + 最大池化", correct: false },
      { text: "卷积 + 全连接", correct: false },
    ],
    explanation:
      "它先用 depthwise 对每个通道单独卷积，再用 1×1 pointwise 跨通道融合，把标准卷积拆两步，计算量可大幅下降。",
    tags: ["CNN", "深度可分离卷积"],
  },
  {
    id: 120,
    type: "multiple",
    question: "深度可分离卷积相比标准卷积的优势包括？",
    options: [
      { text: "计算量显著减少", correct: true },
      { text: "参数量显著减少", correct: true },
      { text: "感受野成倍扩大", correct: false },
      { text: "精度必然提升", correct: false },
    ],
    explanation:
      "拆分后计算量约为标准卷积的 1/(K*K)+1/C，MobileNet 就是靠它在移动端跑得动。但感受野不变，精度未必更高，是靠计算换效率。",
    tags: ["CNN", "深度可分离卷积"],
  },
  {
    id: 121,
    type: "multiple",
    question: "残差连接（residual connection）为深层网络带来了哪些收益？",
    options: [
      { text: "缓解深层网络难以优化、退化（degradation）的问题", correct: true },
      { text: "让梯度能经恒等捷径直通浅层，缓解梯度消失", correct: true },
      { text: "解决数据类别不平衡的问题", correct: false },
      { text: "消除卷积核权重初始化的随机性", correct: false },
    ],
    explanation:
      "恒等捷径让梯度能直通浅层，既缓解梯度消失，也改善深层性能退化，这是 ResNet 能堆到上百层的基石。",
    tags: ["CNN", "ResNet", "残差连接"],
  },
  {
    id: 122,
    type: "single",
    question: "LeNet 主要由哪几类层构成？",
    options: [
      { text: "卷积层 + 池化层 + 全连接层", correct: true },
      { text: "只有全连接层", correct: false },
      { text: "自注意力层 + 卷积层", correct: false },
      { text: "RNN + 卷积层", correct: false },
    ],
    explanation:
      "LeNet 是最早成功的 CNN 之一，结构就是「卷积+池化」交替提取特征，末尾接全连接分类，后来 CNN 的基本套路。",
    tags: ["CNN", "LeNet"],
  },
  {
    id: 123,
    type: "multiple",
    question: "AlexNet 相对早期网络的关键创新有哪些？",
    options: [
      { text: "用 ReLU 替代 sigmoid/tanh 缓解梯度消失", correct: true },
      { text: "引入 Dropout 减少过拟合", correct: true },
      { text: "使用最大池化 + LRN 归一化", correct: true },
      { text: "引入注意力机制做长距离依赖", correct: false },
    ],
    explanation:
      "AlexNet 带来 ReLU、Dropout、GPU 并行等一批实用技巧，让深度学习在 2012 年 ImageNet 上一鸣惊人。它并没有注意力机制，那是后来的事。",
    tags: ["CNN", "AlexNet"],
  },
  {
    id: 124,
    type: "single",
    question: "VGG 的核心设计思想是什么？",
    options: [
      { text: "用一堆小 3×3 卷积堆叠加深网络，保持简单结构", correct: true },
      { text: "引入残差捷径加深网络", correct: false },
      { text: "用自注意力替换所有卷积", correct: false },
      { text: "用大卷积核减少层数", correct: false },
    ],
    explanation:
      "VGG 验证了「更深的简单 3×3 卷积」能显著涨点，结构规整、易于复现，但参数量偏大、计算开销高是它的短板。",
    tags: ["CNN", "VGG"],
  },
  {
    id: 125,
    type: "single",
    question: "CNN 用于目标检测（如 YOLO、Faster R-CNN）时，卷积骨干的作用是？",
    options: [
      { text: "提取多尺度特征图供检测头定位和分类", correct: true },
      { text: "直接输出目标的边界框坐标", correct: false },
      { text: "对整幅图做像素级分类", correct: false },
      { text: "生成图像的文本描述", correct: false },
    ],
    explanation:
      "检测网络用 CNN backbone 抽出特征，检测头（如 anchor/region proposal）再在特征图上找目标位置和类别。它不直接产框，框是后续头模块干的。",
    tags: ["CNN", "目标检测"],
  },
  {
    id: 126,
    type: "single",
    question: "语义分割（semantic segmentation）对网络的核心要求是？",
    options: [
      { text: "输出与输入同分辨率的逐像素类别预测", correct: true },
      { text: "只输出一个类别标签", correct: false },
      { text: "输出多个候选框", correct: false },
      { text: "生成一段描述文本", correct: false },
    ],
    explanation:
      "语义分割要每个像素都给出类别，因此网络用编码器下采样 + 解码器/上采样（如转置卷积、空洞卷积）把特征恢复到原图尺寸。",
    tags: ["CNN", "语义分割"],
  },
  {
    id: 127,
    type: "multiple",
    question: "图像分类、目标检测、语义分割三者的区别，正确的是？",
    options: [
      { text: "分类只输出整图的类别", correct: true },
      { text: "检测输出目标的位置框 + 类别", correct: true },
      { text: "分割是对每个像素做类别预测", correct: true },
      { text: "三种任务输出格式完全相同", correct: false },
    ],
    explanation:
      "三者的输出粒度不同：分类是整图一个标签，检测是「框+类别」集合，分割是像素级标签图。任务粒度越细，网络通常越复杂。",
    tags: ["CNN", "任务对比"],
  },
  {
    id: 128,
    type: "single",
    question: "在 CNN 分类网络中，全连接层通常放在哪里？",
    options: [
      { text: "网络末尾，把特征图展平映射到类别得分", correct: true },
      { text: "网络开头，先对原始像素做变换", correct: false },
      { text: "卷积层之间的每一层", correct: false },
      { text: "池化层之前负责下采样", correct: false },
    ],
    explanation:
      "全连接层位于尾部，把展平后的高维特征映射到类别数。现代网络也常用全局平均池化替代它，进一步省参数。",
    tags: ["CNN", "全连接层"],
  },
  {
    id: 129,
    type: "single",
    question: "batch normalization（BN）放在卷积层之后的主要作用是？",
    options: [
      { text: "对每个通道的激活做归一化，稳定训练、加速收敛", correct: true },
      { text: "增加卷积核的深度", correct: false },
      { text: "改变特征图的空间分辨率", correct: false },
      { text: "替代 dropout 的全部功能", correct: false },
    ],
    explanation:
      "BN 对每个通道在 batch 上归一化均值方差，缓解内部协变量偏移，允许用更大学习率、更稳定地训练深层 CNN。",
    tags: ["CNN", "归一化"],
  },
  {
    id: 130,
    type: "single",
    question: "卷积层为什么天然具有“平移等变性”（translation equivariance）？",
    options: [
      { text: "因为同一卷积核在图像各处共享权重，输入平移则输出特征同步平移", correct: true },
      { text: "因为 padding 固定了图像边缘", correct: false },
      { text: "因为池化保留了最大值位置", correct: false },
      { text: "因为卷积核的权重会随输入动态变化", correct: false },
    ],
    explanation:
      "参数共享 + 滑动卷积意味着物体挪位置，激活图也相应挪位，即输出随输入平移而平移。这是 CNN 理解图像位置的天然特性。",
    tags: ["CNN", "卷积原理"],
  },
  {
    id: 131,
    type: "single",
    question: "CNN 相对 Transformer 在图像处理上最突出的归纳偏置是？",
    options: [
      { text: "局部性与平移等变性（locality & equivariance）", correct: true },
      { text: "全局自注意力建模长距离关系", correct: false },
      { text: "位置编码注入空间顺序", correct: false },
      { text: "多头注意力并行计算", correct: false },
    ],
    explanation:
      "CNN 天生只在局部邻域做卷积，参数共享 + 局部连接给它带来强归纳偏置，数据少时更易学、更省样本。Transformer 则靠注意力覆盖全局，偏置弱但上限高。",
    tags: ["CNN", "CNN vs Transformer"],
  },
  {
    id: 132,
    type: "multiple",
    question: "对比 CNN 与 ViT，下列说法正确的是？",
    options: [
      { text: "CNN 依赖局部归纳偏置，ViT 用自注意力捕捉全局依赖", correct: true },
      { text: "ViT 通常需要更大的数据集或预训练才能收敛好", correct: true },
      { text: "CNN 完全不涉及全局信息", correct: false },
      { text: "ViT 不需要任何位置信息", correct: false },
    ],
    explanation:
      "ViT 没有 CNN 的局部偏置，靠自注意力看全局，但因此更吃数据；它仍需位置编码（position embedding）告诉模型 patch 的顺序。",
    tags: ["CNN", "CNN vs Transformer"],
  },
  {
    id: 133,
    type: "single",
    question: "在视觉语言模型（VLM，如 CLIP、BLIP）中，CNN 骨干（如 ResNet）扮演什么角色？",
    options: [
      { text: "图像塔（vision tower），把图像编码成视觉特征", correct: true },
      { text: "把文本编码成 token", correct: false },
      { text: "生成图像", correct: false },
      { text: "只负责分类文本情感", correct: false },
    ],
    explanation:
      "VLM 通常双塔结构：图像塔用 CNN（或 ViT）编码图片特征，文本塔用 Transformer 编码文本，再对齐二者。CNN 负责把原始像素变成有语义的视觉表示。",
    tags: ["CNN", "VLM"],
  },
  {
    id: 134,
    type: "multiple",
    question: "CNN 相比全连接网络的核心优势有哪些？",
    options: [
      { text: "参数共享大幅减少参数量", correct: true },
      { text: "局部连接契合图像局部结构", correct: true },
      { text: "天然支持变长序列输入", correct: false },
      { text: "天然具备平移等变性", correct: true },
    ],
    explanation:
      "参数共享、局部连接、平移等变是 CNN 的三大法宝，让它高效处理图像。但 CNN 输入尺寸通常是固定的网格，天然支持变长序列是 RNN/Transformer 的优势。",
    tags: ["CNN", "卷积原理"],
  },
  {
    id: 135,
    type: "single",
    question: "stride=1、padding='same' 的 3×3 卷积后，特征图尺寸与输入相比？",
    options: [
      { text: "保持不变", correct: true },
      { text: "缩小 2 像素", correct: false },
      { text: "放大 2 像素", correct: false },
      { text: "减半", correct: false },
    ],
    explanation:
      "same padding 会补足够的零，让 stride=1 的卷积保持 H、W 不变；只有 stride>1 或 valid 卷积才会缩小。",
    tags: ["CNN", "padding"],
  },
  {
    id: 136,
    type: "multiple",
    question: "为了扩大 CNN 的感受野，可行的手段包括？",
    options: [
      { text: "堆叠更多卷积层", correct: true },
      { text: "使用更大尺寸的卷积核", correct: true },
      { text: "使用空洞卷积", correct: true },
      { text: "增加池化的窗口大小但减少层数", correct: false },
    ],
    explanation:
      "堆层、加大核、用空洞卷积都能扩大感受野。单纯把池化窗口变大只是单层下采样，若不同时堆叠层，感受野提升有限。",
    tags: ["CNN", "感受野"],
  },
  {
    id: 137,
    type: "single",
    question: "Global Average Pooling（全局平均池化）相比传统全连接分类头的优势是？",
    options: [
      { text: "把每个通道压成一个值，参数更少且不易过拟合", correct: true },
      { text: "显著提升特征图的分辨率", correct: false },
      { text: "能让网络接受任意通道数", correct: false },
      { text: "替代了卷积层的全部功能", correct: false },
    ],
    explanation:
      "GAP 对每个通道做全局平均，得到与通道数等长的向量直接接分类，去掉了展平+全连接的大量参数，也更鲁棒。",
    tags: ["CNN", "全连接层"],
  },
  {
    id: 138,
    type: "single",
    question: "在 U-Net 结构中，跳跃连接（skip connection）的作用是？",
    options: [
      { text: "把编码器的细粒度空间信息传给解码器，保住边缘细节", correct: true },
      { text: "把解码器输出直接当输入", correct: false },
      { text: "减少网络的参数量", correct: false },
      { text: "加速卷积运算的并行度", correct: false },
    ],
    explanation:
      "分割网络下采样时丢了精细位置信息，跳跃连接把编码器同尺度特征拼到解码器，恢复细节，是 U-Net 在医学分割表现好的关键。",
    tags: ["CNN", "U-Net", "语义分割"],
  },
  {
    id: 139,
    type: "single",
    question: "Mobilenet、EfficientNet 等轻量网络关注的核心指标是？",
    options: [
      { text: "在尽量少参数/算力的前提下保持精度", correct: true },
      { text: "只要精度最高，不管算力", correct: false },
      { text: "尽可能增加网络深度", correct: false },
      { text: "只处理灰度图像", correct: false },
    ],
    explanation:
      "移动端/边缘设备算力有限，这些网络用深度可分离卷积、NAS 搜索等手法在精度与效率间取平衡。",
    tags: ["CNN", "轻量网络"],
  },
  {
    id: 140,
    type: "multiple",
    question: "防止 CNN 训练过拟合的常见手段包括？",
    options: [
      { text: "数据增强（翻转、裁剪、颜色扰动）", correct: true },
      { text: "Dropout 随机丢弃神经元", correct: true },
      { text: "权重衰减（weight decay / L2）", correct: true },
      { text: "不加任何正则直接加深网络", correct: false },
    ],
    explanation:
      "数据增强、Dropout、权重衰减都是经典正则手段，能提高泛化。单纯加深网络反而更易过拟合，不解决根本问题。",
    tags: ["CNN", "正则化"],
  },
  {
    id: 141,
    type: "single",
    question: "CNN 分类网络最后一层通常用什么激活函数输出类别概率？",
    options: [
      { text: "Softmax", correct: true },
      { text: "ReLU", correct: false },
      { text: "Sigmoid（单输出）", correct: false },
      { text: "Tanh", correct: false },
    ],
    explanation:
      "多分类用 softmax 把 logits 归一化成各类别上的概率分布；二分类也可用 sigmoid。ReLU、tanh 常在隐藏层用。",
    tags: ["CNN", "分类"],
  },
  {
    id: 142,
    type: "multiple",
    question: "关于感受野、参数共享、下采样三者的关系，正确的是？",
    options: [
      { text: "下采样（stride/池化）能扩大后续层的感受野", correct: true },
      { text: "参数共享不改变感受野大小", correct: true },
      { text: "参数共享会缩小感受野", correct: false },
      { text: "感受野越大一定分辨率越低", correct: false },
    ],
    explanation:
      "下采样让每个高层像素覆盖更大输入区域，从而扩大感受野；参数共享只是复用权重，不影响感受野。感受野大≠分辨率低，二者是不同概念。",
    tags: ["CNN", "感受野"],
  },
  {
    id: 143,
    type: "single",
    question: "在 Transformer 时代，CNN 骨干在图像任务中仍然重要的原因之一是？",
    options: [
      { text: "作为特征金字塔或下采样模块提供多尺度与效率，且数据效率更高", correct: true },
      { text: "完全被注意力取代、已无任何用途", correct: false },
      { text: "只能处理小图不能处理大图", correct: false },
      { text: "不能做特征提取", correct: false },
    ],
    explanation:
      "很多 ViT 系网络仍把 CNN 当下采样 stem（如 Patch Embedding 前的卷积）、或做多尺度金字塔（如 HRNet），且 CNN 在数据少时收敛更好，作用并未消失。",
    tags: ["CNN", "CNN vs Transformer"],
  },

// 题型说明：
//   type: "single" 单选（只有一个正确选项）
//   type: "multiple" 多选（有多个正确选项）
//   options: 选项列表，{ text: 选项内容, correct: 是否为正确选项 }
//   explanation: 答案解析，答完后展示
//   tags: 题目所属知识点标签（用于展示）


  {
    id: 144,
    type: "single",
    question: "因果语言建模（causal LM）与掩码语言建模（masked LM）的核心区别是什么？",
    options: [
      { text: "因果 LM 只能看左侧上文、不能看未来 token；掩码 LM 能看到双向上下文", correct: true },
      { text: "因果 LM 用全词掩码，掩码 LM 用子词掩码", correct: false },
      { text: "因果 LM 不需要 tokenizer，掩码 LM 需要", correct: false },
      { text: "因果 LM 只做分类，掩码 LM 只做生成", correct: false },
    ],
    explanation:
      "因果 LM 做 next token prediction，注意力只允许看左侧，防止「偷看答案」；掩码 LM（如 BERT）随机盖住 token 后从双向上下文还原，两者建模目标完全不同。",
    tags: ["LLM", "Pretraining"],
  },
  {
    id: 145,
    type: "single",
    question: "预训练阶段的训练目标（objective）通常是下列哪一项？",
    options: [
      { text: "最大化下一个 token 的对数似然", correct: true },
      { text: "最小化整段文本的重构均方误差", correct: false },
      { text: "最大化句子的余弦相似度", correct: false },
      { text: "最小化输出分布的 KL 散度到某个教师模型", correct: false },
    ],
    explanation:
      "预训练就是自回归式最大化每个位置下一个 token 的对数似然，等价于最小化交叉熵损失，目标是学会语言本身的统计规律。",
    tags: ["LLM", "Pretraining"],
  },
  {
    id: 146,
    type: "single",
    question: "训练一个 epoch 的含义是？",
    options: [
      { text: "模型对全部训练数据完整过一遍", correct: true },
      { text: "跑完一个 batch 的梯度更新", correct: false },
      { text: "学习率衰减一轮", correct: false },
      { text: "训练完一层网络", correct: false },
    ],
    explanation:
      "一个 epoch 就是把整个训练集完整过一遍。预训练数据量极大，通常远不到一个 epoch 就收敛，所以常用「看到多少 token」而非 epoch 来计量。",
    tags: ["LLM", "Training"],
  },
  {
    id: 147,
    type: "single",
    question: "预训练语言模型通常「看到 token 数」远大于 1 个 epoch，原因是？",
    options: [
      { text: "数据量极大，重复过完一遍成本过高且容易过拟合", correct: true },
      { text: "必须保证每个样本只被看到一次", correct: false },
      { text: "epoch 数量与模型参数量成正比", correct: false },
      { text: "训练数据是流式实时生成的", correct: false },
    ],
    explanation:
      "预训练语料是 TB 级海量文本，完整过一遍成本高，且模型通常没看完一遍就够学了，重复太多反而过拟合（chinchilla 定律就是权衡数据量）。",
    tags: ["LLM", "Training"],
  },
  {
    id: 148,
    type: "single",
    question: "AdamW 相比 Adam 的核心改动是什么？",
    options: [
      { text: "把权重衰减从梯度中解耦，直接作用在参数上", correct: true },
      { text: "引入动量项平滑梯度", correct: false },
      { text: "用滑动平均替代梯度的单调累加", correct: false },
      { text: "为每个参数分配独立的学习率", correct: false },
    ],
    explanation:
      "Adam 把权重衰减混进梯度，会被自适应缩放干扰；AdamW 把 weight decay 单独从参数上扣，与自适应机制解耦，训练更稳。",
    tags: ["Optimizer", "Training"],
  },
  {
    id: 149,
    type: "single",
    question: "权重衰减（weight decay）的主要作用是什么？",
    options: [
      { text: "惩罚过大的权重，抑制过拟合", correct: true },
      { text: "加速梯度收敛到局部极小", correct: false },
      { text: "扩大模型容量以拟合更多数据", correct: false },
      { text: "把权重归一化到单位范数", correct: false },
    ],
    explanation:
      "weight decay 在更新时按比例把权重往零压，相当于给参数加了 L2 正则，权重不轻易变大，从而抑制过拟合。",
    tags: ["Optimizer", "Regularization"],
  },
  {
    id: 150,
    type: "single",
    question: "学习率 warmup 阶段，学习率的变化趋势是？",
    options: [
      { text: "从很小的值逐渐上升到一个目标值", correct: true },
      { text: "从目标值线性衰减到零", correct: false },
      { text: "全程保持不变", correct: false },
      { text: "从最大值突然降到零再回升", correct: false },
    ],
    explanation:
      "warmup 让学习率从 0（或很小）线性升到峰值，避免训练初期参数还没稳就大步走，导致 loss 冲高甚至发散。",
    tags: ["Optimizer", "Training"],
  },
  {
    id: 151,
    type: "single",
    question: "为什么大模型训练需要学习率 warmup？",
    options: [
      { text: "初期 Adam 的二阶矩估计不准，大步更新易导致 loss spike", correct: true },
      { text: "可以跳过梯度裁剪步骤", correct: false },
      { text: "为了在更短时间跑完更多数据", correct: false },
      { text: "让 GPU 显存占用先降下来", correct: false },
    ],
    explanation:
      "训练起步时梯度和二阶矩统计都是噪声，自适应优化器估计不准，直接上大步长容易让 loss 飙升（spike），warmup 就是先小步走稳再加速。",
    tags: ["Optimizer", "Training"],
  },
  {
    id: 152,
    type: "single",
    question: "余弦退火（cosine schedule）学习率调度的特征是？",
    options: [
      { text: "学习率先保持稳定，再按余弦曲线平滑降到零", correct: true },
      { text: "学习率全程指数增长", correct: false },
      { text: "学习率在每一轮 epoch 末尾突然翻倍", correct: false },
      { text: "学习率只升不降", correct: false },
    ],
    explanation:
      "cosine schedule 通常先 warmup 到峰值，再沿余弦曲线平滑衰减到接近 0，后期小步长有助于收敛到更优的平坦区，泛化更好。",
    tags: ["Optimizer", "Training"],
  },
  {
    id: 153,
    type: "single",
    question: "梯度累积（gradient accumulation）解决的核心问题是什么？",
    options: [
      { text: "单个 batch 过大放不进显存，用多个小 batch 累加梯度模拟大 batch", correct: true },
      { text: "减少需要训练的 epoch 数量", correct: false },
      { text: "避免梯度消失", correct: false },
      { text: "替代权重初始化", correct: false },
    ],
    explanation:
      "大 batch 一次放不进 GPU 显存时，就把 batch 拆成几个 micro-batch 分别前反向，把梯度累加若干步再统一更新参数，效果近似等于一个大 batch。",
    tags: ["Training", "Efficiency"],
  },
  {
    id: 154,
    type: "single",
    question: "关于梯度累积，下列说法正确的是？",
    options: [
      { text: "累加 N 个 micro-batch 的梯度后统一做一次参数更新", correct: true },
      { text: "每个 micro-batch 都单独更新一次参数", correct: false },
      { text: "累加的是 loss 而不是梯度", correct: false },
      { text: "梯度累积后无需再除以累积步数", correct: false },
    ],
    explanation:
      "梯度累积是攒 N 步梯度、除 N 取平均后再更新，模拟大 batch；若每步都更新就退化成普通小 batch 了。",
    tags: ["Training", "Efficiency"],
  },
  {
    id: 155,
    type: "single",
    question: "梯度裁剪（gradient clipping）的主要目的是？",
    options: [
      { text: "限制梯度范数，防止梯度爆炸导致训练不稳定", correct: true },
      { text: "让梯度变得完全为零", correct: false },
      { text: "提高混合精度下的计算速度", correct: false },
      { text: "强制所有参数更新量一致", correct: false },
    ],
    explanation:
      "一旦梯度范数超过阈值就整体缩放截断，把最大更新量控制在安全范围内，防止 loss 冲高（spike）甚至 NaN。",
    tags: ["Training", "Stability"],
  },
  {
    id: 156,
    type: "single",
    question: "bf16 与 fp16 同为 16 位浮点，二者的本质区别是什么？",
    options: [
      { text: "bf16 指数量更多、范围大但精度差；fp16 尾数更多、范围小但精度好", correct: true },
      { text: "bf16 是定点数，fp16 是浮点数", correct: false },
      { text: "bf16 只能用于训练，fp16 只能用于推理", correct: false },
      { text: "二者数值范围与精度完全相同，仅命名不同", correct: false },
    ],
    explanation:
      "bf16 用 8 指数位换了大范围但尾数少（精度差）；fp16 尾数多精度好但范围小。一句话：bf16 用精度换范围，fp16 用范围换精度。",
    tags: ["Mixed Precision", "Training"],
  },
  {
    id: 157,
    type: "single",
    question: "混合精度训练（如 bf16）中，通常需要一份「主权重」保持在什么精度？",
    options: [
      { text: "fp32，用于累积更新，避免精度丢失", correct: true },
      { text: "bf16，与计算精度一致", correct: false },
      { text: "int8，进一步省显存", correct: false },
      { text: "fp64，追求最高精度", correct: false },
    ],
    explanation:
      "16 位前反向计算省显存，但权重更新累积在小精度上误差会越滚越大，所以保留一份 fp32 主权重做更新，推理或保存时再转回低精度。",
    tags: ["Mixed Precision", "Training"],
  },
  {
    id: 158,
    type: "single",
    question: "损失函数（loss）在前向计算中通常由什么产生？",
    options: [
      { text: "对每个位置计算预测分布与真实 token 的交叉熵", correct: true },
      { text: "对所有参数做 L2 范数求和", correct: false },
      { text: "对整句做词袋余弦距离", correct: false },
      { text: "对输出向量做 softmax 前的平均", correct: false },
    ],
    explanation:
      "自回归模型每个位置都要预测下一个 token，用交叉熵度量预测分布和真实 token 的差距，通常对整段求平均作为 loss。",
    tags: ["LLM", "Loss"],
  },
  {
    id: 159,
    type: "single",
    question: "交叉熵损失用于分类时，其正确形式通常是？",
    options: [
      { text: "对真实类别的负对数概率求和", correct: true },
      { text: "对所有类别的概率求和", correct: false },
      { text: "对真实类别概率取绝对值", correct: false },
      { text: "对类别概率做平方差", correct: false },
    ],
    explanation:
      "交叉熵 = -Σ y·log(p)，只有真实类别 y=1 项保留，就是「负对数概率」，最小化它等价于让正确类别的概率最大。",
    tags: ["Loss", "LLM"],
  },
  {
    id: 160,
    type: "single",
    question: "tokenizer 中「词表（vocabulary）」指的是什么？",
    options: [
      { text: "模型能表示的所有 token 的集合", correct: true },
      { text: "训练时看到的所有句子", correct: false },
      { text: "所有词向量矩阵的行列数量", correct: false },
      { text: "只包含标点符号的集合", correct: false },
    ],
    explanation:
      "词表就是 tokenizer 定义的那套基础单位（子词/subword），文本被切成词表里的 token 才能送入模型，embedding 和输出层的维度就由词表大小决定。",
    tags: ["Tokenizer", "LLM"],
  },
  {
    id: 161,
    type: "single",
    question: "用 BPE（Byte Pair Encoding）切词相比整词切分的主要好处是？",
    options: [
      { text: "能在更小词表下覆盖海量词汇，兼顾 OOV（未登录词）", correct: true },
      { text: "让每个 token 恰好等于一个汉字", correct: false },
      { text: "彻底消除所有子词", correct: false },
      { text: "把词表大小增加到无限", correct: false },
    ],
    explanation:
      "BPE 从字符不断合并高频对，生成子词级词表，常用词整词、生僻词拆成子词，能大幅压缩词表又能表达任何文本。",
    tags: ["Tokenizer", "LLM"],
  },
  {
    id: 162,
    type: "multiple",
    question: "关于上下文窗口（context window），下列哪些说法正确？（多选）",
    options: [
      { text: "它限制模型一次能处理的输入 token 数量", correct: true },
      { text: "超过窗口的文本无法进入注意力层", correct: true },
      { text: "它等于模型的隐藏层参数量", correct: false },
      { text: "它决定了单次推理的输出词表大小", correct: false },
    ],
    explanation:
      "上下文窗口限制输入序列长度，超过的部分根本送不进注意力层，长文本处理上限由它决定，与参数量、词表无关。",
    tags: ["LLM", "Context"],
  },
  {
    id: 163,
    type: "single",
    question: "位置编码（position encoding）要解决的核心问题是？",
    options: [
      { text: "让模型区分相同 token 在不同位置的含义", correct: true },
      { text: "消除注意力里的矩阵维度不一致", correct: false },
      { text: "降低训练的内存占用", correct: false },
      { text: "把词表大小压缩一半", correct: false },
    ],
    explanation:
      "自注意力是置换不变的，同一批 token 换个顺序结果一样，这不行。位置编码把位置信息注入输入，让「我在第 3 位」和「我在第 8 位」能被区分开。",
    tags: ["LLM", "Position Encoding"],
  },
  {
    id: 164,
    type: "single",
    question: "RoPE（旋转位置编码）的核心思想是？",
    options: [
      { text: "把位置信息编码成向量旋转角度，相对位置通过角度差表达", correct: true },
      { text: "把位置信息加到词向量上", correct: false },
      { text: "用正弦余弦表拼接所有位置", correct: false },
      { text: "完全去掉位置编码", correct: false },
    ],
    explanation:
      "RoPE 把每个位置当成一个旋转角，让 query/key 按位置旋转，于是相对位置（差多少位）自然变成角度差，且对长度外推更友好，现在主流 LLM 基本都在用。",
    tags: ["LLM", "Position Encoding"],
  },
  {
    id: 165,
    type: "single",
    question: "KV cache 的核心作用是什么？",
    options: [
      { text: "缓存已算好的 Key 和 Value，推理时避免重复计算前面 token 的注意力", correct: true },
      { text: "缓存训练时的梯度", correct: false },
      { text: "保存优化器的动量状态", correct: false },
      { text: "存储 tokenizer 的词表", correct: false },
    ],
    explanation:
      "生成每个新 token 时，前面 token 的 Key/Value 其实没变，缓存起来就不用每步重算整段，这是推理加速的关键。注意它是推理优化，与训练本身无直接关系。",
    tags: ["LLM", "Inference"],
  },
  {
    id: 166,
    type: "multiple",
    question: "关于激活重计算/梯度检查点，下列哪些说法正确？（多选）",
    options: [
      { text: "它不保存全部中间激活，从而节省显存", correct: true },
      { text: "反向传播前需要重新前向来恢复激活", correct: true },
      { text: "它同时降低显存占用和训练时间", correct: false },
      { text: "它是训练超大模型的常用省显存手段", correct: true },
    ],
    explanation:
      "checkpointing 不存中间激活、反传时重算一遍，省显存但多耗算力，是显存与速度的权衡，大模型训练几乎必用。",
    tags: ["Training", "Memory"],
  },
  {
    id: 167,
    type: "multiple",
    question: "关于梯度检查点（gradient checkpointing）的代价与收益，下列哪些说法正确？（多选）",
    options: [
      { text: "以增加部分计算量为代价换取显存下降", correct: true },
      { text: "反向传播前要重新前向以恢复激活", correct: true },
      { text: "它会降低模型精度", correct: false },
      { text: "它常用于训练放不下的超大模型", correct: true },
    ],
    explanation:
      "反向传播前要重新前向一次恢复激活，计算量上升但显存大幅下降，不损精度，是省显存训大模型的核心手段。",
    tags: ["Training", "Memory"],
  },
  {
    id: 168,
    type: "multiple",
    question: "关于 MoE（专家混合）稀疏训练，下列哪些说法正确？（多选）",
    options: [
      { text: "每个 token 只经过被路由器选中的少量专家", correct: true },
      { text: "路由器（router）给 token 选 top-k 个专家", correct: true },
      { text: "每个 token 必须经过全部专家层", correct: false },
      { text: "参数量大但单 token 计算量相对小", correct: true },
    ],
    explanation:
      "MoE 用 router 给每个 token 选 top-k 专家，计算只走这部分，参数多但每 token 激活少，用稀疏激活换大容量。",
    tags: ["MoE", "Training"],
  },
  {
    id: 169,
    type: "single",
    question: "MoE 训练中常见的「专家负载均衡（load balancing）」问题是？",
    options: [
      { text: "部分专家被频繁选中、另一些被冷落，导致负载不均", correct: true },
      { text: "专家层无法计算梯度", correct: false },
      { text: "专家之间参数共享导致退化", correct: false },
      { text: "专家层不参与反向传播", correct: false },
    ],
    explanation:
      "router 容易「越选越顺手」让少数专家被狂选，其余专家荒废。所以要加负载均衡损失（aux loss）鼓励均匀分配，防止训练失衡。",
    tags: ["MoE", "Training"],
  },
  {
    id: 170,
    type: "multiple",
    question: "关于数据并行（data parallelism），下列哪些说法正确？（多选）",
    options: [
      { text: "每张卡持有一份完整模型副本，各处理不同的数据分片", correct: true },
      { text: "通过 all-reduce 同步各卡的梯度", correct: true },
      { text: "数据并行能有效扩展等效 batch size", correct: true },
      { text: "每张卡只负责模型的一部分层", correct: false },
    ],
    explanation:
      "数据并行把数据切给多卡，每卡一份完整模型、各算各的 batch，再 all-reduce 同步梯度，等效 batch 变大。只分层的并行是流水线并行。",
    tags: ["Parallelism", "Distributed"],
  },
  {
    id: 171,
    type: "multiple",
    question: "关于张量并行（tensor parallelism），下列哪些说法正确？（多选）",
    options: [
      { text: "把单个权重矩阵（如注意力/FFN）切分到多张卡", correct: true },
      { text: "切分后需通信合并部分计算结果", correct: true },
      { text: "用于应对单卡放不下单个超大权重", correct: true },
      { text: "它是把训练数据分给多卡", correct: false },
    ],
    explanation:
      "张量并行把单个大矩阵切块放多卡、算完通信合并，专治「单个权重放不下一张卡」；分数据的属于数据并行。",
    tags: ["Parallelism", "Distributed"],
  },
  {
    id: 172,
    type: "multiple",
    question: "关于流水线并行（pipeline parallelism），下列哪些说法正确？（多选）",
    options: [
      { text: "把模型按层切分成若干段分给多卡", correct: true },
      { text: "数据按段顺序流经各卡", correct: true },
      { text: "每张卡只需容纳一部分层，能装下巨型模型", correct: true },
      { text: "每张卡处理不同的独立样本", correct: false },
    ],
    explanation:
      "流水线并行把模型按层分段，卡 A 跑前段、卡 B 跑后段，数据顺序流过，每卡只扛一部分层；各卡处理不同样本是数据并行。",
    tags: ["Parallelism", "Distributed"],
  },
  {
    id: 173,
    type: "single",
    question: "ZeRO（Zero Redundancy Optimizer）主要优化的是哪部分显存？",
    options: [
      { text: "优化器状态、梯度乃至参数在数据并行各副本间去冗余分片", correct: true },
      { text: "把权重全部转成 int8", correct: false },
      { text: "只压缩中间激活值", correct: false },
      { text: "减少训练数据的加载", correct: false },
    ],
    explanation:
      "数据并行每卡都存一份完整的优化器状态、梯度和参数，冗余严重；ZeRO 把它们按卡切分、各管各的，从根源上省掉大量显存。",
    tags: ["Memory", "Distributed"],
  },
  {
    id: 174,
    type: "single",
    question: "ZeRO-3 相比 ZeRO-1 还额外分片了哪部分？",
    options: [
      { text: "把模型参数也切分到各卡，用时再收集", correct: true },
      { text: "把训练数据也分片", correct: false },
      { text: "把词表也分片", correct: false },
      { text: "把损失函数也分片", correct: false },
    ],
    explanation:
      "ZeRO-1 只分片优化器状态，ZeRO-3 连参数都分片（用时 all-gather），能训远超单卡容量的模型，代价是通信量更大。",
    tags: ["Memory", "Distributed"],
  },
  {
    id: 175,
    type: "single",
    question: "判断过拟合最直观的依据是？",
    options: [
      { text: "训练 loss 持续下降但验证 loss 开始上升", correct: true },
      { text: "训练和验证 loss 同时下降", correct: false },
      { text: "训练 loss 为负数", correct: false },
      { text: "训练 loss 与验证 loss 完全相同", correct: false },
    ],
    explanation:
      "过拟合就是模型把训练集背下来了：训练集表现越来越好（loss 降），一到没见过的验证集就露馅（loss 涨），说明泛化差。",
    tags: ["Generalization", "Training"],
  },
  {
    id: 176,
    type: "single",
    question: "SFT（指令微调）的核心训练方式是什么？",
    options: [
      { text: "在指令-回答数据上继续做监督学习，学习遵循指令的输出", correct: true },
      { text: "只用强化学习信号调整", correct: false },
      { text: "不更新任何参数", correct: false },
      { text: "只在未标注数据上自监督", correct: false },
    ],
    explanation:
      "SFT 用 (指令, 期望回答) 这种监督数据继续训练（通常是 next token 交叉熵），让模型从「会接文本」变成「会听指令回答」。",
    tags: ["SFT", "Fine-tuning"],
  },
  {
    id: 177,
    type: "single",
    question: "RLHF（基于人类反馈的强化学习）相比纯 SFT 的额外价值是？",
    options: [
      { text: "用人类偏好信号进一步对齐价值观与有用性，而非仅模仿文本", correct: true },
      { text: "让模型参数量翻倍", correct: false },
      { text: "替代 tokenizer", correct: false },
      { text: "消除所有位置编码", correct: false },
    ],
    explanation:
      "SFT 只是模仿「人怎么写」，RLHF 还根据「人更喜欢哪个回答」用奖励信号优化，能更好地对齐真实偏好。",
    tags: ["RLHF", "Alignment"],
  },
  {
    id: 178,
    type: "multiple",
    question: "关于 DPO（直接偏好优化）与 PPO，下列哪些说法正确？（多选）",
    options: [
      { text: "DPO 用闭式损失直接优化偏好，无需单独训练奖励模型", correct: true },
      { text: "DPO 直接基于偏好对（chosen/rejected）构造目标", correct: true },
      { text: "PPO 通常需要额外训练一个奖励模型", correct: true },
      { text: "DPO 完全不需要任何偏好数据", correct: false },
    ],
    explanation:
      "PPO 得先训奖励模型再 RL 采样训练，繁琐；DPO 直接基于偏好对推导闭式损失，一步到位对齐偏好，更简单高效。",
    tags: ["DPO", "Alignment"],
  },
  {
    id: 179,
    type: "single",
    question: "Rejection Sampling（拒绝采样）在训练中的作用是？",
    options: [
      { text: "对同一 prompt 采样多个回答，只选奖励最高的作为训练目标", correct: true },
      { text: "随机丢弃掉一半训练数据", correct: false },
      { text: "拒绝梯度大于阈值的更新", correct: false },
      { text: "强制模型只输出单选答案", correct: false },
    ],
    explanation:
      "同一个问题让模型采样多个候选回答，用奖励模型打分，只挑最高分的去当正例继续训练（如用于迭代式 RLHF/SFT），质量更高。",
    tags: ["RLHF", "Training"],
  },
  {
    id: 180,
    type: "single",
    question: "蒸馏（distillation）中「教师模型」的作用是？",
    options: [
      { text: "提供软标签/概率分布作为学生模型的学习目标", correct: true },
      { text: "提供最终推理算力", correct: false },
      { text: "负责给训练数据打标签做预训练", correct: false },
      { text: "替代优化器", correct: false },
    ],
    explanation:
      "教师模型是「高材生」，输出软化的概率分布（不只是硬标签）给学生学，学生用更小规模逼近教师的泛化能力。",
    tags: ["Distillation", "LLM"],
  },
  {
    id: 181,
    type: "multiple",
    question: "关于 LoRA 微调，下列哪些说法正确？（多选）",
    options: [
      { text: "冻结原始权重，只训练低秩增量矩阵 ΔW", correct: true },
      { text: "可训练参数量极小，省显存", correct: true },
      { text: "推理时可将 ΔW 合并回原权重，不增推理成本", correct: true },
      { text: "必须全量重训整个基座模型", correct: false },
    ],
    explanation:
      "LoRA 冻结原权重、只训 ΔW=B×A，参数量骤减；推理时把 ΔW 合并回 W 不增加开销，换任务只需换低秩矩阵。",
    tags: ["LoRA", "PEFT"],
  },
  {
    id: 182,
    type: "single",
    question: "关于 LoRA 注入的低秩矩阵 ΔW，正确的是？",
    options: [
      { text: "ΔW 是两个低秩矩阵的乘积，训练完可合并回原权重", correct: true },
      { text: "ΔW 是满秩且与 W 同尺寸", correct: false },
      { text: "ΔW 直接覆盖掉原始权重", correct: false },
      { text: "ΔW 不参与任何梯度计算", correct: false },
    ],
    explanation:
      "ΔW = B×A，B、A 都是低秩小矩阵，只训这俩；推理时可把 ΔW 合并回 W，不增加推理成本。",
    tags: ["LoRA", "PEFT"],
  },
  {
    id: 183,
    type: "single",
    question: "PEFT（参数高效微调）的共同点是？",
    options: [
      { text: "只更新一小部分参数，大幅降低微调算力与显存", correct: true },
      { text: "冻结全部参数不做任何调整", correct: false },
      { text: "必须从头重训整个模型", correct: false },
      { text: "只适用于图像模型", correct: false },
    ],
    explanation:
      "PEFT 家族（LoRA、prefix tuning、adapters 等）思路一致：不动主干、只调极少量参数，用极小成本适配新任务。",
    tags: ["PEFT", "Fine-tuning"],
  },
  {
    id: 184,
    type: "single",
    question: "训练中发生梯度爆炸（gradient explosion）最典型的现象是？",
    options: [
      { text: "梯度范数剧增，loss 突变到极大值或出现 NaN", correct: true },
      { text: "梯度始终为零", correct: false },
      { text: "学习率自动变为负数", correct: false },
      { text: "词表自动扩展", correct: false },
    ],
    explanation:
      "梯度爆炸就是反向传播里梯度越传越大，导致参数一步跳飞，loss 冲高甚至溢出成 NaN，通常靠梯度裁剪兜底。",
    tags: ["Stability", "Training"],
  },
  {
    id: 185,
    type: "single",
    question: "深网络里梯度消失（gradient vanishing）的常见缓解手段是？",
    options: [
      { text: "残差连接（Residual）+ 合理的初始化 + 归一化层", correct: true },
      { text: "把所有隐藏层删掉", correct: false },
      { text: "把学习率设成无穷大", correct: false },
      { text: "去掉所有位置编码", correct: false },
    ],
    explanation:
      "残差连接让梯度有一条「近路」直通浅层，配合 LayerNorm 和合适的初始化，能有效缓解梯度随着层数越传越小的消失问题。",
    tags: ["Stability", "Training"],
  },
  {
    id: 186,
    type: "single",
    question: "训练中途出现 loss spike（loss 突然冲高）后，合理的处理是？",
    options: [
      { text: "暂缓学习率/回滚到 spike 前的检查点，而非直接让训练崩掉", correct: true },
      { text: "直接忽略，loss 一定会自动恢复", correct: false },
      { text: "把学习率调到最大强行冲过去", correct: false },
      { text: "立即清空训练数据", correct: false },
    ],
    explanation:
      "loss spike 常见原因是学习率过大或梯度爆炸，正确做法是回滚到 spike 前的 checkpoint、适当降学习率再续训，别让它在坏状态里越走越歪。",
    tags: ["Stability", "Training"],
  },
  {
    id: 187,
    type: "single",
    question: "perplexity（困惑度）与交叉熵损失的关系是？",
    options: [
      { text: "perplexity = exp(平均交叉熵)", correct: true },
      { text: "perplexity = 1 / 交叉熵", correct: false },
      { text: "perplexity = 交叉熵的平方", correct: false },
      { text: "两者完全无关", correct: false },
    ],
    explanation:
      "perplexity 是平均交叉熵取指数，数学上等于「模型平均情况下对下一个 token 有多不确定」，值越小越好，约等于有效候选词数。",
    tags: ["Evaluation", "LLM"],
  },
  {
    id: 188,
    type: "single",
    question: "pass@k 衡量的是模型什么能力？",
    options: [
      { text: "一次生成 k 个答案中至少有一个正确的概率", correct: true },
      { text: "模型参数总量除以 k", correct: false },
      { text: "训练集通过率除以验证集通过率", correct: false },
      { text: "每 k 个 epoch 的 loss 下降量", correct: false },
    ],
    explanation:
      "pass@k 是采样 k 个候选，看至少有一个通过测试的比例，常用于代码生成，衡量模型的「上限能力」而非单次表现。",
    tags: ["Evaluation", "Code"],
  },
  {
    id: 189,
    type: "single",
    question: "在大模型规模定律（scaling law）中，模型参数量增大通常带来？",
    options: [
      { text: "在足够数据下 loss 持续下降，能力更强", correct: true },
      { text: "必然过拟合任何数据", correct: false },
      { text: "必须同时减少训练数据", correct: false },
      { text: "推理速度一定变快", correct: false },
    ],
    explanation:
      "参数越多、数据越多，loss 越低、能力越强，但有个前提：数据量要跟得上，否则就浪费参数（参数量和数据量要按比例涨）。",
    tags: ["Scaling", "Training"],
  },
  {
    id: 190,
    type: "multiple",
    question: "关于预训练数据的组织，下列哪些做法合理？（多选）",
    options: [
      { text: "把质量低、重复多的文本清洗过滤掉", correct: true },
      { text: "对同一文本反复重采样来撑大数据量", correct: false },
      { text: "跨文档避免相邻样本来自同一篇，减少记忆污染", correct: true },
      { text: "只保留英文单一语种", correct: false },
    ],
    explanation:
      "高质量清洗和多语种覆盖是预训练数据的关键，重复去重要保留多样性；反复重采样同文本文档会损害多样性与泛化。",
    tags: ["Data", "Pretraining"],
  },
  {
    id: 191,
    type: "multiple",
    question: "关于学习率调度，下列哪些说法正确？（多选）",
    options: [
      { text: "warmup 能减少训练初期的 loss spike", correct: true },
      { text: "后期把学习率衰减到很低有助于稳定收敛", correct: true },
      { text: "学习率全程固定不变通常是最好的选择", correct: false },
      { text: "学习率越大越容易导致发散", correct: true },
    ],
    explanation:
      "主流做法是 warmup + 后期衰减（cosine 等），全程固定不利于收敛；学习率过大梯度和噪声被放大，容易发散。",
    tags: ["Optimizer", "Training"],
  },
  {
    id: 192,
    type: "multiple",
    question: "关于混合精度训练，下列哪些说法正确？（多选）",
    options: [
      { text: "bf16 因指数位多，对梯度更新范围更友好", correct: true },
      { text: "通常保留一份 fp32 主权重用于参数更新", correct: true },
      { text: "混合精度一定比 fp32 训练效果更好", correct: false },
      { text: "fp16 范围小，容易出现上溢/下溢，需 loss scaling", correct: true },
    ],
    explanation:
      "bf16 范围大适合训练，fp16 范围小常用 loss scaling 兜底；两者都要靠 fp32 主权重保证更新精度，但精度不等于效果必然更好。",
    tags: ["Mixed Precision", "Training"],
  },
  {
    id: 193,
    type: "multiple",
    question: "关于梯度累积，下列哪些说法正确？（多选）",
    options: [
      { text: "可以在不增大显存的情况下模拟更大的 batch size", correct: true },
      { text: "累积 N 步后更新一次参数，等效 batch 变大 N 倍", correct: true },
      { text: "累积的每一步都必须更新参数", correct: false },
      { text: "累积会显著降低梯度计算的总量", correct: false },
    ],
    explanation:
      "梯度累积用多个 micro-batch 累加梯度再统一更新，等效大 batch 且不占更多显存；但总计算量不变，且只有最后一步才更新参数。",
    tags: ["Training", "Efficiency"],
  },
  {
    id: 194,
    type: "multiple",
    question: "关于梯度裁剪，下列哪些说法正确？（多选）",
    options: [
      { text: "能限制单次更新的最大幅度，防梯度爆炸", correct: true },
      { text: "按全局范数裁剪（global norm clipping）考虑整批梯度", correct: true },
      { text: "梯度裁剪会把梯度全部清零", correct: false },
      { text: "梯度裁剪能让收敛速度无限提升", correct: false },
    ],
    explanation:
      "梯度裁剪把超阈值梯度的范数缩回安全范围，常用全局范数裁剪；它不置零、也不直接提速，只是保稳定。",
    tags: ["Stability", "Training"],
  },
  {
    id: 195,
    type: "multiple",
    question: "关于 KV cache 与推理，下列哪些说法正确？（多选）",
    options: [
      { text: "KV cache 缓存的是前面 token 的 Key 和 Value", correct: true },
      { text: "KV cache 可以显著减少推理时重复的注意力计算", correct: true },
      { text: "KV cache 与预训练时的参数更新直接相关", correct: false },
      { text: "KV cache 占用会随序列长度线性增长", correct: true },
    ],
    explanation:
      "KV cache 缓存历史 Key/Value 加速自回归推理，内存随长度线性涨；它属于推理优化，与训练参数更新无关。",
    tags: ["Inference", "LLM"],
  },
  {
    id: 196,
    type: "multiple",
    question: "关于激活重计算/梯度检查点，下列哪些说法正确？（多选）",
    options: [
      { text: "通过不保存全部中间激活来节省显存", correct: true },
      { text: "反向传播前需要重新前向以恢复激活", correct: true },
      { text: "能同时降低显存占用和训练时间", correct: false },
      { text: "是训练超大模型时常见的省显存手段", correct: true },
    ],
    explanation:
      "检查点以「多算一次前向」为代价省显存，显存下降但时间上升，不是双赢；大模型训练几乎必用它。",
    tags: ["Training", "Memory"],
  },
  {
    id: 197,
    type: "multiple",
    question: "关于 MoE 稀疏训练，下列哪些说法正确？（多选）",
    options: [
      { text: "每个 token 只经过少量被选中的专家", correct: true },
      { text: "需要负载均衡损失避免部分专家被冷落", correct: true },
      { text: "MoE 会显著降低参数量", correct: false },
      { text: "MoE 总参数多但单 token 计算量小", correct: true },
    ],
    explanation:
      "MoE 是「参数多、激活少」：总参数量很大，但每个 token 只走 top-k 专家，单步计算量小，还需 aux loss 做负载均衡。",
    tags: ["MoE", "Training"],
  },
  {
    id: 198,
    type: "multiple",
    question: "关于后训练对齐（RLHF/DPO/SFT），下列哪些说法正确？（多选）",
    options: [
      { text: "SFT 用监督的指令-回答数据继续训练", correct: true },
      { text: "DPO 可直接基于偏好对优化，无需单独奖励模型", correct: true },
      { text: "RLHF 的人类偏好信号能更好地对齐价值观", correct: true },
      { text: "对齐后模型就不需要预训练了", correct: false },
    ],
    explanation:
      "预训练学能力，后训练（SFT→RLHF/DPO）学「听指令、合偏好」，各司其职；对齐是在预训练基础上做的，不能取代预训练。",
    tags: ["RLHF", "Alignment"],
  },


  {
    id: 199,
    type: "single",
    question: "CLIP 的核心训练目标是什么？",
    options: [
      { text: "让文本编码器输出与视觉编码器完全相同的向量", correct: false },
      { text: "拉近图文配对的表示、推远不配对的表示", correct: true },
      { text: "让模型学会逐像素重建图像", correct: false },
      { text: "用图像特征直接预测文本序列", correct: false },
    ],
    explanation: "CLIP 用对比学习（contrastive learning）对齐图文，核心就一句：配对的对齐、不配对的推开，让二者表示进入同一语义空间。",
    tags: ["Multimodal", "CLIP"],
  },
  {
    id: 200,
    type: "single",
    question: "InfoNCE 这类对比损失，负样本数量对训练效果的影响是？",
    options: [
      { text: "负样本越多越好，能提供更丰富的判别信号", correct: true },
      { text: "负样本越少越好，越多越容易过拟合", correct: false },
      { text: "负样本数量对训练完全无影响", correct: false },
      { text: "负样本只需要一个就足够", correct: false },
    ],
    explanation: "对比损失本质是靠负样本「逼」模型把正样本从干扰中分出来，负样本越多，hard negative 越丰富，判别力越强——所以 CLIP 拼命堆 batch size。",
    tags: ["Multimodal", "Contrastive"],
  },
  {
    id: 201,
    type: "single",
    question: "CLIP 训练中，为什么要把 batch size 拉得很大？",
    options: [
      { text: "为了减小显存占用", correct: false },
      { text: "因为 batch 内其余样本互为负样本，batch 越大负例越多", correct: true },
      { text: "为了提升单个 step 的推理速度", correct: false },
      { text: "为了让温度参数自动更新", correct: false },
    ],
    explanation: "CLIP 不额外构造负样本，batch 内其他图文对被当作负样本，batch 越大，可用的正/负对越多，对比学习越有效。",
    tags: ["Multimodal", "CLIP"],
  },
  {
    id: 202,
    type: "single",
    question: "对比学习中的 temperature（温度参数）作用是什么？",
    options: [
      { text: "控制模型对相似度的敏感程度，调节分布的尖锐度", correct: true },
      { text: "控制学习率的大小", correct: false },
      { text: "控制负样本的数量", correct: false },
      { text: "控制 batch size 的大小", correct: false },
    ],
    explanation: "温度参数缩放 logits 后再 softmax，温度越低分布越尖锐（更像 one-hot），越能放大难样本的梯度；太小则训练易不稳定。",
    tags: ["Multimodal", "Contrastive"],
  },
  {
    id: 203,
    type: "single",
    question: "多模态模型中 Projector（投影层）的主要作用是什么？",
    options: [
      { text: "把视觉特征映射到语言模型的输入空间，与文本 token 对齐", correct: true },
      { text: "对图像做数据增强", correct: false },
      { text: "压缩图像的通道数以减少存储", correct: false },
      { text: "把文本特征变成图片", correct: false },
    ],
    explanation: "视觉塔输出的特征和文本 embedding 维度/语义不一致，Projector 负责把它变换成语言模型能「读懂」的 token 序列。",
    tags: ["Multimodal", "Projector"],
  },
  {
    id: 204,
    type: "single",
    question: "为什么多模态模型通常需要一个独立的 Projector 而非直接拼接视觉特征？",
    options: [
      { text: "因为视觉特征与文本词向量维度、分布不一致，需要线性变换对齐", correct: true },
      { text: "因为 Projector 可以增强图片分辨率", correct: false },
      { text: "因为语言模型不能接受任何向量输入", correct: false },
      { text: "因为 Projector 用来给模型增加随机噪声", correct: false },
    ],
    explanation: "视觉特征来自 CNN/ViT，语义空间和 LLM 的文本 embedding 不同，Projector（常是 MLP）负责降维、对齐，让语言模型能正确消费。",
    tags: ["Multimodal", "Projector"],
  },
  {
    id: 205,
    type: "single",
    question: "图文交错（interleaved）数据训练相比纯「图-文」对训练的优势是？",
    options: [
      { text: "能学到图片在文本上下文中出现的时序与关联关系", correct: true },
      { text: "能显著减少训练所需的参数量", correct: false },
      { text: "能完全消除多模态幻觉", correct: false },
      { text: "能替代所有文本预训练", correct: false },
    ],
    explanation: "交错数据里图片嵌在段落中，模型学到的不只是「图配一段文字」，而是图文在上下文里如何交织、互相指代，理解更深。",
    tags: ["Multimodal", "Training"],
  },
  {
    id: 206,
    type: "single",
    question: "视觉编码器（视觉塔）通常基于什么架构？",
    options: [
      { text: "ViT 或 CNN 这类图像特征提取网络", correct: true },
      { text: "纯 RNN 循环网络", correct: false },
      { text: "纯词嵌入层", correct: false },
      { text: "注意力打分函数", correct: false },
    ],
    explanation: "视觉塔负责把原始图像编码成特征序列，主流是 ViT（Vision Transformer）或 ResNet 这类卷积网络，早期 CLIP 两者都试过。",
    tags: ["Multimodal", "Encoder"],
  },
  {
    id: 207,
    type: "single",
    question: "在 LLaVA 这类模型中，视觉 token 通常是怎样送入语言模型的？",
    options: [
      { text: "先经 Projector 映射成与文本等价的 embedding 再拼接", correct: true },
      { text: "直接输入像素值矩阵", correct: false },
      { text: "通过单独的硬编码通道", correct: false },
      { text: "完全绕开语言模型自回归", correct: false },
    ],
    explanation: "视觉 token 被投影成和文本 embedding 同维的向量，插进输入序列里，语言模型就能把它们当普通 token 做自回归。",
    tags: ["Multimodal", "Architecture"],
  },
  {
    id: 208,
    type: "single",
    question: "多模态幻觉（hallucination）在图像描述任务中最典型的表现是？",
    options: [
      { text: "描述出图片中并不存在的物体或事实", correct: true },
      { text: "输出过于简短", correct: false },
      { text: "无法识别任何物体", correct: false },
      { text: "训练时间过长", correct: false },
    ],
    explanation: "模型在输出时「脑补」出图里没有的内容，比如没苹果却说有苹果，是图像描述和 VQA 里常见又难缠的问题。",
    tags: ["Multimodal", "Hallucination"],
  },
  {
    id: 209,
    type: "single",
    question: "多模态幻觉产生的一个常见原因是？",
    options: [
      { text: "文本先验过强，模型依赖语言惯性而忽略视觉证据", correct: true },
      { text: "模型参数量太大", correct: false },
      { text: "图片分辨率过高", correct: false },
      { text: "训练数据里文本太多", correct: false },
    ],
    explanation: "LLM 语言先验（语言 model 惯性）很强，训练时若图文对里物体常共现，模型会按语言惯性补全，忽视真实图像内容。",
    tags: ["Multimodal", "Hallucination"],
  },
  {
    id: 210,
    type: "single",
    question: "跨模态检索（如图搜文、文搜图）评估通常用什么指标？",
    options: [
      { text: "Recall@K（Top-K 命中率）", correct: true },
      { text: "BLEU", correct: false },
      { text: "Perplexity", correct: false },
      { text: "交叉熵损失", correct: false },
    ],
    explanation: "检索任务看的是「把正确答案排在多靠前」，所以用 Recall@K（前 K 个命中率）这类排序指标，而不是生成类指标。",
    tags: ["Multimodal", "Retrieval"],
  },
  {
    id: 211,
    type: "single",
    question: "CLIP 训练中文本编码器的作用是？",
    options: [
      { text: "把文本编码成与图像可对齐的表示向量", correct: true },
      { text: "生成图像像素", correct: false },
      { text: "压缩文本长度", correct: false },
      { text: "仅用于分词", correct: false },
    ],
    explanation: "文本编码器（通常是 Transformer）把一句话编码成一个向量，和图像编码器输出的向量做对比对齐。",
    tags: ["Multimodal", "Encoder"],
  },
  {
    id: 212,
    type: "single",
    question: "CLIP 训练时，图像和文本编码器的输出如何比较相似度？",
    options: [
      { text: "先各自归一化，再算二者点积作为相似度", correct: true },
      { text: "直接算 L2 距离取反", correct: false },
      { text: "用余弦损失替代交叉熵", correct: false },
      { text: "计算 KL 散度", correct: false },
    ],
    explanation: "把两个向量 L2 归一化后做点积，等价于余弦相似度，再配合温度缩放进 softmax 算交叉熵，就组成了对比损失。",
    tags: ["Multimodal", "CLIP"],
  },
  {
    id: 213,
    type: "single",
    question: "微调多模态模型时，只冻结视觉塔、只训练 LLM 和 Projector 的主要目的是？",
    options: [
      { text: "降低计算与显存开销，同时保留视觉特征通用性", correct: true },
      { text: "让视觉塔完全不参与推理", correct: false },
      { text: "消除所有幻觉", correct: false },
      { text: "增加模型参数量", correct: false },
    ],
    explanation: "视觉塔在大规模预训练后特征很通用，冻结它既省显存（不用算它的梯度）又避免在小数据上过拟合，这是常见微调策略。",
    tags: ["Multimodal", "Finetune"],
  },
  {
    id: 214,
    type: "single",
    question: "LoRA 应用于多模态模型微调时，通常把低秩矩阵加在哪些层上？",
    options: [
      { text: "LLM 的自注意力与 FFN 投影层", correct: true },
      { text: "数据加载器的维度", correct: false },
      { text: "图像像素的空间位置", correct: false },
      { text: "标签索引", correct: false },
    ],
    explanation: "LoRA 只给大模型里占参数的线性层（Q/K/V/O、FFN）加可训练低秩增量，视觉塔往往也冻结，需要微调的权重大大减少。",
    tags: ["Multimodal", "LoRA"],
  },
  {
    id: 215,
    type: "single",
    question: "LoRA 的核心思想是？",
    options: [
      { text: "用低秩矩阵近似权重的增量，冻结原权重", correct: true },
      { text: "复制一份完整模型来微调", correct: false },
      { text: "只调整偏置项", correct: false },
      { text: "对权重做量化压缩", correct: false },
    ],
    explanation: "LoRA 假设微调时权重的变化量近似低秩，用两个小矩阵 ΔW=A·B 表示增量，原权重冻结，训练时只更新 A、B，省显存省参数。",
    tags: ["Multimodal", "LoRA"],
  },
  {
    id: 216,
    type: "single",
    question: "多模态评测基准 GQA 主要测什么？",
    options: [
      { text: "基于图像的推理问答（VQA）能力", correct: true },
      { text: "纯文本语义相似度", correct: false },
      { text: "图像超分辨率", correct: false },
      { text: "视频帧率", correct: false },
    ],
    explanation: "GQA 是图像问答/推理基准，重点测模型能不能结合图像做多步推理，回答看图才能答的问题。",
    tags: ["Multimodal", "Benchmark"],
  },
  {
    id: 217,
    type: "single",
    question: "MMMU 这个多模态基准的特点是？",
    options: [
      { text: "覆盖大学级别多学科，难度高、重推理", correct: true },
      { text: "只测图片分类", correct: false },
      { text: "只测语音识别", correct: false },
      { text: "只含单选且无图", correct: false },
    ],
    explanation: "MMMU 汇聚大学多学科试题，覆盖艺术、医学、STEM 等，题目难、要真懂图并推理，是目前评判多模态模型的重要门槛。",
    tags: ["Multimodal", "Benchmark"],
  },
  {
    id: 218,
    type: "single",
    question: "视频多模态任务相比图像多模态，额外引入的关键信息是？",
    options: [
      { text: "时间维度上的运动与因果依赖", correct: true },
      { text: "更多可用的像素颜色", correct: false },
      { text: "更大的 batch size", correct: false },
      { text: "更长的标签文本", correct: false },
    ],
    explanation: "视频多了时间轴，模型得理解帧之间怎么动、事件怎么因果推进，所以要多帧采样 + 时序建模，比静态图难一个维度。",
    tags: ["Multimodal", "Video"],
  },
  {
    id: 219,
    type: "single",
    question: "训练多模态模型时，为何常出现模态间序列长度不平衡？",
    options: [
      { text: "图像编码后 token 数往往远多于文本 token", correct: true },
      { text: "因为文本永远比图像长", correct: false },
      { text: "因为图像没有 token", correct: false },
      { text: "因为投影层会删除文本", correct: false },
    ],
    explanation: "一张图切成若干 patch 能编码出几百上千个视觉 token，而配的一句话才几十个词，长度差异大，常需 token 裁剪或降采样。",
    tags: ["Multimodal", "Training"],
  },
  {
    id: 220,
    type: "single",
    question: "缓解多模态中视觉 token 过多导致序列过长的常用手段是？",
    options: [
      { text: "视觉 token 池化/降采样或缩减 patch 数", correct: true },
      { text: "把图片直接删掉", correct: false },
      { text: "给文本多加 token", correct: false },
      { text: "提高所有学习率", correct: false },
    ],
    explanation: "视觉 token 太多会拖慢自回归并爆显存，常见做法是池化、token 合并（token merging）或减少 patch 数量来压缩。",
    tags: ["Multimodal", "Training"],
  },
  {
    id: 221,
    type: "single",
    question: "多模态模型输出图像的 caption 时，常用什么生成指标？",
    options: [
      { text: "CIDEr / BLEU / ROUGE 这类文本匹配指标", correct: true },
      { text: "Recall@K", correct: false },
      { text: "mAP", correct: false },
      { text: "帧率", correct: false },
    ],
    explanation: "caption 是生成文本，衡量它与参考描述的重合度，常用 CIDEr、BLEU、ROUGE 这类基于 n-gram 匹配的指标。",
    tags: ["Multimodal", "Caption"],
  },
  {
    id: 222,
    type: "single",
    question: "OCR 属于多模态中哪一类典型任务？",
    options: [
      { text: "从图像中识别文字（图像到文本）", correct: true },
      { text: "纯语音转文字", correct: false },
      { text: "纯文本翻译", correct: false },
      { text: "视频压缩", correct: false },
    ],
    explanation: "OCR（光学字符识别）是典型的跨模态任务，输入图像、输出其中的文字序列，考验模型对图形里文字的感知。",
    tags: ["Multimodal", "OCR"],
  },
  {
    id: 223,
    type: "single",
    question: "CLIP 用 InfoNCE 训练时，损失主要惩罚什么？",
    options: [
      { text: "让配对的相似度低、让不配对的相似度高", correct: false },
      { text: "让配对的相似度高、不配对的相似度低", correct: true },
      { text: "让所有样本相似度都为 0", correct: false },
      { text: "让相似度分布均匀", correct: false },
    ],
    explanation: "InfoNCE 目标是最大化正样本对的相似度、最小化负样本对的相似度，本质上是个多分类：在一组候选里认出正确配对。",
    tags: ["Multimodal", "Contrastive"],
  },
  {
    id: 224,
    type: "single",
    question: "为什么多模态训练数据中「图文对」的质量比数量更重要？",
    options: [
      { text: "噪声大、图文不匹配的样本会教坏模型对齐", correct: true },
      { text: "数量多了反而过拟合", correct: false },
      { text: "因为硬性要求数据必须很简短", correct: false },
      { text: "因为模型不看数据", correct: false },
    ],
    explanation: "如果大量样本图不对文（弱匹配、标注噪声大），模型学到的是错误对齐，负样本还会变「伪负样本」，质量差再多也白搭。",
    tags: ["Multimodal", "Data"],
  },
  {
    id: 225,
    type: "single",
    question: "训练 CLIP 时温度参数通常设得比较小（如 0.07），原因是？",
    options: [
      { text: "让相似度分布更尖锐，拉大正负样本差距", correct: true },
      { text: "让分布更平坦避免过拟合", correct: false },
      { text: "让学习率更大", correct: false },
      { text: "让梯度消失", correct: false },
    ],
    explanation: "温度小相当于把 logits 放大，softmax 后分布更尖锐，正样本概率更突出，梯度信号更清晰——太小则训练不稳。",
    tags: ["Multimodal", "CLIP"],
  },
  {
    id: 226,
    type: "multiple",
    question: "关于多模态模型中模态融合（fusion）的说法正确的有？",
    options: [
      { text: "让不同模态信息交互、互补，联合决策", correct: true },
      { text: "常通过 cross-attention 等方式实现模态间交互", correct: true },
      { text: "目的是删除多余模态以省算力", correct: false },
      { text: "把多模态退化回单模态", correct: false },
    ],
    explanation: "融合让视觉、文本等信息互相交叉（如 cross-attention）实现联合推理；它不是删模态或退化回单模态，而是让各模态互补。",
    tags: ["Multimodal", "Architecture"],
  },
  {
    id: 227,
    type: "multiple",
    question: "关于跨模态注意力（cross-attention）的说法正确的有？",
    options: [
      { text: "query 来自一个模态，key/value 来自另一模态", correct: true },
      { text: "用于让模型按文本信息去关注图像相应区域", correct: true },
      { text: "它和自注意力完全等价", correct: false },
      { text: "只作用于图像不作用于文本", correct: false },
    ],
    explanation: "cross-attention 让「提问方」的 query 去「检索方」的 key/value 取信息，实现跨模态关注；与序列内部的自注意力不同，也并非只针对图像。",
    tags: ["Multimodal", "Architecture"],
  },
  {
    id: 228,
    type: "single",
    question: "训练多模态数据时，模态缺失（modality missing）是指？",
    options: [
      { text: "某些样本缺图像或缺文本，模型需学会容错", correct: true },
      { text: "batch 里缺少正样本", correct: false },
      { text: "缺少学习率", correct: false },
      { text: "缺少损失函数", correct: false },
    ],
    explanation: "真实数据常某个模态缺失（如图没配文、或只有文没图），训练要让模型在这类不完整输入下也能合理工作。",
    tags: ["Multimodal", "Training"],
  },
  {
    id: 229,
    type: "single",
    question: "模态失配（modality mismatch）在多模态训练中的含义是？",
    options: [
      { text: "图文的配对关系不对应或语义不一致", correct: true },
      { text: "两个模态参数量不一样", correct: false },
      { text: "输入分辨率不统一", correct: false },
      { text: "学习率与损失不匹配", correct: false },
    ],
    explanation: "失配指图像和配文并不真正对应（如广告图配了不相干的文案），这种伪正样本会污染对齐，训练时要注意清洗。",
    tags: ["Multimodal", "Training"],
  },
  {
    id: 230,
    type: "single",
    question: "多模态模型微调时，若目标任务数据量很小，最稳妥的做法是？",
    options: [
      { text: "冻结大部分骨干，用 LoRA 等只训少量参数", correct: true },
      { text: "全量从头训练所有参数", correct: false },
      { text: "完全不训练只推理", correct: false },
      { text: "把学习率调得极大", correct: false },
    ],
    explanation: "数据少时全量微调容易过拟合且显存压力大，冻结骨干 + LoRA 只改少量参数是既稳又省的标准做法。",
    tags: ["Multimodal", "Finetune"],
  },
  {
    id: 231,
    type: "single",
    question: "CLIP 类模型的一个已知局限是？",
    options: [
      { text: "对细粒度属性、空间关系等语义理解偏弱", correct: true },
      { text: "无法编码任何图像", correct: false },
      { text: "不能处理文本", correct: false },
      { text: "输出永远错误", correct: false },
    ],
    explanation: "CLIP 学到的是图文「粗对齐」，对颜色细节、空间位置、计数等细粒度语义常常抓不准，所以后续模型都补了更大规模数据和更强视觉塔。",
    tags: ["Multimodal", "CLIP"],
  },
  {
    id: 232,
    type: "single",
    question: "多模态量化（quantization）的主要目的是？",
    options: [
      { text: "降低模型体积与推理显存/延迟", correct: true },
      { text: "提高图片分辨率", correct: false },
      { text: "增加训练数据", correct: false },
      { text: "提升文本长度", correct: false },
    ],
    explanation: "量化把权重/激活从 fp16 压到 int8 等低位，模型更小更快，适合部署；代价是可能有精度损失，多模态模型同样适用。",
    tags: ["Multimodal", "Quantization"],
  },
  {
    id: 233,
    type: "single",
    question: "训练精度上，多模态大模型预训练常用 bf16 而非 fp16 的主要原因是？",
    options: [
      { text: "bf16 指数范围大，不易溢出，训练更稳", correct: true },
      { text: "bf16 精度比 fp32 还高", correct: false },
      { text: "fp16 无法在 GPU 上运行", correct: false },
      { text: "bf16 不需要梯度", correct: false },
    ],
    explanation: "bf16 与 fp32 指数范围相同，梯度不易上溢/下溢；fp16 范围小，大模型训练常因溢出而不稳，所以多用 bf16。",
    tags: ["Multimodal", "Training"],
  },
  {
    id: 234,
    type: "single",
    question: "多模态训练中平衡「各模态数据比例」是为了？",
    options: [
      { text: "避免模型偏向某一模态、丢失其他模态能力", correct: true },
      { text: "减少总数据量", correct: false },
      { text: "让模型更快过拟合", correct: false },
      { text: "完全舍弃图像", correct: false },
    ],
    explanation: "若图像数据远多/远少于文本，模型会对「多数模态」过拟合而退化另一模态，所以要控制比例、分阶段混合训练。",
    tags: ["Multimodal", "Data"],
  },
  {
    id: 235,
    type: "single",
    question: "多模态模型输出视频描述时，相比静态图描述的最大难点是？",
    options: [
      { text: "需理解跨帧的时间动态与事件流", correct: true },
      { text: "画面颜色太多", correct: false },
      { text: "不需要看视频内容", correct: false },
      { text: "只有一帧可看", correct: false },
    ],
    explanation: "视频描述要求模型跨帧追踪动作、事件和因果，而非只描述单帧静态内容，时序理解是关键难点。",
    tags: ["Multimodal", "Video"],
  },
  {
    id: 236,
    type: "multiple",
    question: "下列哪些属于多模态训练数据的常见形态？",
    options: [
      { text: "图像-文本对", correct: true },
      { text: "图文交错文档", correct: true },
      { text: "纯音频文件无任何标注", correct: false },
      { text: "视频-字幕对", correct: true },
    ],
    explanation: "图像-文本对、图文交错、视频-字幕/音频-文本都是多模态训练的原料；无标注纯音频不算「配对数据」。",
    tags: ["Multimodal", "Data"],
  },
  {
    id: 237,
    type: "multiple",
    question: "关于 CLIP 的训练设置，下列说法正确的有？",
    options: [
      { text: "batch size 要尽量大以增加负样本", correct: true },
      { text: "温度参数需要作为可学习参数训练", correct: true },
      { text: "图文编码器完全冻结不训练", correct: false },
      { text: "负样本来自 batch 内的其他图文对", correct: true },
    ],
    explanation: "CLIP 用大 batch 扩充负样本、温度参数可学习；图像与文本两个编码器都是要一起训练的，不是冻结的。",
    tags: ["Multimodal", "CLIP"],
  },
  {
    id: 238,
    type: "multiple",
    question: "多模态幻觉可能由哪些原因导致？",
    options: [
      { text: "语言先验过强，模型按惯性补全", correct: true },
      { text: "训练数据中物体常共现造成虚假关联", correct: true },
      { text: "模型无法读取任何图像信息", correct: false },
      { text: "采样/解码策略放大了错误", correct: true },
    ],
    explanation: "语言先验、数据中的虚假共现、以及贪婪/beam 解码放大都可能导致幻觉；「看不到图」是极端情况并非主因。",
    tags: ["Multimodal", "Hallucination"],
  },
  {
    id: 239,
    type: "multiple",
    question: "下列哪些属于评估多模态模型的标准任务？",
    options: [
      { text: "图像描述（image captioning）", correct: true },
      { text: "视觉问答（VQA）", correct: true },
      { text: "纯文本情感分类", correct: false },
      { text: "跨模态检索", correct: true },
    ],
    explanation: "caption、VQA、跨模态检索都是多模态核心任务；纯文本情感分类不涉及图像，不算多模态评估。",
    tags: ["Multimodal", "Task"],
  },
  {
    id: 240,
    type: "multiple",
    question: "关于 Projector 的说法正确的有？",
    options: [
      { text: "它把视觉特征变换到语言模型可理解的 embedding 空间", correct: true },
      { text: "常由 MLP 或线性层实现", correct: true },
      { text: "它负责生成图像", correct: false },
      { text: "它的训练通常和模型其他部分一起完成对齐", correct: true },
    ],
    explanation: "Projector 把视觉特征对齐到文本 embedding 空间，常用 MLP 实现，并在多模态预训练/微调阶段参与训练完成对齐。",
    tags: ["Multimodal", "Projector"],
  },
  {
    id: 241,
    type: "multiple",
    question: "缓解多模态训练中模态间序列长度不平衡的手段有？",
    options: [
      { text: "对视觉 token 做池化或降采样", correct: true },
      { text: "合并/裁剪冗余视觉 token", correct: true },
      { text: "故意把文本 token 无限拉长", correct: false },
      { text: "缩小输入图像 patch 数", correct: true },
    ],
    explanation: "压缩视觉 token（池化、合并、减 patch）都是常规手段；无意义拉长文本只会恶化序列不平衡，不可取。",
    tags: ["Multimodal", "Training"],
  },
  {
    id: 242,
    type: "multiple",
    question: "CLIP 对比学习损失中，下列叙述正确的有？",
    options: [
      { text: "对每个图文对，都要让它与 batch 内其他对区分开", correct: true },
      { text: "损失同时优化图像→文本和文本→图像两个方向", correct: true },
      { text: "不需要任何负样本", correct: false },
      { text: "目标是最大化正对相似度并最小化负对相似度", correct: true },
    ],
    explanation: "InfoNCE 做对称的图文双向对齐，正对拉近、负对推远，batch 内其余对充当负样本，缺了负样本训练就会退化。",
    tags: ["Multimodal", "Contrastive"],
  },
  {
    id: 243,
    type: "multiple",
    question: "LoRA 应用在多模态模型微调时的正确描述有？",
    options: [
      { text: "可冻结视觉塔，只训练低秩注入层", correct: true },
      { text: "能显著减少可训练参数量", correct: true },
      { text: "通常加在注意力与 FFN 的线性投影上", correct: true },
      { text: "必须从头训练所有原始权重", correct: false },
    ],
    explanation: "LoRA 只给线性层注入低秩增量，可冻结视觉塔和 LLM 原权重，训练参数量大幅减少，并不改动原始权重。",
    tags: ["Multimodal", "LoRA"],
  },
  {
    id: 244,
    type: "multiple",
    question: "视频多模态模型相对图像模型需要额外处理的是？",
    options: [
      { text: "多帧采样以捕捉时间动态", correct: true },
      { text: "时序建模（如 3D 卷积/时序注意力）", correct: true },
      { text: "运动与因果关系的理解", correct: true },
      { text: "只处理单帧静态图", correct: false },
    ],
    explanation: "视频要处理时间轴，需多帧采样、时序建模、理解运动因果；只取单帧就退化成图像模型，丢失动态信息。",
    tags: ["Multimodal", "Video"],
  },
  {
    id: 245,
    type: "multiple",
    question: "关于多模态评测基准的描述正确的有？",
    options: [
      { text: "GQA 侧重看图推理问答", correct: true },
      { text: "MMMU 覆盖大学级多学科、难度高", correct: true },
      { text: "只用单一正确率就能完全衡量模型", correct: false },
      { text: "基准能暴露模型的细粒度理解与幻觉问题", correct: true },
    ],
    explanation: "GQA、MMMU 都是综合多模态基准，能测出细粒度与推理短板、反映幻觉；单指标无法全面衡量多模态能力。",
    tags: ["Multimodal", "Benchmark"],
  },
  {
    id: 246,
    type: "multiple",
    question: "训练多模态模型时，关于数据与模态的正确认识有？",
    options: [
      { text: "图文对的质量（匹配度）比单纯数量更重要", correct: true },
      { text: "模态缺失的样本也应让模型学会容错", correct: true },
      { text: "要控制各模态数据比例避免模态失衡", correct: true },
      { text: "数据越乱越好，能提升泛化", correct: false },
    ],
    explanation: "数据要重质量、允许模态缺失训练容错、并平衡各模态比例；脏乱数据只会教坏对齐，绝非越多越好。",
    tags: ["Multimodal", "Data"],
  },
  {
    id: 247,
    type: "multiple",
    question: "关于多模态模型架构与量化的说法正确的有？",
    options: [
      { text: "Projector 是连接视觉塔与语言模型的关键模块", correct: true },
      { text: "cross-attention 用于模态间的信息交互", correct: true },
      { text: "bf16 相比 fp16 指数范围更大、训练更稳", correct: true },
      { text: "量化只适用于文本模型，不适用于多模态", correct: false },
    ],
    explanation: "Projector、cross-attention 是架构核心；bf16 范围大更稳；量化同样能压缩多模态模型用于部署。",
    tags: ["Multimodal", "Architecture"],
  },
  {
    id: 248,
    type: "multiple",
    question: "多模态模型微调（含图文交错数据训练）时正确的做法有？",
    options: [
      { text: "数据少时可冻结骨干并配合 LoRA 微调", correct: true },
      { text: "用图文交错数据可学习图文在上下文中的关联", correct: true },
      { text: "视觉 token 过多时应压缩以控制序列长度", correct: true },
      { text: "应无脑加大学习率以快速收敛", correct: false },
    ],
    explanation: "小数据用冻结+LoRA、用交错数据学关联、压缩视觉 token 都合理；无脑加大学习率只会让训练发散，不可取。",
    tags: ["Multimodal", "Finetune"],
  },


  {
    id: 249,
    type: "single",
    question: "主流 VLM 的基本架构通常按什么顺序组织？",
    options: [
      { text: "LLM → Projector → 视觉编码器", correct: false },
      { text: "视觉编码器 → Projector → LLM", correct: true },
      { text: "Projector → 视觉编码器 → LLM", correct: false },
      { text: "LLM → 视觉编码器 → Projector", correct: false },
    ],
    explanation: "图像先进视觉编码器（如 ViT/CLIP）提特征，再过 Projector 投影到 LLM 的词嵌入空间，最后交给 LLM 理解生成。",
    tags: ["VLM", "Architecture"],
  },
  {
    id: 250,
    type: "single",
    question: "VLM 中 Projector（投影层）的核心作用是什么？",
    options: [
      { text: "对图像做数据增强", correct: false },
      { text: "把视觉特征维度映射到 LLM 的词嵌入空间", correct: true },
      { text: "负责文本的 token 化", correct: false },
      { text: "压缩图像文件体积", correct: false },
    ],
    explanation: "视觉编码器输出的特征维度和 LLM 词嵌入维度不一致，Projector 负责对齐映射，最简单的就是一层线性层（LLaVA 的做法）。",
    tags: ["VLM", "Architecture"],
  },
  {
    id: 251,
    type: "single",
    question: "视觉塔（视觉编码器）在 VLM 中主要承担什么任务？",
    options: [
      { text: "只负责文本序列的生成", correct: false },
      { text: "把像素编码成具有语义的视觉特征向量", correct: true },
      { text: "负责算注意力分数", correct: false },
      { text: "做 token 的概率采样", correct: false },
    ],
    explanation: "视觉塔就是视觉编码器（如 CLIP/ViT），把图像像素转成富含语义的特征，供后续投影和对齐。",
    tags: ["VLM", "Architecture"],
  },
  {
    id: 252,
    type: "single",
    question: "为什么把图像转成「视觉 token」而不是当整体向量直接用？",
    options: [
      { text: "token 化后才能让 LLM 在空间上定位、逐区域理解", correct: true },
      { text: "整体向量占用的显存更小", correct: false },
      { text: "token 化能省去 Projector", correct: false },
      { text: "视觉 token 不需要参与注意力", correct: false },
    ],
    explanation: "图像拆成 patch 级别的 token，LLM 才能逐区域推理、定位具体位置，类似文本 token 的粒度。",
    tags: ["VLM", "Tokenization"],
  },
  {
    id: 253,
    type: "multiple",
    question: "关于 ViT 和 CLIP 作为视觉编码器的说法，正确的有？（多选）",
    options: [
      { text: "CLIP 通过图文对比学习对齐图像与文本", correct: true },
      { text: "ViT 用 self-attention 建模 patch 之间的关系", correct: true },
      { text: "CLIP 视觉塔的输出天然带有全局语义", correct: true },
      { text: "ViT 完全无法提取语义特征", correct: false },
    ],
    explanation: "ViT 用注意力处理 patch，CLIP 在此之上做图文对比对齐，让视觉塔输出既含局部结构也含全局语义。",
    tags: ["VLM", "Architecture"],
  },
  {
    id: 254,
    type: "single",
    question: "一张 224×224 的图像按 14×14 patch 切分，会得到多少视觉 token？",
    options: [
      { text: "16", correct: false },
      { text: "256", correct: true },
      { text: "1024", correct: false },
      { text: "224", correct: false },
    ],
    explanation: "224/14=16，横竖各 16 块，16×16=256 个 token，这就是 CLIP 视觉塔的常见 token 量。",
    tags: ["VLM", "Tokenization"],
  },
  {
    id: 255,
    type: "single",
    question: "相比固定分辨率切图，动态分辨率（如 Qwen2-VL）的核心收益是什么？",
    options: [
      { text: "能保留高分辨率细粒度信息，减少信息丢失", correct: true },
      { text: "让所有图像都变成正方形", correct: false },
      { text: "取消了视觉编码器", correct: false },
      { text: "只能处理纯文字", correct: false },
    ],
    explanation: "动态分辨率按原图比例切分成多块、逐块编码再拼接，能保住文档、图表里的细节，而不是粗暴缩放丢信息。",
    tags: ["VLM", "Resolution"],
  },
  {
    id: 256,
    type: "multiple",
    question: "高分辨率图像给 VLM 带来的挑战有哪些？（多选）",
    options: [
      { text: "视觉 token 数量暴增，显存与计算上升", correct: true },
      { text: "KV cache 随之增大", correct: true },
      { text: "细节信息天然比低分辨率更丰富", correct: true },
      { text: "高分辨率完全不影响计算成本", correct: false },
    ],
    explanation: "分辨率越高 token 越多，注意力计算和 KV cache 都跟着涨，所以才有 token 压缩等手段来缓解。",
    tags: ["VLM", "Resolution"],
  },
  {
    id: 257,
    type: "single",
    question: "InternVL 通常用什么策略来应对高分辨率图像？",
    options: [
      { text: "只保留全局缩略图一个 token", correct: false },
      { text: "动态切块 + 全局缩略图结合的多尺度 token", correct: true },
      { text: "完全丢弃高分辨率分支", correct: false },
      { text: "把所有像素拼成一个 token", correct: false },
    ],
    explanation: "InternVL 融合局部高分辨率切块与全局缩略图，兼顾细节与整体语义，是典型的多尺度动态方案。",
    tags: ["VLM", "Resolution"],
  },
  {
    id: 258,
    type: "single",
    question: "视觉 token 需要带位置编码，最主要的原因是什么？",
    options: [
      { text: "attention 本身是置换不变的，需要位置编码标识空间位置", correct: true },
      { text: "视觉 token 不需要位置也能理解", correct: false },
      { text: "位置编码只为文本而设计", correct: false },
      { text: "位置编码用来给图像上色", correct: false },
    ],
    explanation: "注意力机制对输入顺序不敏感，视觉 token 是 patch 序列，必须靠位置编码才能让模型知道空间布局。",
    tags: ["VLM", "PositionEncoding"],
  },
  {
    id: 259,
    type: "single",
    question: "视觉特征与文本对齐不当时，最直接的后果是什么？",
    options: [
      { text: "模型可能看图不看或答非所问", correct: true },
      { text: "模型运行速度反而更快", correct: false },
      { text: "图像文件变小", correct: false },
      { text: "没有任何影响", correct: false },
    ],
    explanation: "视觉特征若没和文本嵌入对齐，LLM 就把图像 token 当噪声，导致忽略图像或生成无关答案，所以 Projector 训练很关键。",
    tags: ["VLM", "Alignment"],
  },
  {
    id: 260,
    type: "multiple",
    question: "多模态推理中，模型理解空间关系（如「桌子在椅子左边」）依赖哪些因素？（多选）",
    options: [
      { text: "视觉 token 的位置编码", correct: true },
      { text: "patch 的空间顺序被保留", correct: true },
      { text: "模型对空间语义的建模能力", correct: true },
      { text: "把图像转成纯文本描述", correct: false },
    ],
    explanation: "空间关系靠 token 位置编码和顺序保留让模型感知布局，加上训练学到的空间语义才能判断方位。",
    tags: ["VLM", "SpatialReasoning"],
  },
  {
    id: 261,
    type: "single",
    question: "VLM 在 OCR（光学字符识别）上表现好，主要得益于什么？",
    options: [
      { text: "视觉塔能精细捕捉字形纹理并映射到字符语义", correct: true },
      { text: "模型本身不做任何图像理解", correct: false },
      { text: "OCR 完全不需要视觉信息", correct: false },
      { text: "高分辨率分支与文本嵌入共享空间", correct: false },
    ],
    explanation: "OCR 靠视觉塔捕捉细微笔画，再靠图文对齐把字形与字符语义关联，高分辨率分支进一步保住小字。",
    tags: ["VLM", "OCR"],
  },
  {
    id: 262,
    type: "multiple",
    question: "让 VLM 读懂复杂图表（柱状图、折线图、表格）通常需要？（多选）",
    options: [
      { text: "高分辨率保留数值与坐标细节", correct: true },
      { text: "视觉 token 数量足够密集", correct: true },
      { text: "模型具备数值/坐标理解能力", correct: true },
      { text: "把图表压成低分辨率缩略图", correct: false },
    ],
    explanation: "图表含大量数字和坐标，细节要够密，模型还得学会数值与空间位置对应，低分辨率会直接糊掉关键数字。",
    tags: ["VLM", "Chart"],
  },
  {
    id: 263,
    type: "single",
    question: "VLM 做 GUI 理解（截图操作、定位按钮）时，最关键的是什么？",
    options: [
      { text: "精确的空间定位能力与高分辨率", correct: true },
      { text: "关闭视觉编码器", correct: false },
      { text: "完全不看像素", correct: false },
      { text: "只依赖语言先验", correct: false },
    ],
    explanation: "GUI 任务要精确点击坐标，必须高分辨率看清元素 + 精确定位，光靠语言先验猜不到按钮在哪。",
    tags: ["VLM", "GUI"],
  },
  {
    id: 264,
    type: "single",
    question: "多模态幻觉（模型说图中不存在的东西）最常见的来源是什么？",
    options: [
      { text: "模型过度依赖语言先验，忽视实际视觉证据", correct: true },
      { text: "图像分辨率太高", correct: false },
      { text: "Projector 被量化导致溢出", correct: false },
      { text: "位置编码缺失", correct: false },
    ],
    explanation: "模型常靠训练数据的统计惯性「脑补」，若视觉证据不强就被语言先验带偏，说出图里没有的内容。",
    tags: ["VLM", "Hallucination"],
  },
  {
    id: 265,
    type: "multiple",
    question: "缓解多模态幻觉的常用手段有哪些？（多选）",
    options: [
      { text: "加强视觉特征在生成中的权重", correct: true },
      { text: "训练时加入对抗性幻觉数据", correct: true },
      { text: "解码时约束生成内容与图像证据一致", correct: true },
      { text: "故意忽略视觉输入", correct: false },
    ],
    explanation: "让模型更信视觉证据、用幻觉数据纠偏、解码阶段做证据约束，都能降低「脑补」的发生率。",
    tags: ["VLM", "Hallucination"],
  },
  {
    id: 266,
    type: "single",
    question: "对 VLM 做权重量化（如 int8/4bit）时，通常最需要注意什么？",
    options: [
      { text: "视觉塔要保持较高精度，量化敏感", correct: true },
      { text: "Projector 必须量化才能跑", correct: false },
      { text: "量化对视觉塔毫无影响", correct: false },
      { text: "只需量化文本嵌入层", correct: false },
    ],
    explanation: "视觉塔对量化很敏感，精度掉得明显；Projector 往往保持高精度，否则对齐崩了视觉理解直接退化。",
    tags: ["VLM", "Quantization"],
  },
  {
    id: 267,
    type: "single",
    question: "为什么 Projector（投影层）通常不宜被量化？",
    options: [
      { text: "它负责视觉到文本的关键对齐，精度损失会放大", correct: true },
      { text: "它的参数量最大", correct: false },
      { text: "它不参与任何计算", correct: false },
      { text: "量化它反而更快", correct: false },
    ],
    explanation: "Projector 是视觉与文本的唯一桥梁，一旦量化精度损失，整个多模态对齐质量就跟着崩，所以常保高精度。",
    tags: ["VLM", "Quantization"],
  },
  {
    id: 268,
    type: "multiple",
    question: "VLM 的量化与精度敏感点有哪些？（多选）",
    options: [
      { text: "视觉塔对低比特量化更敏感", correct: true },
      { text: "多模态对齐对精度要求高于纯文本任务", correct: true },
      { text: "高分辨率推理放大了显存压力", correct: true },
      { text: "量化对任何层效果完全一样", correct: false },
    ],
    explanation: "视觉塔与对齐层都吃精度，量化方案要分层处理，且高分辨率带来的显存压力也让量化更受关注。",
    tags: ["VLM", "Quantization"],
  },
  {
    id: 269,
    type: "single",
    question: "长图像序列（如多张图或高分辨率多切块）推理时，KV cache 主要带来什么问题？",
    options: [
      { text: "KV cache 显存占用随 token 数线性增长", correct: true },
      { text: "KV cache 让推理更快且省显存", correct: false },
      { text: "KV cache 与 token 数无关", correct: false },
      { text: "KV cache 只存在于训练阶段", correct: false },
    ],
    explanation: "每个新增视觉 token 都要存 K/V，token 一多 KV cache 显存暴涨，所以长图/多图需要压缩或外存 KV。",
    tags: ["VLM", "KVcache"],
  },
  {
    id: 270,
    type: "multiple",
    question: "缓解 VLM 长视觉序列 KV cache 压力/计算量的常见做法有？（多选）",
    options: [
      { text: "合并或压缩视觉 token", correct: true },
      { text: "窗口/局部注意力限制关注范围", correct: true },
      { text: "token pruning 丢弃冗余视觉 token", correct: true },
      { text: "无限扩大 KV cache 容量", correct: false },
    ],
    explanation: "token 合并、局部注意力、剪枝都是降低 token 量级的手段，能直接缓解 KV 增长和注意力开销。",
    tags: ["VLM", "KVcache"],
  },
  {
    id: 271,
    type: "single",
    question: "跨模态检索（用文本找图、用图找文）依赖的核心机制是什么？",
    options: [
      { text: "把图文映射到同一向量空间比较相似度", correct: true },
      { text: "把图像编码成音频", correct: false },
      { text: "只比较文件大小", correct: false },
      { text: "完全随机返回结果", correct: false },
    ],
    explanation: "CLIP 这类模型把图文统一映射到共享向量空间，检索就是算相似度，这也是 VLM 视觉塔的经典训练方式。",
    tags: ["VLM", "Retrieval"],
  },
  {
    id: 272,
    type: "single",
    question: "VLM 评测集 MMMU 主要测什么？",
    options: [
      { text: "跨学科的多模态大学水平知识推理", correct: true },
      { text: "模型生成速度", correct: false },
      { text: "模型参数量大小", correct: false },
      { text: "纯文本的语法纠错", correct: false },
    ],
    explanation: "MMMU 是跨 30+ 学科的多模态评测，题目带图还要多步推理，考察大学水平的知识与视觉联合推理能力。",
    tags: ["VLM", "Evaluation"],
  },
  {
    id: 273,
    type: "single",
    question: "POPE 评测主要针对 VLM 的哪个问题？",
    options: [
      { text: "多模态幻觉", correct: true },
      { text: "图像分辨率", correct: false },
      { text: "推理速度", correct: false },
      { text: "KV cache 占用", correct: false },
    ],
    explanation: "POPE 是探测 VLM 幻觉的基准，通过提问图中是否存在某对象（含不存在的），测模型会不会「脑补」出错误对象。",
    tags: ["VLM", "Evaluation"],
  },
  {
    id: 274,
    type: "multiple",
    question: "VLM 常见评测维度包括哪些？（多选）",
    options: [
      { text: "知识推理（MMMU 等）", correct: true },
      { text: "幻觉（POPE 等）", correct: true },
      { text: "细粒度感知（OCR/计数/空间）", correct: true },
      { text: "模型商用价格", correct: false },
    ],
    explanation: "VLM 评测横跨知识推理、幻觉、细粒度感知（OCR、计数、空间定位）等维度，价格不属于能力评测。",
    tags: ["VLM", "Evaluation"],
  },
  {
    id: 275,
    type: "single",
    question: "VLM 微调最常用的高效参数方法（PEFT）是什么？",
    options: [
      { text: "LoRA（低秩适配）", correct: true },
      { text: "全量重训视觉编码器", correct: false },
      { text: "直接重训整个大模型", correct: false },
      { text: "把权重全部归零", correct: false },
    ],
    explanation: "LoRA 冻结原权重、只训低秩增量矩阵，参数极少但效果好，是 VLM 微调的事实标准。",
    tags: ["VLM", "FineTuning"],
  },
  {
    id: 276,
    type: "single",
    question: "VLM 用 LoRA 微调时，通常把注意力放在哪些层上？",
    options: [
      { text: "LLM 主干的自注意力权重", correct: true },
      { text: "图像文件格式", correct: false },
      { text: "位置编码常量", correct: false },
      { text: "损失函数本身", correct: false },
    ],
    explanation: "LoRA 一般注入 LLM 的 attention 层的 q/k/v/o 权重上，让语言模型学会处理新的视觉指令，改动小、见效快。",
    tags: ["VLM", "FineTuning"],
  },
  {
    id: 277,
    type: "multiple",
    question: "关于 VLM 微调，说法正确的有？（多选）",
    options: [
      { text: "可冻结视觉塔只微调 LLM 与 Projector", correct: true },
      { text: "LoRA 能显著降低可训练参数量", correct: true },
      { text: "视觉指令数据质量影响微调效果", correct: true },
      { text: "微调必须重训全部参数", correct: false },
    ],
    explanation: "很多 VLM 冻结视觉塔只训投影与 LLM，配合 LoRA 降参数，而数据质量直接决定微调效果。",
    tags: ["VLM", "FineTuning"],
  },
  {
    id: 278,
    type: "single",
    question: "LLaVA 的视觉指令微调（visual instruction tuning）核心思路是什么？",
    options: [
      { text: "用图文指令数据让模型学会跟随图像提问作答", correct: true },
      { text: "只做纯文本问答", correct: false },
      { text: "禁用所有视觉输入", correct: false },
      { text: "用随机噪声训练", correct: false },
    ],
    explanation: "LLaVA 用大量「图像+人类指令+答案」的数据微调，教会模型看图后按指令作答，是视觉指令微调的里程碑。",
    tags: ["VLM", "FineTuning"],
  },
  {
    id: 279,
    type: "single",
    question: "VLM 的 few-shot / 多图输入是怎么实现的？",
    options: [
      { text: "把多张图都编码成视觉 token 拼进同一序列", correct: true },
      { text: "每次只允许一张图", correct: false },
      { text: "把图片转成音频", correct: false },
      { text: "用纯文字描述代替图像", correct: false },
    ],
    explanation: "多图就是各自编码成 token 后拼接进 prompt 序列，让模型同时参考多张图做推理，是 few-shot 的一种形态。",
    tags: ["VLM", "MultiImage"],
  },
  {
    id: 280,
    type: "multiple",
    question: "多图输入相比单图，VLM 面临的额外挑战有哪些？（多选）",
    options: [
      { text: "视觉 token 总量更大，序列更长", correct: true },
      { text: "KV cache 占用上升", correct: true },
      { text: "需要理解跨图像的关系", correct: true },
      { text: "图越多推理反而越省显存", correct: false },
    ],
    explanation: "多图叠加 token 变多、KV 变大，还得学会跨图比较和关联，都是多图场景的难点。",
    tags: ["VLM", "MultiImage"],
  },
  {
    id: 281,
    type: "single",
    question: "VLM 与扩散模型（如文生图/图文编辑）结合的主要思路是什么？",
    options: [
      { text: "用 VLM 提供细粒度语义/布局指导扩散生成", correct: true },
      { text: "让扩散模型取代 LLM", correct: false },
      { text: "完全丢弃视觉理解", correct: false },
      { text: "把图像转成纯文本再生成", correct: false },
    ],
    explanation: "VLM 负责理解指令和内容，扩散模型负责生成像素，两者配合实现更可控的文生图或图生图。",
    tags: ["VLM", "Diffusion"],
  },
  {
    id: 282,
    type: "multiple",
    question: "VLM 与扩散结合的常见应用场景包括？（多选）",
    options: [
      { text: "图像编辑/局部重绘", correct: true },
      { text: "根据文本与参考图生成新图", correct: true },
      { text: "多轮图文交互式创作", correct: true },
      { text: "纯文本拼写检查", correct: false },
    ],
    explanation: "结合场景普遍在图文编辑、参考图生成和交互式创作，核心是理解语义再控制生成。",
    tags: ["VLM", "Diffusion"],
  },
  {
    id: 283,
    type: "single",
    question: "CLIP 的对比学习训练目标的核心是什么？",
    options: [
      { text: "拉近匹配图文对、推远不匹配对", correct: true },
      { text: "让所有图文距离都为 0", correct: false },
      { text: "只最小化图像损失", correct: false },
      { text: "只训练文本编码器", correct: false },
    ],
    explanation: "CLIP 用 InfoNCE 式对比损失，让正确图文对相似度高、错误对低，从而学出对齐的共享向量空间。",
    tags: ["VLM", "CLIP"],
  },
  {
    id: 284,
    type: "single",
    question: "为什么多数 VLM 的视觉塔参数远小于 LLM，却能起关键作用？",
    options: [
      { text: "它负责把像素编码成可被 LLM 利用的高质量语义", correct: true },
      { text: "参数量大与小无关紧要", correct: false },
      { text: "视觉塔从不参与推理", correct: false },
      { text: "LLM 自己就能直接看像素", correct: false },
    ],
    explanation: "视觉塔虽小但负责「翻译」像素为语义，是信息入口，质量直接决定 LLM 能看到什么。",
    tags: ["VLM", "Architecture"],
  },
  {
    id: 285,
    type: "single",
    question: "视觉 token 过密（如超高分辨率）对训练/推理最直接的影响是？",
    options: [
      { text: "注意力计算量二次增长，显存与耗时上升", correct: true },
      { text: "反而让训练更省资源", correct: false },
      { text: "token 数不影响注意力复杂度", correct: false },
      { text: "只要调低学习率就无影响", correct: false },
    ],
    explanation: "注意力对 token 数是 O(n²)，token 一多计算和显存都大幅上涨，所以需要分辨率与 token 数的权衡。",
    tags: ["VLM", "Efficiency"],
  },
  {
    id: 286,
    type: "multiple",
    question: "VLM 中减少视觉 token 数量的技术有哪些？（多选）",
    options: [
      { text: "token 合并/池化", correct: true },
      { text: "pruning 丢弃冗余 token", correct: true },
      { text: "用更小 patch 增加细节", correct: true },
      { text: "把 token 数量加倍", correct: false },
    ],
    explanation: "合并、剪枝都能降 token 数，小 patch 虽增加细节但也增加 token，方向相反，需按需求权衡。",
    tags: ["VLM", "Efficiency"],
  },
  {
    id: 287,
    type: "single",
    question: "VLM 在做「指代理解」（如「请指出红框里的物体」）时依赖什么？",
    options: [
      { text: "视觉 token 的空间位置与坐标定位能力", correct: true },
      { text: "只依赖语言先验", correct: false },
      { text: "关闭视觉输入", correct: false },
      { text: "随机猜测位置", correct: false },
    ],
    explanation: "指代要精准对应图像区域，靠视觉 token 的空间位置编码和定位能力，纯语言先验猜不中具体位置。",
    tags: ["VLM", "SpatialReasoning"],
  },
  {
    id: 288,
    type: "single",
    question: "VLM 训练通常分哪两阶段进行？",
    options: [
      { text: "先对齐（预训练）后指令微调", correct: true },
      { text: "先指令微调后对齐", correct: false },
      { text: "只做随机初始化", correct: false },
      { text: "无需任何训练", correct: false },
    ],
    explanation: "先做图文对齐预训练让视觉塔与 LLM 连上，再做视觉指令微调提升按指令看图作答能力，是 LLaVA 的经典流程。",
    tags: ["VLM", "Training"],
  },
  {
    id: 289,
    type: "multiple",
    question: "视觉指令微调的数据通常包含哪些成分？（多选）",
    options: [
      { text: "图像与对应指令", correct: true },
      { text: "期望的参考回答", correct: true },
      { text: "细粒度定位/关系标注", correct: true },
      { text: "纯随机噪声", correct: false },
    ],
    explanation: "指令数据要含图、指令、答案，越丰富越能覆盖 OCR、定位、关系等能力，随机噪声毫无帮助。",
    tags: ["VLM", "Training"],
  },
  {
    id: 290,
    type: "single",
    question: "为什么纯文本预训练的大模型直接叠加视觉塔往往效果差？",
    options: [
      { text: "缺少图文对齐训练，视觉特征无法被 LLM 解读", correct: true },
      { text: "模型参数过多", correct: false },
      { text: "视觉塔太小", correct: false },
      { text: "图像格式问题", correct: false },
    ],
    explanation: "LLM 只学过文本嵌入，视觉特征若未对齐就是「外语」，必须经 Projector 和对齐训练才能被理解。",
    tags: ["VLM", "Alignment"],
  },
  {
    id: 291,
    type: "single",
    question: "VLM 中 CLS token（分类 token）常被用于什么？",
    options: [
      { text: "汇总整图信息做全局图文匹配", correct: true },
      { text: "代表每个 patch 的局部信息", correct: false },
      { text: "只用于文本生成", correct: false },
      { text: "存储图像文件名", correct: false },
    ],
    explanation: "ViT 的 CLS token 聚集整图全局信息，CLIP 用它和文本 embedding 对比算相似度，用于图文匹配。",
    tags: ["VLM", "CLIP"],
  },
  {
    id: 292,
    type: "multiple",
    question: "多模态幻觉的典型表现有哪些？（多选）",
    options: [
      { text: "说图里不存在的物体", correct: true },
      { text: "编造图中没有的文字或数字", correct: true },
      { text: "对图像区域给出错误归属", correct: true },
      { text: "精确复述图里真实内容", correct: false },
    ],
    explanation: "幻觉就是输出与图像证据不符，包括虚构对象、编造文字、错判区域，真实复述反而是正确行为。",
    tags: ["VLM", "Hallucination"],
  },
  {
    id: 293,
    type: "single",
    question: "VLM 做视觉问答（VQA）时，图像 token 通常放在 prompt 的什么位置？",
    options: [
      { text: "作为序列的一部分拼接在文本之前", correct: true },
      { text: "只能放在序列末尾", correct: false },
      { text: "完全脱离序列", correct: false },
      { text: "与文本分开单独算", correct: false },
    ],
    explanation: "图像 token 一般作为前缀拼进输入序列，再跟指令和问题，LLM 用自回归统一生成回答。",
    tags: ["VLM", "Architecture"],
  },
  {
    id: 294,
    type: "single",
    question: "决定 VLM 长文本+多图推理资源开销的最核心变量是什么？",
    options: [
      { text: "总序列长度（文本+视觉 token 数）", correct: true },
      { text: "图像文件扩展名", correct: false },
      { text: "训练时的学习率", correct: false },
      { text: "模型发布版本号", correct: false },
    ],
    explanation: "序列越长，注意力 O(n²) 与 KV cache 都越大，总序列长度是长上下文多图推理资源开销的最核心变量。",
    tags: ["VLM", "Efficiency"],
  },
  {
    id: 295,
    type: "single",
    question: "VLM 相比纯文本 LLM，在输入侧最本质的差异是什么？",
    options: [
      { text: "多了把像素转成 token 的视觉编码与投影链路", correct: true },
      { text: "语言模型完全不一样", correct: false },
      { text: "不需要 tokenizer", correct: false },
      { text: "没有注意力机制", correct: false },
    ],
    explanation: "差异在于多了一条「像素→视觉特征→投影 token」的输入链路，语言理解部分仍由 LLM 负责。",
    tags: ["VLM", "Architecture"],
  },
  {
    id: 296,
    type: "single",
    question: "InternVL 视觉塔通常基于什么预训练模型初始化？",
    options: [
      { text: "InternViT（大规模视觉对比预训练）", correct: true },
      { text: "随机权重", correct: false },
      { text: "纯文本词表", correct: false },
      { text: "音频模型", correct: false },
    ],
    explanation: "InternVL 用专门大规模图文对比预训练的 InternViT 作为视觉塔，视觉能力强，配合动态切图效果好。",
    tags: ["VLM", "Architecture"],
  },
  {
    id: 297,
    type: "multiple",
    question: "VLM 部署到边缘/低显存设备可采用的策略包括？（多选）",
    options: [
      { text: "量化视觉塔与 LLM 权重", correct: true },
      { text: "token 压缩降低序列长度", correct: true },
      { text: "用 KV cache 压缩技术", correct: true },
      { text: "把序列无限拉长", correct: false },
    ],
    explanation: "量化、token 压缩、KV 压缩都是降显存的实用手段，拉长序列只会让显存更紧张。",
    tags: ["VLM", "Deployment"],
  },
  {
    id: 298,
    type: "single",
    question: "要判断一个 VLM 是否真正「看见」图像而非只靠语言先验，最可靠的方式是什么？",
    options: [
      { text: "用视觉强依赖、语言先验无效的对抗/细节样例测试", correct: true },
      { text: "只看模型参数量", correct: false },
      { text: "只问常识问题", correct: false },
      { text: "看它有没有网络连接", correct: false },
    ],
    explanation: "设计语言猜不中的视觉样例（如数字、布局、位置），如果模型答对就说明真在利用视觉信息，这也是防幻觉的思路。",
    tags: ["VLM", "Evaluation"],
  },

// 题型说明：
//   type: "single" 单选（只有一个正确选项）
//   type: "multiple" 多选（有多个正确选项）
//   options: 选项列表，{ text: 选项内容, correct: 是否为正确选项 }
//   explanation: 答案解析，答完后展示
//   tags: 题目所属知识点标签（用于展示）


  {
    id: 299,
    type: "multiple",
    question: "以下哪些属于 Agent 相对普通 LLM 调用的典型特征？",
    options: [
      { text: "能循环执行「思考-行动-观察」并主动调用工具", correct: true },
      { text: "根据反馈迭代修正，直到达成目标", correct: true },
      { text: "必须运行在更大的模型上", correct: false },
      { text: "只能处理纯文本输入", correct: false },
    ],
    explanation:
      "Agent 在 LLM 外面套了「决策-行动-观察」的循环，能反复调工具、根据反馈修正。它不一定用更大模型，也不限纯文本。",
    tags: ["Agent"],
  },
  {
    id: 300,
    type: "single",
    question: "ReAct 中的「Act」指的是什么？",
    options: [
      { text: "模型对输入做注意力计算", correct: false },
      { text: "根据推理结果采取行动，通常是调用工具", correct: true },
      { text: "对历史对话做衰减", correct: false },
      { text: "更新模型的梯度", correct: false },
    ],
    explanation:
      "ReAct 循环是 Thought（推理）→ Action（行动，一般是调工具）→ Observation（观察结果）。Act 就是那个「动手调工具」的环节。",
    tags: ["Agent", "ReAct"],
  },
  {
    id: 301,
    type: "single",
    question: "Toolformer 方法的核心思想是什么？",
    options: [
      { text: "用强化学习训练 Agent 的长期记忆", correct: false },
      { text: "让模型自我监督地学会在合适位置插入 API 调用，而不是固定死工具流程", correct: true },
      { text: "把多个模型蒸馏成一个", correct: false },
      { text: "用工具替代模型的所有推理", correct: false },
    ],
    explanation:
      "Toolformer 用自监督方式给文本标注出「该调用哪个工具」的位置并训练模型插入调用，让模型自己学会何时用工具，而不是人为写死调用规则。",
    tags: ["Agent", "Toolformer"],
  },
  {
    id: 302,
    type: "single",
    question: "Function Calling 中，模型「返回工具调用」在 API 层面的表现形式是？",
    options: [
      { text: "把工具输出拼接到系统提示里", correct: false },
      { text: "在响应中返回结构化的 function call，包含函数名和参数，而不是最终文本", correct: true },
      { text: "修改模型的权重", correct: false },
      { text: "直接生成一段可执行脚本", correct: false },
    ],
    explanation:
      "Function Calling 的关键是模型先不吐最终答案，而是返回一个结构化工具调用（函数名+参数 JSON），由宿主代码去执行再回喂结果。",
    tags: ["Agent", "Function Calling"],
  },
  {
    id: 303,
    type: "single",
    question: "在 Agent 循环中，`think-act-observe` 里的 `observe` 环节通常发生在什么之后？",
    options: [
      { text: "在第一次思考之前", correct: false },
      { text: "工具被实际执行并返回结果之后", correct: true },
      { text: "在最终答案生成之后", correct: false },
      { text: "在初始化上下文之后", correct: false },
    ],
    explanation:
      "顺序是先想、再调用工具，工具执行完返回输出，这就是 observe。观察结果会作为新的输入喂回给模型，驱动下一轮思考。",
    tags: ["Agent"],
  },
  {
    id: 304,
    type: "single",
    question: "Agent 出现「死循环」的典型表现是？",
    options: [
      { text: "模型拒绝回答", correct: false },
      { text: "反复调用同一个工具或步骤，得不到进展也停不下来", correct: true },
      { text: "输出内容过短", correct: false },
      { text: "一次只调用一个工具", correct: false },
    ],
    explanation:
      "死循环是 Agent 的常见失败模式：模型不断重复调用同个工具或陷入相同步骤，因为没有终止条件或工具结果没让它前进。",
    tags: ["Agent", "失败模式"],
  },
  {
    id: 305,
    type: "single",
    question: "缓解 Agent 死循环的最直接手段是什么？",
    options: [
      { text: "增大模型参数量", correct: false },
      { text: "设置最大迭代步数上限，超限强制终止", correct: true },
      { text: "去掉工具调用功能", correct: false },
      { text: "让模型一次输出更多 token", correct: false },
    ],
    explanation:
      "给循环加个最大步数限制是最实用也最稳的一招——无论模型怎么绕，步数一到就强制结束并报错或给部分结果。",
    tags: ["Agent", "失败模式"],
  },
  {
    id: 306,
    type: "single",
    question: "Agent 的「规划器/执行器分离」（Planner-Executor）设计中，规划器的主要职责是？",
    options: [
      { text: "执行每个工具调用", correct: false },
      { text: "把大目标拆解成子任务并排序", correct: true },
      { text: "维护数据库连接", correct: false },
      { text: "格式化最终输出", correct: false },
    ],
    explanation:
      "规划器只负责拆解和排布任务、决定「做什么」，执行器再一步步把子任务跑掉。职责分离让 Agent 更可控。",
    tags: ["Agent", "规划"],
  },
  {
    id: 307,
    type: "single",
    question: "关于「单步推理 vs 多步推理」，下列说法正确的是？",
    options: [
      { text: "多步推理必然比单步更准，且永远更好", correct: false },
      { text: "多步推理通过逐步推理+中间反馈能处理更复杂问题，但成本更高、更易出错", correct: true },
      { text: "单步推理无法做任何工具调用", correct: false },
      { text: "多步推理只适用于文本问答", correct: false },
    ],
    explanation:
      "多步推理能拆解难题、结合中间结果，但每步都有出错累积风险，token 和延迟成本也更高，不是无条件更好。",
    tags: ["Agent", "推理"],
  },
  {
    id: 308,
    type: "single",
    question: "Agent 的系统提示（system prompt）里最不适合放入什么？",
    options: [
      { text: "工具的使用规则和边界", correct: false },
      { text: "可动态变化的长对话内容", correct: true },
      { text: "输出格式要求", correct: false },
      { text: "安全与授权约束", correct: false },
    ],
    explanation:
      "系统提示是静态指令，不该塞频繁变动的长内容（那些放用户消息或额外上下文里），否则既浪费窗口又可能被忽略。",
    tags: ["Agent", "系统提示"],
  },
  {
    id: 309,
    type: "single",
    question: "Agent 上下文窗口有限的直接后果是？",
    options: [
      { text: "模型训练变慢", correct: false },
      { text: "多轮工具调用后，早期关键信息可能被截断或遗忘", correct: true },
      { text: "工具完全无法被调用", correct: false },
      { text: "输出 token 数量被固定", correct: false },
    ],
    explanation:
      "Agent 多轮循环会累积大量观察结果，窗口满了早期信息就得被丢。这推动了压缩、摘要、检索等记忆方案的出现。",
    tags: ["Agent", "上下文窗口"],
  },
  {
    id: 310,
    type: "single",
    question: "长期记忆（long-term memory）在 Agent 中通常用来解决什么问题？",
    options: [
      { text: "加快单次推理速度", correct: false },
      { text: "让 Agent 跨会话/跨任务持久化保存和回忆经验", correct: true },
      { text: "压缩短期工作记忆", correct: false },
      { text: "替代工具调用", correct: false },
    ],
    explanation:
      "长期记忆把经验存到外部（向量库、数据库等），让 Agent 在后续任务里能重新取用，应对上下文窗口有限的短板。",
    tags: ["Agent", "记忆"],
  },
  {
    id: 311,
    type: "single",
    question: "短期记忆（short-term memory）在 Agent 中最直接对应的是什么？",
    options: [
      { text: "模型训练数据", correct: false },
      { text: "当前会话的上下文窗口内容", correct: true },
      { text: "磁盘上的长期存储", correct: false },
      { text: "参数权重", correct: false },
    ],
    explanation:
      "短期记忆就是当前正在处理的上下文——对话历史、中间观察结果。它容量有限，随轮次增加会被淘汰。",
    tags: ["Agent", "记忆"],
  },
  {
    id: 312,
    type: "single",
    question: "RAG 与 Agent 结合的核心价值是？",
    options: [
      { text: "减少 Agent 的推理步骤", correct: false },
      { text: "让 Agent 按需检索外部知识来补充事实，减少幻觉", correct: true },
      { text: "替代 Agent 的记忆系统", correct: false },
      { text: "让模型参数变大", correct: false },
    ],
    explanation:
      "RAG 给 Agent 提供按需取用外部资料的能力，让它在不确定时先去查，而不是硬编，从而降低幻觉、补齐事实。",
    tags: ["Agent", "RAG"],
  },
  {
    id: 313,
    type: "single",
    question: "多 Agent 协作中最常见的组织方式是？",
    options: [
      { text: "所有 Agent 完全独立、互不通信", correct: false },
      { text: "按角色分工（如规划者、执行者、审查者），相互传递信息", correct: true },
      { text: "所有 Agent 共享同一份参数", correct: false },
      { text: "只有一个 Agent 能调用工具", correct: false },
    ],
    explanation:
      "多 Agent 常按不同角色拆解（规划/执行/审查/反思），通过消息或共享存储协作，像团队一样分工配合。",
    tags: ["Agent", "多Agent"],
  },
  {
    id: 314,
    type: "single",
    question: "Agent 授权与安全中最需要警惕的「工具越权」风险是什么？",
    options: [
      { text: "模型生成了格式错误的 JSON", correct: false },
      { text: "模型被诱导去调用危险/高权限工具（如删数据、转账）", correct: true },
      { text: "工具响应太长", correct: false },
      { text: "模型输出包含 emoji", correct: false },
    ],
    explanation:
      "安全核心是权限边界：危险工具要加审批、白名单和最小权限，防止模型被 prompt 注入或误操作触发高风险动作。",
    tags: ["Agent", "安全"],
  },
  {
    id: 315,
    type: "single",
    question: "Prompt Injection（提示注入）在 Agent 场景下的主要危害是？",
    options: [
      { text: "让模型训练崩溃", correct: false },
      { text: "攻击者通过外部内容（网页、工具返回）劫持 Agent 行为去执行恶意指令", correct: true },
      { text: "模型输出变短", correct: false },
      { text: "增大上下文开销", correct: false },
    ],
    explanation:
      "Agent 会读取外部内容（网页、搜索结果），攻击者把恶意指令藏进去，就能让模型「被劫持」执行不该做的动作。",
    tags: ["Agent", "安全"],
  },
  {
    id: 316,
    type: "single",
    question: "结构化输出（如 JSON / function schema）对 Agent 的作用是？",
    options: [
      { text: "只影响界面显示，与功能无关", correct: false },
      { text: "让模型按约定 schema 返回，方便程序可靠地解析和驱动工具调用", correct: true },
      { text: "提高模型训练速度", correct: false },
      { text: "减少模型参数量", correct: false },
    ],
    explanation:
      "Agent 要「被程序消费」，必须输出机器可解析的结构（JSON、function call）。结构化成约定后程序才能稳定解析、执行工具。",
    tags: ["Agent", "结构化输出"],
  },
  {
    id: 317,
    type: "single",
    question: "评估一个 Agent 是否好用，最合理的方式是？",
    options: [
      { text: "只看它单次回答是否正确", correct: false },
      { text: "用覆盖多任务、能自动判分的评测集，看任务完成率和成功率", correct: true },
      { text: "让作者主观读一遍", correct: false },
      { text: "只看模型参数量大小", correct: false },
    ],
    explanation:
      "Agent 是流程型系统，评测应看多任务下的整体成功率、完成率、步数效率等，用可自动判分的基准而不是拍脑袋。",
    tags: ["Agent", "评测"],
  },
  {
    id: 318,
    type: "single",
    question: "Agent 状态管理（state management）主要指的是？",
    options: [
      { text: "管理操作系统进程状态", correct: false },
      { text: "跟踪 Agent 执行过程中的目标、步骤进度、中间结果和已用工具", correct: true },
      { text: "修改模型权重", correct: false },
      { text: "管理数据库事务", correct: false },
    ],
    explanation:
      "Agent 是状态机：要维护「进行到哪一步、还有哪些待办、中间产出是什么」，跨步骤/跨会话保持一致，否则容易迷失。",
    tags: ["Agent", "状态管理"],
  },
  {
    id: 319,
    type: "single",
    question: "Agent 的「反思」（Reflection）机制一般指什么？",
    options: [
      { text: "模型自我审视已完成的步骤，发现问题并修正策略", correct: true },
      { text: "模型反复输出同一个答案", correct: false },
      { text: "模型降低上下文长度", correct: false },
      { text: "模型提前终止任务", correct: false },
    ],
    explanation:
      "反思是 Agent 对自身执行过程做复盘，发现哪里错了、怎么改进，通常基于结果反馈来提升后续决策质量。",
    tags: ["Agent", "反思"],
  },
  {
    id: 320,
    type: "single",
    question: "在 Function Calling 中，工具参数通常以什么形式传给模型约定？",
    options: [
      { text: "自然语言段落", correct: false },
      { text: "JSON Schema 描述每个参数的类型和约束", correct: true },
      { text: "二进制文件", correct: false },
      { text: "Markdown 表格", correct: false },
    ],
    explanation:
      "工具用 JSON Schema 声明参数结构（名称、类型、必填项、枚举等），模型据此生成符合规范的参数，保证可解析。",
    tags: ["Agent", "Function Calling"],
  },
  {
    id: 321,
    type: "single",
    question: "Agent「工具选择」能力弱的表现是？",
    options: [
      { text: "总能选到最优工具", correct: false },
      { text: "不知道该用哪个工具、频繁选错或给出非法参数", correct: true },
      { text: "工具执行过快", correct: false },
      { text: "工具说明写得太长", correct: false },
    ],
    explanation:
      "工具选择依赖模型理解工具说明和当前目标。选错工具或填错参数说明模型对工具语义理解不到位。",
    tags: ["Agent", "工具选择"],
  },
  {
    id: 322,
    type: "single",
    question: "Agent 幻觉（hallucination）在工具调用场景最常见的表现是？",
    options: [
      { text: "模型诚实承认自己不会", correct: false },
      { text: "虚构一个不存在的工具名或参数、凭空编造工具结果", correct: true },
      { text: "把参数填完整", correct: false },
      { text: "跳过所有工具直接回答", correct: false },
    ],
    explanation:
      "Agent 幻觉指模型编造不存在的工具或结果——比如调用一个根本没注册的函数，或对观察结果「脑补」。这是评测中的典型失败点。",
    tags: ["Agent", "失败模式"],
  },
  {
    id: 323,
    type: "single",
    question: "Agent 评测中「成功率」（success rate）通常衡量的是？",
    options: [
      { text: "单次 token 生成速度", correct: false },
      { text: "Agent 能否在允许的步骤内正确完成目标任务的占比", correct: true },
      { text: "工具返回内容长度", correct: false },
      { text: "模型吞吐量", correct: false },
    ],
    explanation:
      "成功率就是「任务最终做成的比例」，是 Agent 评测最核心的指标之一，常配合最大步数限制一起统计。",
    tags: ["Agent", "评测"],
  },
  {
    id: 324,
    type: "single",
    question: "Agent 中「记忆压缩」（summarize/condense）的主要目的是？",
    options: [
      { text: "提高模型参数量", correct: false },
      { text: "在上下文窗口有限的情况下保留关键信息、继续推进长任务", correct: true },
      { text: "让工具跑得更快", correct: false },
      { text: "替代系统提示", correct: false },
    ],
    explanation:
      "长任务里上下文会爆，压缩/摘要把冗长历史提炼成要点塞回窗口，牺牲一点细节换可继续运行。",
    tags: ["Agent", "记忆"],
  },
  {
    id: 325,
    type: "single",
    question: "Agent「规划」环节失败，最常见的后果是？",
    options: [
      { text: "模型无法启动", correct: false },
      { text: "任务拆分错误、目标偏离，做了无用步骤或走弯路", correct: true },
      { text: "工具永远不会被调用", correct: false },
      { text: "上下文一定溢出", correct: false },
    ],
    explanation:
      "规划负责「做什么、怎么做」。规划错了后面全跑偏，这也是评测里看任务是否真正达成的重要原因。",
    tags: ["Agent", "规划"],
  },
  {
    id: 326,
    type: "single",
    question: "让 Agent 更准确地选择工具，通常最有效的做法是？",
    options: [
      { text: "把每个工具写成清晰的名称+描述+参数 schema，并给出示例", correct: true },
      { text: "给工具用最简短的命名", correct: false },
      { text: "隐藏工具描述只留函数名", correct: false },
      { text: "把所有工具合并成一个", correct: false },
    ],
    explanation:
      "工具说明质量直接决定选择准确率：清晰的描述、参数 schema 和示例能让模型知道何时用、参数怎么填。",
    tags: ["Agent", "工具选择"],
  },
  {
    id: 327,
    type: "single",
    question: "Agent 中「反思后重试」相比直接重试的关键区别是？",
    options: [
      { text: "反思后重试先分析失败原因再调整策略，而非盲目重跑", correct: true },
      { text: "直接重试一定更准", correct: false },
      { text: "反思后重试会禁用工具", correct: false },
      { text: "两种没有区别", correct: false },
    ],
    explanation:
      "盲目重试重复同样的错误；带反思的重试会先复盘「为什么失败」再改策略，通常能显著提升成功率。",
    tags: ["Agent", "反思"],
  },
  {
    id: 328,
    type: "single",
    question: "Agent 授权（tool authorization）中「最小权限原则」指？",
    options: [
      { text: "给 Agent 尽量多的权限以完成任务", correct: false },
      { text: "只授予完成当前任务所必需的最小权限，降低风险", correct: true },
      { text: "所有工具永久免审批", correct: false },
      { text: "权限越高越好", correct: false },
    ],
    explanation:
      "最小权限就是「够用就好」，高风险操作加审批或限制，避免 Agent 因注入或误操作造成大范围破坏。",
    tags: ["Agent", "安全"],
  },
  {
    id: 329,
    type: "single",
    question: "Agent 循环里「最大迭代步数」为什么是必要的？",
    options: [
      { text: "它能让模型输出更长的答案", correct: false },
      { text: "防止死循环并控制延迟与成本", correct: true },
      { text: "它决定了模型参数量", correct: false },
      { text: "它用于训练数据切分", correct: false },
    ],
    explanation:
      "设上限既是兜底（防死循环），也是预算控制（限制步数=控制 token 和时间成本），是工程上的常规护栏。",
    tags: ["Agent", "失败模式"],
  },
  {
    id: 330,
    type: "single",
    question: "Agent 与 LLM 在「单次调用」上的关系，最准确的说法是？",
    options: [
      { text: "Agent 就是一次 LLM 调用", correct: false },
      { text: "Agent 以 LLM 为推理核心，通过循环多次调用 LLM 并驱动工具", correct: true },
      { text: "Agent 完全不需要 LLM", correct: false },
      { text: "LLM 只能用于 Agent 的输入", correct: false },
    ],
    explanation:
      "LLM 是 Agent 的「大脑」，Agent 每轮思考都是一次 LLM 调用，核心是围绕它构建循环和工具能力，而不是一次性问答。",
    tags: ["Agent"],
  },
  {
    id: 331,
    type: "single",
    question: "ReAct 名字里的「Re」代表 Reasoning，它的作用是？",
    options: [
      { text: "决定是否停止工具调用", correct: false },
      { text: "先想清楚再决定行动，让行动有依据", correct: true },
      { text: "压缩上下文", correct: false },
      { text: "分配 GPU 资源", correct: false },
    ],
    explanation:
      "Re=Reasoning（推理）：模型先产出思考过程，基于推理再选行动。这样每个动作都有「为什么」，提高可控性和正确率。",
    tags: ["Agent", "ReAct"],
  },
  {
    id: 332,
    type: "single",
    question: "当 Agent 的工具返回「空结果或报错」时，合理的处理是？",
    options: [
      { text: "直接把错误当最终答案", correct: false },
      { text: "让模型解析错误信息，调整参数或换工具重试", correct: true },
      { text: "无视结果继续下一步", correct: false },
      { text: "立刻终止整个 Agent", correct: false },
    ],
    explanation:
      "错误本身是信息：模型应读取报错，判断是参数不对还是选错工具，据此修正重试，这是 Agent 鲁棒性的关键。",
    tags: ["Agent", "工具调用"],
  },
  {
    id: 333,
    type: "single",
    question: "Agent 中「规划器」把大任务拆成子任务后，下一步通常交给谁？",
    options: [
      { text: "交给规划器自己重新规划", correct: false },
      { text: "交给执行器（executor）逐个执行", correct: true },
      { text: "交给数据库存储", correct: false },
      { text: "交给用户手动完成", correct: false },
    ],
    explanation:
      "规划器只管拆和排，执行器负责实际跑子任务。两者解耦让「想」和「做」各司其职，便于并行和监控。",
    tags: ["Agent", "规划"],
  },
  {
    id: 334,
    type: "single",
    question: "多 Agent 协作中，「主从/管理」（manager-worker）模式的典型特征是？",
    options: [
      { text: "所有 Agent 权力平等无层级", correct: false },
      { text: "一个管理 Agent 负责任务拆解和调度，多个工作 Agent 执行", correct: true },
      { text: "只有一个 Agent 存在", correct: false },
      { text: "所有 Agent 共享同一输出", correct: false },
    ],
    explanation:
      "manager 拆任务、分配、收结果；worker 各自执行。层级清晰，适合把复杂任务纵向拆给多个子 Agent 并行处理。",
    tags: ["Agent", "多Agent"],
  },
  {
    id: 335,
    type: "multiple",
    question: "以下哪些属于 Agent 的常见失败模式？",
    options: [
      { text: "陷入死循环", correct: true },
      { text: "幻觉、虚构不存在的工具或结果", correct: true },
      { text: "输出过长导致超时", correct: false },
      { text: "上下文溢出导致早期信息丢失", correct: true },
    ],
    explanation:
      "死循环、幻觉、上下文溢出都是典型失败点。输出过长一般是成本问题，不算独立失败模式。",
    tags: ["Agent", "失败模式"],
  },
  {
    id: 336,
    type: "multiple",
    question: "Agent 的「记忆」体系通常包含哪些层次？",
    options: [
      { text: "短期记忆（当前上下文）", correct: true },
      { text: "长期记忆（跨会话外部存储）", correct: true },
      { text: "GPU 显存", correct: false },
      { text: "压缩/摘要后的工作记忆", correct: true },
    ],
    explanation:
      "记忆分短期、长期，还有为了省窗口做的压缩摘要。显存是硬件，跟语义记忆没关系。",
    tags: ["Agent", "记忆"],
  },
  {
    id: 337,
    type: "multiple",
    question: "以下哪些措施有助于缓解 Agent 的 prompt injection 风险？",
    options: [
      { text: "对高权限工具强制人工审批", correct: true },
      { text: "把外部内容与可信指令在提示中明确分隔", correct: true },
      { text: "给 Agent 分配最小权限", correct: true },
      { text: "完全禁止 Agent 读取任何外部内容", correct: false },
    ],
    explanation:
      "审批、分隔指令、最小权限都能降低注入危害。禁止读外部内容会砍掉 Agent 核心能力，不现实。",
    tags: ["Agent", "安全"],
  },
  {
    id: 338,
    type: "multiple",
    question: "关于 Agent 评测，以下哪些指标是合理的？",
    options: [
      { text: "任务成功率", correct: true },
      { text: "平均所需步数/成本", correct: true },
      { text: "模型训练损失", correct: false },
      { text: "在多大程度上正确使用了预期工具", correct: true },
    ],
    explanation:
      "成功率、步数/成本、工具使用正确性都是实用指标。训练损失是训练期的，不是部署后 Agent 评测关注点。",
    tags: ["Agent", "评测"],
  },
  {
    id: 339,
    type: "multiple",
    question: "Agent 把 RAG 结合进来的常见做法包括？",
    options: [
      { text: "用检索工具在需要时查询知识库", correct: true },
      { text: "把检索结果作为上下文补充给推理", correct: true },
      { text: "用检索代替所有记忆", correct: false },
      { text: "把搜索结果喂回模型减少幻觉", correct: true },
    ],
    explanation:
      "RAG 作为工具按需检索、结果回填上下文，是主流。但它不替代记忆体系，只是其中一环。",
    tags: ["Agent", "RAG"],
  },
  {
    id: 340,
    type: "multiple",
    question: "Function Calling 相对「让模型自由输出 JSON」的优势包括？",
    options: [
      { text: "原生校验参数是否符合 schema", correct: true },
      { text: "工具按声明自动被发现和约束，减少格式错误", correct: true },
      { text: "无需任何解析容错", correct: false },
      { text: "与结构化输出严格对齐，可靠驱动工具执行", correct: true },
    ],
    explanation:
      "Function Calling 靠 schema 约束和校验，格式更稳、更可靠。但它仍可能输出非法参数，不能完全省掉容错。",
    tags: ["Agent", "Function Calling"],
  },
  {
    id: 341,
    type: "multiple",
    question: "Agent 上下文溢出时，可以采用的缓解策略有？",
    options: [
      { text: "对历史做摘要压缩", correct: true },
      { text: "把不相关旧内容移到外部存储", correct: true },
      { text: "用 RAG 检索关键信息", correct: true },
      { text: "无脑丢弃所有早期内容", correct: false },
    ],
    explanation:
      "压缩、外移、检索都能续命。无脑全丢会丢失关键状态，属于粗暴做法。",
    tags: ["Agent", "上下文窗口"],
  },
  {
    id: 342,
    type: "multiple",
    question: "多 Agent 协作相比单 Agent 的潜在收益有？",
    options: [
      { text: "可通过分工并行处理，降低单点上下文压力", correct: true },
      { text: "不同角色可专注不同子任务，提升专业性", correct: true },
      { text: "一定比单 Agent 更快更准", correct: false },
      { text: "可用审查/反思角色交叉检查错误", correct: true },
    ],
    explanation:
      "分工、并行、交叉审查是收益；但多 Agent 通信和协调本身有开销，并不保证一定更快更准。",
    tags: ["Agent", "多Agent"],
  },
  {
    id: 343,
    type: "multiple",
    question: "Agent 的状态管理需要记录哪些信息？",
    options: [
      { text: "当前目标与子任务进度", correct: true },
      { text: "已调用过的工具和中间结果", correct: true },
      { text: "剩余待办事项", correct: true },
      { text: "仅最终答案", correct: false },
    ],
    explanation:
      "状态要覆盖目标、进度、已用工具、待办，跨步骤保持一致。只记最终答案无法支撑多步执行。",
    tags: ["Agent", "状态管理"],
  },
  {
    id: 344,
    type: "multiple",
    question: "Agent 反思机制可以借助哪些反馈来改进后续决策？",
    options: [
      { text: "工具执行结果是否成功", correct: true },
      { text: "最终答案是否符合目标", correct: true },
      { text: "失败的具体原因分析", correct: true },
      { text: "模型参数量", correct: false },
    ],
    explanation:
      "反思基于执行结果、目标符合度、失败原因做复盘。参数量是模型属性，不是反馈来源。",
    tags: ["Agent", "反思"],
  },
  {
    id: 345,
    type: "multiple",
    question: "规划器/执行器分离设计的优点包括？",
    options: [
      { text: "规划与执行可独立优化", correct: true },
      { text: "便于并行和监控进度", correct: true },
      { text: "完全消除 Agent 的错误", correct: false },
      { text: "降低单一模型同时承担想与做的复杂度", correct: true },
    ],
    explanation:
      "解耦后想与做各司其职、可独立调优。但它不会消除错误，只是让结构更清晰可控。",
    tags: ["Agent", "规划"],
  },
  {
    id: 346,
    type: "multiple",
    question: "以下哪些是 Agent「工具参数填充」正确性的关键因素？",
    options: [
      { text: "参数 schema 定义是否清晰", correct: true },
      { text: "工具描述是否说明参数含义", correct: true },
      { text: "是否有参数校验和错误重试", correct: true },
      { text: "工具名称是否足够长", correct: false },
    ],
    explanation:
      "清晰的 schema、描述和校验重试能保证参数正确。名称长度与正确性无关。",
    tags: ["Agent", "工具调用"],
  },
  {
    id: 347,
    type: "multiple",
    question: "Agent 安全设计中，针对高风险工具可以采取哪些措施？",
    options: [
      { text: "加入白名单和权限校验", correct: true },
      { text: "执行前人工审批", correct: true },
      { text: "对敏感操作记录审计日志", correct: true },
      { text: "给 Agent 授予超级权限以完成任务", correct: false },
    ],
    explanation:
      "白名单、审批、审计都是安全手段。授予超级权限反而放大风险，违背最小权限原则。",
    tags: ["Agent", "安全"],
  },
  {
    id: 348,
    type: "multiple",
    question: "关于 Agent 的思考与行动，以下说法正确的是？",
    options: [
      { text: "多步推理允许结合中间观察结果逐步修正", correct: true },
      { text: "先思考再行动的 ReAct 模式提升可控性", correct: true },
      { text: "思考越多越好，没有成本", correct: false },
      { text: "每一步都应基于当前状态做决策", correct: true },
    ],
    explanation:
      "多步推理能修正，ReAct 提升可控性，每步基于状态决策都是对的。但思考有 token 和延迟成本，不是越多越好。",
    tags: ["Agent", "推理"],
  },


  {
    id: 349,
    type: "single",
    question: "LangGraph 与普通 LangChain Chain 最核心的区别是什么？",
    options: [
      { text: "LangGraph 只支持同步执行，Chain 支持异步", correct: false },
      { text: "LangGraph 的图可以包含循环和条件分支，Chain 是单向线性流水线", correct: true },
      { text: "LangGraph 只能调用工具，Chain 不能", correct: false },
      { text: "二者没有本质区别，只是 API 命名不同", correct: false },
    ],
    explanation:
      "普通 Chain 是固定的线性 DAG（有向无环图），数据从上往下单向流过。LangGraph 的核心是引入循环（loops），让图能根据状态反复执行节点，这是实现 agent 的关键（agent 得能反复思考-行动）。",
    tags: ["LangGraph"],
  },
  {
    id: 350,
    type: "single",
    question: "在 StateGraph 中，节点（Node）的主要职责是什么？",
    options: [
      { text: "只负责调度其他节点，不处理数据", correct: false },
      { text: "接收当前状态，处理后返回更新字典，被 reducer 合并进状态", correct: true },
      { text: "每个节点都必须独立维护自己的数据库", correct: false },
      { text: "节点只能读状态，不能修改状态", correct: false },
    ],
    explanation:
      "节点是带状态的函数：输入是当前 State，返回值是一份 update dict（更新字典），LangGraph 用各 channel 的 reducer（归并函数）把它合并回共享状态。节点本身不直接持有状态，它只产出增量。",
    tags: ["LangGraph", "StateGraph"],
  },
  {
    id: 351,
    type: "single",
    question: "Graph 中的 START 和 END 节点分别代表什么？",
    options: [
      { text: "START 是入口虚拟节点，END 是出口虚拟节点，不执行真实逻辑", correct: true },
      { text: "START 和 END 都是真实执行的计算节点", correct: false },
      { text: "START 必须连接到 END，中间不能有别的节点", correct: false },
      { text: "END 节点负责压缩状态", correct: false },
    ],
    explanation:
      "START（入口）和 END（出口）是图里的两个特殊虚拟节点，用来标记执行的起点和终点，本身不跑逻辑。一个图必须能从 START 出发、最终到达 END，否则要么到不了（死循环）要么没出口。",
    tags: ["LangGraph", "StateGraph"],
  },
  {
    id: 352,
    type: "multiple",
    question: "关于 StateGraph 的状态管理，下列说法正确的有？（多选）",
    options: [
      { text: "状态是图执行过程中共享的全局数据", correct: true },
      { text: "多个节点可以并行读取同一个状态", correct: true },
      { text: "状态必须用字典形式保存，不能自定义类", correct: false },
      { text: "每个节点返回的字段必须与状态 schema 完全一致才能运行", correct: false },
    ],
    explanation:
      "State 是图的全局共享数据，由各个节点读写，并行分支也能共享读它。schema（状态定义）可以用 TypedDict，也可用自定义类（需要 __getitem__/__setitem__ 等接口）。节点只返回要更新的字段即可，不必把整个 schema 全返回。",
    tags: ["LangGraph", "State"],
  },
  {
    id: 353,
    type: "single",
    question: "状态里定义 channel 时用的 reducer 主要解决什么问题？",
    options: [
      { text: "解决节点执行速度不够快的问题", correct: false },
      { text: "解决多个节点同时写同一个 key 时如何合并更新", correct: true },
      { text: "给状态字段加上类型检查", correct: false },
      { text: "负责把状态保存到数据库", correct: false },
    ],
    explanation:
      "reducer（归并函数）回答「当多个来源同时更新同一个 channel 时该听谁的」。比如消息用 add_messages 做追加而不是覆盖，这样多个分支产生的消息能累加。默认无 reducer 的 key 是直接覆盖。",
    tags: ["LangGraph", "State"],
  },
  {
    id: 354,
    type: "single",
    question: "add_messages 这个 reducer 的典型作用是什么？",
    options: [
      { text: "给每条消息打上时间戳", correct: false },
      { text: "把新消息追加到已有消息列表，而不是覆盖", correct: true },
      { text: "过滤掉重复的模型输出", correct: false },
      { text: "把多轮对话压缩成一轮", correct: false },
    ],
    explanation:
      "add_messages（消息追加归并函数）把节点返回的新消息追加到列表末尾，同时合并相同 id 的消息。它是构建多轮对话 agent 的关键，让对话历史逐步累积，而不是被后一个节点整体覆盖。",
    tags: ["LangGraph", "State"],
  },
  {
    id: 355,
    type: "multiple",
    question: "关于条件边（conditional edges），下列说法正确的有？（多选）",
    options: [
      { text: "在运行时根据当前状态决定下一步走哪个分支", correct: true },
      { text: "常用于实现 agent 的「有工具就继续，否则结束」循环", correct: true },
      { text: "可以返回多个目标节点实现 fan-out 分支", correct: true },
      { text: "只能返回固定一个节点，无法动态选路", correct: false },
    ],
    explanation:
      "条件边是运行时根据状态决定路由的函数，可返回单个节点名，也可返回一组节点（配合 Send API 做 fan-out）。它是实现 agent 循环分支和并行分发的核心原语。",
    tags: ["LangGraph", "conditional edges"],
  },
  {
    id: 356,
    type: "single",
    question: "在 Agent 循环建模中，典型的「行动-观察-思考」循环用哪种结构实现？",
    options: [
      { text: "条件边把「执行工具」节点和「生成回复」节点连成环", correct: true },
      { text: "用多个独立的 Chain 首尾串联", correct: false },
      { text: "把所有逻辑塞进一个节点里顺序执行", correct: false },
      { text: "用正则表达式匹配输出", correct: false },
    ],
    explanation:
      "agent 是循环：模型生成 tool call → ToolNode 执行并返回结果 → 模型再基于结果继续。LangGraph 用条件边判断「还有工具调用就回到模型节点，没有就走到 END」，把这几个节点连成环。",
    tags: ["LangGraph", "Agent", "ToolNode"],
  },
  {
    id: 357,
    type: "single",
    question: "ToolNode 的核心职责是什么？",
    options: [
      { text: "把模型输出中带 tool_calls 的请求映射到对应工具执行", correct: true },
      { text: "负责训练一个工具选择器", correct: false },
      { text: "代替模型直接生成回答", correct: false },
      { text: "把工具结果加密存储", correct: false },
    ],
    explanation:
      "ToolNode 是现成的工具执行节点：它读状态里的消息，找出带 tool_calls（工具调用）的消息，按名称调用注册的工具，再把执行结果作为 ToolMessage 追加回状态。省得你自己手写「解析调用-执行-回填」的样板代码。",
    tags: ["LangGraph", "ToolNode"],
  },
  {
    id: 358,
    type: "multiple",
    question: "关于 LangGraph 的持久化与 checkpoint，下列正确的有？（多选）",
    options: [
      { text: "checkpoint 记录图在每一步的完整状态快照", correct: true },
      { text: "配置 checkpointer 后图支持从断点恢复执行", correct: true },
      { text: "checkpoint 只能保存在内存中，无法落盘", correct: false },
      { text: "checkpoint 会保存每个节点的完整日志文本", correct: false },
    ],
    explanation:
      "checkpointer（检查点器）在每一步保存状态快照，从而支持断点续跑、回放、分支。它既可以用内存实现，也可以用 SQLite 等持久化后端。它保存的是结构化状态，不是日志文本。",
    tags: ["LangGraph", "persistence", "checkpoint"],
  },
  {
    id: 359,
    type: "single",
    question: "interrupt 机制在 LangGraph 中主要用于实现什么能力？",
    options: [
      { text: "让图在指定节点暂停，等外部人工确认后再继续", correct: true },
      { text: "让模型在回答中途随机换一个主题", correct: false },
      { text: "把长图拆成多个短图", correct: false },
      { text: "提高流式输出的速度", correct: false },
    ],
    explanation:
      "interrupt（中断）让执行在某个节点停下来、把状态交出去，等待外部输入（比如人批准一笔操作）再继续。它是实现 Human-in-the-loop 和权限审批的核心原语。",
    tags: ["LangGraph", "interrupt", "human-in-the-loop"],
  },
  {
    id: 360,
    type: "single",
    question: "Human-in-the-loop 场景通常配合哪个机制实现「人工确认后再执行工具」？",
    options: [
      { text: "interrupt 挂起图，审核通过后再恢复", correct: true },
      { text: "把工具权限设为永远拒绝", correct: false },
      { text: "用多个 agent 并行竞争", correct: false },
      { text: "加大模型温度让输出更随机", correct: false },
    ],
    explanation:
      "这类「敏感操作需人工把关」的做法，是用 interrupt 在工具执行前挂起图、返回待审批状态，人确认后图从该点恢复继续执行。常见于支付、删除、外发请求等高风险动作。",
    tags: ["LangGraph", "human-in-the-loop", "interrupt"],
  },
  {
    id: 361,
    type: "multiple",
    question: "关于并行分支（fan-out），下列说法正确的有？（多选）",
    options: [
      { text: "一个节点通向多个后续节点，LangGraph 自动并行执行它们", correct: true },
      { text: "并行分支共享同一份状态，通过 reducer 合并各自写入", correct: true },
      { text: "互不依赖的分支可以并行调度以提升吞吐", correct: true },
      { text: "并行分支必须串行等待，一次只能跑一个", correct: false },
    ],
    explanation:
      "fan-out（扇出）即一个源节点分叉到多个目标节点，互不依赖时 LangGraph 会并行调度。各分支都写同一份状态，用 reducer（如 add_messages）合并写入；fan-in（聚合）则等所有分支完成后再继续。",
    tags: ["LangGraph", "parallel", "fan-out"],
  },
  {
    id: 362,
    type: "single",
    question: "fan-in（聚合）阶段需要把多个分支的结果合并时，主要靠什么机制？",
    options: [
      { text: "靠各 channel 的 reducer 合并不同分支的写入", correct: true },
      { text: "用 Python 的 threading 锁手动同步", correct: false },
      { text: "把结果拼成一个大字符串再切分", correct: false },
      { text: "重新跑一遍所有分支", correct: false },
    ],
    explanation:
      "多个并行分支各自写状态后，下游聚合节点会读到合并结果，而「怎么合并」由对应 channel 的 reducer 决定。比如用 add_messages 把各分支的产出都追加进列表，就实现了结果的汇聚。",
    tags: ["LangGraph", "fan-in", "State"],
  },
  {
    id: 363,
    type: "single",
    question: "图编译（compile）之后调用返回的是什么？",
    options: [
      { text: "一个可调用的 CompiledGraph，支持 invoke/stream 等方法", correct: true },
      { text: "一个 JSON 配置文件", correct: false },
      { text: "一组 Python 原生函数，只能手动逐个调用", correct: false },
      { text: "一个包含所有节点源码的对象", correct: false },
    ],
    explanation:
      "graph.compile() 把节点和边编译成一个 CompiledGraph（编译后的图对象），它有 invoke、stream、ainvoke 等统一接口，还带上 checkpointer 等运行时配置。之后就通过它来实际运行图。",
    tags: ["LangGraph", "compile"],
  },
  {
    id: 364,
    type: "multiple",
    question: "关于 LangGraph 的流式输出（stream），下列正确的有？（多选）",
    options: [
      { text: "默认模式按节点为单位产出 chunk，能看到每个节点输出的 token", correct: true },
      { text: "messages 模式专门流式输出模型生成的消息 token", correct: true },
      { text: "流式输出会把完整状态在每次更新时都全量重发", correct: false },
      { text: "stream 无法和 checkpoint 一起使用", correct: false },
    ],
    explanation:
      "默认 stream 模式按节点粒度输出（每个节点产出后给一坨状态更新）；stream_mode=\"messages\" 则聚焦于 LLM 的 token 级输出，适合打字机效果。它发的是增量更新而非全量快照，也能和 checkpointer 配合。",
    tags: ["LangGraph", "stream"],
  },
  {
    id: 365,
    type: "single",
    question: "构建多 Agent 图时，最常见的组织方式是什么？",
    options: [
      { text: "用一个 supervisor 节点决定把任务交给哪个子 agent", correct: true },
      { text: "所有 agent 同时执行并覆盖彼此状态", correct: false },
      { text: "让所有 agent 永远并行且不通信", correct: false },
      { text: "每个 agent 独立跑一个完整的图，互不相干", correct: false },
    ],
    explanation:
      "多 Agent 常用「supervisor（主管）」模式：一个调度节点看当前状态，决定下一步让哪个子 agent 干活，子 agent 又把结果写回，supervisor 再判断是否继续。整个结构在单个 StateGraph 里表达。",
    tags: ["LangGraph", "multi-agent"],
  },
  {
    id: 366,
    type: "single",
    question: "要让 LangGraph 图跨会话/跨多次运行记住状态，应该怎么做？",
    options: [
      { text: "配置持久化 checkpointer，并按 thread_id 区分不同会话", correct: true },
      { text: "把状态硬编码写死在代码里", correct: false },
      { text: "每次运行前清空全局变量", correct: false },
      { text: "用线程调度参数调节运行次数", correct: false },
    ],
    explanation:
      "持久化靠 checkpointer（如 SQLite、Postgres）把每个 checkpoint 存盘，thread_id（会话/线程 id）用来隔离不同对话的状态。这样图能跨多次 invoke 恢复记忆，实现「记住上次聊到哪」。",
    tags: ["LangGraph", "memory", "persistence"],
  },
  {
    id: 367,
    type: "single",
    question: "一个图「无法到达 END」通常意味着什么？",
    options: [
      { text: "图可能陷入死循环或缺少到达终点的路径", correct: true },
      { text: "图会跳过某些节点直接结束", correct: false },
      { text: "图会自动从 START 重新开始", correct: false },
      { text: "这只是性能提示，不影响运行", correct: false },
    ],
    explanation:
      "如果所有执行路径都无法收敛到 END，图要么会无限循环（比如条件边永远返回同一分支），要么布局不合法。图结构要求从任意路径都能最终走到 END 才算完整。",
    tags: ["LangGraph", "graph structure"],
  },
  {
    id: 368,
    type: "single",
    question: "为什么 LangGraph 需要「循环」，而传统 Chain 不需要？",
    options: [
      { text: "agent 需要能反复「思考→调工具→观察结果→再思考」，是动态的", correct: true },
      { text: "循环能让模型输出更长更完整", correct: false },
      { text: "循环只是为了让代码更简短", correct: false },
      { text: "因为图必须包含环才叫图", correct: false },
    ],
    explanation:
      "agent 的行为是动态的：不知道要调几次工具才能完成任务，得根据中间结果反复决策。线性 Chain 无法表达「再试一次」，所以需要循环结构让执行在模型节点和工具节点之间来回。",
    tags: ["LangGraph", "Agent"],
  },
  {
    id: 369,
    type: "multiple",
    question: "下列哪些属于 LangGraph 处理「状态 schema」时的正确做法？（多选）",
    options: [
      { text: "可以用 TypedDict 声明状态字段及其 reducer", correct: true },
      { text: "支持用 Pydantic 模型（v2 需要 model_config）定义状态", correct: true },
      { text: "状态字段的值类型可以任意变化，无需定义", correct: false },
      { text: "reducer 可以定义在 schema 字段上，控制合并逻辑", correct: true },
    ],
    explanation:
      "状态 schema 既可用 TypedDict 声明（字段上可挂 reducer），也可用 Pydantic 模型定义。字段和合并逻辑应当明确声明，而不是放任类型随意漂移——reducer 就挂在字段上指定怎么合并。",
    tags: ["LangGraph", "State"],
  },
  {
    id: 370,
    type: "single",
    question: "在 agent 中，模型节点返回的 AIMessage 带 tool_calls 时，下一步通常是谁？",
    options: [
      { text: "执行对应工具的 ToolNode", correct: true },
      { text: "直接跳到 END 结束", correct: false },
      { text: "重新加载训练数据", correct: false },
      { text: "把消息丢弃并重跑一次", correct: false },
    ],
    explanation:
      "模型一旦请求调用工具，说明它需要外部信息，下一步就该交给 ToolNode 去执行这个调用，把结果作为 ToolMessage 喂回来。只有模型不再要求工具时才会走向 END。",
    tags: ["LangGraph", "ToolNode", "Agent"],
  },
  {
    id: 371,
    type: "single",
    question: "使用 checkpointer 时，`thread_id` 的作用是什么？",
    options: [
      { text: "把不同的对话会话状态隔离开来", correct: true },
      { text: "设置图执行的并行线程数", correct: false },
      { text: "指定模型随机种子", correct: false },
      { text: "控制流式输出的刷新频率", correct: false },
    ],
    explanation:
      "thread_id（线程/会话 id）是 checkpoint 的命名空间：不同 thread_id 对应不同会话的状态历史。同一 thread_id 连续调用会接着上次的状态跑，实现对话记忆；不同 id 则互不影响。",
    tags: ["LangGraph", "checkpoint", "thread"],
  },
  {
    id: 372,
    type: "multiple",
    question: "关于 LangGraph 图结构的边（Edge），下列说法正确的有？（多选）",
    options: [
      { text: "普通边表示无条件地从一个节点到另一个节点", correct: true },
      { text: "条件边在运行时根据状态决定走向", correct: true },
      { text: "边只能连接相邻的两个节点，不能跳级", correct: false },
      { text: "START 只能作为边的起点，END 只能作为终点", correct: true },
    ],
    explanation:
      "普通边是固定跳转，条件边按状态动态选路；边可以任意连接（甚至可以跳过中间层）。START 和 END 有固定角色：START 只作源、END 只作目标。这些都是图布局的基本规则。",
    tags: ["LangGraph", "Edge"],
  },
  {
    id: 373,
    type: "single",
    question: "Node 返回的 update 与状态合并时，若该 key 没有定义 reducer 会怎样？",
    options: [
      { text: "新值直接覆盖旧值", correct: true },
      { text: "旧值保留，新值被丢弃", correct: false },
      { text: "系统抛异常并终止", correct: false },
      { text: "自动把新旧值拼接成数组", correct: false },
    ],
    explanation:
      "没有 reducer 的 channel 默认采用「覆盖」语义：后写入的值直接替换旧值。所以如果你想让多个分支累加或合并，必须显式给字段配上 add_messages 之类的 reducer，否则只会互相覆盖。",
    tags: ["LangGraph", "State", "reducer"],
  },
  {
    id: 374,
    type: "single",
    question: "LangGraph 之所以适合做 agent 框架，关键设计在于什么？",
    options: [
      { text: "显式建模循环、状态和可控执行流程", correct: true },
      { text: "内置了完整的大模型权重", correct: false },
      { text: "只依赖规则匹配，不用大模型", correct: false },
      { text: "把所有逻辑都封装成黑盒", correct: false },
    ],
    explanation:
      "LangGraph 的价值在于把 agent 的「状态」「循环」「分支」「持久化」显式建模成图，让人能精确控制执行流，而不是把一切塞进黑盒 prompt 循环。这也是它和一层包装的 agent 库的根本差异。",
    tags: ["LangGraph", "Agent"],
  },
  {
    id: 375,
    type: "single",
    question: "想让图在被人工接管后「从上次中断的地方继续」，核心是依赖什么？",
    options: [
      { text: "checkpoint 保存的断点 + interrupt 提供的挂起点", correct: true },
      { text: "每次都从 START 重新完整执行", correct: false },
      { text: "把状态存成纯文本再人工粘贴", correct: false },
      { text: "增大模型的 max_tokens", correct: false },
    ],
    explanation:
      "interrupt 挂起时，checkpointer 已保存该点的完整状态；恢复就是带着新输入从这个 checkpoint 继续往后走，不必重跑前半程。二者配合是「中断-恢复」工作流的基石。",
    tags: ["LangGraph", "interrupt", "checkpoint"],
  },
  {
    id: 376,
    type: "multiple",
    question: "下列哪些场景适合用 LangGraph 的多 Agent / 图编排来解决？（多选）",
    options: [
      { text: "需要多个专家模型分工协作的复杂任务", correct: true },
      { text: "需要人工审批介入的关键操作流程", correct: true },
      { text: "需要跨会话记住上下文的对话应用", correct: true },
      { text: "纯静态文本模板替换", correct: false },
    ],
    explanation:
      "多模型分工、人工审批介入、跨会话记忆这些都需要「状态+循环+可控执行流」，正是图编排的强项。而纯静态模板替换用一条链甚至一个函数就够了，杀鸡用不着牛刀。",
    tags: ["LangGraph", "multi-agent"],
  },
  {
    id: 377,
    type: "multiple",
    question: "关于节点抛出异常时的处理，下列说法正确的有？（多选）",
    options: [
      { text: "默认异常会向上抛出，终止本次运行", correct: true },
      { text: "可以在外部用 try/except 或 await 捕获处理", correct: true },
      { text: "配合重试策略（retry policy）可自动重试失败节点", correct: true },
      { text: "图会自动无限重试直到成功，无需干预", correct: false },
    ],
    explanation:
      "默认异常向上抛出、终止本次运行，由外部捕获。若要自动重试，需在节点上配置重试策略（如 retry_policy 指定次数和退避），图本身不会无限重跑。",
    tags: ["LangGraph", "retry"],
  },
  {
    id: 378,
    type: "single",
    question: "给图做「循环次数上限」控制，最直接的做法是？",
    options: [
      { text: "在条件边里判断次数/消息数，达到上限走 END", correct: true },
      { text: "无限循环然后手动 kill 进程", correct: false },
      { text: "在图中加入 sleep 让模型变慢", correct: false },
      { text: "通过降低温度减少循环概率", correct: false },
    ],
    explanation:
      "通常维护一个计数器（比如消息数或调用轮数）放在状态里，条件边据此判断：次数超了就返回 END，否则继续循环。这能防止模型卡在反复调工具的无限循环里。",
    tags: ["LangGraph", "conditional edges"],
  },
  {
    id: 379,
    type: "multiple",
    question: "关于 LangGraph 状态中字段的可选性（Optional/默认值），正确的有？（多选）",
    options: [
      { text: "标注为可选的字段允许在初始状态里缺省", correct: true },
      { text: "给字段设默认值可简化初始状态构造", correct: true },
      { text: "所有字段都必须显式初始化，否则报错", correct: false },
      { text: "可选字段在缺省时无法用 reducer 合并", correct: false },
    ],
    explanation:
      "把状态字段标成 Optional（可选）或设默认值，初始状态里就不用硬给全。可选字段照样能配 reducer——add_messages 常配在一个可选的 messages 字段上，初始没有也照常工作。",
    tags: ["LangGraph", "State"],
  },
  {
    id: 380,
    type: "single",
    question: "LangGraph 与「只调用一次 LLM 的单轮 Chain」本质差异体现在哪？",
    options: [
      { text: "LangGraph 可跨多轮保持状态并支持循环与人工介入", correct: true },
      { text: "LangGraph 不用大模型，Chain 用", correct: false },
      { text: "两者输出格式完全一致，只是名字不同", correct: false },
      { text: "Chain 支持循环，LangGraph 不支持", correct: false },
    ],
    explanation:
      "单轮 Chain 是无状态的一次性流水线：输进去、调一次模型、出结果。LangGraph 是带状态、可循环、可持久化、可中断的执行引擎——这才支撑起真正多轮、可交互的 agent。",
    tags: ["LangGraph"],
  },
  {
    id: 381,
    type: "single",
    question: "当图处于 interrupt 挂起状态时，状态已经被保存，这属于哪种能力的体现？",
    options: [
      { text: "持久化与断点恢复能力", correct: true },
      { text: "并行调度能力", correct: false },
      { text: "模型量化能力", correct: false },
      { text: "流式输出能力", correct: false },
    ],
    explanation:
      "挂起时能完整保存状态、之后无缝恢复，靠的是 checkpoint 的持久化能力。它把「运行到一半的图」固化成可恢复的快照，这正是 human-in-the-loop 能落地的技术基础。",
    tags: ["LangGraph", "checkpoint", "persistence"],
  },
  {
    id: 382,
    type: "multiple",
    question: "下列哪些是 LangGraph 解决 agent 常见痛点的典型手段？（多选）",
    options: [
      { text: "用条件边 + 计数器防止 agent 无限循环", correct: true },
      { text: "用 checkpointer 实现跨会话记忆与断点恢复", correct: true },
      { text: "用 interrupt 实现敏感操作的人工审批", correct: true },
      { text: "用加大 max_tokens 替代循环控制", correct: false },
    ],
    explanation:
      "无限循环、失忆、越权操作是 agent 三大痛点，对应方案分别是条件边+次数上限、checkpointer 持久化、interrupt 审批。加大 max_tokens 只是让单次生成更长，并不能解决结构性问题。",
    tags: ["LangGraph", "Agent"],
  },
  {
    id: 383,
    type: "single",
    question: "关于 LangGraph 的异步运行，下列说法正确的是？",
    options: [
      { text: "图支持 ainvoke/astream 等异步接口，能并发调度节点", correct: true },
      { text: "LangGraph 只能同步运行，不支持异步", correct: false },
      { text: "异步接口只是同步的别名，无并发能力", correct: false },
      { text: "异步只能用于单个节点，图整体必须同步", correct: false },
    ],
    explanation:
      "CompiledGraph 提供 ainvoke、astream 等 async 接口，配合 asyncio 让并行分支真正并发执行、也便于在异步服务里集成。它和同步接口在图上是一致的语义。",
    tags: ["LangGraph", "async"],
  },


  {
    id: 384,
    type: "single",
    question: "扩散模型的前向（forward）过程做的事情是？",
    options: [
      { text: "逐步向数据加入高斯噪声，最终接近纯噪声", correct: true },
      { text: "从纯噪声逐步去噪还原出图像", correct: false },
      { text: "直接用 VAE 把图像编码成向量", correct: false },
      { text: "用分类器判断图像类别", correct: false },
    ],
    explanation:
      "前向过程也叫加噪过程，按给定的噪声调度（noise schedule，即每一步的 beta 值）逐步往干净数据里加高斯噪声，加到最后基本退化成标准高斯分布。",
    tags: ["Diffusion"],
  },
  {
    id: 385,
    type: "single",
    question: "扩散模型的反向（reverse）过程目标是学习什么？",
    options: [
      { text: "从噪声逐步去噪还原出真实数据分布", correct: true },
      { text: "给图像加更多噪声", correct: false },
      { text: "预测图像的类别标签", correct: false },
      { text: "压缩图像的编码", correct: false },
    ],
    explanation:
      "反向过程是「去噪」，模型学习每一步的逆变换，从纯噪声一步步还原出干净的图像，也就是真正的生成阶段。",
    tags: ["Diffusion"],
  },
  {
    id: 386,
    type: "single",
    question: "DDPM 训练时的核心损失是对什么做监督？",
    options: [
      { text: "预测加到图像上的噪声，与真实噪声算 MSE", correct: true },
      { text: "预测图像的像素类别", correct: false },
      { text: "比较前向与反向的分布差异用 KL 散度", correct: false },
      { text: "预测图像的语义分割掩码", correct: false },
    ],
    explanation:
      "DDPM 简化后就是让网络预测加入的噪声 ε，再用预测值和真实噪声算 MSE。看似在预测噪声，等价于隐式地学「如何一步步去噪」。",
    tags: ["Diffusion", "DDPM"],
  },
  {
    id: 387,
    type: "single",
    question: "噪声调度（noise schedule / beta schedule）的作用是？",
    options: [
      { text: "决定每一加噪步注入噪声的强度，控制前向退化速率", correct: true },
      { text: "决定去噪网络的参数量", correct: false },
      { text: "决定输出图像的尺寸", correct: false },
      { text: "决定训练数据的数量", correct: false },
    ],
    explanation:
      "noise schedule 是每步加入噪声的 beta 序列，beta 大退化快、beta 小退化慢，它决定了噪声如何从 0 平滑增加到接近全噪声，也间接影响采样的质量。",
    tags: ["Diffusion", "DDPM"],
  },
  {
    id: 388,
    type: "single",
    question: "DDPM 的采样通常需要上千步，主要因为？",
    options: [
      { text: "每一步去噪幅度小，逐步逼近才能得到清晰结果", correct: true },
      { text: "模型本身必须跑上千层", correct: false },
      { text: "噪声调度只支持这么多步", correct: false },
      { text: "VAE 编码器的限制", correct: false },
    ],
    explanation:
      "DDPM 每步只能去一点点噪，步子太大就发散，所以要从纯噪声一步步迭代上千步才能生成像样的图，这也是它慢的主因。",
    tags: ["Diffusion", "DDPM"],
  },
  {
    id: 389,
    type: "single",
    question: "DDIM 相比 DDPM 的核心改进是？",
    options: [
      { text: "把采样过程变成确定性、可用更少步数采样", correct: true },
      { text: "用 VAE 替代去噪网络", correct: false },
      { text: "去掉了所有噪声注入", correct: false },
      { text: "改用 GAN 的判别器训练", correct: false },
    ],
    explanation:
      "DDIM 是「去噪扩散隐式模型」，它把原随机采样改成确定性映射（等价于求一个常微分方程的解），从而几步甚至几十步就能高质量采样，是加速采样器的代表。",
    tags: ["Diffusion", "Sampling", "DDIM"],
  },
  {
    id: 390,
    type: "single",
    question: "扩散模型「预测噪声」与「预测 x0」两种目标的关系是？",
    options: [
      { text: "两者数学上可互相换算，常常等价或可导来导去", correct: true },
      { text: "预测噪声只能用噪声数据，预测 x0 只能用于干净数据", correct: false },
      { text: "二者完全无关", correct: false },
      { text: "预测 x0 不需要任何网络", correct: false },
    ],
    explanation:
      "给定加噪公式，噪声和 x0（原始干净图）之间是线性可换算的关系，所以很多实现预测 x0，再用公式折算出噪声，本质上训练同一个网络。",
    tags: ["Diffusion", "Objective"],
  },
  {
    id: 391,
    type: "single",
    question: "Latent Diffusion（Stable Diffusion）的核心思路是？",
    options: [
      { text: "先用 VAE 把图像压缩到低维潜在空间，再在潜在空间扩散", correct: true },
      { text: "直接在高分辨率像素空间做扩散", correct: false },
      { text: "去掉所有编码器只用纯扩散", correct: false },
      { text: "用文本直接生成像素不经过任何压缩", correct: false },
    ],
    explanation:
      "Stable Diffusion 先用 VAE 编码器把图像压缩成小得多的 latent（潜在表示），扩散和去噪都在这低维空间做，最后用 VAE 解码器还原成图，大大省显存和算力。",
    tags: ["Diffusion", "Latent Diffusion"],
  },
  {
    id: 392,
    type: "single",
    question: "Latent Diffusion 相比像素空间扩散的收益主要是？",
    options: [
      { text: "在低维潜在空间扩散，计算量小、可训高分辨率", correct: true },
      { text: "让去噪网络层数更多", correct: false },
      { text: "无需任何训练数据", correct: false },
      { text: "彻底去掉噪声调度的过程", correct: false },
    ],
    explanation:
      "潜在空间维度远小于像素空间，扩散的计算和显存开销都大幅下降，这也是它能在消费级显卡上训练、生成高分辨率图的原因。",
    tags: ["Diffusion", "Latent Diffusion"],
  },
  {
    id: 393,
    type: "single",
    question: "Stable Diffusion 中 VAE 解码器的作用是？",
    options: [
      { text: "把去噪后的潜在向量还原成真实图像", correct: true },
      { text: "把真实图像压缩成潜在向量", correct: false },
      { text: "负责加噪过程", correct: false },
      { text: "判断图像真假", correct: false },
    ],
    explanation:
      "VAE 解码器把模型在 latent 空间去噪得到的干净表示「翻译」回像素空间，得到最终可见图像；编码器则相反，把图压进 latent。",
    tags: ["Diffusion", "Latent Diffusion", "VAE"],
  },
  {
    id: 394,
    type: "single",
    question: "text-to-image 中文本条件主要通过什么方式注入去噪网络？",
    options: [
      { text: "文本经编码后作为 cross-attention 的 Key/Value 参与去噪", correct: true },
      { text: "把文本直接拼到图像像素上", correct: false },
      { text: "文本只用于损失函数", correct: false },
      { text: "文本只用于初始化噪声", correct: false },
    ],
    explanation:
      "文本（常经 CLIP 等编码）转成条件 token，作为 cross-attention（跨注意力）的 K/V，去噪网络用自身特征作 Q 去检索文本信息，从而让生成受文本引导。",
    tags: ["Diffusion", "Text-to-Image", "Cross-Attention"],
  },
  {
    id: 395,
    type: "single",
    question: "Classifier-Free Guidance（CFG）的核心思想是？",
    options: [
      { text: "同时训练有条件与无条件两个分支，采样时拉大两者差异", correct: true },
      { text: "训练一个额外的分类器来引导", correct: false },
      { text: "完全去掉文本条件", correct: false },
      { text: "用 GAN 判别器替换去噪网络", correct: false },
    ],
    explanation:
      "CFG 在训练时随机丢弃条件，让模型既能按条件生成也能无条件生成；采样时把条件输出往「偏离无条件」的方向推（w = 1 + scale），从而增强与提示词的一致性，不用额外的分类器。",
    tags: ["Diffusion", "CFG"],
  },
  {
    id: 396,
    type: "single",
    question: "CFG 的引导强度（guidance scale）过高会导致？",
    options: [
      { text: "图像过度饱和、失真、出现伪影", correct: true },
      { text: "图像变得更模糊", correct: false },
      { text: "生成速度更快", correct: false },
      { text: "完全忽略文本", correct: false },
    ],
    explanation:
      "guidance scale 太大等于拼命远离无条件分支、死磕条件，容易过饱和、颜色发艳、边缘产生伪影，破坏自然度。一般取 7~9 左右较平衡。",
    tags: ["Diffusion", "CFG"],
  },
  {
    id: 397,
    type: "single",
    question: "CFG 采样时的公式本质是在做什么？",
    options: [
      { text: "把条件预测拉远于无条件预测，按 scale 加权", correct: true },
      { text: "把两个预测取平均", correct: false },
      { text: "随机选其中一个预测", correct: false },
      { text: "只用无条件预测", correct: false },
    ],
    explanation:
      "CFG 的预测 ≈ 无条件预测 + scale×(条件预测 − 无条件预测)，就是让结果朝「满足条件」的方向多走一段，增强对提示词的响应。",
    tags: ["Diffusion", "CFG"],
  },
  {
    id: 398,
    type: "single",
    question: "ControlNet 的核心设计是？",
    options: [
      { text: "复制去噪网络并加一个可训练的 ControlNet 分支，注入条件同时冻结原权重", correct: true },
      { text: "重新训练一个全新的扩散模型", correct: false },
      { text: "只改变噪声调度", correct: false },
      { text: "去掉文本条件", correct: false },
    ],
    explanation:
      "ControlNet 复制一套网络作为控制分支，接收边缘/深度/姿势等条件，再把它加回冻结的原始去噪网络，这样既保有原模型能力又能精确控制生成。",
    tags: ["Diffusion", "ControlNet"],
  },
  {
    id: 399,
    type: "single",
    question: "ControlNet 常用的控制条件不包括？",
    options: [
      { text: "文本描述的词向量", correct: true },
      { text: "边缘图（Canny）", correct: false },
      { text: "深度图", correct: false },
      { text: "人体姿势骨架（OpenPose）", correct: false },
    ],
    explanation:
      "ControlNet 常用边缘、深度、姿态等空间结构条件来控制生成；文本描述本身走的是另一条 cross-attention 条件通道，不是 ControlNet 的空间控制输入。",
    tags: ["Diffusion", "ControlNet"],
  },
  {
    id: 400,
    type: "single",
    question: "为什么 ControlNet 要冻结原模型的权重？",
    options: [
      { text: "保留原模型能力，避免在特定任务上训练时遗忘", correct: true },
      { text: "让模型无法生成图像", correct: false },
      { text: "降低输入分辨率", correct: false },
      { text: "去掉噪声过程", correct: false },
    ],
    explanation:
      "冻结原权重只需训练轻量的控制分支，既省显存和算力，又能避免在小数据上微调把基础模型学坏，保留通用生成能力。",
    tags: ["Diffusion", "ControlNet"],
  },
  {
    id: 401,
    type: "single",
    question: "扩散模型常用的去噪主干网络是？",
    options: [
      { text: "U-Net（或 DiT 等）", correct: true },
      { text: "纯全连接网络", correct: false },
      { text: "循环神经网络", correct: false },
      { text: "简单的线性层", correct: false },
    ],
    explanation:
      "扩散模型主流用 U-Net（带下采样上采样和 skip connection 的编解码网络）来预测噪声；近年也有 DiT（Diffusion Transformer，用 Transformer 替代 U-Net）的方案。",
    tags: ["Diffusion", "U-Net", "DiT"],
  },
  {
    id: 402,
    type: "single",
    question: "扩散模型中的时间步（timestep）信息如何进入网络？",
    options: [
      { text: "编码成 embedding 后加/乘到特征上，告诉网络当前在哪一步", correct: true },
      { text: "直接作为损失值", correct: false },
      { text: "只用于输出层", correct: false },
      { text: "时间步不参与网络计算", correct: false },
    ],
    explanation:
      "时间步被编码成位置类 embedding，通过加法/Scale 等方式注入网络的各层，让同一个网络知道「现在该去噪多狠」，这是扩散网络能处理所有步的关键。",
    tags: ["Diffusion", "Timestep"],
  },
  {
    id: 403,
    type: "single",
    question: "DiT（Diffusion Transformer）相比 U-Net 的改进是？",
    options: [
      { text: "用 Transformer 结构替代卷积 U-Net，利于规模化", correct: true },
      { text: "彻底去掉时间步信息", correct: false },
      { text: "不再需要噪声预测", correct: false },
      { text: "只能生成文本", correct: false },
    ],
    explanation:
      "DiT 把图像切成 patch（图像块）后扔进 Transformer 做扩散去噪，继承了 Transformer 的可扩展性，scale up 后效果常优于 U-Net，是新一代扩散模型（如 Sora）的基础。",
    tags: ["Diffusion", "DiT"],
  },
  {
    id: 404,
    type: "single",
    question: "图像分辨率对扩散模型生成的影响是？",
    options: [
      { text: "分辨率越高显存和计算开销越大，且训练与推理分辨率需匹配", correct: true },
      { text: "分辨率与显存完全无关", correct: false },
      { text: "分辨率只影响文本编码", correct: false },
      { text: "分辨率越高反而越快", correct: false },
    ],
    explanation:
      "像素越多 latent 越大，扩散网络的计算量和显存随之上升；而且若在低分辨率训练、高分辨率推理，会出现网格状伪影或主体重复，需要专门的分辨率适配。",
    tags: ["Diffusion", "Resolution"],
  },
  {
    id: 405,
    type: "single",
    question: "扩散模型中「确定性采样」的含义是？",
    options: [
      { text: "给定同一噪声，每次采样结果一致（如 DDIM）", correct: true },
      { text: "每次采样结果随机不同", correct: false },
      { text: "不使用任何噪声", correct: false },
      { text: "只采样一步就结束", correct: false },
    ],
    explanation:
      "确定性采样去掉随机重采样噪声，让从同一初始噪声出发得到唯一确定的轨迹，DDIM 就是典型，利于可控复现和图像编辑（如噪声往返映射）。",
    tags: ["Diffusion", "Deterministic"],
  },
  {
    id: 406,
    type: "single",
    question: "加速采样器（如 DDIM、DPM-Solver）降低步数后常见代价是？",
    options: [
      { text: "步数过少可能损失细节、产生伪影", correct: true },
      { text: "生成的图必然完全一样", correct: false },
      { text: "不需要任何去噪网络", correct: false },
      { text: "显存占用必然翻倍", correct: false },
    ],
    explanation:
      "高阶级数采样器能在几步内逼近多步效果，但步数太少时离散误差变大，细节会变糊或出现伪影，通常要在速度与质量间做取舍。",
    tags: ["Diffusion", "Sampling"],
  },
  {
    id: 407,
    type: "single",
    question: "扩散模型与 VLM（多模态大模型）结合常见于？",
    options: [
      { text: "用 VLM 的文本/图文编码提升条件理解，或做编辑、评估引导", correct: true },
      { text: "让 VLM 替代去噪网络做扩散", correct: false },
      { text: "两者完全不相干", correct: false },
      { text: "VLM 负责加噪过程", correct: false },
    ],
    explanation:
      "VLM 能更好地理解复杂指令和图文关系，可被当作更强的文本编码器、用于文字引导的图像编辑（如把 VLM 输出转化为编辑方向），或作为生成质量评估器。",
    tags: ["Diffusion", "VLM"],
  },
  {
    id: 408,
    type: "single",
    question: "FID 指标衡量的是？",
    options: [
      { text: "生成图像分布与真实图像分布的差距", correct: true },
      { text: "单张图像与单张图的逐像素差异", correct: false },
      { text: "文本与图像的语义相似度", correct: false },
      { text: "模型参数的数量", correct: false },
    ],
    explanation:
      "FID（Fréchet Inception Distance）把真实图和生成图分别用特征提取器映射到特征空间，再比较两个高斯分布的均值协方差距离，数值越低代表生成分布越接近真实分布。",
    tags: ["Diffusion", "FID", "Evaluation"],
  },
  {
    id: 409,
    type: "single",
    question: "FID 数值越小表示？",
    options: [
      { text: "生成分布越接近真实分布，质量越好", correct: true },
      { text: "生成图越大", correct: false },
      { text: "模型训练越慢", correct: false },
      { text: "噪声越多", correct: false },
    ],
    explanation:
      "FID 是距离类指标，越小说明生成分布和真实分布越接近，通常代表质量与多样性更好，是扩散模型最常用的评估指标之一。",
    tags: ["Diffusion", "FID", "Evaluation"],
  },
  {
    id: 410,
    type: "single",
    question: "Inpainting（图像修复）指的是？",
    options: [
      { text: "用扩散模型填补图像中被遮罩（mask）的区域", correct: true },
      { text: "把图像放大若干倍", correct: false },
      { text: "把图像缩小", correct: false },
      { text: "去除图像的噪声", correct: false },
    ],
    explanation:
      "Inpainting 是给定一张图和一块遮罩，模型只在遮罩区域内重新生成内容，而保持遮罩外区域不变，常用于去水印、补残缺、改局部物体。",
    tags: ["Diffusion", "Inpainting"],
  },
  {
    id: 411,
    type: "single",
    question: "Outpainting（扩展画布）的核心做法是？",
    options: [
      { text: "在已知图像周围扩出画布并生成与之衔接的新内容", correct: true },
      { text: "把图像等比缩小", correct: false },
      { text: "只重绘图像中心", correct: false },
      { text: "把整张图重新随机生成", correct: false },
    ],
    explanation:
      "Outpainting 保留原有内容，把遮罩放在边缘外部，让扩散模型在扩出的区域生成与已有内容风格衔接的新画面，实现「无中生有」地扩图。",
    tags: ["Diffusion", "Outpainting"],
  },
  {
    id: 412,
    type: "single",
    question: "视频扩散（video diffusion）与图像扩散的关键区别是？",
    options: [
      { text: "需在时间维度上建模，保证帧间时序连贯", correct: true },
      { text: "完全不需要空间建模", correct: false },
      { text: "无法用任何去噪网络", correct: false },
      { text: "只能用单帧生成", correct: false },
    ],
    explanation:
      "视频扩散在空间之外还要处理时间维，网络需同时考虑相邻帧的运动一致性，避免闪烁和动作不连贯，代价是计算量、显存和训练数据要求都大幅上升。",
    tags: ["Diffusion", "Video"],
  },
  {
    id: 413,
    type: "single",
    question: "扩散模型推理加速（推理优化）的常见手段是？",
    options: [
      { text: "蒸馏成少步模型、低精度量化、优化采样器", correct: true },
      { text: "增大采样步数", correct: false },
      { text: "提高输入分辨率", correct: false },
      { text: "增加去噪网络层数", correct: false },
    ],
    explanation:
      "常见加速：用蒸馏（如 LCM/一致性蒸馏）把模型压到几步出图、对模型做量化省显存和计算、用更高效的采样器或减少 CFG 重复计算，而不是去增步数或加大模型。",
    tags: ["Diffusion", "Inference", "Acceleration"],
  },
  {
    id: 414,
    type: "single",
    question: "扩散模型推理时显存压力主要来自？",
    options: [
      { text: "去噪网络 + 中间特征/激活 + VAE，且逐步重复前向", correct: true },
      { text: "只有输入文本", correct: false },
      { text: "噪声调度表本身", correct: false },
      { text: "损失函数", correct: false },
    ],
    explanation:
      "扩散推理要多次前向去噪网络，加上 latent 特征和激活都占显存，条件注入的 cross-attention 也耗资源，高分辨率下压力尤为明显。",
    tags: ["Diffusion", "Memory", "Inference"],
  },
  {
    id: 415,
    type: "single",
    question: "训练噪声预测网络时，x0 与噪声的混合规则是？",
    options: [
      { text: "按当前时间步的调度系数把 x0 与噪声加权混合", correct: true },
      { text: "完全随机拼图", correct: false },
      { text: "只保留噪声不要 x0", correct: false },
      { text: "只保留 x0 不要噪声", correct: false },
    ],
    explanation:
      "加噪样本 = sqrt(ᾱ_t)·x0 + sqrt(1−ᾱ_t)·ε，ᾱ_t 由噪声调度累积得到。时间步越靠后 x0 权重越小、噪声占比越大，网络就据此学会预测每一步的噪声。",
    tags: ["Diffusion", "Training"],
  },
  {
    id: 416,
    type: "multiple",
    question: "下列哪些属于扩散模型前向加噪过程的特征？（多选）",
    options: [
      { text: "逐步加入高斯噪声", correct: true },
      { text: "最终接近标准高斯分布", correct: true },
      { text: "过程不依赖任何可训练参数", correct: true },
      { text: "必须用可训练网络完成", correct: false },
    ],
    explanation:
      "前向过程是按固定噪声调度加噪，不需要训练、参数固定，加到末尾退化到高斯噪声；可训练的是反向去噪网络。",
    tags: ["Diffusion"],
  },
  {
    id: 417,
    type: "multiple",
    question: "关于 DDPM 与 DDIM，正确的有？（多选）",
    options: [
      { text: "DDPM 采样偏随机、步数多", correct: true },
      { text: "DDIM 采样确定性更强、可用少步数", correct: true },
      { text: "DDIM 常比 DDPM 采样更快", correct: true },
      { text: "两者训练目标完全不同", correct: false },
    ],
    explanation:
      "DDPM 随机采样需上千步，DDIM 确定性采样几步到几十步即可，两者共享同一套训练好的去噪网络，只是采样方式不同。",
    tags: ["Diffusion", "DDPM", "DDIM"],
  },
  {
    id: 418,
    type: "multiple",
    question: "Latent Diffusion 中 VAE 承担哪些角色？（多选）",
    options: [
      { text: "编码器把图像压缩到潜在空间", correct: true },
      { text: "解码器把潜在向量还原成图像", correct: true },
      { text: "潜在空间维度远小于像素空间", correct: true },
      { text: "VAE 负责加噪过程", correct: false },
    ],
    explanation:
      "VAE 负责图像与 latent 之间的编解码，latent 维度小所以省资源；加噪去噪是在 latent 上由扩散网络完成，不归 VAE 管。",
    tags: ["Diffusion", "Latent Diffusion", "VAE"],
  },
  {
    id: 419,
    type: "multiple",
    question: "关于 CFG，正确的有？（多选）",
    options: [
      { text: "训练时随机丢弃条件模拟无条件分支", correct: true },
      { text: "采样时结合条件与无条件预测", correct: true },
      { text: "不需要额外的分类器", correct: true },
      { text: "guidance scale 越大一定越好", correct: false },
    ],
    explanation:
      "CFG 通过训练时的条件丢弃和采样时拉大差异实现，无需分类器；scale 过大会过饱和失真，不是越大越好。",
    tags: ["Diffusion", "CFG"],
  },
  {
    id: 420,
    type: "multiple",
    question: "下列哪些可以作为 ControlNet 的结构条件？（多选）",
    options: [
      { text: "Canny 边缘图", correct: true },
      { text: "深度图", correct: true },
      { text: "人体姿态关键点", correct: true },
      { text: "纯文本描述", correct: false },
    ],
    explanation:
      "ControlNet 吃的是边缘、深度、姿势等空间结构条件；纯文本走的是 cross-attention 条件通道，不通过 ControlNet 注入。",
    tags: ["Diffusion", "ControlNet"],
  },
  {
    id: 421,
    type: "multiple",
    question: "关于扩散模型的去噪网络，正确的有？（多选）",
    options: [
      { text: "U-Net 是最常见的骨干", correct: true },
      { text: "DiT 用 Transformer 替代 U-Net", correct: true },
      { text: "需要接收时间步信息", correct: true },
      { text: "只能处理固定尺寸输入", correct: false },
    ],
    explanation:
      "U-Net 是主流，DiT 是 Transformer 版，两者都要注入时间步 embedding 来区分去噪阶段；通过位置编码等也能处理不同尺寸。",
    tags: ["Diffusion", "U-Net", "DiT"],
  },
  {
    id: 422,
    type: "multiple",
    question: "扩散模型生成质量评估常用哪些指标？（多选）",
    options: [
      { text: "FID（分布距离）", correct: true },
      { text: "IS（Inception Score）", correct: true },
      { text: "CLIP Score（图文一致性）", correct: true },
      { text: "模型参数量", correct: false },
    ],
    explanation:
      "FID 看生成分布与真实分布的距离，IS 看单张图质量和多样性，CLIP Score 衡量图文语义一致性；参数量是资源指标不是质量指标。",
    tags: ["Diffusion", "Evaluation", "FID"],
  },
  {
    id: 423,
    type: "multiple",
    question: "扩散模型推理加速的可行方案有？（多选）",
    options: [
      { text: "蒸馏成少步模型", correct: true },
      { text: "模型量化", correct: true },
      { text: "用高效采样器减步数", correct: true },
      { text: "增加采样步数提高速度", correct: false },
    ],
    explanation:
      "蒸馏、量化、高效采样器都能提速；增加采样步数会变慢，与加速目标相反。",
    tags: ["Diffusion", "Inference", "Acceleration"],
  },
  {
    id: 424,
    type: "multiple",
    question: "关于图像分辨率与 latent，正确的有？（多选）",
    options: [
      { text: "latent 分辨率通常约为像素分辨率的 1/8", correct: true },
      { text: "分辨率越高显存和计算越大", correct: true },
      { text: "训练与推理分辨率差异过大会产生伪影", correct: true },
      { text: "latent 和像素分辨率完全相等", correct: false },
    ],
    explanation:
      "Stable Diffusion 的 VAE 下采样 8 倍，latent 是像素的 1/8；分辨率高则开销大，且训练/推理分辨率不匹配会出现网格伪影、主体重复。",
    tags: ["Diffusion", "Resolution", "Latent"],
  },
  {
    id: 425,
    type: "multiple",
    question: "关于 Inpainting/Outpainting，正确的有？（多选）",
    options: [
      { text: "两者都基于遮罩（mask）控制生成区域", correct: true },
      { text: "Inpainting 填补图像内部被遮罩区域", correct: true },
      { text: "Outpainting 扩展画布生成衔接内容", correct: true },
      { text: "两者都会破坏遮罩外的内容", correct: false },
    ],
    explanation:
      "两者都用遮罩限定哪些区域重新生成，遮罩外内容保持不变；Inpainting 补内部、Outpainting 扩外部。",
    tags: ["Diffusion", "Inpainting", "Outpainting"],
  },
  {
    id: 426,
    type: "multiple",
    question: "视频扩散面临的挑战包括？（多选）",
    options: [
      { text: "帧间时序连贯性", correct: true },
      { text: "计算量和显存开销大", correct: true },
      { text: "需要大量视频训练数据", correct: true },
      { text: "只需要处理单帧", correct: false },
    ],
    explanation:
      "视频扩散要多维建模保证运动连贯，算力显存和数据需求都远高于图像，不只是处理单帧。",
    tags: ["Diffusion", "Video"],
  },
  {
    id: 427,
    type: "multiple",
    question: "扩散模型与 VLM 结合的意义包括？（多选）",
    options: [
      { text: "VLM 提供更强的图文理解作为条件", correct: true },
      { text: "可用于文本引导的图像编辑", correct: true },
      { text: "VLM 可作为生成质量评估器", correct: true },
      { text: "VLM 用于执行加噪过程", correct: false },
    ],
    explanation:
      "VLM 强化条件理解、辅助编辑、充当评估器都能提升扩散系统；加噪是固定数学过程，不需要 VLM。",
    tags: ["Diffusion", "VLM"],
  },
  {
    id: 428,
    type: "multiple",
    question: "关于噪声预测网络的训练，正确的有？（多选）",
    options: [
      { text: "目标通常是预测所加噪声，与真实噪声算 MSE", correct: true },
      { text: "需要给不同时间步的样本训练", correct: true },
      { text: "等价于间接学习去噪", correct: true },
      { text: "必须预测图像像素分类标签", correct: false },
    ],
    explanation:
      "训练目标是还原所加的噪声（或用 x0 目标等价换算），需要在各种时间步的加噪样本上训练，让网络学会逆向去噪；不是做像素分类。",
    tags: ["Diffusion", "Training", "Objective"],
  },


  {
    id: 429,
    type: "single",
    question: "投机解码（speculative decoding）的核心思想是什么？",
    options: [
      { text: "用一个更大的模型替换小模型来加速推理", correct: false },
      { text: "用快的小模型先草拟多个 token，大模型并行验证", correct: true },
      { text: "把浮点精度从 FP16 降到 INT8 来提速", correct: false },
      { text: "提前缓存所有层的输出以跳过计算", correct: false },
    ],
    explanation: "核心是「小模型 draft、大模型 verify」：草稿模型一口气猜 N 个 token，目标模型并行校验，一次验证吃下多个 token。",
    tags: ["Speculative"],
  },
  {
    id: 430,
    type: "single",
    question: "在投机解码中，草稿模型（draft model）和验证模型（target model）各自的职责是？",
    options: [
      { text: "两者都逐 token 串行生成", correct: false },
      { text: "草稿模型快速生成候选序列，目标模型并行验证", correct: true },
      { text: "目标模型负责猜，草稿模型负责纠错", correct: false },
      { text: "两者职责完全相同，只是参数不同", correct: false },
    ],
    explanation: "draft 负责快、target 负责准：小模型并行成本低，负责猜一长串，大模型再并行打分校正。",
    tags: ["Speculative"],
  },
  {
    id: 431,
    type: "single",
    question: "投机解码能够加速的根本原因是？",
    options: [
      { text: "把模型的串行自回归生成变成了批量并行验证", correct: true },
      { text: "显著降低了大模型本身的单次前向计算量", correct: false },
      { text: "压缩了权重显存从而加快访存", correct: false },
      { text: "省掉了 KV cache 的全部读写开销", correct: false },
    ],
    explanation: "瓶颈在串行：每步只能出一个 token。投机解码让一次前向（并行）验证多个 token，把时间换成空间，绕过串行天花板。",
    tags: ["Speculative"],
  },
  {
    id: 432,
    type: "single",
    question: "「接受率」（acceptance rate）指的是什么？",
    options: [
      { text: "草稿模型生成 token 的速度", correct: false },
      { text: "草稿 token 被目标模型接受并采纳的比例", correct: true },
      { text: "目标模型处理一个 batch 的吞吐量", correct: false },
      { text: "模型在 benchmark 上的准确率", correct: false },
    ],
    explanation: "接受率 = 被采纳的草稿 token 数 / 草稿 token 总数，它直接决定能省多少目标模型前向步数。",
    tags: ["Speculative"],
  },
  {
    id: 433,
    type: "single",
    question: "接受率越高，投机解码的理论加速倍数通常？",
    options: [
      { text: "越低", correct: false },
      { text: "越高", correct: true },
      { text: "完全无关", correct: false },
      { text: "恒定不变", correct: false },
    ],
    explanation: "接受率越高，单次目标前向能验证通过的 token 越多，需要的大模型步数越少，加速越明显。",
    tags: ["Speculative"],
  },
  {
    id: 434,
    type: "single",
    question: "投机解码为什么能保证「无损」，即输出分布与目标模型一致？",
    options: [
      { text: "因为草稿模型总是输出正确的 token", correct: false },
      { text: "因为采用 rejection sampling 修正，被拒 token 按目标分布重新采样", correct: true },
      { text: "因为直接取目标模型最高概率 token", correct: false },
      { text: "因为跳过了所有不确定的采样步骤", correct: false },
    ],
    explanation: "拒绝采样（rejection sampling）在验证失败时按目标模型分布重新抽样，保证最终序列的分布与原模型一致，因此无损。",
    tags: ["Speculative"],
  },
  {
    id: 435,
    type: "single",
    question: "当一个草稿 token 被目标模型拒绝时，投机解码通常如何回退？",
    options: [
      { text: "回退到序列开头重新生成", correct: false },
      { text: "回退到最后一个被接受的 token，从那里重新生成", correct: true },
      { text: "直接丢弃该 token 继续用草稿生成", correct: false },
      { text: "重新用草稿模型从头预测一次", correct: false },
    ],
    explanation: "只回退到最近被接受的 token，在其后按目标分布重采样，既保证分布一致又不浪费已接受的成果。",
    tags: ["Speculative"],
  },
  {
    id: 436,
    type: "single",
    question: "投机解码中猜测的 token 数量（n 值）设置得越大，通常？",
    options: [
      { text: "一定越加速", correct: false },
      { text: "收益递减甚至可能变慢", correct: true },
      { text: "对性能毫无影响", correct: false },
      { text: "只增加草稿模型负担，目标模型不变", correct: false },
    ],
    explanation: "n 越大草稿被拒的概率越高、验证的尾部 token 越容易白算，边际收益递减，过大反而拖累。存在最优 n。",
    tags: ["Speculative"],
  },
  {
    id: 437,
    type: "single",
    question: "投机解码中 KV cache 的复用通常是怎样的？",
    options: [
      { text: "每次验证都重建全部 KV cache", correct: false },
      { text: "草稿和验证共享、累积缓存前缀，回退后截断复用", correct: true },
      { text: "只复用草稿模型的 KV cache", correct: false },
      { text: "KV cache 与投机解码无关", correct: false },
    ],
    explanation: "草稿模型每步推进都会更新 KV cache，目标模型并行验证时基于共享前缀累积缓存，回退时截断到最近接受处复用。",
    tags: ["Speculative"],
  },
  {
    id: 438,
    type: "single",
    question: "如果草稿模型和目标模型的词表（vocabulary）不一致，会有什么问题？",
    options: [
      { text: "没有影响，验证照常进行", correct: false },
      { text: "无法直接对比 token 概率，需要在共享词表子集上采样验证", correct: true },
      { text: "只能让词表完全相同的模型参与", correct: false },
      { text: "需要把草稿模型重新训练一遍", correct: false },
    ],
    explanation: "两模型词表不同时，验证只能落在两者共享的词表子集上，否则 token 对不上号、概率无法比较。",
    tags: ["Speculative"],
  },
  {
    id: 439,
    type: "single",
    question: "理想接受率 r 下，投机解码的理论加速上限（大模型前向步数减少比例）约为？",
    options: [
      { text: "r/(1+r)", correct: true },
      { text: "1/(1+r)", correct: false },
      { text: "2r", correct: false },
      { text: "1−r", correct: false },
    ],
    explanation: "平均每步能推进的 token 数近似 1/(1−r)，对应的期望加速约 r/(1+r)，这是理论上限，实际还要算草稿开销。",
    tags: ["Speculative"],
  },
  {
    id: 440,
    type: "single",
    question: "投机解码相比「单纯把模型量化到更低精度」的区别在于？",
    options: [
      { text: "两者本质相同，都是压缩权重", correct: false },
      { text: "量化改变计算精度，投机解码靠模型协作保持原精度且无损", correct: true },
      { text: "量化更无损，投机解码有损", correct: false },
      { text: "投机解码只改硬件，不涉及算法", correct: false },
    ],
    explanation: "量化是压缩权重精度（可能掉点），投机解码则通过大小模型协作在不改变输出分布的情况下加速，两者可叠加。",
    tags: ["Speculative"],
  },
  {
    id: 441,
    type: "single",
    question: "EAGLE 等投机解码变体（如 EAGLE/EAGLE-2）的核心改进思路是？",
    options: [
      { text: "直接用更大的草稿模型", correct: false },
      { text: "利用目标模型的隐藏层特征做更精准的 draft，提高接受率", correct: true },
      { text: "放弃验证只信任草稿输出", correct: false },
      { text: "把草稿模型换成任意语言模型", correct: false },
    ],
    explanation: "EAGLE 用目标模型的隐藏状态作为草稿依据，草稿更贴合目标分布，接受率更高，从而更快。",
    tags: ["Speculative"],
  },
  {
    id: 442,
    type: "single",
    question: "在投机解码中，草稿模型的选择一般倾向于？",
    options: [
      { text: "与目标模型相同大小以保证一致", correct: false },
      { text: "更小、更快、分布尽量贴近目标模型", correct: true },
      { text: "越大越好，反正要被验证", correct: false },
      { text: "任意模型即可，无需对齐", correct: false },
    ],
    explanation: "草稿模型要小、要快，且与目标分布越接近越好——否则接受率低，草稿的开销就白花了。",
    tags: ["Speculative"],
  },
  {
    id: 443,
    type: "single",
    question: "草稿模型生成 token 的开销相对于目标模型，在投机解码中如何考虑？",
    options: [
      { text: "完全忽略，不影响性能", correct: false },
      { text: "必须计入，否则实际加速会被夸大", correct: true },
      { text: "草稿总是免费的", correct: false },
      { text: "只在草稿被拒时才计算", correct: false },
    ],
    explanation: "草稿模型自己也要跑前向，虽然便宜但不是零成本，真正收益要扣除草稿开销后才算数。",
    tags: ["Speculative"],
  },
  {
    id: 444,
    type: "single",
    question: "投机解码中，目标模型对草稿序列的验证方式通常是？",
    options: [
      { text: "逐个 token 串行验证", correct: false },
      { text: "在 batch 内并行计算所有草稿位置的 logits 并对比概率", correct: true },
      { text: "只在最后一个 token 处验证", correct: false },
      { text: "随机挑几个 token 抽验", correct: false },
    ],
    explanation: "把草稿序列拼成一个 batch 一次性前向，目标模型同时得到每个位置的分布，再逐位置做接受/拒绝判断。",
    tags: ["Speculative"],
  },
  {
    id: 445,
    type: "single",
    question: "当草稿 token 被接受时，目标模型的输出与草稿模型的关系是？",
    options: [
      { text: "必须完全替换为目标模型的预测", correct: false },
      { text: "保留草稿 token，仅用目标模型概率决定是否拒绝", correct: true },
      { text: "丢弃草稿，重新采样", correct: false },
      { text: "取两者概率的平均", correct: false },
    ],
    explanation: "接受时保留草稿 token，目标模型只在概率过低（被拒）时才介入重采样，从而省下目标前向步数。",
    tags: ["Speculative"],
  },
  {
    id: 446,
    type: "single",
    question: "投机解码对硬件（如 GPU）的主要要求/影响是？",
    options: [
      { text: "需要更多的顺序依赖、不利于并行", correct: false },
      { text: "让计算更可并行化，常配合 KV cache 与批处理发挥效果", correct: true },
      { text: "必须换更快的显存", correct: false },
      { text: "只能跑在 CPU 上", correct: false },
    ],
    explanation: "投机解码把单步串行改成批量并行验证，正好吃满 GPU 的并行算力，配合 KV cache 缓存能进一步降开销。",
    tags: ["Speculative"],
  },
  {
    id: 447,
    type: "single",
    question: "当草稿序列全部被接受时，投机解码做了什么额外工作？",
    options: [
      { text: "什么都不用做，直接输出", correct: false },
      { text: "仍需在最后一个位置用目标模型分布采样一个 token 作为下一步草稿起点", correct: true },
      { text: "重新跑一遍草稿模型", correct: false },
      { text: "清除所有 KV cache", correct: false },
    ],
    explanation: "全被接受时仍需在末尾用目标模型分布采样一个新 token，既保证分布正确，又作为下一轮草稿的起点。",
    tags: ["Speculative"],
  },
  {
    id: 448,
    type: "single",
    question: "投机解码通常对什么样的生成场景收益更明显？",
    options: [
      { text: "每一步都很难预测、接受率极低", correct: false },
      { text: "重复性高、可预测性强的文本", correct: true },
      { text: "非常短的生成", correct: false },
      { text: "与草稿模型完全无关的场景", correct: false },
    ],
    explanation: "可预测的内容接受率高、一次能通过多个 token，长上下文里省下的大模型步数更多，收益更明显。",
    tags: ["Speculative"],
  },
  {
    id: 449,
    type: "multiple",
    question: "投机解码需要哪些关键要素才能正确工作？",
    options: [
      { text: "一个快速的小草稿模型", correct: true },
      { text: "一个用于验证的目标模型", correct: true },
      { text: "保证分布一致性的拒绝采样机制", correct: true },
      { text: "必须用与目标完全相同的词表", correct: false },
    ],
    explanation: "草稿模型、目标验证模型、以及无损的拒绝采样三者缺一不可；词表可只在共享子集上验证，不必完全相同。",
    tags: ["Speculative"],
  },
  {
    id: 450,
    type: "multiple",
    question: "以下哪些因素会显著影响投机解码的实际加速效果？",
    options: [
      { text: "草稿模型的接受率", correct: true },
      { text: "猜测的 token 数量 n 的选择", correct: true },
      { text: "草稿模型本身的计算开销", correct: true },
      { text: "目标模型的词表大小一定越小越好", correct: false },
    ],
    explanation: "接受率、n 值、草稿开销都会直接决定净加速；词表大小本身不直接决定加速，别被带偏。",
    tags: ["Speculative"],
  },
  {
    id: 451,
    type: "multiple",
    question: "关于投机解码的无损性，下列说法正确的有？",
    options: [
      { text: "输出分布与单独运行目标模型完全一致", correct: true },
      { text: "靠拒绝采样修正被拒 token", correct: true },
      { text: "一定会引入额外的近似误差", correct: false },
      { text: "与草稿模型的分布无关", correct: false },
    ],
    explanation: "拒绝采样保证采样分布一致，因此无损；草稿质量只影响加速，不影响最终分布的正确性。",
    tags: ["Speculative"],
  },
  {
    id: 452,
    type: "multiple",
    question: "与投机解码配合使用、能进一步提升效果的常见手段有？",
    options: [
      { text: "KV cache 复用", correct: true },
      { text: "批量并行验证", correct: true },
      { text: "与量化等压缩手段叠加", correct: true },
      { text: "把目标模型替换成草稿模型", correct: false },
    ],
    explanation: "KV cache 复用、批量验证、叠加量化都能放大收益；把目标换成草稿就失去验证、也失去无损保证了。",
    tags: ["Speculative"],
  },
  {
    id: 453,
    type: "multiple",
    question: "草稿模型在投机解码中的理想特征包括？",
    options: [
      { text: "推理速度快、开销小", correct: true },
      { text: "概率分布与目标模型接近", correct: true },
      { text: "参数规模尽可能接近目标模型", correct: false },
      { text: "支持流式/并行生成草稿", correct: true },
    ],
    explanation: "草稿要小而快、分布贴近目标、能高效生成序列；参数越大越接近目标，反而失去草稿的意义。",
    tags: ["Speculative"],
  },
  {
    id: 454,
    type: "multiple",
    question: "投机解码相比传统自回归逐 token 生成，主要在哪些方面带来提升？",
    options: [
      { text: "减少目标模型的前向推理次数", correct: true },
      { text: "提升单位时间内的 token 吞吐", correct: true },
      { text: "改变最终输出分布", correct: false },
      { text: "必须牺牲生成质量来换速度", correct: false },
    ],
    explanation: "它减少目标前向步数、提高吞吐，但保持输出分布不变，不用牺牲质量——这就是无损加速的卖点。",
    tags: ["Speculative"],
  },
  {
    id: 455,
    type: "multiple",
    question: "关于投机解码中的拒绝采样机制，以下描述正确的有？",
    options: [
      { text: "被拒时按目标模型分布重新采样", correct: true },
      { text: "保证了生成分布与目标模型一致", correct: true },
      { text: "被拒后回退到最近接受 token 处", correct: true },
      { text: "被拒后直接丢弃该 token 继续草稿", correct: false },
    ],
    explanation: "拒绝采样在被拒时按目标分布重采样并回退到最近接受处，保证无损；不能直接丢弃继续草稿，那会破坏分布。",
    tags: ["Speculative"],
  },
  {
    id: 456,
    type: "multiple",
    question: "相比直接量化压缩模型，投机解码的特点有哪些？",
    options: [
      { text: "通过模型协作提速而非压缩权重", correct: true },
      { text: "能保持与原模型一致的输出分布", correct: true },
      { text: "与量化可以叠加使用", correct: true },
      { text: "属于对权重精度的修改", correct: false },
    ],
    explanation: "投机解码靠协作加速且无损，量化改精度；两者并不互斥，常一起用。",
    tags: ["Speculative"],
  },
  {
    id: 457,
    type: "multiple",
    question: "在确定投机解码的猜测长度 n 时，需要考虑哪些权衡？",
    options: [
      { text: "n 过大尾部草稿更容易被拒、验证白算", correct: true },
      { text: "n 过小单次验证吃下的 token 太少", correct: true },
      { text: "n 与草稿模型的生成开销相关", correct: true },
      { text: "n 越大越快，无需权衡", correct: false },
    ],
    explanation: "n 要平衡「一次验证吃多少」和「尾部被拒的风险」，还要考虑草稿生成成本，存在最优值而非越大越好。",
    tags: ["Speculative"],
  },
  {
    id: 458,
    type: "single",
    question: "EAGLE 这类基于草稿的变体，其核心改进点在于？",
    options: [
      { text: "用更大的草稿模型提升准确性", correct: false },
      { text: "利用目标模型隐藏特征做更精准的草稿，提高接受率", correct: true },
      { text: "放弃拒绝采样以换取更快速度", correct: false },
      { text: "改变输出分布以降低计算量", correct: false },
    ],
    explanation: "EAGLE 用目标模型隐藏状态生成更贴合分布的草稿，接受率更高、加速更快，同时仍在拒绝采样框架内保持无损。",
    tags: ["Speculative"],
  },


  {
    id: 459,
    type: "single",
    question: "对大规模爬取的网页文本做精确去重（exact dedup），最常用的哈希做法是？",
    options: [
      { text: "用 SimHash 计算相似度指纹", correct: false },
      { text: "对每篇文档取 MD5/SHA 等哈希，哈希碰撞则判重", correct: true },
      { text: "逐字逐句做字符串比较", correct: false },
      { text: "用词频向量做余弦相似度聚类", correct: false },
    ],
    explanation:
      "精确去重就是取文档哈希（如 SHA），哈希相同的视为重复直接丢弃。SimHash 是近重复/近似去重用的，能容忍小改动，精确去重不需要。",
    tags: ["Data"],
  },
  {
    id: 460,
    type: "single",
    question: "MinHash / SimHash 这类方法解决的是什么去重问题？",
    options: [
      { text: "完全相同的文档去重", correct: false },
      { text: "近似重复（只改几个词、增删句子）的文档去重", correct: true },
      { text: "图片像素级去重", correct: false },
      { text: "跨语言文档对齐", correct: false },
    ],
    explanation:
      "内容几乎一样但略有改动的近重复文档，精确哈希对不上。SimHash/MinHash 把文档压成指纹，指纹汉明距离近就判为近重复，专门处理这种。",
    tags: ["Data"],
  },
  {
    id: 461,
    type: "multiple",
    question: "对 LLM 训练数据做去重，为什么重要？（多选）",
    options: [
      { text: "去重能降低过拟合风险", correct: true },
      { text: "去重能减少训练 token 总量、省算力", correct: true },
      { text: "去重能消除测试集与训练集的重叠泄露", correct: true },
      { text: "去重能提升数据的随机数熵值", correct: false },
    ],
    explanation:
      "重复数据会让模型把高频内容背下来而非泛化（过拟合），还浪费算力；测试集若和训练集重叠，评测分数会虚高（泄露）。和随机数无关。",
    tags: ["Data"],
  },
  {
    id: 462,
    type: "multiple",
    question: "相比整文档哈希，按 n-gram 窗口去重有哪些优势？（多选）",
    options: [
      { text: "能捕捉到文档内部只有一小段的局部重复", correct: true },
      { text: "长文档整体哈希对不上时也能删掉局部重复", correct: true },
      { text: "完全不需要任何哈希函数", correct: false },
      { text: "能把近似重复的文档也当精确重复删除", correct: false },
    ],
    explanation:
      "整文档哈希只在全文完全一致时判重，长文档只有一小段重复就会漏。按 n-gram 窗口切分，能精确删掉这些局部重复片段；但 n-gram 仍算精确匹配，不是 SimHash 那类近似去重。",
    tags: ["Data"],
  },
  {
    id: 463,
    type: "single",
    question: "数据质量评估里，衡量「数据标注是否靠谱」最常用的指标是？",
    options: [
      { text: "标签覆盖率", correct: false },
      { text: "标注一致性（inter-annotator agreement）", correct: true },
      { text: "数据吞吐量", correct: false },
      { text: "文件压缩率", correct: false },
    ],
    explanation:
      "标注质量最核心的看多个标注者之间是否一致，常用 Cohen's Kappa / Fleiss' Kappa。一致性低说明标注标准模糊或任务太难，标签本身不可信。",
    tags: ["Data"],
  },
  {
    id: 464,
    type: "single",
    question: "标注数据时，2 个标注者对 100 条样本判了 90 条一致，最简单的「一致率」是？",
    options: [
      { text: "90%", correct: true },
      { text: "80%", correct: false },
      { text: "45%", correct: false },
      { text: "无法计算", correct: false },
    ],
    explanation:
      "一致率就是意见相同的比例：90/100 = 90%。但注意它没扣掉「瞎猜也可能碰对」的部分，严谨时要用 Kappa 排除随机一致。",
    tags: ["Data"],
  },
  {
    id: 465,
    type: "multiple",
    question: "Cohen's Kappa 相比简单一致率，改进体现在哪些方面？（多选）",
    options: [
      { text: "扣除了两个标注者可能随机碰对的部分", correct: true },
      { text: "在类别不均衡时比简单一致率更可信", correct: true },
      { text: "能直接输出每个样本的最优标签", correct: false },
      { text: "与标注者人数多少完全无关", correct: false },
    ],
    explanation:
      "Kappa 用公式扣掉了双方凭运气（随机）也会一致的比例，得到「比随机好多少」。简单一致率在类不均衡时会被虚高，Kappa 更可信。",
    tags: ["Data"],
  },
  {
    id: 466,
    type: "single",
    question: "数据漂移（data drift）最典型的检测手段是？",
    options: [
      { text: "对比线上实时数据与训练数据的分布差异", correct: true },
      { text: "增大训练集规模", correct: false },
      { text: "提高模型层数", correct: false },
      { text: "把学习率调大", correct: false },
    ],
    explanation:
      "漂移就是线上数据和训练时分布不一致。检测靠统计检验或分布距离指标（PSI、KS 检验、KL 散度等）对比两个分布，及时预警重训。",
    tags: ["Data"],
  },
  {
    id: 467,
    type: "single",
    question: "「训练分布不变，但输入输出的条件关系变了」——这在数据漂移里通常叫？",
    options: [
      { text: "协变量漂移（covariate shift）", correct: true },
      { text: "概念漂移（concept drift）", correct: false },
      { text: "标签噪声", correct: false },
      { text: "样本选择偏差", correct: false },
    ],
    explanation:
      "P(x) 变了而 P(y|x) 没变叫协变量漂移；反之 P(x) 不变但 P(y|x) 的关系变了叫概念漂移（比如金融风控规则变了）。两者对策不同。",
    tags: ["Data"],
  },
  {
    id: 468,
    type: "single",
    question: "训练集中类别 A:类别 B 为 100:1，直接训练分类器最可能出的问题是什么？",
    options: [
      { text: "模型偏向把样本预测为数量多的类别 A", correct: true },
      { text: "模型对类别 A 的预测概率过低", correct: false },
      { text: "训练速度变快", correct: false },
      { text: "特征维度升高", correct: false },
    ],
    explanation:
      "类别严重不均衡时，只要全猜多数类就能拿很高准确率，梯度也偏向多数类，导致模型忽略少数类。对策包括过采样、欠采样、类别加权等。",
    tags: ["Data"],
  },
  {
    id: 469,
    type: "multiple",
    question: "处理类别不均衡，下列哪些是常见且有效的做法？（多选）",
    options: [
      { text: "对少数类做过采样（如 SMOTE）", correct: true },
      { text: "对多数类做欠采样", correct: true },
      { text: "给少数类样本更大的损失权重", correct: true },
      { text: "把多数类样本全部复制 100 倍", correct: false },
    ],
    explanation:
      "SMOTE 在少数类样本间插值合成新样本、欠采样抽掉部分多数类、或按类别反比加权损失，都管用。无脑复制多数类不会改变相对比例，等于没做。",
    tags: ["Data"],
  },
  {
    id: 470,
    type: "single",
    question: "在训练/验证/测试集划分时，能反复用测试集调参选模型吗？",
    options: [
      { text: "不能，测试集只能看一次，否则信息泄漏、分数虚高", correct: true },
      { text: "可以，测越多越准", correct: false },
      { text: "只有对深度学习不能用", correct: false },
      { text: "可以，只要用交叉验证就没事", correct: false },
    ],
    explanation:
      "测试集代表「没见过的新数据」。反复用它在上面调，等于把测试分布信息偷偷导进模型（信息泄漏），最终线上性能会明显低于测试分数。",
    tags: ["Data"],
  },
  {
    id: 471,
    type: "single",
    question: "小样本训练时，希望更稳妥地评估模型泛化能力，通常用？",
    options: [
      { text: "K 折交叉验证", correct: true },
      { text: "把全部数据当测试集", correct: false },
      { text: "只在训练集上报准确率", correct: false },
      { text: "只看单个测试样本", correct: false },
    ],
    explanation:
      "数据少时一次划分太抖。K 折把数据切 K 份轮流当验证，K 次结果取平均，评估更稳。全当测试或只看训练集都不具泛化意义。",
    tags: ["Data"],
  },
  {
    id: 472,
    type: "single",
    question: "切验证集时最容易犯的严重错误是？",
    options: [
      { text: "验证集和训练集重叠（同一条样本两边都有）", correct: true },
      { text: "验证集文件用了压缩格式", correct: false },
      { text: "验证集样本数太多", correct: false },
      { text: "标签用中文字符", correct: false },
    ],
    explanation:
      "验证集/测试集和训练集重叠=信息泄漏，模型见过答案，指标虚高。时序数据尤其要用时间切分而非随机切，避免未来信息泄漏。",
    tags: ["Data"],
  },
  {
    id: 473,
    type: "single",
    question: "DVC（数据版本管理）解决的核心痛点是什么？",
    options: [
      { text: "让大文件、数据集也能像代码一样被版本管理和回溯", correct: true },
      { text: "让 GPU 跑得更快", correct: false },
      { text: "给 Python 换解释器", correct: false },
      { text: "压缩训练日志", correct: false },
    ],
    explanation:
      "git 不适合存大文件（仓库会爆）。DVC 把数据存到外部存储，用轻量元数据文件记录版本和内容哈希，实现「数据和代码一起版本化、可复现」。",
    tags: ["Data"],
  },
  {
    id: 474,
    type: "single",
    question: "DVC 里 .dvc 文件存的内容本质上是？",
    options: [
      { text: "数据文件在远程存储的哈希与指针", correct: true },
      { text: "数据的完整二进制内容", correct: false },
      { text: "模型训练日志", correct: false },
      { text: "Python 依赖列表", correct: false },
    ],
    explanation:
      "真实数据放远程（S3、本地目录等），.dvc 只是个几 KB 的指针文件，记录内容哈希。要哪个版本 checkout 时按哈希取回对应数据，所以仓库很小。",
    tags: ["Data"],
  },
  {
    id: 475,
    type: "single",
    question: "ETL 三个字母分别对应什么？",
    options: [
      { text: "Extract（抽取）、Transform（转换）、Load（加载）", correct: true },
      { text: "Extend、Test、Log", correct: false },
      { text: "Encode、Train、Learn", correct: false },
      { text: "Extract、Tag、Label", correct: false },
    ],
    explanation:
      "ETL 就是先从源系统抽出数据（Extract），再做清洗/转换/特征处理（Transform），最后写入目标存储（Load）。是数据管道的基本骨架。",
    tags: ["Data"],
  },
  {
    id: 476,
    type: "single",
    question: "相比传统 ETL，ELT 的核心差异在于？",
    options: [
      { text: "先把原始数据原样加载到目标仓库，再做转换", correct: true },
      { text: "先转换再抽取", correct: false },
      { text: "不需要数据存储", correct: false },
      { text: "只用流式不用批处理", correct: false },
    ],
    explanation:
      "ELT 是先 Load 原始数据进数仓/湖，再在仓库里做 Transform。借助 MPP 引擎的算力在存储端大规模转换，适合海量数据和云数仓场景。",
    tags: ["Data"],
  },
  {
    id: 477,
    type: "multiple",
    question: "特征存储（feature store）主要解决哪些问题？（多选）",
    options: [
      { text: "让训练和在线推理使用同一套、同一版本的特征", correct: true },
      { text: "统一管理并复用特征，避免重复计算", correct: true },
      { text: "替代模型推理引擎", correct: false },
      { text: "生成合成数据", correct: false },
    ],
    explanation:
      "没有特征存储时，训练和线上经常各自算特征，结果对不上（training-serving skew）。特征存储统一管理、复用并保证一致性，避免线上线下特征不一致。",
    tags: ["Data"],
  },
  {
    id: 478,
    type: "single",
    question: "训练时在线服务时特征不一致（training-serving skew），最直接的危害是？",
    options: [
      { text: "线上推理的特征分布和训练时不同，模型表现变差", correct: true },
      { text: "训练速度变慢", correct: false },
      { text: "模型参数量变大", correct: false },
      { text: "损失函数失效", correct: false },
    ],
    explanation:
      "模型按训练时的特征分布学习。线上若特征口径、缺失值处理、时间点不同，输入分布偏移，模型预测自然不准。特征存储是缓解手段之一。",
    tags: ["Data"],
  },
  {
    id: 479,
    type: "single",
    question: "处理缺失值时，什么做法风险最大？",
    options: [
      { text: "直接删除整行样本", correct: false },
      { text: "在填充前先做训练/测试集划分，测试集也被填成了训练分布", correct: true },
      { text: "用中位数填充", correct: false },
      { text: "用众数填充", correct: false },
    ],
    explanation:
      "用中位数/众数填充本身没问题，但若用「全量数据」算统计量再填，就相当于把测试信息泄漏进了训练。填充统计量必须只用训练集算。",
    tags: ["Data"],
  },
  {
    id: 480,
    type: "single",
    question: "特征标准化（z-score）时，标准化的均值和标准差应该从哪算？",
    options: [
      { text: "只用训练集", correct: true },
      { text: "全量数据（含测试集）", correct: false },
      { text: "只用测试集", correct: false },
      { text: "随机数", correct: false },
    ],
    explanation:
      "用测试集算统计量等于让测试信息参与了训练，是数据泄漏。正确做法：fit 只用训练集，再对验证/测试集用同一组均值和方差 transform。",
    tags: ["Data"],
  },
  {
    id: 481,
    type: "single",
    question: "Token 化（tokenization）时对文本做 padding 的目的是？",
    options: [
      { text: "把同一 batch 内的序列对齐到相同长度以便并行计算", correct: true },
      { text: "压缩文本内容", correct: false },
      { text: "加密训练数据", correct: false },
      { text: "减少词表大小", correct: false },
    ],
    explanation:
      "batch 里文本长度不齐，GPU 并行要固定形状，所以用 padding 补齐到 batch 内最大长度。配套 attention mask 让模型忽略 padding 位置。",
    tags: ["Data"],
  },
  {
    id: 482,
    type: "single",
    question: "训练时 padding token 对应的位置，attention mask 应该置为？",
    options: [
      { text: "0（不允许被注意力关注）", correct: true },
      { text: "1（正常参与注意力）", correct: false },
      { text: "随机值", correct: false },
      { text: "-1", correct: false },
    ],
    explanation:
      "padding 位置是人为填充、无语义，必须用 mask=0 屏蔽，否则模型会把无效 token 也当有效信息学习。这是并行训练必须配套的一步。",
    tags: ["Data"],
  },
  {
    id: 483,
    type: "multiple",
    question: "MLM（masked language modeling）里 mask 的作用包含哪些？（多选）",
    options: [
      { text: "随机遮挡部分 token，让模型预测被遮内容", correct: true },
      { text: "让模型在双向上下文中建模语义", correct: true },
      { text: "删除所有低频词以压缩词表", correct: false },
      { text: "给 token 排序对齐长度", correct: false },
    ],
    explanation:
      "把输入中随机比例 token 换成 [MASK]，训练目标是预测这些被遮的原词。这样让模型学到双向上下文语义。预训练时代最常用的自监督方式之一。",
    tags: ["Data"],
  },
  {
    id: 484,
    type: "single",
    question: "长尾分布（long-tail）数据里，多数样本集中在？",
    options: [
      { text: "头部少数几个高频类别/特征上", correct: true },
      { text: "均匀分布在整个特征空间", correct: false },
      { text: "尾部少数类别上", correct: false },
      { text: "全为稀有样本", correct: false },
    ],
    explanation:
      "长尾就是「头重脚轻」：少数高频类占了大多数样本，大量长尾类样本极少。头部类学得充分，尾部类数据不足、容易欠拟合。",
    tags: ["Data"],
  },
  {
    id: 485,
    type: "single",
    question: "训练集中含 PII（个人身份信息）数据，直接明文存储的主要风险是？",
    options: [
      { text: "隐私泄露、不合规（如 GDPR/个保法），可能被滥用", correct: true },
      { text: "训练速度下降", correct: false },
      { text: "模型无法收敛", correct: false },
      { text: "文件变大", correct: false },
    ],
    explanation:
      "PII（姓名、证件号、住址等）泄露会侵犯个人隐私且违法。必须脱敏：掩码、替换、泛化（把「张三」替换成「某用户」），并控制访问权限。",
    tags: ["Data"],
  },
  {
    id: 486,
    type: "multiple",
    question: "对训练数据做脱敏/匿名化，下列哪些是常见手段？（多选）",
    options: [
      { text: "对姓名/身份证号做掩码或替换", correct: true },
      { text: "对年龄做区间泛化（如 25 → 20-30）", correct: true },
      { text: "给 PII 字段加随机噪声扰动", correct: true },
      { text: "把数据直接公开分享以证明合规", correct: false },
    ],
    explanation:
      "掩码替换、泛化（把精确值变区间）、加噪声（差分隐私思路）都是脱敏手段。公开分享恰恰是泄露，方向反了。",
    tags: ["Data"],
  },
  {
    id: 487,
    type: "multiple",
    question: "关于差分隐私（differential privacy），下列说法正确的有？（多选）",
    options: [
      { text: "在查询/训练结果里注入受控随机噪声", correct: true },
      { text: "使单条样本的存在与否几乎不影响输出", correct: true },
      { text: "删除所有敏感字段是其主要手段", correct: false },
      { text: "加密全部数据文件就能实现差分隐私", correct: false },
    ],
    explanation:
      "差分隐私通过在查询/训练结果里注入受控噪声，让攻击者无法从输出反推某个人是否存在，从而保护个体隐私。核心是「噪声」而非删除。",
    tags: ["Data"],
  },
  {
    id: 488,
    type: "multiple",
    question: "关于合成数据（synthetic data），下列说法正确的有？（多选）",
    options: [
      { text: "可批量生成、可控分布", correct: true },
      { text: "可规避隐私限制、补齐稀缺样本", correct: true },
      { text: "性能一定优于真实数据", correct: false },
      { text: "完全取代真实数据且无需评估", correct: false },
    ],
    explanation:
      "合成数据能按需造出均衡、不涉及隐私、且难收集的样本。但它可能带生成模型的偏置，不代表「一定更好」，质量需评估。",
    tags: ["Data"],
  },
  {
    id: 489,
    type: "single",
    question: "用合成数据训练时最常见的陷阱是？",
    options: [
      { text: "合成分布与真实分布不一致，导致模型在真实场景掉点", correct: true },
      { text: "合成数据总是过拟合", correct: false },
      { text: "合成数据无法转成 tensor", correct: false },
      { text: "合成数据内存一定翻倍", correct: false },
    ],
    explanation:
      "生成器可能只学会了训练它的数据分布，甚至放大偏置，导致合成分布偏离真实分布。上线前要用真实数据评估，不能只信合成数据上的指标。",
    tags: ["Data"],
  },
  {
    id: 490,
    type: "single",
    question: "数据增强（data augmentation）的目标是？",
    options: [
      { text: "用对样本的合理变换造出更多有效样本，提升泛化", correct: true },
      { text: "无脑复制样本把数据集撑大", correct: false },
      { text: "删除所有噪声样本", correct: false },
      { text: "缩小数据量节省算力", correct: false },
    ],
    explanation:
      "增强通过旋转、裁剪、加噪、同义替换等保留语义的变换增加数据多样性，抑制过拟合、提升泛化。无脑复制不增加多样性，没用。",
    tags: ["Data"],
  },
  {
    id: 491,
    type: "multiple",
    question: "关于数据增强的边界，下列说法正确的有？（多选）",
    options: [
      { text: "数字识别里把「6」旋转 180 度会改变语义", correct: true },
      { text: "对语义对称的图像随机翻转通常保持语义", correct: true },
      { text: "增强越多越好，无需考虑是否改变语义", correct: false },
      { text: "任何亮度扰动都必然改变语义", correct: false },
    ],
    explanation:
      "增强的前提是「语义不变」。数字 6 旋转 180 度会变 9，标签就错了；而亮度、翻转（若语义对称）、裁剪一般保持语义。增强要贴合任务。",
    tags: ["Data"],
  },
  {
    id: 492,
    type: "single",
    question: "训练/验证/测试集划分中，数据增强通常只应用在哪个集合上？",
    options: [
      { text: "仅训练集", correct: true },
      { text: "训练集和验证集都做", correct: false },
      { text: "三个集合都做", correct: false },
      { text: "仅测试集", correct: false },
    ],
    explanation:
      "验证/测试集要反映「真实分布」，不能做会改变分布的数据增强，否则评估失真。增强只用于训练集以提升泛化。",
    tags: ["Data"],
  },
  {
    id: 493,
    type: "single",
    question: "采样时想强制让 batch 里每个类别样本均衡，最直接的做法是？",
    options: [
      { text: "按类别分层采样（stratified sampling）", correct: true },
      { text: "随机均匀采样", correct: false },
      { text: "按文件大小采样", correct: false },
      { text: "按 token 长度采样", correct: false },
    ],
    explanation:
      "分层采样保证每个 batch 中各类别比例基本一致，避免某些 batch 全是多数类。随机采样在类别不均衡时会出现批次内类别失衡。",
    tags: ["Data"],
  },
  {
    id: 494,
    type: "multiple",
    question: "困难样本挖掘（hard example mining）通常挑选哪类样本？（多选）",
    options: [
      { text: "模型当前预测错误或置信度低的样本", correct: true },
      { text: "接近决策边界、容易混淆的样本", correct: true },
      { text: "和已有样本完全重复的样本", correct: false },
      { text: "样本量最大的多数类样本", correct: false },
    ],
    explanation:
      "困难样本就是当前模型容易搞错、接近决策边界的样本。集中训练这些，能快速逼模型把边界学对，常用于目标检测和对比学习等。",
    tags: ["Data"],
  },
  {
    id: 495,
    type: "multiple",
    question: "构建图文对（image-text pair）数据时，下列哪些是正确的清洗动作？（多选）",
    options: [
      { text: "过滤掉图文语义不匹配的样本", correct: true },
      { text: "剔除包含 PII 的文本", correct: true },
      { text: "用 CLIP 相似度评分筛选高质量对", correct: true },
      { text: "文本和图像随意乱配也能用于对齐训练", correct: false },
    ],
    explanation:
      "图文对齐训练要求图文真的对应。语义不匹配、含隐私、相似度极低（说明对不上）的都该过滤。乱配会严重污染跨模态对齐。",
    tags: ["Data"],
  },
  {
    id: 496,
    type: "single",
    question: "图文对数据里若大量出现「文本描述通用、图片千奇百怪」的对，可能导致？",
    options: [
      { text: "模型学到文本与图像弱相关，对齐能力差", correct: true },
      { text: "模型训练变快", correct: false },
      { text: "模型参数量自动变小", correct: false },
      { text: "不需要清洗", correct: false },
    ],
    explanation:
      "弱对齐的图文对会教模型「随便什么图配什么文都行」，削弱图文语义关联。CLIP 这类模型对数据质量极其敏感，低质数据直接拖累对齐。",
    tags: ["Data"],
  },
  {
    id: 497,
    type: "single",
    question: "scaling law 里关于「数据量」的经典规律是？",
    options: [
      { text: "模型性能随训练数据量按幂律大致提升", correct: true },
      { text: "数据越多性能线性下降", correct: false },
      { text: "数据量只影响速度不影响效果", correct: false },
      { text: "数据量超过 10 亿就完全无提升", correct: false },
    ],
    explanation:
      "Chinchilla 等研究显示，在算力约束下性能随数据量按幂律（幂次通常小于 1）增长，且存在「模型大小与数据量要按比例一起放大」的平衡点。",
    tags: ["Data"],
  },
  {
    id: 498,
    type: "single",
    question: "「数据量翻倍、性能提升边际递减」，这体现 scaling law 的哪个特征？",
    options: [
      { text: "幂律增长的收益逐渐趋缓", correct: true },
      { text: "线性增长", correct: false },
      { text: "无规律随机波动", correct: false },
      { text: "先下降后上升", correct: false },
    ],
    explanation:
      "scaling law 是幂律（约 data 的 0.095 次方），指数小于 1，所以数据翻倍带来的提升是递减的。这也是为什么要在数据质量和数量间权衡。",
    tags: ["Data"],
  },
  {
    id: 499,
    type: "multiple",
    question: "影响 LLM 最终能力的「数据三要素」通常指？（多选）",
    options: [
      { text: "数据数量（规模）", correct: true },
      { text: "数据质量", correct: true },
      { text: "数据多样性", correct: true },
      { text: "文件名的长短", correct: false },
    ],
    explanation:
      "数量管规模、质量管下限、多样性管泛化，三者共同决定模型上限。仅堆量不重质（如重复数据）收益很低，质量与多样性同样关键。",
    tags: ["Data"],
  },
  {
    id: 500,
    type: "single",
    question: "使用他人受版权保护的数据训练模型，核心合规风险是？",
    options: [
      { text: "版权/授权问题，可能构成侵权", correct: true },
      { text: "文件一定无法解析", correct: false },
      { text: "模型必然崩溃", correct: false },
      { text: "数据量自动翻倍", correct: false },
    ],
    explanation:
      "爬取并训练受版权内容，核心是授权问题——是否获许可、是否违反服务条款、是否合理使用。数据合规审查需要记录数据来源与授权状态。",
    tags: ["Data"],
  },
  {
    id: 501,
    type: "single",
    question: "构建训练数据集时，理想的做法是给每条数据记录来源和授权信息，主要目的是？",
    options: [
      { text: "保证可追溯、便于合规审计和后续维权", correct: true },
      { text: "让文件更小", correct: false },
      { text: "加快模型收敛", correct: false },
      { text: "替代模型训练", correct: false },
    ],
    explanation:
      "记录数据来源/授权/许可证（provenance），一旦有版权争议可追溯来源、证明合法性，是数据合规治理的基本要求。",
    tags: ["Data"],
  },
  {
    id: 502,
    type: "single",
    question: "数据管道（pipeline）里某一步骤偶发失败，最健壮的处理方式是？",
    options: [
      { text: "加入失败重试（retry）与失败记录/死信队列", correct: true },
      { text: "直接静默忽略错误继续跑", correct: false },
      { text: "删库重跑", correct: false },
      { text: "关掉日志", correct: false },
    ],
    explanation:
      "偶发失败（网络抖动、超时）用带退避的重试处理；多次仍失败的进死信队列（DLQ）记录待查，而不是静默吞掉错误导致数据静默缺失。",
    tags: ["Data"],
  },
  {
    id: 503,
    type: "multiple",
    question: "要防止管道静默产出脏数据，下列哪些是合理做法？（多选）",
    options: [
      { text: "在关键步骤做数据质量断言，失败即中断", correct: true },
      { text: "校验行数、null 比例、schema 等是否达标", correct: true },
      { text: "静默忽略校验错误继续执行", correct: false },
      { text: "只在全部跑完后统一检查一次", correct: false },
    ],
    explanation:
      "静默产生脏数据比显式报错更危险。在每步对行数、null 比例、schema 做校验断言，不达标就 fail-fast，让问题尽早暴露而不是污染下游。",
    tags: ["Data"],
  },
  {
    id: 504,
    type: "single",
    question: "海量结构化数据做列式存储，最常推荐哪种格式？",
    options: [
      { text: "Parquet", correct: true },
      { text: "JSON Lines", correct: false },
      { text: "纯文本 CSV", correct: false },
      { text: "XML", correct: false },
    ],
    explanation:
      "Parquet 是列式存储，按列压缩、只读需要的列，I/O 效率高，还有内嵌 schema。CSV/JSONL 是行式，扫描慢且占空间。",
    tags: ["Data"],
  },
  {
    id: 505,
    type: "multiple",
    question: "相比行式存储（如 CSV），列式存储（如 Parquet）的主要优势有？（多选）",
    options: [
      { text: "同类数据相邻，压缩率更高", correct: true },
      { text: "只需读取查询涉及的列，I/O 更少", correct: true },
      { text: "内嵌 schema，类型更清晰", correct: true },
      { text: "数据天然加密不可读", correct: false },
    ],
    explanation:
      "列式存储同类值聚在一起压缩率高；分析查询只取若干列，避免整行全读；自带 schema 比 CSV 的字符串推断更可靠。加密与格式无关。",
    tags: ["Data"],
  },
  {
    id: 506,
    type: "single",
    question: "训练迭代中「数据加载」成为瓶颈、GPU 空等，最常用的解法是？",
    options: [
      { text: "用多进程/预取（prefetch）+ 异步加载流水线", correct: true },
      { text: "把模型改小", correct: false },
      { text: "提高学习率", correct: false },
      { text: "删掉部分数据", correct: false },
    ],
    explanation:
      "数据加载慢于计算就是 I/O 瓶颈。用多进程加载 + 预取到队列，让 CPU 准备数据和 GPU 计算重叠（pipeline），GPU 不再空等。",
    tags: ["Data"],
  },
  {
    id: 507,
    type: "single",
    question: "数据漂移监测发现线上分布明显偏离训练分布，最合理的下一步是？",
    options: [
      { text: "定位漂移来源，用新分布数据补充/重训并评估", correct: true },
      { text: "忽略它继续上线", correct: false },
      { text: "把学习率调为零", correct: false },
      { text: "删除所有旧数据", correct: false },
    ],
    explanation:
      "漂移要追根因（是数据源变了还是概念变了），针对性补充或重训，并在真实分布上评估。直接忽略或瞎调超参都不解决问题。",
    tags: ["Data"],
  },
  {
    id: 508,
    type: "single",
    question: "标注数据时引入「标注者个人主观偏好」导致标签系统性地偏离真实，这种问题属于？",
    options: [
      { text: "标注偏置（label bias）", correct: true },
      { text: "模型过拟合", correct: false },
      { text: "数据泄露", correct: false },
      { text: "token 截断", correct: false },
    ],
    explanation:
      "标注者个人理解、先验观念造成标签系统偏移，叫标注偏置。缓解靠制定清晰标注规范、多人交叉标注、用多数投票或专家复核。",
    tags: ["Data"],
  },

];