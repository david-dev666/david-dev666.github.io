export const siteConfig = {
  name: "博文的博客",
  title: "聚焦VLM多模态技术",
  description: "博文的博客 - 探索视觉语言模型与人工智能",
  accentColor: "#7c3aed",
  // 社交链接（暂时关闭）
  // social: {
  //   email: "your-email@example.com",
  //   github: "https://github.com/david-dev-lab",
  // },
  aboutMe: "你好，我是段博文，目前专注于VLM实践。",
  // 技能标签（暂时关闭）
  // skills: ["Python", "PyTorch", "Transformers", "Computer Vision", "NLP", "Deep Learning"],
  skills: [],
  projects: [
    {
      name: "CLIP",
      description:
        "图文跨模态检索与零样本图像分类",
      link: "/projects/clip",
      tags: ["VLM", "CLIP"],
      isExternal: true,
    },
    {
      name: "ControlNet",
      description:
        "为文生图扩散模型添加精确的空间条件控制",
      link: "/projects/controlnet",
      tags: ["Diffusion Model", "Image Generation"],
      isExternal: true,
    },
    {
      name: "MiniLLM",
      description:
        "从零构建大语言模型：深入解析左填充、内存映射与对齐算法",
      link: "/projects/build_MiniLLM_from_scratch",
      tags: ["LLM", "SFT", "DPO"],
      isExternal: true,
    },
    // 以下项目暂时关闭
    // {
    //   name: "多模态问答系统",
    //   description:
    //     "结合图像理解和文本生成的问答系统，可以回答关于图片内容的各种问题",
    //   link: "#",
    //   skills: ["Python", "CLIP", "LLM"],
    // },
    // {
    //   name: "VLM 学习笔记",
    //   description:
    //     "整理的视觉语言模型学习资料和实践心得，包括论文阅读笔记和代码实现",
    //   link: "#",
    //   skills: ["Deep Learning", "VLM", "Research"],
    // },
  ],
  experience: [
    {
      company: "",
      title: "VLM 研究与实践",
      dateRange: "2024 - 至今",
      bullets: [
        "深入学习 CLIP、BLIP、LLaVA 等主流视觉语言模型",
        "复现经典论文并进行实验验证",
        "构建个人项目积累实践经验",
      ],
    },
  ],
  // 学习板块（暂时关闭）
  // education: [
  //   {
  //     school: "自学",
  //     degree: "人工智能与深度学习",
  //     dateRange: "持续学习中",
  //     achievements: [
  //       "系统学习深度学习基础理论",
  //       "掌握 PyTorch 框架和 Transformers 库",
  //       "专注于视觉语言模型（VLM）领域研究",
  //     ],
  //   },
  // ],
  education: [],
};
