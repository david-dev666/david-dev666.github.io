// 题库数据类型定义
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
