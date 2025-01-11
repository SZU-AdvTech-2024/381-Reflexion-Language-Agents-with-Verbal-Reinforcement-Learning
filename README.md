# 🤖 ReAct-Reflexion 复现实验

这是一个复现 ReAct-Reflexion 论文的项目。该项目实现了一个能够在问答过程中进行自我反思的 Agent。

## 🎯 项目目标

- 复现 ReAct 和 Cot 推理框架
- 实现 Reflexion 反思机制
- 在 HotpotQA 数据集上进行实验评估
- 对比不同反思策略的效果

## 🏗️ 项目结构

├── agents/ # Agent 实现
│ ├── react_agent.py # 基础 ReAct Agent
│ ├── react_reflect_agent.py # 带反思机制的 Agent
│ └──  cot_agent.py # Chain-of-Thought Agent
├── utils/ # 工具函数
│ ├── llms.py # LLM 调用封装
│ ├── prompt.py # 提示词模板
│ └── fewshots.py # Few-shot 示例
├── data/ # 数据集
├── output/ # 输出结果
├── run_hotpot_cot.py # 运行 HotpotQA 数据集上的实验，基于 Cot 的推理框架
├── run_hotpot_react.py # 运行 HotpotQA 数据集上的实验，基于 ReAct 的推理框架
├── figures.ipynb # 绘制实验结果图表
└── .env # 环境变量

## 🛠️ 使用方法

1. 安装依赖:

bash
pip install -r requirements.txt

2. 配置环境变量:

修改 .env 文件，设置 OPENAI_LLM_API_KEY 和 LOCAL_LLM_BASE_URL 等环境变量

3. 运行实验（可以跳过）:

bash
python run_hotpot_cot.py
python run_hotpot_react.py

结果将输出到 output/ 目录下

4. 分析结果:

bash
jupyter notebook figures.ipynb


## 📝 注意事项

- 需要配置 LLM API 密钥
- 实验结果可能因模型和参数设置而异
- 建议使用 Python 3.8+ 版本

## 🔗 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [HotpotQA Dataset](https://hotpotqa.github.io/)

