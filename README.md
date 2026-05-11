# A3 Java Programming Assistant

这是一个用于 INFS4205/7205 A3 的 **Personalised Multimodal Java Programming Assistant** 项目模板。
系统使用个人 Java PDF 资料和图片资料构建多模态知识库，将 PDF 文本和图片 OCR/caption 内容写入 Chroma 向量数据库，然后用 LangGraph Agent 完成问题分类、检索规划、证据检索、回答生成、答案验证和修复。

## 1. Project Goal

本项目的研究问题是：

> Compared with a plain LLM and text-only RAG, can a LangGraph-based agent using hybrid retrieval over Java notes, project documents, debugging screenshots, and architecture diagrams provide more accurate and grounded Java programming assistance?

中文理解：

> 相比普通 LLM 和纯文本 RAG，结合 Java PDF 文档、报错截图、架构图，并使用 LangGraph 进行检索规划与答案验证，是否能提供更准确、更有证据支撑的 Java 程序员问答服务？

## 2. Directory Structure

```text
A3_Java_Programming_Assistant/
│
├── data/
│   ├── raw/
│   │   ├── pdfs/               # 放 Java PDF 笔记、项目文档、报错总结
│   │   └── images/             # 放 IDE 截图、报错截图、架构图、UML 图
│   └── processed/              # ingest.py 生成的 chunks 和 metadata
│
├── chroma_db/                  # Chroma 本地向量数据库
│
├── src/
│   ├── config.py               # 读取 .env 配置
│   ├── llm_client.py            # OpenAI-compatible API 封装，适配阿里云百炼
│   ├── utils.py                 # JSONL、chunk、清洗等工具函数
│   ├── ingest.py                # PDF/image 处理并构建 Chroma index
│   ├── vector_store.py          # Chroma 封装
│   ├── retriever.py             # text/image/hybrid 检索策略
│   ├── graph_agent.py           # LangGraph final agent workflow
│   ├── baselines.py             # B0/B1/B2/B3 对比系统
│   └── evaluation.py            # 自动评估脚本
│
├── eval/
│   └── benchmark_questions.json # structured benchmark questions
│
├── outputs/
│   ├── eval_results.csv         # 每题每系统详细结果
│   ├── eval_summary.json        # 汇总指标
│   └── agent_traces.jsonl       # LangGraph trace
│
├── main.py                      # CLI 问答入口
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 3. Data Preparation

请把你的真实资料放入：

```text
data/raw/pdfs/
data/raw/images/
```

推荐 PDF：

```text
java_core_notes.pdf
spring_boot_notes.pdf
java_interview_notes.pdf
common_java_errors.pdf
```

推荐图片：

```text
controller_service_mapper_architecture.png
spring_boot_project_structure.png
maven_dependency_error.png
mysql_connection_error.png
```

注意：图片不是只放进去就结束。`src/ingest.py` 会对图片进行 OCR 和 VLM caption，然后生成 `image_chunks.jsonl` 并写入 Chroma。

## 4. Install Dependencies

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果使用 OCR，需要本机安装 Tesseract。没有 Tesseract 也可以运行，因为系统主要依赖 Vision model caption；OCR 失败会自动跳过。

## 5. Configure API Key

复制模板：

```bash
cp .env.example .env
```

然后在 `.env` 中填入你的阿里云百炼 API Key：

```env
OPENAI_API_KEY=your_bailian_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
EMBEDDING_MODEL=text-embedding-v4
VISION_MODEL_NAME=qwen-vl-plus
```

本项目使用 **OpenAI-compatible mode**。API Key 和 BASE_URL 保持一致，不同用途通过不同 model name 区分。

## 6. Build Vector Database

```bash
python -m src.ingest
```

该命令会完成：

1. 读取 `data/raw/pdfs/`
2. 提取 PDF 文本并切分 chunks
3. 读取 `data/raw/images/`
4. 使用 OCR / Vision model 生成图片文本化证据
5. 生成 `data/processed/text_chunks.jsonl`
6. 生成 `data/processed/image_chunks.jsonl`
7. 生成 `data/processed/all_chunks.jsonl`
8. 将所有 chunks 向量化并写入 Chroma

## 7. Run Final LangGraph Agent

交互式运行：

```bash
python main.py
```

单问题运行：

```bash
python main.py --question "According to my Spring Boot notes, what does the Controller layer do?"
```

输出包括：

- final answer
- query type
- retrieval plan
- groundedness
- token usage
- retrieved evidence

## 8. Run Baselines

```bash
python main.py --system B0_plain_llm --question "What does my Maven error screenshot show?"
python main.py --system B1_text_only_rag --question "What does my Maven error screenshot show?"
python main.py --system B2_multimodal_rag --question "What does my Maven error screenshot show?"
python main.py --system B3_final_langgraph_agent --question "What does my Maven error screenshot show?"
```

系统版本：

| System | Meaning |
|---|---|
| B0_plain_llm | 不使用知识库，直接问 LLM |
| B1_text_only_rag | 只检索 PDF 文本 chunks |
| B2_multimodal_rag | 检索 PDF + image caption/OCR chunks |
| B3_final_langgraph_agent | LangGraph routing + retrieval + verification |

## 9. Run Evaluation

```bash
python -m src.evaluation
```

输出文件：

```text
outputs/eval_results.csv
outputs/eval_summary.json
outputs/agent_traces.jsonl
```

评估指标包括：

- Recall@3
- Task Success
- Groundedness
- Latency
- Token Usage
- Tool Calls

`eval/benchmark_questions.json` 是示例测试集。请根据你自己放入的 PDF 和图片文件名修改 `gold_evidence`，否则 Recall@3 可能不准确。

## 10. Report Writing Notes

报告中可以写：

- Knowledge Base: PDF text + image-derived chunks
- Retrieval: Chroma vector DB with metadata, supporting text/image/hybrid retrieval
- Agent: LangGraph workflow with classify → plan → retrieve → generate → verify → repair
- Evaluation: B0 Plain LLM, B1 Text-only RAG, B2 Multimodal RAG, B3 Final Agent
- Metrics: Recall@3, Task Success, Groundedness, Latency, Token Usage, Tool Calls

## 11. Important Submission Warning

不要提交真实 `.env` 文件和 API Key。

提交前确认：

```text
[ ] data/raw/pdfs/ 有真实 PDF 资料
[ ] data/raw/images/ 有真实图片/截图/图表
[ ] python -m src.ingest 成功运行
[ ] python main.py 可以回答问题
[ ] python -m src.evaluation 生成 outputs 文件
[ ] .env 没有被打包提交
[ ] .env.example 保留在提交包中
```
