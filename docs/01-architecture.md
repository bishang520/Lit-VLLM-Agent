# 架构设计

## 1. 总体目标

面向长篇幅科研论文与技术文档，提供"问题 → 检索 → 精读 → 结构化回答"的完整链路。系统由四层构成：

1. **数据层**：ArXiv 采集、版面解析、层次化切块、清洗、GPT-4 辅助造数、BPE 词表扩充。
2. **训练层**：QLoRA SFT 多轮指令微调 → DPO 偏好对齐 → 合并导出 → AWQ 量化。
3. **检索层**：Milvus（稠密）+ BM25（稀疏）混合检索 → BGE Reranker 重排 → Self-RAG 自评估。
4. **服务层**：vLLM 推理引擎 + Guided Decoding JSON Schema 约束 + FastAPI OpenAI 兼容接口（流式）。

## 2. 关键设计决策

| 决策点 | 选择 | 理由 |
| --- | --- | --- |
| 基座模型 | Qwen2.5-7B-Instruct（可切 3B/14B/Llama） | 中文+英文学术场景均衡、许可友好、vLLM 生态成熟 |
| 微调方式 | QLoRA（4-bit NF4）+ DeepSpeed ZeRO-2 | 单卡可训 7B，显存/效果折中 |
| 词表扩充 | BPE 训练 + 压缩收益筛选 + embedding resize | 降低学术术语切分损耗，需随 SFT 一起训练新 token |
| 稠密向量 | BGE 系列（bge-small 演示 / bge-m3 生产） | 学术语义检索效果好、中文英文均覆盖 |
| 混合检索 | RRF（Reciprocal Rank Fusion） | 无需调分数尺度，稳定且鲁棒 |
| 重排序 | 交叉编码器（bge-reranker / MiniLM） | 精排精度显著高于双塔 |
| 抗幻觉 | Self-RAG 三段式：相关性门控 + 主张级 NLI 校验 + 引用可溯源性 | 可解释、可量化为指标 |
| 约束解码 | vLLM Guided Decoding（xgrammar）按 JSON Schema 强制生成 | 服务端保证输出结构合法 |
| 量化 | AWQ 4-bit | 显存减半、延迟降低、精度损失小 |
| 服务接口 | OpenAI 兼容 | 便于接入 LangChain / 前端 / 评测工具 |

## 3. 数据流

```text
ArXiv PDF
  -> 版面解析（PyMuPDF；生产可切 Marker）
  -> 文档树（title / abstract / sections / paragraphs）
  -> 层级切块（chunk 携带 doc_id、section_path、page、parent_id）
  -> 清洗去重
      -> 稠密索引（Milvus）+ 稀疏索引（BM25）共享同一份 chunk
      -> GPT-4 辅助生成 QA / CoT / 偏好对 -> SFT、DPO 数据集
      -> BPE 语料 -> 领域词表扩充
```

## 4. 服务链路

```text
POST /v1/chat/completions
  -> 混合检索 top-k（RRF 融合 20+20 -> 精排 5）
  -> Self-RAG 相关性门控
  -> 拼接上下文（章节标题 + 原文 + 页码）进 prompt
  -> vLLM guided_json=Schema 生成 {answer, reasoning, citations, confidence, is_grounded}
  -> 主张级忠实度校验，不达标则标记/拒答
  -> SSE 流式返回
```

## 5. 目录与模块职责

见 [README](../README.md#目录结构)。核心包：

- `src/data`：从 PDF 到数据集与词表的一切离线工作。
- `src/rag`：检索与自评估，纯 Python 可运行，依赖 Milvus Lite 或内存向量库。
- `src/train`：SFT / DPO 训练入口，面向 GPU 服务器。
- `src/serving`：服务化层，vLLM 可替换为 mock 便于本地联调。
- `src/eval`：忠实度与引用评测，为简历指标提供量化依据。

## 6. 环境矩阵

| 环境 | 用途 | 说明 |
| --- | --- | --- |
| 本机 Windows（CPU） | 数据管道、RAG、评测、接口联调（mock 模式） | 安装 requirements-base/rag |
| GPU 服务器（Linux） | SFT、DPO、vLLM 部署 | 安装 requirements-train/serving |

vLLM 不支持原生 Windows；若本机有 NVIDIA GPU，可在 WSL2（Ubuntu + CUDA）中运行。
