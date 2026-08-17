# 阶段报告：工程落地 + RAG 端到端演示

> 项目：混合 RAG 与 vLLM 加速的学术文档 Agent 系统 · 2025.10 – 2026.3
> 更新：2026-08-17

## 一、本阶段目标与结果

完成项目工程化骨架与四大技术支柱的代码实现，并在本机（CPU）跑通
**数据采集 → 版面解析 → 层级切块 → 混合检索 → 重排 → API 服务**的完整 RAG 链路。

## 二、代码交付物

| 模块 | 位置 | 状态 |
| --- | --- | --- |
| 架构与设计文档 | `docs/`（5 篇） | 完成 |
| 数据管道（采集/解析/清洗/造数/词表扩充） | `src/data/` + `scripts/data/` | 完成，本机可跑 |
| RAG（切块/Milvus+BM25/重排/Self-RAG） | `src/rag/` + `scripts/rag/` | 完成，本机已跑通 |
| 训练（QLoRA SFT + DPO） | `src/train/` + `scripts/train/` | 完成，待 GPU 服务器运行 |
| 部署（vLLM + Guided Decoding + FastAPI） | `src/serving/` + `scripts/serving/` | 完成，本机 mock 模式已跑通 |
| 评测（忠实度/引用） | `src/eval/` | 完成 |
| 配置（YAML/JSON） | `configs/` | 完成 |

## 三、本机演示结果

### 数据

- 下载并解析 2 篇 ArXiv 论文：Attention Is All You Need（1706.03762）、Retrieval-Augmented Generation（2005.11401）
- 版面解析成功还原标题、摘要、章节层级与页码
- 层级切块得到 **266 个片段**（章节父块 + 段落子块，带 section_path / page / parent_id）

### 检索（Milvus 稠密 + BM25 稀疏 + RRF 融合 + 交叉编码器重排）

查询 "How is scaled dot-product attention computed in the Transformer?" 的 top-5 结果全部命中
原论文 Model Architecture 章节，其中一条直接命中公式原文：

> "we compute the dot products of the query with all keys, divide each by √d_k, and apply a softmax function..."

### API 服务（mock 模式，本机可联调）

- `GET /health` → 正常
- `POST /v1/retrieve` → 返回片段、章节、页码、分数、来源
- `POST /v1/chat/completions` → 返回符合 JSON Schema 的结构化回答：
  `{answer, reasoning, citations:[{chunk_id, confidence}], confidence, is_grounded}`
- 客户端脚本 `scripts/serving/smoke_test.py` 一次跑通三接口

## 四、环境结论

- 本机：Windows 11、CPU、无 NVIDIA GPU、磁盘充足
- 本机可完整跑：数据管道、RAG、评测、API mock 联调
- 需 GPU 服务器（Linux，建议 ≥24GB 显存）运行：SFT、DPO、vLLM 部署
- 已处理的本机开发问题：Windows 长路径限制（虚拟环境放短路径）、HuggingFace 官方域名不通
  （使用 hf-mirror.com 镜像）、ArXiv 接口 429 限流（自动退避 + 直连 PDF 降级）

## 五、下一步

1. 确认 GPU 来源（云 GPU / 实验室服务器 / WSL2）
2. （可选）配置 OpenAI API Key，启动 GPT-4 造数
3. 扩充语料（目标数百篇）→ BPE 词表扩充 → SFT → DPO → AWQ → vLLM 部署
4. 评测：检索 Recall@K、忠实度、引用命中、延迟

复现方式见项目根目录 `README.md`。

---

# 阶段报告（更新）：AutoDL 7B 全流程训练与部署完成

> 更新：2026-08-17（AutoDL RTX 4090 D / 24G，约 3 小时完成）

## 一、训练数据

- ArXiv 论文：42 篇（cs.CL/AI/LG 领域，含最新 2026 论文）
- SFT 数据集：153 条多轮问答（mock 模式构建，含章节上下文与引用标注）
- DPO 数据集：306 条偏好对（幻觉/无引用/推理错序三类负样本）
- BPE 领域词表扩充：141 个学术 token（arXiv、Theorem、Corollary 等）

## 二、训练结果（Qwen2.5-7B-Instruct）

### SFT（QLoRA 4-bit，400 步，46 分钟）

- train_loss：0.197
- eval_loss：0.106
- 合并后完整模型：`work/models/sft-qwen7b-merged`（7.3GB）

### DPO（LoRA 4-bit，200 步，52 分钟）

- train_loss：0.24 → 0.082
- rewards/chosen：-2.5（偏高为优），rewards/rejected：-5.4
- rewards/accuracies：98.75% - 100%（偏好排序准确率）
- rewards/margins：约 2.8（chosen 明显优于 rejected）

## 三、量化与部署

- AWQ 4-bit 量化：28 层 8 分钟，产出 `work/models/dpo-qwen7b-awq`（5.2GB）
- vLLM 0.8.5 + xgrammar 引导解码：按 JSON Schema 强约束生成
- FastAPI OpenAI 兼容服务：`POST /v1/chat/completions` 返回
  `{answer, reasoning, citations, confidence, is_grounded}`，实测输出合法 JSON
- 流式接口：SSE 协议完整（含 `[DONE]`）

## 四、实测回答示例（训练后模型）

> 问：Explain how the retriever and generator work together in RAG.
>
> 答：检索器在大规模知识库中检索相关信息增强输入，生成器利用检索结果
> 生成连贯且与上下文相关的输出……（is_grounded=True, confidence=0.9）

## 五、环境要点（供复现）

- torch 2.6.0+cu124（驱动 550 上限 CUDA 12.4，不能装 cu126+）
- transformers 5.15（Trainer 用 processing_class；词表掩码用偏移映射自算）
- vLLM 0.8.5（GuidedDecodingParams；AWQ 必须 float16；离线接口无 stream，已做 SSE 兼容回退）
- autoawq 0.2.9（校准数据需切成短样本）
- wandb 需 <0.19（兼容 TRL）

## 六、下一步

- 服务器上接入 RAG 检索器（Milvus/BM25 索引已在本机建好，可上传）
- 扩充语料到数百篇后重训，指标会明显更好
- 跑评测：检索 Recall@K、忠实度（NLI 支持率）、引用命中、延迟
- 保存 AutoDL 镜像，避免重装系统丢失训练产物

---

# 阶段报告（更新 2）：服务器端 RAG 完整接入

> 更新：2026-08-17

## 一、服务器检索索引

- 语料：44 篇 ArXiv 论文（服务器侧采集）
- 层级切块：8,771 个片段（章节父块 + 段落子块，含页码/章节路径）
- 稠密索引：Milvus（milvus-lite 3.2.0 + pymilvus 3.0.1，COSINE，8,771 条）
- 稀疏索引：BM25（k1=1.5, b=0.75，pickle 缓存）
- 重排：cross-encoder MiniLM（top-5）

## 二、端到端实测（训练后 7B + RAG）

问：What is multi-head attention and why does the Transformer use it?

- 检索 top-3：全部命中 Attention 论文 Model Architecture 章节
- 回答：合法 JSON（answer/reasoning/citations/confidence/is_grounded）
- 引用：[1706.03762__00030, 1706.03762__00016]（真实存在于索引中）
- is_grounded=True, confidence=0.9

## 三、当前服务器服务

```text
http://<实例>:8000
GET  /health                      # 健康检查（retriever=true）
POST /v1/retrieve                 # 检索调试（片段+分数+来源）
POST /v1/chat/completions         # RAG 问答（引导解码 + 流式 SSE）
```

## 四、技术备注

- pymilvus 3.x 的 MilvusClient API 与 2.x 兼容，现有 store.py 无需改动
- vLLM 离线接口不支持逐 token 流式，SSE 已做单块回退；生产真流式可直连
  `scripts/serving/run_vllm.sh` 的 OpenAI 服务

---

# 阶段报告（更新 3）：正式评测指标

> 更新：2026-08-17 · 评测脚本 `scripts/eval/run_eval.py` · 原始数据 `outputs/eval_report.json`

## 一、检索指标（50 个单句查询，gold=来源片段）

通过 API 真实服务链路评测（Milvus + BM25 + RRF + 重排）：

| 方法 | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- |
| 稠密（bge-small） | 0.90 | 0.90 | 0.83 |
| 稀疏（BM25） | 0.98 | 1.00 | 0.95 |
| **混合（RRF+重排）** | **1.00** | **1.00** | **0.99** |

结论：混合检索在召回与排序上均优于单一检索，验证了混合策略的增益。

## 二、RAG 生成指标（10 个查询，训练后 7B 模型）

| 指标 | 数值 |
| --- | --- |
| 忠实度（主张级 NLI 支持率） | 0.90 |
| 引用召回（Citation Recall） | 0.60 |
| 引用精确率（Citation Precision） | 0.55 |
| 引用-检索对齐率（grounding） | 0.80 |
| 平均生成延迟 | 13.6 s/query |
| 平均检索延迟（hybrid，CPU 重排） | 2.0 s/query |

## 三、解读与下一步优化

- 检索召回已到 100%（本评测集），说明切块与混合检索链路有效
- 忠实度 90%：模型大部分主张能被检索证据支持，符合 DPO 对齐目标
- 引用精确率偏低（55%）：模型偶尔引用未检索到的片段或漏引；改进方向：
  1. 在 DPO 数据中加强"只能引用给定片段"的负样本
  2. 引导解码 schema 中限制 citations 上限（如 maxItems: 3）
- 延迟优化：重排模型改跑 GPU、嵌入换 bge-m3、vLLM 开启 prefix caching 可显著降低
- 数据规模（44 篇/153 条）为演示级，扩到数百篇后指标可进一步提升
