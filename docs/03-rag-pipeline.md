# RAG 管道设计

## 1. 层次化切块（`src/rag/chunker.py`）

保留论文结构信息：

- 以章节为单位，段落合并到目标块大小（默认 600 token），带 80 token 重叠；
- 每个 chunk 携带：`doc_id`、`chunk_id`、`section_path`、`headings`、`page`、`parent_id`；
- 支持按 `section` 过滤与父块回跳（检索到子块可返回整节上下文）。

## 2. 稠密索引：Milvus（`src/rag/store.py`）

- 默认使用 Milvus Lite（单文件嵌入式，零运维）；
- 字段：`id / chunk_id / doc_id / text / section / page / parent_id / vector / metadata`；
- 距离度量：COSINE；演示用 FLAT，数据量大后可换 HNSW。
- 无 Milvus 时自动降级为内存 NumPy 向量库（`InMemoryStore`），便于本机演示。

## 3. 稀疏索引：BM25（`src/rag/bm25.py`）

- `rank_bm25` 实现，k1=1.5、b=0.75；
- 轻量分词（小写 + 字母数字 token + 学术常见缩写）；
- 索引持久化到 pickle，`ingest` 后可直接加载。

## 4. 混合检索与融合（`src/rag/retriever.py`）

稠密与稀疏各取 top-20，采用 **RRF** 融合：

```text
score(chunk) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)
```

- 可调权重，默认 1:1，RRF k=60；
- 融合后取 top-30 进入重排。

## 5. 重排序（`src/rag/reranker.py`）

- 交叉编码器：`cross-encoder/ms-marco-MiniLM-L-6-v2`（演示）/ `BAAI/bge-reranker-v2-m3`（生产）；
- 对 (query, chunk) 逐对打分，取 top-k（默认 5）进入 LLM 上下文。

## 6. Self-RAG 自评估（`src/rag/self_rag.py`）

三段式抗幻觉：

1. **相关性门控**：query 与 top-k chunk 相似度低于阈值 → 判"检索不足"，提示用户换问法或扩展检索；
2. **主张级忠实度**：将回答切分为主张（句子级），用 NLI 交叉编码器逐条判定"证据是否蕴含主张"，低于阈值的判为不忠实；
3. **引用可溯源性**：每条主张必须映射到某个 chunk（`citation_threshold`），无法映射则要求拒答或标注不确定。

输出 `SelfRagReport`：相关性判定、主张列表与各自判定、忠实比例、引用列表。

## 7. 评测（`src/eval/`）

- 检索：Recall@K / MRR；
- 忠实度：主张 NLI 支持率；
- 引用：Citation Recall / Precision；
- 端到端：人工 + LLM-as-Judge 评分（可选）。
