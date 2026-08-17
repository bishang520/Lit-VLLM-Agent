# 数据管道设计

## 1. 数据源与采集

- 数据源：ArXiv 开源论文（默认分类 cs.CL / cs.AI / cs.LG / math.LG）。
- 采集方式：`arxiv` Python 包（API），支持按 arXiv ID 或查询词批量下载 PDF。
- 产物：PDF 文件 + 解析后的结构化文本。

```bash
python scripts/data/fetch_papers.py --ids 1706.03762,2005.11401
python scripts/data/fetch_papers.py --query "retrieval augmented generation" --max-results 200
```

## 2. 版面解析（`src/data/layout.py`）

目标：从 PDF 还原"论文 → 章节 → 段落"层级，而非纯文本流。

- 默认后端：PyMuPDF，按字体大小区分标题/正文，按页聚合段落。
- 生产可选后端：Marker / Unstructured，处理表格、公式、双栏等复杂版面。
- 产物：`LayoutDocument(title, abstract, blocks[])` 与 `Section` 树。

## 3. 清洗与质量过滤（`src/data/cleaning.py`）

- Unicode 归一化（NFKC）、空白/换行折叠。
- LaTeX 残留与 arXiv 页眉页脚清理。
- 质量过滤：最短长度、最少句子数、重复字符比例、语言粗略判定。
- 去重：正文 shingle 哈希。

## 4. GPT-4 辅助造数（`src/data/sft_builder.py` / `dpo_builder.py`）

### SFT 数据集

每条样本为多轮对话，要求模型：

- 基于给定原文回答，必须给出引用标注 `[chunk_id]`；
- 输出 Chain-of-Thought 推理；
- 不确定时明确说明，禁止编造。

生成方式：GPT-4 根据论文章节生成 QA 对；无 API Key 时可 `--mock` 用模板离线造数。

### DPO 数据集

每条样本为 `(prompt, chosen, rejected)`：

- `chosen`：CoT 完整 + 引用准确 + 忠实原文。
- `rejected` 构造策略：
  1. 幻觉型：替换关键实体/数字（同义但错误）；
  2. 无引用型：回答正确但无引用来源；
  3. 推理错序型：结论正确但中间推理步骤乱序/缺失。

## 5. BPE 领域词表扩充（`src/data/vocab.py`）

1. 在学术语料上训练 BPE（`tokenizers` 库，约 32k merges）。
2. 对每个候选 token 计算**压缩收益**：`freq × (基座分词片断数 - 1)`，选出收益最高的 N 个学术术语。
3. 跳过基座已存在的 token；新增 token 写入 `added_tokens` 并保存扩展 tokenizer。
4. SFT 阶段 `resize_token_embeddings` 并只解冻/训练新增 embedding 行（QLoRA 中随全参 LoRA 一并训练）。

```bash
python scripts/data/expand_vocab.py --corpus-dir work/data/arxiv_txt \
    --base-tokenizer Qwen/Qwen2.5-7B-Instruct \
    --out work/models/extended_tokenizer --top-n 2000
```

## 6. 产物格式

- `sft_train.jsonl`：`{"messages": [{"role": ..., "content": ...}]}`
- `dpo_train.jsonl`：`{"prompt": ..., "chosen": ..., "rejected": ...}`
- 扩展词表目录：`tokenizer.json` / `added_tokens.json` / `domain_tokens.txt`
