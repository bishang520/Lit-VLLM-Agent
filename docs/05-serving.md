# 服务化部署设计

## 1. vLLM 推理引擎

```bash
bash scripts/serving/run_vllm.sh
```

默认启用：

- **PagedAttention**：vLLM 默认，显存按页管理，避免 KV cache 碎片；
- **Continuous Batching**：vLLM 默认，请求级调度，吞吐显著高于静态 batch；
- **AWQ 4-bit**：`--quantization awq`，显存与延迟双降；
- **Guided Decoding**：`--guided-decoding-backend xgrammar`，按 JSON Schema 语法树约束生成。

## 2. JSON Schema 强约束

`configs/serving/answer_schema.json` 定义响应结构：

```json
{"answer": "...", "reasoning": "...", "citations": [{"chunk_id": "..."}], "confidence": 0.9, "is_grounded": true}
```

- 服务端通过 `SamplingParams(guided_json=schema)` 保证输出合法；
- 本地 mock 模式通过 `jsonschema.validate` 兜底校验。

## 3. FastAPI 服务（`src/serving/app.py`）

接口（OpenAI 兼容）：

- `POST /v1/chat/completions`：RAG 检索 → 拼接上下文 → 引导解码 → SSE 流式返回；
- `POST /v1/retrieve`：检索调试（返回 chunk 与分数）；
- `GET /health`：健康检查。

启动方式：

```bash
# GPU 服务器：真实 vLLM
python scripts/serving/start_api.py --config configs/serving/vllm.yaml

# 本机联调：mock 模式（不加载模型，返回模板答案 + 真实检索引用）
python scripts/serving/start_api.py --mock --config configs/serving/vllm.yaml --port 8001
```

## 4. AWQ 量化导出

```bash
python scripts/serving/quantize_awq.py --model work/models/dpo-merged-qwen7b --out work/models/dpo-merged-qwen7b-awq
```

## 5. 客户端（`src/serving/client.py`）

提供 OpenAI 兼容客户端，便于评测脚本与前端接入：

```python
from src.serving.client import AcademicAgentClient
client = AcademicAgentClient("http://localhost:8000")
resp = client.chat([{"role": "user", "content": "..."}])
```
