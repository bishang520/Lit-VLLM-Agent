#!/usr/bin/env bash
# vLLM 服务启动：AWQ 4-bit + Guided Decoding (JSON Schema)
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-work/models/dpo-merged-qwen7b-awq}"
PORT="${PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization awq \
  --guided-decoding-backend xgrammar \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port "$PORT"
