#!/usr/bin/env bash
# QLoRA SFT 训练（Linux GPU 服务器）
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m src.train.sft --config configs/train/sft_qlora.yaml
