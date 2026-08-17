#!/usr/bin/env bash
# DPO 偏好对齐（Linux GPU 服务器，需先完成 SFT）
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m src.train.dpo --config configs/train/dpo.yaml
