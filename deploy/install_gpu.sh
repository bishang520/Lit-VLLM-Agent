#!/usr/bin/env bash
# AutoDL / Linux GPU 服务器一键环境安装
# 用法：bash deploy/install_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "未找到 $PYTHON，请使用 AutoDL 的 PyTorch 镜像"; exit 1; }

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "==> 系统 Python: $($PYTHON --version)"

if $PYTHON -c "import torch" 2>/dev/null; then
  echo "==> torch 已存在: $($PYTHON -c 'import torch; print(torch.__version__)')"
else
  echo "==> 安装 CUDA 版 torch"
  $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

$PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'" \
  || { echo "GPU/CUDA 不可用：请确认实例已开机且选择了带 CUDA 的 PyTorch 镜像"; exit 1; }

echo "==> 升级 pip"
$PYTHON -m pip install --upgrade pip

echo "==> 安装训练依赖"
$PYTHON -m pip install -r requirements/requirements-train.txt

echo "==> 安装部署依赖"
$PYTHON -m pip install -r requirements/requirements-serving.txt

echo "==> 安装 AWQ 量化工具"
$PYTHON -m pip install autoawq

echo "==> 最终验证"
$PYTHON -c "
import torch, transformers, trl, vllm, deepspeed, bitsandbytes
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('trl         ', trl.__version__)
print('vllm        ', vllm.__version__)
print('gpu         ', torch.cuda.get_device_name(0))
"

echo "==> 环境就绪。下一步：bash deploy/README.md 中的训练流水线"
