# AutoDL 部署指南

## 1. 租用建议

- 机型：**RTX 4090（24G 显存）按量计费**起步；预算充足可换 A100/A800（40G/80G）。
- 镜像：**PyTorch 2.x + CUDA 12.x**，Python 3.10/3.12。
- 开机前开通 AutoDL 的「学术加速」（HuggingFace / GitHub 下载加速）。
- 只做数据准备、不用 GPU 时用「无卡模式」省钱；训练/部署时再开机。
- 关机后 GPU 停止计费，但数据盘继续计费，注意及时清理。

## 2. 两种协作方式

### 方式 A：我（Codex）远程操作

把你租好后 AutoDL 控制台显示的**登录指令**发给我即可，形如：

```text
ssh -p <端口> root@<区域>.autodl.com
密码：<实例密码>
```

本机已具备 SSH 客户端；每次需要联网操作时你在弹窗里点同意即可。
密码属于敏感信息，用完可以改掉，或直接换新实例。

### 方式 B：你自己在服务器终端操作

1. AutoDL 控制台 → 文件 → 上传：把项目根目录打包上传（排除 `work/`、`.venv`、`__pycache__`、`.git`）。
2. 在终端执行 `bash deploy/install_gpu.sh` 一键装环境。
3. 遇到任何报错，把输出贴给我，我远程帮你排查。

## 3. 训练流水线（装完环境后）

```bash
# 0) 准备语料：把 ArXiv 论文解析文本放到 work/data/arxiv_txt（本机跑完直接上传，或在服务器重新采集）
python scripts/data/fetch_papers.py --ids 1706.03762,2005.11401 --out work/data/arxiv_pdfs --extract

# 1) BPE 领域词表扩充
python scripts/data/expand_vocab.py \
  --corpus-dir work/data/arxiv_txt \
  --base-tokenizer Qwen/Qwen2.5-7B-Instruct \
  --out work/models/extended_tokenizer

# 2) QLoRA SFT
bash scripts/train/run_sft.sh
# 单卡 4090 建议：把 configs/train/sft_qlora.yaml 中 deepspeed 置 null 直接跑 QLoRA；
# 显存不足时 per_device_train_batch_size 降到 1 + 打开 offload。

# 3) DPO 偏好对齐
bash scripts/train/run_dpo.sh

# 4) 合并权重 + AWQ 4-bit 量化
python scripts/serving/quantize_awq.py \
  --model work/models/dpo-merged-qwen7b \
  --out work/models/dpo-merged-qwen7b-awq

# 5) vLLM 部署（PagedAttention / Continuous Batching / Guided Decoding 默认开启）
bash scripts/serving/run_vllm.sh

# 6) 业务 API（RAG 检索 + 引导解码 + 流式）
python scripts/serving/start_api.py --config configs/serving/vllm.yaml
```

## 4. 常见问题

- **torch 版本不对 / CUDA 不可用**：`pip uninstall -y torch` 后用 CUDA 索引重装：
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **显存不足**：调小 `per_device_train_batch_size`（1）、打开 `gradient_checkpointing`
  （默认开）、DeepSpeed offload，或换 Qwen2.5-3B-Instruct。
- **HuggingFace 下载慢**：确认「学术加速」已开，或 `export HF_ENDPOINT=https://hf-mirror.com`。
- **bitsandbytes 报错**：镜像 CUDA 版本与 torch 不匹配，重装匹配版本即可。
