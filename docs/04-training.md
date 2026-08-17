# 训练设计

## 1. SFT：QLoRA + DeepSpeed ZeRO-2

入口：`python -m src.train.sft --config configs/train/sft_qlora.yaml`

- 基座：Qwen2.5-7B-Instruct（可在配置中切换）；
- 量化基座：4-bit NF4（bitsandbytes），LoRA rank=16/alpha=32，覆盖全部注意力与 FFN 投影层；
- 优化器状态由 DeepSpeed ZeRO-2 分片，可与 offload 组合以降低显存；
- 多轮数据：`apply_chat_template` + 仅 assistant 位置计算 loss（`assistant_masks`）；
- 领域词表：加载扩展 tokenizer 后 `resize_token_embeddings`，新 token 随 LoRA 训练；
- 训练后可选 `merge_and_unload` 导出全量权重。

## 2. DPO：偏好对齐

入口：`python -m src.train.dpo --config configs/train/dpo.yaml`

- 基于 SFT 产物（base + LoRA adapter）继续训练，`beta=0.1`；
- 数据列：`prompt / chosen / rejected`，TRL `DPOTrainer` 自动做 chat template 与 masking；
- 参考模型默认复用 base（`ref_model=None` 时 TRL 用 base 模型），显存紧张可 `peft_config` 共享 LoRA 微调参考模型；
- 训练指标：`rewards/chosen`、`rewards/rejected`、`logps`。

## 3. 硬件与显存预估（7B 模型）

| 阶段 | 显存建议 |
| --- | --- |
| QLoRA SFT（ZeRO-2 + offload） | ≥ 16GB 可训，24GB 舒适 |
| DPO（LoRA） | ≥ 24GB 舒适 |
| AWQ 4-bit 推理 | 约 6-8GB |

显存不足时：换 Qwen2.5-3B-Instruct、降低 batch、开启 offload、减小 max_length。

## 4. 训练产物

```text
work/models/
├── extended_tokenizer/       # 扩展词表
├── sft-qlora-qwen7b/         # SFT LoRA adapter
├── sft-merged-qwen7b/        # SFT 合并权重
├── dpo-qwen7b/               # DPO LoRA adapter
├── dpo-merged-qwen7b/        # DPO 合并权重
└── dpo-merged-qwen7b-awq/    # AWQ 4-bit 量化
```

## 5. 常见问题

- **Windows 无法训练**：请使用 Linux GPU 服务器或 WSL2。
- **bitsandbytes 安装失败**：确保 Python 3.10-3.12 + CUDA 匹配版本。
- **loss 不下降**：检查 assistant mask 是否正确（`src/train/tokenize.py`）。
- **显存溢出**：开启 `gradient_checkpointing`、降低 `per_device_train_batch_size`、增加 `gradient_accumulation_steps`、开启 offload。
