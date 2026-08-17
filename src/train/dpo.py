"""DPO 偏好对齐训练入口（TRL，需 NVIDIA GPU / Linux）。"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.train.tokenize import format_dpo_example, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO 偏好对齐训练")
    parser.add_argument("--config", default="configs/train/dpo.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    try:
        import torch
    except ImportError as e:
        raise SystemExit("缺少 torch，请按 requirements-train.txt 在 GPU 服务器安装") from e
    if not torch.cuda.is_available():
        raise SystemExit("DPO 训练需要 NVIDIA GPU（请在 Linux GPU 服务器运行）")

    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from trl import DPOConfig, DPOTrainer

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    qcfg = cfg["qlora"]
    tcfg = cfg["training"]
    dpo_cfg = cfg["dpo"]

    tokenizer = load_tokenizer(mcfg["tokenizer"], mcfg.get("extended_tokenizer_dir"))
    train_file = dcfg["train_file"]
    if not Path(train_file).is_absolute():
        train_file = str(PROJECT_ROOT / train_file)
    dataset = load_dataset("json", data_files={"train": train_file})
    dataset = dataset.map(
        partial(format_dpo_example, tokenizer=tokenizer),
        remove_columns=["doc_id", "prompt", "chosen", "rejected"],
    )

    compute_dtype = torch.bfloat16 if qcfg.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
    )
    import transformers

    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base"],
        quantization_config=bnb,
        device_map="auto",
        **{dtype_key: compute_dtype},
        trust_remote_code=True,
    )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        print(f"[info] 检测到扩展词表，resize embedding: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    sft_adapter = mcfg.get("sft_adapter")
    if sft_adapter:
        p = Path(sft_adapter)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        print(f"[info] 加载 SFT LoRA adapter: {p}")
        model = PeftModel.from_pretrained(model, str(p))
        print("[info] 合并 SFT adapter 为基座权重，供 DPO 重新挂载 LoRA")
        model = model.merge_and_unload()

    peft_config = LoraConfig(
        r=qcfg.get("r", 8),
        lora_alpha=qcfg.get("alpha", 16),
        lora_dropout=qcfg.get("dropout", 0.05),
        target_modules=qcfg.get("target_modules"),
        task_type="CAUSAL_LM",
    )

    import inspect

    excluded = {"output_dir"}
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters) | set(
        inspect.signature(DPOConfig.__init__).parameters
    )
    dpo_kwargs = {
        k: v
        for k, v in tcfg.items()
        if k not in excluded and k in valid_params
    }
    skipped = [k for k in tcfg if k not in excluded and k not in valid_params]
    if skipped:
        print(f"[warn] 配置项不被当前 transformers 版本支持，已忽略: {skipped}")
    dpo_args = DPOConfig(
        output_dir=tcfg["output_dir"],
        beta=dpo_cfg.get("beta", 0.1),
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        max_length=dcfg.get("max_length", 2048),
        **dpo_kwargs,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(dpo_args.output_dir)

    merge_cfg = cfg.get("merge", {})
    if merge_cfg.get("save_merged"):
        merged_dir = merge_cfg.get("merged_dir", dpo_args.output_dir + "-merged")
        print(f"[info] 合并 LoRA 权重并导出到 {merged_dir}")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)


if __name__ == "__main__":
    sys.exit(main())
