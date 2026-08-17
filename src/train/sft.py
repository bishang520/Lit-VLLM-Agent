"""QLoRA SFT 训练入口（DeepSpeed ZeRO-2 兼容，需 NVIDIA GPU / Linux）。"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.train.tokenize import ChatDataCollator, load_tokenizer, tokenize_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT 训练")
    parser.add_argument("--config", default="configs/train/sft_qlora.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    try:
        import torch
    except ImportError as e:
        raise SystemExit("缺少 torch，请按 requirements-train.txt 在 GPU 服务器安装") from e
    if not torch.cuda.is_available():
        raise SystemExit("SFT 训练需要 NVIDIA GPU（本机检测无 GPU，请在 Linux GPU 服务器运行）")

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    import inspect

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    qcfg = cfg["qlora"]
    tcfg = cfg["training"]

    tokenizer = load_tokenizer(mcfg["tokenizer"], mcfg.get("extended_tokenizer_dir"))
    train_file = dcfg["train_file"]
    if not Path(train_file).is_absolute():
        train_file = str(PROJECT_ROOT / train_file)
    eval_file = dcfg.get("eval_file")
    if eval_file and not Path(eval_file).is_absolute():
        eval_file = str(PROJECT_ROOT / eval_file)
    data_files = {"train": train_file}
    if eval_file:
        data_files["eval"] = eval_file
    dataset = load_dataset("json", data_files=data_files)
    tokenize_fn = partial(
        tokenize_chat, tokenizer=tokenizer, max_length=dcfg.get("max_length", 2048)
    )
    dataset = dataset.map(tokenize_fn, remove_columns=["messages", "doc_id"])

    compute_dtype = torch.bfloat16 if qcfg.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
    )
    import transformers

    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model_kwargs = {
        "quantization_config": bnb,
        "device_map": "auto",
        dtype_key: compute_dtype,
        "trust_remote_code": True,
    }
    if mcfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = mcfg["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(mcfg["base"], **model_kwargs)
    model = prepare_model_for_kbit_training(model)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        print(f"[info] 检测到扩展词表，resize embedding: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    lora = LoraConfig(
        r=qcfg.get("r", 16),
        lora_alpha=qcfg.get("alpha", 32),
        lora_dropout=qcfg.get("dropout", 0.05),
        target_modules=qcfg.get("target_modules"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds_path = tcfg.get("deepspeed")
    if ds_path and not Path(ds_path).is_absolute():
        ds_path = str(PROJECT_ROOT / ds_path)
    excluded = {"output_dir", "deepspeed"}
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    train_kwargs = {
        k: v
        for k, v in tcfg.items()
        if k not in excluded and k in valid_params
    }
    skipped = [k for k in tcfg if k not in excluded and k not in valid_params]
    if skipped:
        print(f"[warn] 配置项不被当前 transformers 版本支持，已忽略: {skipped}")
    training_args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        deepspeed=ds_path,
        **train_kwargs,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("eval"),
        "data_collator": ChatDataCollator(tokenizer),
    }
    trainer_params = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    merge_cfg = cfg.get("merge", {})
    if merge_cfg.get("save_merged"):
        merged_dir = merge_cfg.get("merged_dir", training_args.output_dir + "-merged")
        print(f"[info] 合并 LoRA 权重并导出到 {merged_dir}")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)


if __name__ == "__main__":
    sys.exit(main())
