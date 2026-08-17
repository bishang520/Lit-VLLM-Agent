"""合并 LoRA adapter 到基座并导出完整权重（GPU 服务器）。

默认以 4-bit 量化基座合并（省显存），产物可继续做 AWQ 量化或直接推理。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA adapter")
    parser.add_argument("--base", required=True, help="基座模型名或路径")
    parser.add_argument("--adapter", required=True, help="LoRA adapter 目录")
    parser.add_argument("--tokenizer-dir", default=None, help="扩展 tokenizer 目录（可选）")
    parser.add_argument("--out", required=True, help="合并模型输出目录")
    parser.add_argument("--dtype", default="4bit", choices=["4bit", "bf16"], help="合并后精度")
    args = parser.parse_args()

    import torch

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if args.tokenizer_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(device_map="auto", trust_remote_code=True)
    if args.dtype == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    print(f"加载基座 {args.base}（{args.dtype}）...")
    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        print(f"[info] resize embedding: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print(f"加载 adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("合并 LoRA ...")
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"合并完成 -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
