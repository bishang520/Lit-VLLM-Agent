"""AWQ 4-bit 量化导出（autoawq，需 GPU 服务器）。"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ 4-bit 量化")
    parser.add_argument("--model", required=True, help="合并后的 HF 模型目录")
    parser.add_argument("--out", required=True, help="量化模型输出目录")
    parser.add_argument("--calib-dir", default=None, help="可选：学术语料目录用于校准")
    parser.add_argument("--calib-samples", type=int, default=128)
    args = parser.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"加载模型 {args.model} ...")
    model = AutoAWQForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    calib_data = "wikitext2"
    if args.calib_dir:
        import glob

        files = sorted(glob.glob(str(args.calib_dir) + "/**/*.txt", recursive=True))
        if files:
            text = "\n".join(
                open(f, encoding="utf-8", errors="ignore").read()[:2000] for f in files[:30]
            )
            # 切成短样本，避免单条过长被 autoawq 过滤为空
            calib_data = [
                text[i : i + 512] for i in range(0, len(text), 512)
            ][: args.calib_samples]

    print("开始 AWQ 量化 ...")
    model.quantize(
        tokenizer, quant_config=quant_config, calib_data=calib_data,
        max_calib_samples=args.calib_samples,
    )
    model.save_quantized(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"量化完成 -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
