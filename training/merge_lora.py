from __future__ import annotations

import argparse
import os

from utils.logging import configure_logging


def merge_lora(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
    *,
    trust_remote_code: bool = True,
    dtype: str = "float16",
    logger=None,
) -> None:
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    os.makedirs(output_dir, exist_ok=True)
    use_cuda = bool(torch.cuda.is_available())
    if not use_cuda and dtype != "float32":
        dtype = "float32"
    torch_dtype = getattr(torch, dtype)

    if logger:
        logger.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=not use_cuda,
    )

    if logger:
        logger.info("Loading adapter: %s", adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)

    if logger:
        logger.info("Merging adapter into base...")
    merged = model.merge_and_unload()

    if logger:
        logger.info("Saving merged model -> %s", output_dir)
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")

    # Tokenizer: prefer adapter dir (often contains tokenizer) else base model.
    tok_src = adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json")) else base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=trust_remote_code, use_fast=True)
    except Exception as exc:
        if logger:
            logger.warning("Adapter tokenizer load failed (%s). Falling back to base tokenizer.", exc)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model and save a merged HF directory.")
    ap.add_argument("--base-model", required=True, help="Base model id or local path (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0).")
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter directory (e.g. output/final_adapter).")
    ap.add_argument("--output", required=True, help="Output directory for merged model (e.g. output/final_merged).")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = configure_logging(args.log_level, "merge_lora")
    merge_lora(
        base_model=args.base_model,
        adapter_dir=args.adapter,
        output_dir=args.output,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        logger=logger,
    )


if __name__ == "__main__":
    main()

