from __future__ import annotations

import argparse

from training.single_stage_sft import run_single_stage_sft
from utils.logging import configure_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-stage TinyLlama roleplay QLoRA+LoRA SFT training.")
    ap.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--output-dir", default="./tinyllama-roleplay-lora")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-samples-per-dataset", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-adapter-dir", default=None, help="Path to an existing LoRA adapter to continue training from.")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = configure_logging(args.log_level, "train_single_stage")
    run_single_stage_sft(
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        max_samples_per_dataset=args.max_samples_per_dataset,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        resume_adapter_dir=args.resume_adapter_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()
