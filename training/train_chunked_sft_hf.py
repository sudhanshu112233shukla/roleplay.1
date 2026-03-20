from __future__ import annotations

import argparse

from training.chunked_sft_hf import ChunkedHfSftConfig, run_chunked_hf_sft
from training.streaming_hf_sft_dataset import HfDatasetSpec
from utils.logging import configure_logging


def _parse_hf_spec(s: str) -> HfDatasetSpec:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty HF dataset spec")
    split = "train"
    name_part = s
    if ":" in s:
        name_part, split_part = s.rsplit(":", 1)
        if split_part.strip():
            split = split_part.strip()

    config_name = None
    name = name_part.strip()
    if "@" in name:
        name, config_name = name.split("@", 1)
        name = name.strip()
        config_name = (config_name or "").strip() or None
    return HfDatasetSpec(name=name, config_name=config_name, split=split)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunked/resumable QLoRA+LoRA SFT from Hugging Face datasets (streaming).")
    ap.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument(
        "--hf-dataset",
        action="append",
        required=True,
        help="Dataset spec 'org/name[@config][:split]'. Repeat to mix multiple datasets.",
    )
    ap.add_argument("--output-root", default="./output_chunked_hf")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-dataset-take", type=int, default=None, help="Optional cap per dataset per cycle (debug).")

    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)

    ap.add_argument("--max-steps", type=int, default=800, help="How many optimizer steps to run this session.")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=20)

    ap.add_argument("--shuffle-buffer", type=int, default=512, help="Approximate shuffle buffer size (0 disables).")
    ap.add_argument("--dataloader-num-workers", type=int, default=0)

    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--resume-adapter-dir", default=None, help="Continue from an existing LoRA adapter directory.")
    ap.add_argument("--resume-from-checkpoint", default=None, help="Continue from a Trainer checkpoint directory.")
    ap.add_argument("--auto-resume", action="store_true", help="Auto-pick the latest checkpoint under output_root/checkpoints.")
    ap.add_argument("--merge-at-end", action="store_true", help="Also write a merged full model to output_root/final_merged.")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    specs = [_parse_hf_spec(s) for s in (args.hf_dataset or [])]

    logger = configure_logging(args.log_level, "train_chunked_sft_hf")
    cfg = ChunkedHfSftConfig(
        base_model=args.base_model,
        output_root=args.output_root,
        max_length=args.max_length,
        seed=args.seed,
        datasets=list(specs),
        per_dataset_take=args.per_dataset_take,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        shuffle_buffer=args.shuffle_buffer,
        dataloader_num_workers=args.dataloader_num_workers,
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
        resume_adapter_dir=args.resume_adapter_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume=args.auto_resume,
        merge_at_end=args.merge_at_end,
        trust_remote_code=args.trust_remote_code,
    )
    run_chunked_hf_sft(cfg, logger=logger)


if __name__ == "__main__":
    main()
