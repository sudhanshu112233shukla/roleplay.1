from __future__ import annotations

import argparse

from training.chunked_sft import ChunkedSftConfig, run_chunked_sft
from utils.logging import configure_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunked/resumable QLoRA+LoRA SFT on a local dataset (JSONL/JSON/CSV).")
    ap.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset-path", required=True, help="Path to JSONL/JSON/CSV dataset (or a directory of such files).")
    ap.add_argument("--output-root", default="./output_chunked")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)

    ap.add_argument("--max-steps", type=int, default=800, help="How many optimizer steps to run this session.")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=20)

    ap.add_argument("--shuffle-buffer", type=int, default=512, help="Approximate shuffle buffer size (0 disables).")
    ap.add_argument("--dataloader-num-workers", type=int, default=0)
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA (CPU-friendly full-precision LoRA).")


    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--resume-adapter-dir", default=None, help="Continue from an existing LoRA adapter directory.")
    ap.add_argument("--resume-from-checkpoint", default=None, help="Continue from a Trainer checkpoint directory.")
    ap.add_argument("--auto-resume", action="store_true", help="Auto-pick the latest checkpoint under output_root/checkpoints.")
    ap.add_argument("--merge-at-end", action="store_true", help="Also write a merged full model to output_root/final_merged.")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = configure_logging(args.log_level, "train_chunked_sft")
    cfg = ChunkedSftConfig(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        max_length=args.max_length,
        seed=args.seed,
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
        use_4bit=not args.no_4bit,
    )
    run_chunked_sft(cfg, logger=logger)


if __name__ == "__main__":
    main()

