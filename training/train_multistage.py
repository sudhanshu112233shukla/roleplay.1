from __future__ import annotations

import argparse
import os

from training.dataset_loader import load_dataset, validate_dataset
from training.merge_lora import merge_lora
from utils.logging import configure_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Roleplay training (LoRA SFT) with a local dataset path.")
    ap.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset-path", required=True, help="Path to JSONL/JSON/CSV dataset with system/user/assistant fields.")
    ap.add_argument("--output-root", default="./output")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-adapter-dir", default=None, help="Path to an existing LoRA adapter to continue training from.")
    ap.add_argument("--resume-from-checkpoint", default=None, help="Path to a Trainer checkpoint directory to resume from.")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = configure_logging(args.log_level, "train_multistage")

    # Load + validate dataset
    rows = validate_dataset(load_dataset(args.dataset_path))
    texts = [{"text": r.to_chat_text()} for r in rows]
    logger.info("Loaded %d validated samples from %s", len(texts), args.dataset_path)

    # Train LoRA adapter (QLoRA)
    import random

    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import DataCollatorForLanguageModeling, TrainingArguments  # type: ignore
    from trl import SFTTrainer  # type: ignore

    from models.loaders import load_qlora_model, load_tokenizer
    from models.lora import LoraSpec, build_lora_config, load_adapter, prepare_kbit_training

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.base_model, use_fast=True)
    model = load_qlora_model(args.base_model, trust_remote_code=args.trust_remote_code)
    model = prepare_kbit_training(model)

    peft_config = build_lora_config(
        LoraSpec(
            r=args.lora_rank,
            lora_alpha=max(16, args.lora_rank),
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    )
    if args.resume_adapter_dir:
        logger.info("Resuming from adapter: %s", args.resume_adapter_dir)
        model = load_adapter(model, args.resume_adapter_dir)
        peft_config = None

    ds = Dataset.from_list(texts).shuffle(seed=args.seed)

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing")

    bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_root, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=20,
        evaluation_strategy="no",
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=not bf16_supported,
        bf16=bf16_supported,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_adapter = os.path.join(args.output_root, "final_adapter")
    os.makedirs(final_adapter, exist_ok=True)
    logger.info("Saving adapter + tokenizer -> %s", final_adapter)
    trainer.model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)

    final_merged = os.path.join(args.output_root, "final_merged")
    logger.info("Merging -> %s", final_merged)
    merge_lora(
        base_model=args.base_model,
        adapter_dir=final_adapter,
        output_dir=final_merged,
        trust_remote_code=args.trust_remote_code,
        dtype="bfloat16" if bf16_supported else "float16",
        logger=logger,
    )


if __name__ == "__main__":
    main()
