from __future__ import annotations

import os
from dataclasses import dataclass, field

from models.loaders import load_qlora_model, load_tokenizer
from models.lora import LoraSpec, attach_lora, build_lora_config, load_adapter, prepare_kbit_training
from training.chunked_sft import find_latest_checkpoint
from training.streaming_hf_sft_dataset import HfDatasetSpec, StreamingHfSftDataset, as_torch_iterable


@dataclass(frozen=True)
class ChunkedHfSftConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_root: str = "./output_hf"
    max_length: int = 1024
    seed: int = 42

    datasets: list[HfDatasetSpec] = field(default_factory=list)
    per_dataset_take: int | None = None

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01

    max_steps: int = 800
    save_steps: int = 100
    logging_steps: int = 20

    shuffle_buffer: int = 512
    dataloader_num_workers: int = 0

    lora_rank: int = 32
    lora_dropout: float = 0.05
    trust_remote_code: bool = True

    resume_adapter_dir: str | None = None
    resume_from_checkpoint: str | None = None
    auto_resume: bool = False

    merge_at_end: bool = False


def run_chunked_hf_sft(cfg: ChunkedHfSftConfig, *, logger=None) -> None:
    import random

    import torch  # type: ignore
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments  # type: ignore

    from models.export import merge_and_save

    if not cfg.datasets:
        raise ValueError("at least one HF dataset must be provided")

    os.makedirs(cfg.output_root, exist_ok=True)
    checkpoints_dir = os.path.join(cfg.output_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if logger:
        logger.info("Loading tokenizer+model: %s", cfg.base_model)
    tokenizer = load_tokenizer(cfg.base_model, use_fast=True)
    model = load_qlora_model(cfg.base_model, trust_remote_code=cfg.trust_remote_code)
    model.config.use_cache = False
    model = prepare_kbit_training(model)

    if cfg.resume_adapter_dir:
        if logger:
            logger.info("Resuming from adapter: %s", cfg.resume_adapter_dir)
        model = load_adapter(model, cfg.resume_adapter_dir)
    else:
        peft_cfg = build_lora_config(
            LoraSpec(
                r=cfg.lora_rank,
                lora_alpha=max(16, cfg.lora_rank),
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        )
        model = attach_lora(model, peft_cfg)

    ds = StreamingHfSftDataset(
        datasets=cfg.datasets,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        seed=cfg.seed,
        shuffle_buffer=cfg.shuffle_buffer,
        repeat=True,
        per_dataset_take=cfg.per_dataset_take,
    )
    train_dataset = as_torch_iterable(ds)

    bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=not bf16_supported,
        bf16=bf16_supported,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    resume_checkpoint = cfg.resume_from_checkpoint
    if cfg.auto_resume and not resume_checkpoint:
        resume_checkpoint = find_latest_checkpoint(checkpoints_dir)
        if resume_checkpoint and logger:
            logger.info("Auto-resume checkpoint: %s", resume_checkpoint)

    if logger:
        logger.info("Training (max_steps=%d)...", cfg.max_steps)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_adapter = os.path.join(cfg.output_root, "final_adapter")
    os.makedirs(final_adapter, exist_ok=True)
    if logger:
        logger.info("Saving adapter + tokenizer -> %s", final_adapter)
    trainer.model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)

    if cfg.merge_at_end:
        final_merged = os.path.join(cfg.output_root, "final_merged")
        if logger:
            logger.info("Merging -> %s", final_merged)
        merge_and_save(trainer.model, final_merged, safe_serialization=True, max_shard_size="2GB")

