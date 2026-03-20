from __future__ import annotations

from training_data.hf_adapters import (
    map_characterai_to_text,
    map_oasst_to_text,
    map_sharegpt_conversations_to_text,
    safe_load_hf_dataset,
)
from models.loaders import load_qlora_model, load_tokenizer
from models.lora import LoraSpec, build_lora_config, load_adapter, prepare_kbit_training


def run_single_stage_sft(
    base_model: str,
    output_dir: str,
    max_seq_len: int = 1024,
    max_samples_per_dataset: int = 30000,
    seed: int = 42,
    trust_remote_code: bool = True,
    resume_adapter_dir: str | None = None,
    logger=None,
) -> None:
    """
    Production-structured equivalent of `roleplay_training.ipynb`.
    """
    import random

    import torch  # type: ignore
    from datasets import concatenate_datasets  # type: ignore
    from transformers import DataCollatorForLanguageModeling, TrainingArguments  # type: ignore
    from trl import SFTTrainer  # type: ignore

    random.seed(seed)
    torch.manual_seed(seed)

    if logger:
        logger.info("Loading tokenizer+model: %s", base_model)

    tokenizer = load_tokenizer(base_model, use_fast=True)
    model = load_qlora_model(base_model, trust_remote_code=trust_remote_code)
    model = prepare_kbit_training(model)

    # Same datasets as the original notebook.
    oasst = safe_load_hf_dataset("OpenAssistant/oasst1", split="train")
    sharegpt = safe_load_hf_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
    characterai = safe_load_hf_dataset("ehartford/characterai", split="train")

    def _map(ds, fn):
        return ds.map(lambda ex: fn(ex) or {"text": ""}, remove_columns=ds.column_names)

    oasst_text = _map(oasst, map_oasst_to_text).filter(lambda ex: bool(ex.get("text")))
    sharegpt_text = _map(sharegpt, lambda ex: map_sharegpt_conversations_to_text(ex, "anon8231489123/ShareGPT_Vicuna_unfiltered")).filter(
        lambda ex: bool(ex.get("text"))
    )
    characterai_text = _map(characterai, map_characterai_to_text).filter(lambda ex: bool(ex.get("text")))

    def cap_shuffle(ds):
        if len(ds) > max_samples_per_dataset:
            ds = ds.shuffle(seed=seed).select(range(max_samples_per_dataset))
        return ds

    train_dataset = concatenate_datasets([cap_shuffle(oasst_text), cap_shuffle(sharegpt_text), cap_shuffle(characterai_text)]).shuffle(seed=seed)

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len, padding=False)

    tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing")
    tokenized = tokenized.train_test_split(test_size=0.02, seed=seed)

    peft_config = build_lora_config(
        LoraSpec(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    )
    if resume_adapter_dir:
        if logger:
            logger.info("Resuming from adapter: %s", resume_adapter_dir)
        model = load_adapter(model, resume_adapter_dir)
        peft_config = None

    bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=not bf16_supported,
        bf16=bf16_supported,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)

    if logger:
        logger.info("Training...")
    trainer.train()

    if logger:
        logger.info("Evaluating...")
    trainer.evaluate()

    if logger:
        logger.info("Saving adapter + tokenizer -> %s", output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
