from __future__ import annotations

import os
from typing import Dict, List, Optional

from training_data.hf_adapters import map_any_to_text, normalize_preference_row, safe_load_hf_dataset
from models.export import merge_and_save
from models.loaders import load_qlora_model, load_tokenizer
from models.lora import LoraSpec, attach_lora, build_lora_config, load_adapter, prepare_kbit_training
from utils.text import normalize_text_basic


STAGE_DATASET_SOURCES = {
    "stage1_instruction": [
        ("OpenAssistant/oasst1", "train"),
        ("RyokoAI/ShareGPT52K", "train"),
    ],
    "stage2_roleplay": [
        ("AlekseyKorshuk/persona-chat", "train"),
        ("Anthropic/hh-rlhf", "train"),
    ],
    "stage3_preference": [
        ("Anthropic/hh-rlhf", "train"),
    ],
}


def run_multistage_pipeline(
    model_id: str,
    output_root: str,
    max_length: int = 1024,
    seed: int = 42,
    trust_remote_code: bool = True,
    resume_adapter_dir: str | None = None,
    logger=None,
) -> None:
    import random

    import torch  # type: ignore
    from datasets import Dataset, concatenate_datasets  # type: ignore
    from transformers import TrainingArguments  # type: ignore
    from trl import DPOConfig, DPOTrainer, SFTTrainer  # type: ignore

    os.makedirs(output_root, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    if logger:
        logger.info("Loading tokenizer+model: %s", model_id)

    tokenizer = load_tokenizer(model_id, use_fast=True)
    model = load_qlora_model(model_id, trust_remote_code=trust_remote_code)
    model = prepare_kbit_training(model)

    if resume_adapter_dir:
        if logger:
            logger.info("Resuming from adapter: %s", resume_adapter_dir)
        model = load_adapter(model, resume_adapter_dir)
    else:
        lora_config = build_lora_config(
            LoraSpec(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        )
        model = attach_lora(model, lora_config)

    loaded: Dict[str, List] = {}
    for stage, specs in STAGE_DATASET_SOURCES.items():
        stage_dsets = []
        for name, split in specs:
            try:
                ds = safe_load_hf_dataset(name, split=split)
                stage_dsets.append(ds)
                if logger:
                    logger.info("Loaded %s:%s -> %d rows", name, split, len(ds))
            except Exception as exc:
                if logger:
                    logger.warning("Failed %s:%s (%s)", name, split, exc)
        loaded[stage] = stage_dsets

    SYSTEM_PREFIX = "You are an immersive roleplay assistant. Stay in character, emotionally consistent, and coherent across turns."

    def to_chat_example(user_text: str, assistant_text: str) -> Optional[Dict[str, str]]:
        user_text = normalize_text_basic(user_text)
        assistant_text = normalize_text_basic(assistant_text)
        if len(user_text) < 2 or len(assistant_text) < 2:
            return None
        prompt = f"<|system|>\n{SYSTEM_PREFIX}\n<|user|>\n{user_text}\n<|assistant|>\n"
        return {"prompt": prompt, "response": assistant_text, "text": prompt + assistant_text}

    def normalize_sft_dataset(ds, stage_name: str) -> "Dataset":
        rows: List[Dict[str, str]] = []
        for ex in ds:
            # Prefer robust multi-schema conversion when possible.
            mapped = map_any_to_text(ex, source=stage_name, system_prefix=SYSTEM_PREFIX)
            if mapped and mapped.get("text"):
                rows.append({"text": mapped["text"]})
                continue

            if stage_name == "stage1_instruction":
                if "messages" in ex and isinstance(ex["messages"], list) and len(ex["messages"]) >= 2:
                    user = str(ex["messages"][-2].get("content", ""))
                    assistant = str(ex["messages"][-1].get("content", ""))
                else:
                    user = ex.get("instruction") or ex.get("prompt") or ex.get("input") or ""
                    assistant = ex.get("output") or ex.get("response") or ""
            else:
                user = ex.get("prompt") or ex.get("human") or ex.get("input") or ""
                assistant = ex.get("chosen") or ex.get("response") or ex.get("output") or ""

            parsed = to_chat_example(str(user), str(assistant))
            if parsed and parsed.get("text"):
                rows.append({"text": parsed["text"]})

        return Dataset.from_list(rows).shuffle(seed=seed)

    stage_sft_data: Dict[str, "Dataset"] = {}
    for stage_name in ["stage1_instruction", "stage2_roleplay"]:
        parts = [normalize_sft_dataset(ds, stage_name) for ds in loaded.get(stage_name, []) if ds is not None]
        if parts:
            stage_sft_data[stage_name] = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            if logger:
                logger.info("%s normalized rows: %d", stage_name, len(stage_sft_data[stage_name]))

    pref_rows = []
    for ds in loaded.get("stage3_preference", []):
        for ex in ds:
            row = normalize_preference_row(ex)
            if row:
                pref_rows.append(row)
    preference_dataset = Dataset.from_list(pref_rows).shuffle(seed=seed) if pref_rows else None
    if logger:
        logger.info("Preference rows: %d", len(preference_dataset) if preference_dataset is not None else 0)

    common_training_args = dict(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=1,
        evaluation_strategy="no",
        logging_steps=25,
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
    )

    def run_sft_stage(stage_name: str, train_ds, model_, tokenizer_):
        stage_dir = os.path.join(output_root, stage_name)
        args = TrainingArguments(output_dir=stage_dir, **common_training_args)
        trainer = SFTTrainer(
            model=model_,
            tokenizer=tokenizer_,
            train_dataset=train_ds,
            dataset_text_field="text",
            max_seq_length=max_length,
            args=args,
        )
        if logger:
            logger.info("Training %s...", stage_name)
        trainer.train()
        trainer.save_model(stage_dir)
        return trainer

    if "stage1_instruction" in stage_sft_data:
        run_sft_stage("stage1_instruction", stage_sft_data["stage1_instruction"], model, tokenizer)
    if "stage2_roleplay" in stage_sft_data:
        run_sft_stage("stage2_roleplay", stage_sft_data["stage2_roleplay"], model, tokenizer)

    if preference_dataset is not None and len(preference_dataset) > 0:
        dpo_dir = os.path.join(output_root, "stage3_dpo")
        dpo_args = DPOConfig(
            output_dir=dpo_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            num_train_epochs=1,
            logging_steps=10,
            fp16=True,
            report_to="none",
        )
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=preference_dataset,
            tokenizer=tokenizer,
        )
        if logger:
            logger.info("Running DPO...")
        dpo_trainer.train()
        dpo_trainer.save_model(dpo_dir)
    else:
        if logger:
            logger.info("Preference dataset unavailable; skipping DPO stage.")

    adapter_dir = os.path.join(output_root, "final_adapter")
    merged_dir = os.path.join(output_root, "final_merged")
    tokenizer_dir = os.path.join(output_root, "tokenizer")
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    merge_and_save(model, merged_dir, safe_serialization=True, max_shard_size="2GB")
