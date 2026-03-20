from __future__ import annotations

import os


def save_adapter_and_tokenizer(model, tokenizer, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def merge_and_save(
    peft_model,
    out_dir: str,
    safe_serialization: bool = True,
    max_shard_size: str = "2GB",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)

