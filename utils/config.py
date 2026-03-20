from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(key)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def env_bool(key: str, default: bool = False) -> bool:
    val = env(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_path: Optional[str] = None
    output_root: str = "./output"
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_length: int = 1024
    lora_rank: int = 32
    gradient_checkpointing: bool = True
    bf16: bool = False
    fp16: bool = True
    resume_adapter_dir: Optional[str] = None

