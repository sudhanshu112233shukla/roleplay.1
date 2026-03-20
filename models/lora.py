from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LoraSpec:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def build_lora_config(spec: LoraSpec):
    from peft import LoraConfig  # type: ignore

    return LoraConfig(
        r=spec.r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        bias=spec.bias,
        task_type=spec.task_type,
        target_modules=spec.target_modules,
    )


def prepare_kbit_training(model):
    from peft import prepare_model_for_kbit_training  # type: ignore

    return prepare_model_for_kbit_training(model)


def attach_lora(model, lora_config):
    from peft import get_peft_model  # type: ignore

    return get_peft_model(model, lora_config)


def load_adapter(model, adapter_dir: str):
    from peft import PeftModel  # type: ignore

    return PeftModel.from_pretrained(model, adapter_dir)
