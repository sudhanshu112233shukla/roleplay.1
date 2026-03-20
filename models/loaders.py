from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


def load_tokenizer(model_id: str, use_fast: bool = True):
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qlora_model(
    model_id: str,
    quant: QuantConfig = QuantConfig(),
    trust_remote_code: bool = True,
):
    """
    Backwards-compatible training loader: strict 4-bit bitsandbytes load.

    Training code expects this to fail fast if bitsandbytes isn't available, rather than silently
    switching to full-precision (which can OOM).
    """
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore

    dtype = getattr(torch, quant.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant.load_in_4bit,
        bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    return model


def load_causal_lm_model(
    model_id: str,
    *,
    quant_4bit: bool | None = None,
    quant: QuantConfig = QuantConfig(),
    dtype: str = "float16",
    trust_remote_code: bool = True,
):
    """
    Loads a CausalLM model for inference.

    - If `quant_4bit` is True, attempts to load via bitsandbytes 4-bit quantization.
      If bitsandbytes isn't available, falls back to non-quantized loading.
    - If `quant_4bit` is None, uses 4-bit when CUDA is available (best-effort).
    """
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore

    use_cuda = bool(torch.cuda.is_available())
    if quant_4bit is None:
        quant_4bit = use_cuda
    if quant_4bit and not use_cuda:
        quant_4bit = False

    if quant_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            compute_dtype = getattr(torch, quant.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant.load_in_4bit,
                bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
            model.eval()
            return model
        except Exception:
            # Best-effort: bitsandbytes isn't always available (e.g. CPU-only setups).
            pass

    if not use_cuda and dtype != "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def load_model_with_adapter(
    base_model_id: str,
    adapter_dir: str,
    *,
    quant_4bit: bool | None = None,
    quant: QuantConfig = QuantConfig(),
    dtype: str = "float16",
    trust_remote_code: bool = True,
):
    """
    Loads a base model and attaches a PEFT adapter for inference (no merge).

    `base_model_id` can be an HF id or local dir.
    `adapter_dir` must be a local dir produced by training (e.g. final_adapter).
    """
    from peft import PeftModel  # type: ignore

    model = load_causal_lm_model(
        base_model_id,
        quant_4bit=quant_4bit,
        quant=quant,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model
