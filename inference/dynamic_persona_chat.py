from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch  # type: ignore


@dataclass
class ChatTurn:
    user: str
    assistant: str


@dataclass
class SessionState:
    persona: str = "a helpful roleplay assistant"
    history: List[ChatTurn] = field(default_factory=list)


def build_system_prompt(persona: str) -> str:
    persona = persona.strip()
    hints = []
    lowered = persona.lower()
    if "iron man" in lowered or "tony stark" in lowered:
        hints.append("Witty, confident, tech-savvy, slightly sarcastic. Refer to armor and tech casually.")
    if "wise mentor" in lowered or "mentor" in lowered:
        hints.append("Calm, encouraging, uses short parables or practical advice.")
    if "funny" in lowered or "friend" in lowered:
        hints.append("Light, playful tone. Short humor or gentle teasing.")

    hint_text = ("\nPersona hints: " + " ".join(hints)) if hints else ""

    return (
        f"You are {persona}.\n"
        "Rules:\n"
        "- Fully embody the persona and speak as them directly.\n"
        "- Never say you are 'trying' or 'pretending'; just be the character.\n"
        "- Do NOT mention being an AI, model, or assistant.\n"
        "- Do NOT break character or explain the role.\n"
        "- If the user asks to switch personas, immediately adopt the new persona.\n"
        "- Keep replies concise unless asked for more detail."
        + hint_text
    )


def split_persona_and_remainder(raw: str) -> Tuple[str, str]:
    cleaned = raw.strip().strip(".")
    lower = cleaned.lower()
    separators = [
        " and ",
        " then ",
        " today",
        " tonight",
        " for ",
        " while ",
        " now ",
        " this ",
        " please ",
        " but ",
    ]
    split_at: Optional[int] = None
    split_len = 0
    for sep in separators:
        idx = lower.find(sep)
        if idx != -1 and (split_at is None or idx < split_at):
            split_at = idx
            split_len = len(sep)
    if split_at is not None:
        persona = cleaned[:split_at].strip(" ,.;:-")
        remainder = cleaned[split_at + split_len :].strip(" ,.;:-")
    else:
        persona, remainder = cleaned, ""

    for lead in ("and ", "then ", "please ", "to "):
        if remainder.lower().startswith(lead):
            remainder = remainder[len(lead) :].strip()
    return persona, remainder


def detect_persona_switch(text: str) -> Tuple[Optional[str], str]:
    """
    Detect persona change requests and return (persona, remainder).
    Examples:
    - 'Be Iron Man today and talk to me the whole day.'
    - 'Act like a wise mentor.'
    """
    patterns = [
        r"^(?:be|become)\s+(?P<name>.+)$",
        r"^act like\s+(?P<name>.+)$",
        r"^act as\s+(?P<name>.+)$",
        r"^talk like\s+(?P<name>.+)$",
        r"^roleplay as\s+(?P<name>.+)$",
        r"^pretend to be\s+(?P<name>.+)$",
    ]
    stripped = text.strip()
    for pat in patterns:
        m = re.match(pat, stripped, flags=re.IGNORECASE)
        if not m:
            continue
        name_raw = (m.group("name") or "").strip()
        name, remainder = split_persona_and_remainder(name_raw)
        if not name:
            continue
        if not remainder or remainder.lower().startswith(("today", "for ", "and ")):
            remainder = "Hello."
        return name, remainder
    return None, text


def render_chat_prompt(tokenizer, system_prompt: str, history: List[ChatTurn], user_text: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})
    messages.append({"role": "user", "content": user_text})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback template
    chunks = [f"<|system|>\n{system_prompt}\n"]
    for turn in history:
        chunks.append(f"<|user|>\n{turn.user}\n")
        chunks.append(f"<|assistant|>\n{turn.assistant}\n")
    chunks.append(f"<|user|>\n{user_text}\n<|assistant|>\n")
    return "".join(chunks)


def load_tokenizer_with_fallback(base_model: str, adapter_dir: Optional[str]):
    from transformers import AutoTokenizer  # type: ignore

    if adapter_dir and os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json")):
        try:
            return AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        except Exception:
            pass
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    base_model: str,
    *,
    dtype: str = "float16",
    quant_4bit: bool = True,
    trust_remote_code: bool = True,
):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore

    use_cuda = bool(torch.cuda.is_available())
    if quant_4bit and not use_cuda:
        quant_4bit = False

    if quant_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
            model.eval()
            return model
        except Exception:
            pass

    if not use_cuda and dtype != "float32":
        dtype = "float32"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if use_cuda else None,
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def load_model_with_adapter(
    base_model: str,
    adapter_dir: str,
    *,
    dtype: str = "float16",
    quant_4bit: bool = True,
    trust_remote_code: bool = True,
):
    from peft import PeftModel  # type: ignore

    model = load_base_model(
        base_model,
        dtype=dtype,
        quant_4bit=quant_4bit,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def merge_adapter_into_base(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
    *,
    dtype: str = "float16",
    trust_remote_code: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from peft import PeftModel  # type: ignore

    os.makedirs(output_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")

    tok = load_tokenizer_with_fallback(base_model, adapter_dir)
    tok.save_pretrained(output_dir)


def generate_reply(model, tokenizer, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic persona chat with a base model + LoRA adapter.")
    ap.add_argument("--base-model", required=True, help="Base model id or local path.")
    ap.add_argument("--adapter", default=None, help="Path to LoRA adapter directory.")
    ap.add_argument("--merge-output", default=None, help="If set, merge adapter into base and save here, then exit.")
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading.")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--persona", default="a helpful roleplay assistant")
    args = ap.parse_args()

    if args.merge_output:
        if not args.adapter:
            raise SystemExit("--merge-output requires --adapter")
        merge_adapter_into_base(
            args.base_model,
            args.adapter,
            args.merge_output,
            dtype=args.dtype,
        )
        print(f"Merged model saved to: {args.merge_output}")
        return

    tokenizer = load_tokenizer_with_fallback(args.base_model, args.adapter)
    if args.adapter:
        model = load_model_with_adapter(
            args.base_model,
            args.adapter,
            dtype=args.dtype,
            quant_4bit=not args.no_4bit,
        )
    else:
        model = load_base_model(
            args.base_model,
            dtype=args.dtype,
            quant_4bit=not args.no_4bit,
        )

    state = SessionState(persona=args.persona)
    print("Dynamic persona chat. Type 'exit' to quit.")
    while True:
        user_text = input("\nYou> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        new_persona, remainder = detect_persona_switch(user_text)
        if new_persona:
            state.persona = new_persona
            user_text = remainder

        system_prompt = build_system_prompt(state.persona)
        prompt = render_chat_prompt(tokenizer, system_prompt, state.history, user_text)
        assistant = generate_reply(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nAssistant> {assistant}")
        state.history.append(ChatTurn(user=user_text, assistant=assistant))


if __name__ == "__main__":
    main()

