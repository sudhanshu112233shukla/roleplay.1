from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ChatTurn:
    user: str
    assistant: str


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


def render_prompt(system_prompt: str, history: List[ChatTurn], user_text: str) -> str:
    eos = "</s>"
    parts: List[str] = [f"<|system|>\n{system_prompt}{eos}\n"]
    for turn in history:
        parts.append(f"<|user|>\n{turn.user}{eos}\n")
        parts.append(f"<|assistant|>\n{turn.assistant}{eos}\n")
    parts.append(f"<|user|>\n{user_text}{eos}\n<|assistant|>\n")
    return "".join(parts)


def load_turns(path: str) -> List[str]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise SystemExit("turns file must be a list of strings")
        return [str(x) for x in data]

    turns: List[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, str):
            turns.append(obj)
            continue
        if isinstance(obj, dict):
            text = obj.get("user") or obj.get("text")
            if text:
                turns.append(str(text))
                continue
        raise SystemExit("Invalid JSONL turns file")
    return turns


def run_llama_completion(
    llama_bin: str,
    model_path: str,
    prompt: str,
    *,
    max_new_tokens: int,
    ctx: int,
    temperature: float,
    stop: str,
    args_threads: int,
    args_threads_batch: int,
) -> str:
    cmd = [
        llama_bin,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(max_new_tokens),
        "-c",
        str(ctx),
        "--temp",
        str(temperature),
        "-no-cnv",
        "--no-display-prompt",
        "-r",
        stop,
        "--no-warmup",
        "--color",
        "off",
        "-t",
        str(args_threads),
        "-tb",
        str(args_threads_batch),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    if not out:
        out = (proc.stderr or "").strip()
    if "load_backend:" in out:
        out = out.split("load_backend:", 1)[0].strip()
    if stop in out:
        out = out.split(stop, 1)[0].strip()
    return out.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-turn GGUF test with dynamic persona switching.")
    ap.add_argument("--model", required=True, help="Path to GGUF model file.")
    ap.add_argument("--llama-bin", default=None, help="Path to llama-completion.exe (default: in prebuilt folder).")
    ap.add_argument("--turns-file", required=True, help="JSON/JSONL file of user turns.")
    ap.add_argument("--persona", default="a helpful roleplay assistant")
    ap.add_argument("--use-remainder", action="store_true", help="Use the remainder after a persona switch as the user message (default: keep full user text).")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--threads-batch", type=int, default=4)
    args = ap.parse_args()

    llama_bin = args.llama_bin or r"e:\roleplay\roleplay001\deployment\llama.cpp-prebuilt\llama-completion.exe"
    turns = load_turns(args.turns_file)

    history: List[ChatTurn] = []
    persona = args.persona

    for user_text in turns:
        new_persona, remainder = detect_persona_switch(user_text)
        if new_persona:
            persona = new_persona
            if args.use_remainder:
                user_text = remainder

        sys_prompt = build_system_prompt(persona)
        prompt = render_prompt(sys_prompt, history, user_text)
        assistant = run_llama_completion(
            llama_bin,
            args.model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            ctx=args.ctx,
            temperature=args.temperature,
            stop="<|user|>",
            args_threads=args.threads,
            args_threads_batch=args.threads_batch,
        )
        print(f"\nUser: {user_text}\nAssistant: {assistant}\n")
        history.append(ChatTurn(user=user_text, assistant=assistant))


if __name__ == "__main__":
    main()

