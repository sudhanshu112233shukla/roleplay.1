from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from characters.profile import CharacterProfile
from inference.backends.base import GenerationConfig, LLMBackend


@dataclass(frozen=True)
class PersonaInstruction:
    name: str
    remainder: str
    raw: str


_PERSONA_PATTERNS = [
    # Common roleplay instructions.
    r"\b(?:be|become)\s+(?:an?\s+|the\s+)?(?P<name>[^,.!?;\n]{2,80})",
    r"\bact\s+like\s+(?:an?\s+|the\s+)?(?P<name>[^,.!?;\n]{2,80})",
    r"\bpretend\s+to\s+be\s+(?:an?\s+|the\s+)?(?P<name>[^,.!?;\n]{2,80})",
    r"\broleplay\s+as\s+(?:an?\s+|the\s+)?(?P<name>[^,.!?;\n]{2,80})",
    r"\byou\s+are\s+(?:an?\s+|the\s+)?(?P<name>[^,.!?;\n]{2,80})",
]


def slugify_character_id(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "character"


def detect_persona_instruction(user_text: str) -> Optional[PersonaInstruction]:
    """
    Tries to detect a persona switch instruction in free-form user text.

    Returns:
      PersonaInstruction(name=<persona>, remainder=<user message minus instruction>)
    """
    t = user_text.strip()
    if not t:
        return None

    lowered = t.lower()
    for pat in _PERSONA_PATTERNS:
        m = re.search(pat, lowered, flags=re.IGNORECASE)
        if not m:
            continue

        raw_tail = t[m.start() :].strip()
        name_segment = t[m.start() : m.end()].strip()

        # Extract the name portion from the matched segment using a second regex on the original text
        m2 = re.search(pat, name_segment, flags=re.IGNORECASE)
        if not m2:
            continue
        name = (m2.group("name") or "").strip()

        # Stop the name at common delimiters like "for today", "and ...", etc.
        name, remainder_from_name = _split_name_and_remainder(name)

        # Remove extremely generic "persona" targets that are likely not roleplay characters.
        if _is_too_generic(name):
            continue

        remainder = remainder_from_name.strip()
        if not remainder:
            # Try to take any remainder from the raw tail ("... and talk to me")
            remainder = _extract_remainder_from_raw_tail(raw_tail).strip()

        if not remainder:
            remainder = "Begin."

        return PersonaInstruction(name=name, remainder=remainder, raw=t)

    return None


def _split_name_and_remainder(name: str) -> Tuple[str, str]:
    n = name.strip()
    if not n:
        return "", ""

    # Normalize common suffixes.
    suffix_markers = [
        " for today ",
        " for now ",
        " in this chat ",
        " in this conversation ",
        " in this roleplay ",
    ]
    lowered = f" {n.lower()} "
    for marker in suffix_markers:
        idx = lowered.find(marker)
        if idx != -1:
            # Keep the part before the marker.
            cut = idx
            return n[:cut].strip(), ""

    # Split at a top-level "and ..." if present (e.g. "Iron Man and talk to me")
    m = re.search(r"\s+\band\b\s+", n, flags=re.IGNORECASE)
    if m:
        persona = n[: m.start()].strip()
        remainder = n[m.end() :].strip()
        return persona, remainder

    return n, ""


def _extract_remainder_from_raw_tail(raw_tail: str) -> str:
    # If user wrote: "Be Iron Man today and talk to me", raw_tail includes the instruction.
    m = re.search(r"\s+\band\b\s+(?P<rest>.+)$", raw_tail, flags=re.IGNORECASE)
    return (m.group("rest") if m else "") or ""


def _is_too_generic(name: str) -> bool:
    # Avoid switching persona on common non-persona "be X" phrases.
    generic = {
        "honest",
        "nice",
        "kind",
        "polite",
        "quiet",
        "serious",
        "helpful",
        "my friend",
    }
    n = name.strip().lower()
    return n in generic or len(n) < 2


def build_dynamic_character_profile(
    backend: LLMBackend,
    persona_name: str,
    user_request: str,
    cfg: Optional[GenerationConfig] = None,
) -> CharacterProfile:
    """
    Uses the current backend to synthesize a CharacterProfile when no YAML exists.

    Falls back to a minimal heuristic profile if JSON parsing fails.
    """
    cfg = cfg or GenerationConfig(max_new_tokens=220, temperature=0.7, top_p=0.9)

    prompt = f"""
<|system|>
You are a strict JSON generator. Output ONLY a single JSON object and nothing else.

Create a roleplay character profile for the user request.

Schema (all fields required; strings):
- name
- role
- personality
- speech_style
- background
- emotions
- motivations
- behavior_rules

Keep it concise and usable as a system prompt. Do not include markdown or code fences.
<|user|>
User request: {user_request!r}
Persona name: {persona_name!r}
<|assistant|>
""".strip()

    raw = backend.generate(prompt, cfg=cfg)
    data = _try_parse_json_object(raw)
    if isinstance(data, dict):
        # Ensure name is consistent with what the user asked for.
        data["name"] = str(data.get("name") or persona_name).strip() or persona_name
        data["behavior_rules"] = str(data.get("behavior_rules") or f"Respond as {data['name']} would.").strip()
        try:
            return CharacterProfile.from_dict(data)  # type: ignore[arg-type]
        except Exception:
            pass

    # Fallback: minimal heuristic profile
    return CharacterProfile(
        name=persona_name.strip() or "Character",
        role=persona_name.strip() or "Roleplay character",
        personality="Stay consistent with the persona implied by the name; be immersive and coherent.",
        speech_style="Match the persona's typical voice; keep responses natural and in-character.",
        background="User-defined persona for this conversation.",
        emotions="Emotionally consistent; react appropriately to the scene and user tone.",
        motivations="Engage the user in roleplay; advance the conversation and maintain continuity.",
        behavior_rules=f"Respond like {persona_name.strip() or 'the persona'}; do not mention being an AI unless asked.",
    )


def _try_parse_json_object(text: str) -> Optional[dict]:
    t = (text or "").strip()
    if not t:
        return None

    # Strip accidental code fences.
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = t[start : end + 1].strip()
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None

