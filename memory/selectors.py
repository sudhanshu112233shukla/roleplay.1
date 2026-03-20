from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from utils.text import normalize_text_basic


IMPORTANT_PATTERNS = [
    re.compile(r"\bmy name is\b", re.I),
    re.compile(r"\bi am\b", re.I),
    re.compile(r"\bi live\b", re.I),
    re.compile(r"\bi like\b", re.I),
    re.compile(r"\bremember\b", re.I),
    re.compile(r"\bnever forget\b", re.I),
    re.compile(r"\bsecret\b", re.I),
    re.compile(r"\bquest\b", re.I),
    re.compile(r"\bartifact\b", re.I),
]


@dataclass
class MemoryCandidate:
    text: str
    score: float


def extract_memory_candidate(user_text: str, assistant_text: str = "") -> Optional[MemoryCandidate]:
    t = normalize_text_basic(user_text)
    if not t:
        return None
    score = 0.0
    for p in IMPORTANT_PATTERNS:
        if p.search(t):
            score += 0.2
    if len(t) > 80:
        score += 0.1
    if "?" not in t:
        score += 0.1
    if score >= 0.3:
        return MemoryCandidate(text=t, score=min(score, 1.0))
    return None

