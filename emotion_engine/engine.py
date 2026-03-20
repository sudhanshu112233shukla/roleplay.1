from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from utils.text import normalize_text_basic


EMOTIONS = ["happy", "angry", "curious", "suspicious", "fearful", "neutral"]


@dataclass
class EmotionState:
    by_character: Dict[str, str]

    def get(self, character_id: str) -> str:
        return self.by_character.get(character_id, "neutral")

    def set(self, character_id: str, emotion: str) -> None:
        self.by_character[character_id] = emotion if emotion in EMOTIONS else "neutral"

    def to_prompt_block(self, character_id: str) -> str:
        return f"{character_id}: {self.get(character_id)}"


class EmotionEngine:
    """
    Edge-friendly emotion updater based on simple lexical cues.

    This is intentionally cheap; it can be replaced later with a small classifier.
    """

    def update(self, state: EmotionState, character_id: str, user_text: str, assistant_text: str = "") -> EmotionState:
        t = (normalize_text_basic(user_text) + " " + normalize_text_basic(assistant_text)).lower()

        if any(w in t for w in ["thank", "great", "wonderful", "love", "glad"]):
            state.set(character_id, "happy")
        elif any(w in t for w in ["angry", "furious", "hate", "idiot", "stupid"]):
            state.set(character_id, "angry")
        elif any(w in t for w in ["why", "how", "what if", "curious", "wonder"]):
            state.set(character_id, "curious")
        elif any(w in t for w in ["liar", "sus", "suspicious", "betray"]):
            state.set(character_id, "suspicious")
        elif any(w in t for w in ["help", "scared", "afraid", "terror", "panic"]):
            state.set(character_id, "fearful")
        else:
            state.set(character_id, state.get(character_id))

        return state

