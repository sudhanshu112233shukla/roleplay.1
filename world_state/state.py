from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.text import normalize_text_basic


@dataclass
class WorldState:
    location: str = "unknown"
    time: str = "unknown"
    characters_present: List[str] = field(default_factory=list)
    current_event: str = "none"
    story_progress: str = "beginning"
    facts: Dict[str, str] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        chars = ", ".join(self.characters_present) if self.characters_present else "(none)"
        facts = "\n".join([f"- {k}: {v}" for k, v in self.facts.items()]) if self.facts else "(none)"
        return (
            f"location: {self.location}\n"
            f"time: {self.time}\n"
            f"characters_present: {chars}\n"
            f"current_event: {self.current_event}\n"
            f"story_progress: {self.story_progress}\n"
            f"facts:\n{facts}"
        )


class WorldStateEngine:
    """
    Lightweight, safe updater for world state.

    Edge-compatible default: heuristic updates. This can be upgraded later to use
    a smaller classifier or a model call, but the interface stays the same.
    """

    def update(self, state: WorldState, user_text: str, assistant_text: Optional[str] = None) -> WorldState:
        user_text_n = normalize_text_basic(user_text).lower()
        assistant_text_n = normalize_text_basic(assistant_text or "").lower()

        if "we are in" in user_text_n:
            state.location = user_text.strip()
        if any(k in user_text_n for k in ["night", "dawn", "midday", "evening"]):
            state.time = user_text.strip()

        if any(k in user_text_n for k in ["quest", "mission", "objective"]):
            state.current_event = user_text.strip()
            state.story_progress = "in_progress"

        if any(k in assistant_text_n for k in ["the chapter ends", "to be continued"]):
            state.story_progress = "cliffhanger"

        return state

