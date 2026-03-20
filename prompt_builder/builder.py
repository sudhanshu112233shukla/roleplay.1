from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from characters.profile import CharacterProfile
from emotion_engine.engine import EmotionState
from utils.errors import PromptBuilderError
from world_state.state import WorldState


DEFAULT_SYSTEM_PREFIX = (
    "You are an immersive roleplay assistant. Stay in character, emotionally consistent, descriptive, and coherent across turns."
)


@dataclass
class PromptParts:
    system: str
    world_state: str
    memory: str
    chat_history: str
    user_input: str

    def render(self) -> str:
        return (
            f"<|system|>\n{self.system}\n"
            f"\nWORLD STATE:\n{self.world_state}\n"
            f"\nMEMORY:\n{self.memory}\n"
            f"\nCHAT HISTORY:\n{self.chat_history}\n"
            f"\n<|user|>\n{self.user_input}\n<|assistant|>\n"
        )


class PromptBuilder:
    def __init__(self, system_prefix: str = DEFAULT_SYSTEM_PREFIX):
        self.system_prefix = system_prefix

    def build(
        self,
        character: CharacterProfile,
        world_state: WorldState,
        emotion_state: EmotionState,
        character_id: str,
        retrieved_memories: Sequence[str],
        chat_history: Sequence[Tuple[str, str]],
        user_input: str,
        max_chars: int = 32_000,
        tokenizer=None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            mem_block = "\n".join([f"- {m}" for m in retrieved_memories]) if retrieved_memories else "(none)"
            chat_block = "\n".join([f"<|user|>\n{u}\n<|assistant|>\n{a}\n" for u, a in chat_history]) or "(none)"

            system = (
                f"{self.system_prefix}\n\n"
                f"CHARACTER:\n{character.to_system_prompt()}\n\n"
                f"EMOTION STATE:\n{emotion_state.to_prompt_block(character_id)}"
            )

            parts = PromptParts(
                system=system,
                world_state=world_state.to_prompt_block(),
                memory=mem_block,
                chat_history=chat_block,
                user_input=user_input,
            )
            prompt = parts.render()

            if max_tokens is not None and tokenizer is not None:
                ids = tokenizer(prompt, return_tensors=None, add_special_tokens=False).get("input_ids", [])
                if len(ids) > max_tokens:
                    ids = ids[-max_tokens:]
                    prompt = tokenizer.decode(ids, skip_special_tokens=False)

            if len(prompt) > max_chars:
                prompt = prompt[-max_chars:]

            return prompt
        except Exception as exc:
            raise PromptBuilderError("Failed to construct prompt") from exc

