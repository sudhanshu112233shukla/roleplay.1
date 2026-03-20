from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from characters.dynamic import build_dynamic_character_profile, detect_persona_instruction, slugify_character_id
from characters.profile import CharacterProfile, load_character_profile_by_id
from emotion_engine.engine import EmotionEngine, EmotionState
from inference.backends.base import GenerationConfig, LLMBackend
from memory.faiss_store import FaissMemoryStore
from memory.selectors import extract_memory_candidate
from prompt_builder.builder import PromptBuilder
from utils.errors import MemoryStoreError, PromptBuilderError
from world_state.state import WorldState, WorldStateEngine


@dataclass
class ChatSession:
    user_id: str
    character_id: str
    backend: LLMBackend
    character: CharacterProfile
    profiles_dir: str = "characters/profiles"
    allow_dynamic_personas: bool = True
    auto_save_dynamic_profiles_dir: Optional[str] = None
    auto_save_after_turns: int = 50
    dynamic_profiles: Dict[str, CharacterProfile] = field(default_factory=dict)
    dynamic_turn_counts: Dict[str, int] = field(default_factory=dict)
    memory: Optional[FaissMemoryStore] = None
    world_state: WorldState = field(default_factory=WorldState)
    emotion_state: EmotionState = field(default_factory=lambda: EmotionState(by_character={}))
    history: List[Tuple[str, str]] = field(default_factory=list)
    prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)
    emotion_engine: EmotionEngine = field(default_factory=EmotionEngine)
    world_engine: WorldStateEngine = field(default_factory=WorldStateEngine)

    def step(self, user_text: str, gen_cfg: Optional[GenerationConfig] = None) -> str:
        user_text = self._maybe_switch_persona(user_text)

        prompt = self._build_prompt(user_text)
        assistant_text = self.backend.generate(prompt, cfg=gen_cfg)
        self._finalize_turn(user_text, assistant_text)
        return assistant_text

    def step_stream(
        self,
        user_text: str,
        gen_cfg: Optional[GenerationConfig] = None,
        *,
        stop_sequences: Sequence[str] = (),
    ) -> Iterator[str]:
        """
        Streams the assistant response while preserving the same memory/world/emotion updates as `step()`.

        Yields text chunks (not tokens).
        """
        user_text = self._maybe_switch_persona(user_text)
        prompt = self._build_prompt(user_text)

        stop_sequences = tuple([s for s in stop_sequences if s])
        max_stop_len = max((len(s) for s in stop_sequences), default=0)

        emitted = 0
        buf = ""
        stop_at: Optional[int] = None

        for chunk in self.backend.stream_generate(prompt, cfg=gen_cfg):
            if not chunk:
                continue
            buf += chunk

            if stop_sequences and stop_at is None:
                stop_at = _find_first_stop(buf, stop_sequences)

            if stop_at is not None:
                if stop_at > emitted:
                    yield buf[emitted:stop_at]
                    emitted = stop_at
                break

            safe_upto = len(buf) - max_stop_len
            if safe_upto > emitted:
                yield buf[emitted:safe_upto]
                emitted = safe_upto

        # Flush any remaining text (no stop found)
        if stop_at is None and emitted < len(buf):
            yield buf[emitted:]

        final_text = buf if stop_at is None else buf[:stop_at]
        self._finalize_turn(user_text, final_text)

    def _maybe_switch_persona(self, user_text: str) -> str:
        if not self.allow_dynamic_personas:
            return user_text

        instr = detect_persona_instruction(user_text)
        if instr is None:
            return user_text

        base_id = slugify_character_id(instr.name)

        # Prefer curated/static YAML if it exists.
        try:
            profile = load_character_profile_by_id(base_id, profiles_dir=self.profiles_dir)
            new_id = base_id
        except Exception:
            new_id = f"dynamic/{base_id}"
            profile = self.dynamic_profiles.get(new_id)
            if profile is None:
                profile = build_dynamic_character_profile(self.backend, persona_name=instr.name, user_request=instr.raw)
                self.dynamic_profiles[new_id] = profile

        if new_id != self.character_id:
            self.character_id = new_id
            self.character = profile
            self.world_state.characters_present = [profile.name]
            if new_id not in self.emotion_state.by_character:
                self.emotion_state.by_character[new_id] = "neutral"

        if new_id.startswith("dynamic/"):
            self.dynamic_turn_counts[new_id] = self.dynamic_turn_counts.get(new_id, 0) + 1
            self._maybe_autosave_dynamic(new_id, profile)

        return instr.remainder

    def _maybe_autosave_dynamic(self, dynamic_id: str, profile: CharacterProfile) -> None:
        out_root = self.auto_save_dynamic_profiles_dir
        if not out_root:
            return
        if self.dynamic_turn_counts.get(dynamic_id, 0) < self.auto_save_after_turns:
            return

        try:
            import os

            import yaml  # type: ignore

            slug = dynamic_id.split("/", 1)[-1]
            out_dir = os.path.join(out_root, "user")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{slug}.yaml")
            if os.path.exists(out_path):
                return

            data = {
                "name": profile.name,
                "role": profile.role,
                "personality": profile.personality,
                "speech_style": profile.speech_style,
                "background": profile.background,
                "emotions": profile.emotions,
                "motivations": profile.motivations,
                "behavior_rules": profile.behavior_rules,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        except Exception:
            # Best-effort only; never break the chat flow for a save.
            return

    def _build_prompt(self, user_text: str) -> str:
        retrieved: List[str] = []
        if self.memory is not None:
            try:
                records = self.memory.retrieve(user_text, user_id=self.user_id, k=4)
                retrieved = [r.text for r in records]
            except MemoryStoreError:
                retrieved = []

        try:
            return self.prompt_builder.build(
                character=self.character,
                world_state=self.world_state,
                emotion_state=self.emotion_state,
                character_id=self.character_id,
                retrieved_memories=retrieved,
                chat_history=self.history[-12:],
                user_input=user_text,
            )
        except PromptBuilderError:
            return f"<|system|>\nYou are a helpful roleplay assistant.\n<|user|>\n{user_text}\n<|assistant|>\n"

    def _finalize_turn(self, user_text: str, assistant_text: str) -> None:
        assistant_text = assistant_text or ""
        self.history.append((user_text, assistant_text))
        self.world_engine.update(self.world_state, user_text, assistant_text=assistant_text)
        self.emotion_engine.update(self.emotion_state, self.character_id, user_text, assistant_text=assistant_text)

        if self.memory is not None:
            cand = extract_memory_candidate(user_text, assistant_text)
            if cand is not None:
                try:
                    self.memory.add(self.user_id, cand.text, meta={"character": self.character_id, "score": cand.score})
                except MemoryStoreError:
                    pass


def _find_first_stop(text: str, stops: Sequence[str]) -> Optional[int]:
    best: Optional[int] = None
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx == -1:
            continue
        if best is None or idx < best:
            best = idx
    return best


def create_session(
    user_id: str,
    character_id: str,
    backend: LLMBackend,
    profiles_dir: str = "characters/profiles",
    memory: Optional[FaissMemoryStore] = None,
    allow_dynamic_personas: bool = True,
    auto_save_dynamic_profiles_dir: Optional[str] = None,
    auto_save_after_turns: int = 50,
) -> ChatSession:
    character = load_character_profile_by_id(character_id, profiles_dir=profiles_dir)
    ws = WorldState(characters_present=[character.name])
    es = EmotionState(by_character={character_id: "neutral"})
    return ChatSession(
        user_id=user_id,
        character_id=character_id,
        backend=backend,
        character=character,
        memory=memory,
        world_state=ws,
        emotion_state=es,
        profiles_dir=profiles_dir,
        allow_dynamic_personas=allow_dynamic_personas,
        auto_save_dynamic_profiles_dir=auto_save_dynamic_profiles_dir,
        auto_save_after_turns=auto_save_after_turns,
    )
