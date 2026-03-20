from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from utils.errors import CharacterProfileError


@dataclass(frozen=True)
class CharacterProfile:
    name: str
    role: str
    personality: str
    speech_style: str
    background: str
    emotions: str
    motivations: str
    behavior_rules: str = ""

    def to_system_prompt(self) -> str:
        lines = [
            f'You are "{self.name}".',
            f"Role: {self.role}",
            f"Personality: {self.personality}",
            f"Speech style: {self.speech_style}",
            f"Background: {self.background}",
            f"Emotions: {self.emotions}",
            f"Motivations: {self.motivations}",
        ]
        if self.behavior_rules:
            lines.append(f"Behavior rules: {self.behavior_rules}")
        lines.append("Never break character unless explicitly instructed by the system.")
        return "\n".join(lines)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CharacterProfile":
        required = ["name", "role", "personality", "speech_style", "background", "emotions", "motivations"]
        missing = [k for k in required if not str(data.get(k, "")).strip()]
        if missing:
            raise CharacterProfileError(f"Missing required character fields: {', '.join(missing)}")
        return CharacterProfile(
            name=str(data["name"]).strip(),
            role=str(data["role"]).strip(),
            personality=str(data["personality"]).strip(),
            speech_style=str(data["speech_style"]).strip(),
            background=str(data["background"]).strip(),
            emotions=str(data["emotions"]).strip(),
            motivations=str(data["motivations"]).strip(),
            behavior_rules=str(data.get("behavior_rules", "")).strip(),
        )


def load_character_profile_yaml(path: str) -> CharacterProfile:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise CharacterProfileError("PyYAML is required to load character profiles") from exc

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        raise CharacterProfileError(f"Failed to read character profile: {path}") from exc

    if not isinstance(data, dict):
        raise CharacterProfileError(f"Character profile must be a mapping: {path}")

    return CharacterProfile.from_dict(data)


def load_character_profile_by_id(character_id: str, profiles_dir: str = "characters/profiles") -> CharacterProfile:
    import os

    # Allow nested IDs like "user/iron_man" while preventing traversal / absolute paths.
    cid = (character_id or "").strip().replace("\\", "/").strip("/")
    if not cid or os.path.isabs(cid) or any(p in {"..", "."} for p in cid.split("/")):
        raise CharacterProfileError(f"Invalid character id: {character_id!r}")

    candidates = [
        os.path.join(profiles_dir, f"{cid}.yaml"),
        os.path.join(profiles_dir, f"{cid}.yml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_character_profile_yaml(path)
    raise CharacterProfileError(f"Character profile not found for id='{cid}' in {profiles_dir}")
