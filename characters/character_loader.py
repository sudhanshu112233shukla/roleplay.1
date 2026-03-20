from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from characters.profile import CharacterProfile, load_character_profile_by_id


@dataclass(frozen=True)
class LoadedCharacter:
    character_id: str
    profile: CharacterProfile
    source: str  # "yaml" | "dynamic"


def load_character(character_id: str, profiles_dir: str = "characters/profiles") -> LoadedCharacter:
    profile = load_character_profile_by_id(character_id, profiles_dir=profiles_dir)
    return LoadedCharacter(character_id=character_id, profile=profile, source="yaml")


def try_load_character(character_id: str, profiles_dir: str = "characters/profiles") -> Optional[LoadedCharacter]:
    try:
        return load_character(character_id, profiles_dir=profiles_dir)
    except Exception:
        return None

