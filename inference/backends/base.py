from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 180
    temperature: float = 0.8
    top_p: float = 0.9


class LLMBackend:
    def generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> str:
        raise NotImplementedError

    def stream_generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> Iterator[str]:
        """
        Streaming generation (best-effort).

        Backends that don't implement streaming will fall back to one-shot `generate()`.
        Yields text chunks (not tokens).
        """
        yield self.generate(prompt, cfg=cfg)
