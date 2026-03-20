from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from inference.backends.base import GenerationConfig, LLMBackend


@dataclass
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0


class LlamaCppBackend(LLMBackend):
    """
    GGUF / llama.cpp backend (edge-friendly).

    Requires the optional `llama-cpp-python` dependency at runtime.
    """

    def __init__(self, cfg: LlamaCppConfig):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("llama-cpp-python is required for LlamaCppBackend") from exc

        self._llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
        )

    def generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> str:
        cfg = cfg or GenerationConfig()
        out = self._llm.create_completion(
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        # llama.cpp returns a completion dict; keep plain text output
        return (out.get("choices") or [{}])[0].get("text", "")

    def stream_generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> Iterator[str]:
        cfg = cfg or GenerationConfig()
        for evt in self._llm.create_completion(
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stream=True,
        ):
            try:
                piece = (evt.get("choices") or [{}])[0].get("text", "")
            except Exception:
                piece = ""
            if piece:
                yield piece
