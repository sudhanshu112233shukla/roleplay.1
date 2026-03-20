from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from inference.backends.base import GenerationConfig, LLMBackend


@dataclass(frozen=True)
class OrtGenAIConfig:
    model_dir: str


class OrtGenAIBackend(LLMBackend):
    """
    ONNX Runtime GenAI backend.

    This implementation is intentionally minimal and backend-agnostic for prompt/memory/world/emotion.
    It requires the optional `onnxruntime-genai` package and an ORT GenAI model directory.
    """

    def __init__(self, cfg: OrtGenAIConfig):
        try:
            import onnxruntime_genai as og  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime-genai is required for OrtGenAIBackend") from exc

        self._og = og
        self._model = og.Model(cfg.model_dir)
        self._tokenizer = og.Tokenizer(self._model)
        self._tokenizer_stream = self._tokenizer.create_stream()

    def generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> str:
        cfg = cfg or GenerationConfig()

        params = self._og.GeneratorParams(self._model)
        # Greedy vs sampling control; keep defaults compatible with chat
        params.set_search_options(
            max_length=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )

        input_ids = self._tokenizer.encode(prompt)
        params.input_ids = input_ids

        generator = self._og.Generator(self._model, params)
        pieces = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            pieces.append(self._tokenizer_stream.decode(token))
        return "".join(pieces)

    def stream_generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> Iterator[str]:
        cfg = cfg or GenerationConfig()

        params = self._og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        input_ids = self._tokenizer.encode(prompt)
        params.input_ids = input_ids

        generator = self._og.Generator(self._model, params)
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            piece = self._tokenizer_stream.decode(token)
            if piece:
                yield piece
