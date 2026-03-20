from __future__ import annotations

from typing import Iterator, Optional

import torch  # type: ignore

from inference.backends.base import GenerationConfig, LLMBackend


class TransformersBackend(LLMBackend):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> str:
        cfg = cfg or GenerationConfig()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_tokens = outputs[0][prompt_len:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    def stream_generate(self, prompt: str, cfg: Optional[GenerationConfig] = None) -> Iterator[str]:
        cfg = cfg or GenerationConfig()

        from threading import Thread

        from transformers import TextIteratorStreamer  # type: ignore

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        kwargs = dict(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        def _run_generate():
            with torch.no_grad():
                self.model.generate(**kwargs)

        t = Thread(target=_run_generate, daemon=True)
        t.start()
        for text in streamer:
            if text:
                yield text
        t.join()
