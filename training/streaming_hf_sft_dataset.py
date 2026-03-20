from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from training_data.hf_adapters import map_any_to_text, safe_load_hf_dataset


@dataclass(frozen=True)
class HfDatasetSpec:
    name: str
    config_name: str | None = None
    split: str = "train"


class StreamingHfSftDataset:
    """
    Streaming iterable dataset for SFT from Hugging Face datasets (streaming=True).

    Notes:
    - This is intended for Colab/Jupyter where the datasets can be huge.
    - Uses best-effort schema mapping via `training_data.hf_adapters.map_any_to_text`.
    - Applies an approximate shuffle via a local shuffle buffer.
    - Repeats indefinitely by default, so `TrainingArguments(max_steps=...)` can be used.
    """

    def __init__(
        self,
        *,
        datasets: Sequence[HfDatasetSpec],
        tokenizer,
        max_length: int = 1024,
        seed: int = 42,
        shuffle_buffer: int = 0,
        repeat: bool = True,
        per_dataset_take: int | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("at least one dataset spec is required")
        self._specs = list(datasets)
        self._tokenizer = tokenizer
        self._max_length = int(max_length)
        self._seed = int(seed)
        self._shuffle_buffer = int(shuffle_buffer)
        self._repeat = bool(repeat)
        self._per_dataset_take = per_dataset_take if per_dataset_take is None else int(per_dataset_take)

    def _iter_texts_once(self) -> Iterable[str]:
        # Round-robin interleave over dataset iterators for basic mixing.
        streams: List[Tuple[str, Iterator[Dict]]] = []
        for spec in self._specs:
            ds = safe_load_hf_dataset(spec.name, spec.split, streaming=True, config_name=spec.config_name)
            streams.append((spec.name, iter(ds)))

        alive = [True] * len(streams)
        seen = [0] * len(streams)
        while any(alive):
            for i, (name, it) in enumerate(streams):
                if not alive[i]:
                    continue
                if self._per_dataset_take is not None and seen[i] >= self._per_dataset_take:
                    alive[i] = False
                    continue
                try:
                    ex = next(it)
                except StopIteration:
                    alive[i] = False
                    continue
                seen[i] += 1
                mapped = map_any_to_text(ex, source=name)
                if mapped and mapped.get("text"):
                    yield str(mapped["text"])

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        import torch  # type: ignore

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        rng = random.Random(self._seed + worker_id)

        def _tokenize(text: str) -> Dict[str, List[int]] | None:
            enc = self._tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                padding=False,
                add_special_tokens=False,
                return_attention_mask=True,
            )
            if not enc.get("input_ids"):
                return None
            return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", [1] * len(enc["input_ids"]))}

        def _texts() -> Iterable[str]:
            if not self._repeat:
                yield from self._iter_texts_once()
                return
            while True:
                yield from self._iter_texts_once()

        if self._shuffle_buffer <= 1:
            for text in _texts():
                item = _tokenize(text)
                if item is not None:
                    yield item
            return

        buffer: List[str] = []
        for text in _texts():
            buffer.append(text)
            if len(buffer) < self._shuffle_buffer:
                continue
            idx = rng.randrange(len(buffer))
            item = _tokenize(buffer.pop(idx))
            if item is not None:
                yield item

        while buffer:
            idx = rng.randrange(len(buffer))
            item = _tokenize(buffer.pop(idx))
            if item is not None:
                yield item


def as_torch_iterable(dataset: StreamingHfSftDataset):
    import torch  # type: ignore

    class _TorchIterable(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(dataset)

    return _TorchIterable()
