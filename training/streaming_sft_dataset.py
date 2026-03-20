from __future__ import annotations

import os
import random
from typing import Dict, Iterable, Iterator, List, Sequence


def _expand_dataset_paths(dataset_path: str) -> List[str]:
    if not dataset_path:
        raise ValueError("dataset_path is required")
    if os.path.isdir(dataset_path):
        out: List[str] = []
        for name in sorted(os.listdir(dataset_path)):
            p = os.path.join(dataset_path, name)
            if os.path.isdir(p):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in {".jsonl", ".jl", ".json", ".csv", ".tsv"}:
                out.append(p)
        if not out:
            raise ValueError(f"no dataset files found under directory: {dataset_path}")
        return out
    return [dataset_path]


class StreamingSftDataset:
    """
    Memory-light iterable dataset for SFT on local JSONL/JSON/CSV datasets.

    - Reads the dataset from disk (JSONL/CSV stream; JSON loads fully).
    - Validates/normalizes each example via `training.dataset_loader.iter_validated_dataset`.
    - Tokenizes on the fly.
    - Supports a shuffle buffer (approximate shuffle).
    - Repeats indefinitely by default, so `TrainingArguments(max_steps=...)` can be used.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        tokenizer,
        max_length: int = 1024,
        seed: int = 42,
        shuffle_buffer: int = 0,
        repeat: bool = True,
        drop_empty: bool = True,
        max_chars: int = 32_000,
        max_turns: int = 12,
    ) -> None:
        self._paths = _expand_dataset_paths(dataset_path)
        self._tokenizer = tokenizer
        self._max_length = int(max_length)
        self._seed = int(seed)
        self._shuffle_buffer = int(shuffle_buffer)
        self._repeat = bool(repeat)
        self._drop_empty = bool(drop_empty)
        self._max_chars = int(max_chars)
        self._max_turns = int(max_turns)

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        import torch  # type: ignore

        from training.dataset_loader import iter_validated_dataset

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

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

        def _iter_texts_once() -> Iterable[str]:
            # Interleave files deterministically and shard by worker to avoid duplication.
            for path in self._paths:
                for i, row in enumerate(
                    iter_validated_dataset(
                        path,
                        drop_empty=self._drop_empty,
                        max_chars=self._max_chars,
                        max_turns=self._max_turns,
                    ),
                    0,
                ):
                    if (i % num_workers) != worker_id:
                        continue
                    yield row.to_chat_text()

        def _iter_texts() -> Iterable[str]:
            if not self._repeat:
                yield from _iter_texts_once()
                return
            while True:
                yield from _iter_texts_once()

        if self._shuffle_buffer <= 1:
            for text in _iter_texts():
                item = _tokenize(text)
                if item is not None:
                    yield item
            return

        buffer: List[str] = []
        for text in _iter_texts():
            buffer.append(text)
            if len(buffer) < self._shuffle_buffer:
                continue
            idx = rng.randrange(len(buffer))
            item = _tokenize(buffer.pop(idx))
            if item is not None:
                yield item

        # In practice we won't reach here with repeat=True; keep for completeness.
        while buffer:
            idx = rng.randrange(len(buffer))
            item = _tokenize(buffer.pop(idx))
            if item is not None:
                yield item


def as_torch_iterable(dataset: StreamingSftDataset):
    """
    Wraps `StreamingSftDataset` into a `torch.utils.data.IterableDataset` without adding a torch dependency at import time.
    """

    import torch  # type: ignore

    class _TorchIterable(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(dataset)

    return _TorchIterable()
