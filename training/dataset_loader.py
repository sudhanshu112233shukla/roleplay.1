from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.errors import DatasetError
from utils.text import normalize_text_aggressive


DEFAULT_SYSTEM = "You are an immersive roleplay assistant. Stay in character, emotionally consistent, descriptive, and coherent across turns."


@dataclass(frozen=True)
class DatasetRow:
    system: str
    turns: List[Tuple[str, str]]

    def to_chat_text(self) -> str:
        system = normalize_text_aggressive(self.system or DEFAULT_SYSTEM)
        chunks = [f"<|system|>\n{system}\n"]
        for u, a in self.turns:
            u_n = normalize_text_aggressive(u)
            a_n = normalize_text_aggressive(a)
            if not u_n or not a_n:
                continue
            chunks.append(f"<|user|>\n{u_n}\n")
            chunks.append(f"<|assistant|>\n{a_n}\n")
        return "".join(chunks)


def load_dataset(path: str) -> List[DatasetRow]:
    """
    Loads a dataset from JSONL / JSON / CSV.

    Supported schemas:
    - {"system": "...", "user": "...", "assistant": "..."} (preferred)
    - {"messages": [{"role": "...", "content": "..."} ...], "system": "..."} (best-effort)
    """
    if not path:
        raise DatasetError("dataset path is required")
    if not os.path.exists(path):
        raise DatasetError(f"dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".jsonl", ".jl"}:
        rows = list(_iter_jsonl_rows(path))
    elif ext == ".json":
        rows = list(_iter_json_rows(path))
    elif ext in {".csv", ".tsv"}:
        rows = list(_iter_csv_rows(path, delimiter="\t" if ext == ".tsv" else ","))
    else:
        raise DatasetError(f"unsupported dataset format: {ext} (expected .jsonl/.json/.csv)")

    out: List[DatasetRow] = []
    for i, obj in enumerate(rows, 1):
        try:
            row = _to_row(obj)
        except Exception as exc:
            raise DatasetError(f"invalid row #{i} in {path}") from exc
        out.append(row)
    return out


def iter_dataset_rows(path: str) -> Iterable[DatasetRow]:
    """
    Streaming iterator over dataset rows.

    - JSONL/CSV/TSV are streamed from disk.
    - JSON is loaded into memory first (JSON arrays are not stream-friendly).

    See `load_dataset()` for supported schemas.
    """
    if not path:
        raise DatasetError("dataset path is required")
    if not os.path.exists(path):
        raise DatasetError(f"dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".jsonl", ".jl"}:
        objs: Iterable[Dict[str, Any]] = _iter_jsonl_rows(path)
    elif ext == ".json":
        objs = _iter_json_rows(path)
    elif ext in {".csv", ".tsv"}:
        objs = _iter_csv_rows(path, delimiter="\t" if ext == ".tsv" else ",")
    else:
        raise DatasetError(f"unsupported dataset format: {ext} (expected .jsonl/.json/.csv)")

    for i, obj in enumerate(objs, 1):
        if not isinstance(obj, dict):
            raise DatasetError(f"row must be an object at {path}:{i}")
        try:
            yield _to_row(obj)
        except Exception as exc:
            raise DatasetError(f"invalid row #{i} in {path}") from exc


def iter_validated_dataset(
    path: str,
    *,
    drop_empty: bool = True,
    max_chars: int = 32_000,
    max_turns: int = 12,
) -> Iterable[DatasetRow]:
    """
    Streaming equivalent of `validate_dataset(load_dataset(...))`.
    """

    for r in iter_dataset_rows(path):
        validated = _validate_row(
            r,
            drop_empty=drop_empty,
            max_chars=max_chars,
            max_turns=max_turns,
        )
        if validated is not None:
            yield validated


def validate_dataset(
    rows: Sequence[DatasetRow],
    *,
    drop_empty: bool = True,
    max_chars: int = 32_000,
    max_turns: int = 12,
) -> List[DatasetRow]:
    """
    - Removes empty messages
    - Truncates extremely long examples (char-based; token-length handled in `tokenize_dataset`)
    - Normalizes formatting
    """
    out: List[DatasetRow] = []
    for r in rows:
        validated = _validate_row(
            r,
            drop_empty=drop_empty,
            max_chars=max_chars,
            max_turns=max_turns,
        )
        if validated is not None:
            out.append(validated)
    if not out:
        raise DatasetError("dataset is empty after validation")
    return out


def tokenize_dataset(
    rows: Sequence[DatasetRow],
    tokenizer,
    *,
    max_length: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Tokenizes into model-ready `input_ids` / `attention_mask` with truncation.
    """
    tokenized: List[Dict[str, Any]] = []
    for r in rows:
        text = r.to_chat_text()
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )
        if not enc.get("input_ids"):
            continue
        tokenized.append(enc)
    if not tokenized:
        raise DatasetError("no tokenized samples (check max_length and dataset content)")
    return tokenized


def _iter_jsonl_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise DatasetError(f"invalid JSONL at {path}:{line_no}") from exc
            if isinstance(obj, dict):
                yield obj
            else:
                raise DatasetError(f"row must be a JSON object at {path}:{line_no}")


def _iter_json_rows(path: str) -> Iterable[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise DatasetError(f"invalid JSON: {path}") from exc

    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                raise DatasetError(f"JSON array must contain objects: {path}")
            yield obj
        return

    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for obj in data["data"]:
            if not isinstance(obj, dict):
                raise DatasetError(f"JSON 'data' must contain objects: {path}")
            yield obj
        return

    raise DatasetError(f"JSON must be an array of objects or an object with a 'data' array: {path}")


def _iter_csv_rows(path: str, delimiter: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield dict(row)


def _to_row(obj: Dict[str, Any]) -> DatasetRow:
    system = str(obj.get("system") or obj.get("instruction") or DEFAULT_SYSTEM)

    if "user" in obj or "assistant" in obj:
        user = str(obj.get("user") or obj.get("prompt") or "")
        assistant = str(obj.get("assistant") or obj.get("response") or obj.get("output") or "")
        return DatasetRow(system=system, turns=[(user, assistant)])

    # Best-effort "messages" format.
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        turns: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or m.get("from") or "").lower()
            content = str(m.get("content") or m.get("value") or m.get("text") or "")
            if role == "system" and content.strip():
                system = content
            elif role in {"user", "human"} and content.strip():
                pending_user = content
            elif role in {"assistant", "gpt"} and pending_user is not None and content.strip():
                turns.append((pending_user, content))
                pending_user = None
        # If no paired turns, try last-pair fallback.
        if not turns:
            user = ""
            assistant = ""
            for m in reversed(msgs):
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or m.get("from") or "").lower()
                content = str(m.get("content") or m.get("value") or m.get("text") or "")
                if role in {"assistant", "gpt"} and not assistant and content.strip():
                    assistant = content
                elif role in {"user", "human"} and assistant and not user and content.strip():
                    user = content
                    break
            turns = [(user, assistant)]

        return DatasetRow(system=system, turns=turns)

    raise DatasetError("row missing required fields (system/user/assistant) or messages[]")


def _validate_row(
    r: DatasetRow,
    *,
    drop_empty: bool,
    max_chars: int,
    max_turns: int,
) -> DatasetRow | None:
    system = (r.system or DEFAULT_SYSTEM).strip()
    turns = [(str(u or "").strip(), str(a or "").strip()) for (u, a) in (r.turns or [])]
    if drop_empty:
        turns = [(u, a) for (u, a) in turns if u and a]
    if max_turns and len(turns) > max_turns:
        turns = turns[-max_turns:]

    if drop_empty and not turns:
        return None

    txt = DatasetRow(system=system, turns=turns).to_chat_text()
    if len(txt) > max_chars:
        # Keep the tail to preserve the assistant response; user instruction is often near the end in logs.
        txt = txt[-max_chars:]
        # Re-wrap as a minimal sample
        return DatasetRow(system=DEFAULT_SYSTEM, turns=[("(truncated)", txt)])

    return DatasetRow(system=system, turns=turns)
