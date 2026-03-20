from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.errors import MemoryStoreError


@dataclass
class MemoryRecord:
    user_id: str
    text: str
    meta: Dict


class FaissMemoryStore:
    """
    Vector memory store (FAISS + SentenceTransformers).

    Safety features:
    - `max_records` cap (drops oldest)
    - persistence failures do not crash runtime
    - retrieval failures raise a typed error
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        max_records: int = 50_000,
        persist_dir: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.dim = dim
        self.max_records = max_records
        self.persist_dir = persist_dir

        self._embedder = None
        self._faiss = None
        self._index = None
        self._records: List[MemoryRecord] = []

        self._init_backends()
        if self.persist_dir:
            self._try_load()

    def _init_backends(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise MemoryStoreError("sentence-transformers is required for memory embeddings") from exc

        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise MemoryStoreError("faiss-cpu is required for vector search") from exc

        self._faiss = faiss
        self._embedder = SentenceTransformer(self.embedding_model)
        self._index = faiss.IndexFlatL2(self.dim)

    @property
    def size(self) -> int:
        return len(self._records)

    def add(self, user_id: str, text: str, meta: Optional[Dict] = None) -> None:
        if not text:
            return
        if self.size >= self.max_records:
            # Drop oldest and rebuild index for consistency.
            self._records = self._records[-(self.max_records - 1) :]
            self._rebuild_index()

        try:
            vec = self._embedder.encode([text], normalize_embeddings=True).astype("float32")
            self._index.add(vec)
            self._records.append(MemoryRecord(user_id=user_id, text=text, meta=meta or {}))
            self._try_persist()
        except Exception as exc:
            raise MemoryStoreError("Failed to add memory record") from exc

    def retrieve(self, query: str, user_id: str, k: int = 4) -> List[MemoryRecord]:
        if not query or self.size == 0:
            return []
        try:
            q = self._embedder.encode([query], normalize_embeddings=True).astype("float32")
            _, idx = self._index.search(q, k=min(k * 3, self.size))
            hits = [self._records[i] for i in idx[0] if i >= 0 and self._records[i].user_id == user_id]
            return hits[:k]
        except Exception as exc:
            raise MemoryStoreError("Failed to retrieve memories") from exc

    def _rebuild_index(self) -> None:
        try:
            self._index = self._faiss.IndexFlatL2(self.dim)
            if not self._records:
                return
            vecs = self._embedder.encode([r.text for r in self._records], normalize_embeddings=True).astype("float32")
            self._index.add(vecs)
        except Exception as exc:
            raise MemoryStoreError("Failed to rebuild FAISS index") from exc

    def _paths(self) -> Tuple[str, str]:
        assert self.persist_dir
        return (
            os.path.join(self.persist_dir, "memory.index"),
            os.path.join(self.persist_dir, "memory.jsonl"),
        )

    def _try_persist(self) -> None:
        if not self.persist_dir:
            return
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            index_path, records_path = self._paths()
            self._faiss.write_index(self._index, index_path)
            with open(records_path, "w", encoding="utf-8") as f:
                for r in self._records:
                    f.write(json.dumps({"user_id": r.user_id, "text": r.text, "meta": r.meta}, ensure_ascii=False) + "\n")
        except Exception:
            # Fail-safe.
            return

    def _try_load(self) -> None:
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            index_path, records_path = self._paths()
            if os.path.exists(records_path):
                records: List[MemoryRecord] = []
                with open(records_path, "r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        records.append(MemoryRecord(user_id=row["user_id"], text=row["text"], meta=row.get("meta") or {}))
                self._records = records
            if os.path.exists(index_path):
                self._index = self._faiss.read_index(index_path)
            else:
                self._rebuild_index()
        except Exception:
            # Fail-safe: start fresh.
            self._records = []
            self._index = self._faiss.IndexFlatL2(self.dim)

