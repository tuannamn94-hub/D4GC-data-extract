"""Local vector store with a narrow interface.

Stores vectors and chunk metadata in a JSON sidecar on disk. Similarity search
uses cosine similarity (dot product on L2-normalized vectors). NumPy is used
for speed when available, with a pure-Python fallback.

The :class:`VectorStore` interface (``upsert`` / ``delete`` / ``search``) is
deliberately small so it can be reimplemented on FAISS, Qdrant, pgvector, etc.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import numpy as _np

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


class SearchHit:
    def __init__(self, chunk_id: str, score: float, text: str, metadata: dict[str, Any]):
        self.chunk_id = chunk_id
        self.score = score
        self.text = text
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"SearchHit(chunk_id={self.chunk_id!r}, score={self.score:.4f})"


class VectorStore:
    """A simple persistent vector store keyed by chunk_id."""

    def __init__(self, path: str):
        self.path = path
        self._file = os.path.join(path, "vectors.json")
        os.makedirs(path, exist_ok=True)
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._texts: dict[str, str] = {}
        self._meta: dict[str, dict] = {}
        self._dim: int | None = None
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        with open(self._file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._dim = data.get("dim")
        for rec in data.get("records", []):
            self._ids.append(rec["chunk_id"])
            self._vectors.append(rec["vector"])
            self._texts[rec["chunk_id"]] = rec["text"]
            self._meta[rec["chunk_id"]] = rec["metadata"]

    def _save(self) -> None:
        records = [
            {
                "chunk_id": cid,
                "vector": vec,
                "text": self._texts.get(cid, ""),
                "metadata": self._meta.get(cid, {}),
            }
            for cid, vec in zip(self._ids, self._vectors)
        ]
        tmp = self._file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"dim": self._dim, "records": records}, fh)
        os.replace(tmp, self._file)

    # -- mutations ----------------------------------------------------------
    def upsert(
        self,
        chunk_ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        if not (len(chunk_ids) == len(vectors) == len(texts) == len(metadatas)):
            raise ValueError("upsert arguments must be equal length")
        for cid, vec, text, meta in zip(chunk_ids, vectors, texts, metadatas):
            if self._dim is None:
                self._dim = len(vec)
            elif len(vec) != self._dim:
                raise ValueError(f"vector dim {len(vec)} != store dim {self._dim}")
            if cid in self._texts:  # replace in place
                i = self._ids.index(cid)
                self._vectors[i] = vec
            else:
                self._ids.append(cid)
                self._vectors.append(vec)
            self._texts[cid] = text
            self._meta[cid] = meta
        self._save()

    def delete(self, chunk_ids: list[str]) -> int:
        to_remove = set(chunk_ids)
        if not to_remove:
            return 0
        removed = 0
        keep_ids: list[str] = []
        keep_vecs: list[list[float]] = []
        for cid, vec in zip(self._ids, self._vectors):
            if cid in to_remove:
                self._texts.pop(cid, None)
                self._meta.pop(cid, None)
                removed += 1
            else:
                keep_ids.append(cid)
                keep_vecs.append(vec)
        self._ids, self._vectors = keep_ids, keep_vecs
        if removed:
            self._save()
        return removed

    # -- query --------------------------------------------------------------
    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchHit]:
        if not self._ids:
            return []
        if _HAVE_NUMPY:
            scores = self._scores_numpy(query_vector)
        else:
            scores = [_dot(query_vector, v) for v in self._vectors]
        ranked = sorted(range(len(self._ids)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = []
        for i in ranked:
            cid = self._ids[i]
            hits.append(SearchHit(cid, float(scores[i]), self._texts.get(cid, ""), self._meta.get(cid, {})))
        return hits

    def _scores_numpy(self, query_vector: list[float]):
        mat = _np.asarray(self._vectors, dtype=_np.float32)
        q = _np.asarray(query_vector, dtype=_np.float32)
        return mat @ q

    def __len__(self) -> int:
        return len(self._ids)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
