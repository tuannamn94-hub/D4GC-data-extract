"""Embedding providers.

Two backends:

* ``hash``  — a deterministic, dependency-free embedder based on hashed token
  n-grams. Offline and reproducible; good for tests and demos, not for semantic
  quality.
* ``sentence-transformers`` — real multilingual embeddings when the optional
  ``sentence-transformers`` package is installed.

All embedders return L2-normalized vectors so cosine similarity reduces to a
dot product.
"""

from __future__ import annotations

import abc
import hashlib
import math
import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class Embedder(abc.ABC):
    name: str = "base"
    dim: int = 0

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic bag-of-ngrams embedder using the hashing trick."""

    name = "hash"

    def __init__(self, dim: int = 256, ngram: int = 2):
        self.dim = dim
        self.ngram = ngram

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        features: list[str] = list(tokens)
        for n in range(2, self.ngram + 1):
            features.extend(
                " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )
        for feat in features:
            h = int.from_bytes(hashlib.md5(feat.encode("utf-8")).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)


class SentenceTransformerEmbedder(Embedder):
    """Wrapper around a multilingual sentence-transformers model."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [v.tolist() for v in vectors]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def get_embedder(name: str = "hash", **kwargs) -> Embedder:
    """Factory: return an embedder by name."""
    if name == "hash":
        return HashingEmbedder(**kwargs)
    if name in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder '{name}' (expected 'hash' or 'sentence-transformers')")
