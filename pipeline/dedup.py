"""Deduplication of chunks.

Removes exact duplicates (by normalized-text checksum) and near-duplicates
using a token-shingle MinHash-style Jaccard estimate. Pure standard library.
"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, k: int = 4) -> set[int]:
    """Return a set of hashed k-word shingles for ``text``."""
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < k:
        return {hash(" ".join(tokens))} if tokens else set()
    out = set()
    for i in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[i : i + k])
        out.add(hash(shingle))
    return out


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def deduplicate(chunks: list[Chunk], near_threshold: float = 0.9) -> list[Chunk]:
    """Return ``chunks`` with exact and near-duplicates removed.

    The first occurrence of a duplicate is kept. ``near_threshold`` is the
    Jaccard similarity above which two chunks are considered near-duplicates.
    Set it to ``1.0`` to disable near-duplicate removal (exact only).
    """
    seen_hashes: set[str] = set()
    kept: list[Chunk] = []
    kept_shingles: list[set[int]] = []

    for chunk in chunks:
        checksum = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
        chunk.checksum = checksum
        if checksum in seen_hashes:
            continue

        if near_threshold < 1.0:
            sh = _shingles(chunk.text)
            if any(_jaccard(sh, other) >= near_threshold for other in kept_shingles):
                continue
            kept_shingles.append(sh)

        seen_hashes.add(checksum)
        kept.append(chunk)

    return kept
