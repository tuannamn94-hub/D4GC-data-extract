"""Shared data structures used across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Document:
    """A source document after extraction, before chunking."""

    doc_id: str
    source: str                 # absolute or relative path / URI of the source
    file_type: str              # "pdf", "html", "markdown", "text", ...
    text: str                   # normalized full text
    checksum: str               # sha256 of the raw source bytes
    language: Optional[str] = None
    version: int = 1
    structure: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit of text derived from a Document."""

    chunk_id: str
    doc_id: str
    source: str
    index: int                  # ordinal position within the document
    text: str
    language: Optional[str] = None
    checksum: str = ""          # sha256 of the chunk text (for dedup)
    structure: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata dict stored alongside the vector."""
        d = asdict(self)
        d.pop("text", None)
        return d


# Status values used in the metadata DB.
STATUS_PENDING = "pending"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
STATUS_DELETED = "deleted"
