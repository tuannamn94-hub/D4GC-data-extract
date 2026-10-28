"""Extractor registry.

Maps canonical file types (from :mod:`pipeline.detect`) to an extractor
instance. Register a new parser by adding it to ``_EXTRACTORS``.
"""

from __future__ import annotations

from .base import Extractor, ExtractionResult
from .text import TextExtractor
from .markdown import MarkdownExtractor
from .html import HTMLExtractor
from .pdf import PDFExtractor, PdfExtractionUnavailable

_INSTANCES: list[Extractor] = [
    TextExtractor(),
    MarkdownExtractor(),
    HTMLExtractor(),
    PDFExtractor(),
]

# Build {file_type: extractor} from each extractor's declared file_types.
_REGISTRY: dict[str, Extractor] = {}
for _ext in _INSTANCES:
    for _ft in _ext.file_types:
        _REGISTRY[_ft] = _ext


def get_extractor(file_type: str) -> Extractor | None:
    """Return the extractor for ``file_type``, or ``None`` if unsupported."""
    return _REGISTRY.get(file_type)


def supported_types() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Extractor",
    "ExtractionResult",
    "PdfExtractionUnavailable",
    "get_extractor",
    "supported_types",
]
