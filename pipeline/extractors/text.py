"""Plain-text extractor."""

from __future__ import annotations

from .base import Extractor, ExtractionResult
from ..detect import TYPE_TEXT


def _read_text(path: str) -> str:
    """Read a text file, tolerating unknown encodings."""
    with open(path, "rb") as fh:
        raw = fh.read()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class TextExtractor(Extractor):
    file_types = (TYPE_TEXT,)

    def extract(self, path: str) -> ExtractionResult:
        text = _read_text(path)
        return ExtractionResult(text=text, structure={"headings": [], "tables": []})
