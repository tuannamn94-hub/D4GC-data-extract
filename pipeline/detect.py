"""File-type detection by extension with a content-sniffing fallback."""

from __future__ import annotations

import os

# Canonical file-type names used throughout the pipeline.
TYPE_PDF = "pdf"
TYPE_HTML = "html"
TYPE_MARKDOWN = "markdown"
TYPE_TEXT = "text"
TYPE_UNKNOWN = "unknown"

_EXTENSION_MAP = {
    ".pdf": TYPE_PDF,
    ".htm": TYPE_HTML,
    ".html": TYPE_HTML,
    ".xhtml": TYPE_HTML,
    ".md": TYPE_MARKDOWN,
    ".markdown": TYPE_MARKDOWN,
    ".mdown": TYPE_MARKDOWN,
    ".txt": TYPE_TEXT,
    ".text": TYPE_TEXT,
    ".log": TYPE_TEXT,
    ".rst": TYPE_TEXT,
}


def detect_type(path: str) -> str:
    """Return the canonical file type for ``path``.

    Detection is by extension first, falling back to lightweight content
    sniffing for files without a recognized extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _EXTENSION_MAP:
        return _EXTENSION_MAP[ext]
    return _sniff(path)


def _sniff(path: str) -> str:
    """Best-effort content sniffing for extension-less files."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(2048)
    except OSError:
        return TYPE_UNKNOWN

    if head[:5] == b"%PDF-":
        return TYPE_PDF

    # Decode a sample as text for the remaining heuristics.
    try:
        sample = head.decode("utf-8", errors="ignore").lstrip().lower()
    except Exception:
        return TYPE_UNKNOWN

    if sample.startswith("<!doctype html") or sample.startswith("<html") or "<body" in sample:
        return TYPE_HTML
    if sample.startswith("#") or "\n#" in sample or sample.startswith("- ") or "](" in sample:
        return TYPE_MARKDOWN
    if head and _looks_textual(head):
        return TYPE_TEXT
    return TYPE_UNKNOWN


def _looks_textual(data: bytes) -> bool:
    """Heuristic: treat data as text if it has few control/NUL bytes."""
    if b"\x00" in data:
        return False
    text_chars = sum(1 for b in data if b == 9 or b == 10 or b == 13 or 32 <= b < 127 or b >= 128)
    return text_chars / max(len(data), 1) > 0.85
