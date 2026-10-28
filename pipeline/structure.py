"""Helpers for working with preserved document structure.

Extractors emit a ``structure`` dict that may contain ``headings``, ``tables``
and ``pages``. These helpers map a character offset in the normalized text back
to the nearest heading and page so chunks can carry that context.
"""

from __future__ import annotations

from typing import Any


def page_for_offset(structure: dict[str, Any], offset: int) -> int | None:
    """Return the 1-based page number containing ``offset``, if known."""
    pages = structure.get("pages")
    if not pages:
        return None
    for page in pages:
        if page["char_start"] <= offset < page["char_end"]:
            return page["page"]
    # Past the last recorded boundary: attribute to the final page.
    return pages[-1]["page"] if pages else None


def heading_for_line(structure: dict[str, Any], line_no: int) -> str | None:
    """Return the most recent heading at or before ``line_no``."""
    headings = structure.get("headings")
    if not headings:
        return None
    current = None
    for h in headings:
        if h.get("line") is None:
            continue
        if h["line"] <= line_no:
            current = h["title"]
        else:
            break
    return current


def summarize(structure: dict[str, Any]) -> dict[str, int]:
    """Compact counts for status reporting."""
    return {
        "headings": len(structure.get("headings", [])),
        "tables": len(structure.get("tables", [])),
        "pages": len(structure.get("pages", [])),
    }
