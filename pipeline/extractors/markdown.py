"""Markdown extractor that preserves headings and simple tables."""

from __future__ import annotations

import re

from .base import Extractor, ExtractionResult
from .text import _read_text
from ..detect import TYPE_MARKDOWN

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


class MarkdownExtractor(Extractor):
    file_types = (TYPE_MARKDOWN,)

    def extract(self, path: str) -> ExtractionResult:
        raw = _read_text(path)
        lines = raw.splitlines()

        headings: list[dict] = []
        tables: list[dict] = []
        text_lines: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            m = _HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                headings.append({"level": level, "title": title, "line": i})
                text_lines.append(title)
                i += 1
                continue

            # Detect a GitHub-style table: header row, separator, body rows.
            if "|" in line and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
                start = i
                header = [c.strip() for c in line.strip().strip("|").split("|")]
                i += 2
                rows: list[list[str]] = []
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                tables.append({"header": header, "rows": rows, "line": start})
                # Linearize the table into text so it remains searchable.
                text_lines.append(" | ".join(header))
                for r in rows:
                    text_lines.append(" | ".join(r))
                continue

            # Strip common inline markdown noise for cleaner embeddings.
            cleaned = re.sub(r"[*_`>]", "", line)
            cleaned = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
            text_lines.append(cleaned)
            i += 1

        structure = {"headings": headings, "tables": tables}
        return ExtractionResult(text="\n".join(text_lines), structure=structure)
