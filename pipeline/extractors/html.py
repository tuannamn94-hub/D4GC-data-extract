"""HTML extractor.

Uses BeautifulSoup when available for robust parsing; otherwise falls back to
a conservative standard-library HTMLParser so the pipeline still runs.
Captures headings and tables as structure.
"""

from __future__ import annotations

from html.parser import HTMLParser

from .base import Extractor, ExtractionResult
from .text import _read_text
from ..detect import TYPE_HTML

try:  # optional, higher quality
    from bs4 import BeautifulSoup  # type: ignore

    _HAVE_BS4 = True
except Exception:  # pragma: no cover - exercised only without bs4
    _HAVE_BS4 = False

_SKIP_TAGS = {"script", "style", "noscript", "template", "head"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "section", "article"} | _HEADING_TAGS


class HTMLExtractor(Extractor):
    file_types = (TYPE_HTML,)

    def extract(self, path: str) -> ExtractionResult:
        raw = _read_text(path)
        if _HAVE_BS4:
            return self._extract_bs4(raw)
        return self._extract_stdlib(raw)

    # -- BeautifulSoup path -------------------------------------------------
    def _extract_bs4(self, raw: str) -> ExtractionResult:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(list(_SKIP_TAGS)):
            tag.decompose()

        headings = [
            {"level": int(h.name[1]), "title": h.get_text(" ", strip=True)}
            for h in soup.find_all(list(_HEADING_TAGS))
            if h.get_text(strip=True)
        ]

        tables = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append({"header": rows[0], "rows": rows[1:]})

        text = soup.get_text("\n", strip=True)
        return ExtractionResult(text=text, structure={"headings": headings, "tables": tables})

    # -- stdlib fallback ----------------------------------------------------
    def _extract_stdlib(self, raw: str) -> ExtractionResult:
        parser = _StdlibHTMLExtractor()
        parser.feed(raw)
        parser.close()
        return ExtractionResult(
            text="\n".join(line for line in parser.lines if line.strip()),
            structure={"headings": parser.headings, "tables": parser.tables},
        )


class _StdlibHTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.headings: list[dict] = []
        self.tables: list[dict] = []
        self._skip_depth = 0
        self._current_heading: int | None = None
        self._buf: list[str] = []
        self._in_table = False
        self._current_row: list[str] | None = None
        self._table_rows: list[list[str]] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _HEADING_TAGS:
            self._flush()
            self._current_heading = int(tag[1])
        elif tag == "table":
            self._in_table = True
            self._table_rows = []
        elif tag == "tr" and self._in_table:
            self._current_row = []
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _HEADING_TAGS:
            text = " ".join(self._buf).strip()
            if text:
                self.headings.append({"level": int(tag[1]), "title": text})
                self.lines.append(text)
            self._buf = []
            self._current_heading = None
        elif tag in ("td", "th") and self._current_row is not None:
            self._current_row.append(" ".join(self._buf).strip())
            self._buf = []
        elif tag == "tr" and self._current_row is not None:
            if any(self._current_row):
                self._table_rows.append(self._current_row)
                self.lines.append(" | ".join(self._current_row))
            self._current_row = None
        elif tag == "table" and self._in_table:
            if self._table_rows:
                self.tables.append({"header": self._table_rows[0], "rows": self._table_rows[1:]})
            self._in_table = False
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._buf.append(text)

    def _flush(self):
        if self._current_heading is not None or self._current_row is not None:
            return
        text = " ".join(self._buf).strip()
        if text:
            self.lines.append(text)
        self._buf = []

    def close(self):
        super().close()
        self._flush()
