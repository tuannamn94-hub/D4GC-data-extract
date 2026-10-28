"""PDF extractor.

Uses ``pypdf`` when installed to extract per-page text and preserve page
numbers. Without it, raises a clear, actionable error rather than producing
garbage, and the runner records the document as failed.
"""

from __future__ import annotations

from .base import Extractor, ExtractionResult
from ..detect import TYPE_PDF

try:  # optional dependency
    from pypdf import PdfReader  # type: ignore

    _HAVE_PYPDF = True
except Exception:  # pragma: no cover
    _HAVE_PYPDF = False


class PdfExtractionUnavailable(RuntimeError):
    """Raised when PDF support is requested but no backend is installed."""


class PDFExtractor(Extractor):
    file_types = (TYPE_PDF,)

    def extract(self, path: str) -> ExtractionResult:
        if not _HAVE_PYPDF:
            raise PdfExtractionUnavailable(
                "PDF extraction requires 'pypdf'. Install it with "
                "`pip install pypdf` (see requirements-optional.txt)."
            )

        reader = PdfReader(path)
        pages: list[dict] = []
        text_parts: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            # Record the character offset range so chunks can map back to a page.
            start = sum(len(p) + 1 for p in text_parts)
            text_parts.append(page_text)
            pages.append(
                {
                    "page": page_number,
                    "char_start": start,
                    "char_end": start + len(page_text),
                }
            )

        text = "\n".join(text_parts)
        structure = {"headings": [], "tables": [], "pages": pages}
        return ExtractionResult(text=text, structure=structure)
