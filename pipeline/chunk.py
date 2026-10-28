"""Structure-aware chunking.

Splits a document's normalized text into overlapping chunks that respect
paragraph and sentence boundaries where possible, and annotates each chunk with
the nearest heading and (for PDFs) page number.
"""

from __future__ import annotations

import re
import uuid

from .models import Chunk, Document
from . import structure as structure_mod

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_RE.split(paragraph.strip())
    return [p for p in parts if p]


def chunk_document(
    doc: Document,
    *,
    target_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Split ``doc`` into overlapping chunks of roughly ``target_chars``.

    Chunks never cut a paragraph in the middle unless the paragraph itself is
    larger than ``target_chars``, in which case it is split on sentence
    boundaries. Consecutive chunks share ``overlap_chars`` of context.
    """
    text = doc.text
    if not text.strip():
        return []

    # Track line numbers so we can attribute headings.
    paragraphs = _paragraphs_with_offsets(text)

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    buf_start_offset = 0

    def flush(end_offset: int) -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunk_text = "\n\n".join(buf).strip()
        if chunk_text:
            chunks.append(_make_chunk(doc, len(chunks), chunk_text, buf_start_offset))
        # Start the next buffer with a tail overlap of the current one.
        if overlap_chars > 0 and chunk_text:
            tail = chunk_text[-overlap_chars:]
            buf = [tail]
            buf_len = len(tail)
        else:
            buf = []
            buf_len = 0

    for para_text, para_offset, _line_no in paragraphs:
        if len(para_text) > target_chars:
            # Oversized paragraph: flush, then split into sentence windows.
            flush(para_offset)
            for sent_chunk in _pack_sentences(para_text, target_chars, overlap_chars):
                chunks.append(_make_chunk(doc, len(chunks), sent_chunk, para_offset))
            buf, buf_len = [], 0
            continue

        if buf_len + len(para_text) > target_chars and buf:
            flush(para_offset)
            buf_start_offset = para_offset

        if not buf:
            buf_start_offset = para_offset
        buf.append(para_text)
        buf_len += len(para_text) + 2

    flush(len(text))
    return chunks


def _paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Yield (paragraph, char_offset, line_no) for each non-empty block."""
    result: list[tuple[str, int, int]] = []
    offset = 0
    line_no = 0
    for block in text.split("\n\n"):
        stripped = block.strip()
        if stripped:
            result.append((stripped, offset, line_no))
        offset += len(block) + 2
        line_no += block.count("\n") + 2
    return result


def _pack_sentences(paragraph: str, target: int, overlap: int) -> list[str]:
    sentences = _split_sentences(paragraph)
    out: list[str] = []
    buf: list[str] = []
    length = 0
    for sent in sentences:
        if length + len(sent) > target and buf:
            out.append(" ".join(buf).strip())
            if overlap > 0:
                tail = " ".join(buf)[-overlap:]
                buf = [tail]
                length = len(tail)
            else:
                buf, length = [], 0
        buf.append(sent)
        length += len(sent) + 1
    if buf:
        out.append(" ".join(buf).strip())
    return [c for c in out if c]


def _make_chunk(doc: Document, index: int, text: str, offset: int) -> Chunk:
    chunk_structure: dict = {}
    page = structure_mod.page_for_offset(doc.structure, offset)
    if page is not None:
        chunk_structure["page"] = page
    heading = structure_mod.heading_for_line(doc.structure, text.count("\n"))
    if heading:
        chunk_structure["heading"] = heading

    return Chunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}:{index}").hex,
        doc_id=doc.doc_id,
        source=doc.source,
        index=index,
        text=text,
        language=doc.language,
        structure=chunk_structure,
        meta={"version": doc.version, "file_type": doc.file_type},
    )
