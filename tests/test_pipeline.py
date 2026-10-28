"""End-to-end and unit tests for the pipeline. Run with: python -m pytest -q"""

from __future__ import annotations

import os

import pytest

from pipeline import detect, normalize
from pipeline.dedup import deduplicate
from pipeline.language import detect_language
from pipeline.models import Document, Chunk
from pipeline.chunk import chunk_document
from pipeline.embeddings import get_embedder
from pipeline.runner import Pipeline


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_detect_type_by_extension(tmp_path):
    assert detect.detect_type(str(tmp_path / "a.md")) == detect.TYPE_MARKDOWN
    assert detect.detect_type(str(tmp_path / "a.html")) == detect.TYPE_HTML
    assert detect.detect_type(str(tmp_path / "a.txt")) == detect.TYPE_TEXT
    assert detect.detect_type(str(tmp_path / "a.pdf")) == detect.TYPE_PDF


def test_detect_type_by_content(tmp_path):
    p = tmp_path / "noext"
    p.write_bytes(b"<!DOCTYPE html><html><body>hi</body></html>")
    assert detect.detect_type(str(p)) == detect.TYPE_HTML


def test_normalize_collapses_whitespace_and_control():
    raw = "Hello\u200b   world\r\n\r\n\r\n\r\nnext\x07line"
    out = normalize.normalize_text(raw)
    assert "\u200b" not in out
    assert "\x07" not in out
    assert "Hello world" in out
    assert out.count("\n\n") == 1


def test_validate_rejects_binary_like():
    with pytest.raises(normalize.ValidationError):
        normalize.validate_text("\x00\x01\x02\x03\x04\x05", min_printable_ratio=0.6)


def test_language_detection_spanish_vs_english():
    assert detect_language("the quick brown fox jumps over the lazy dog and runs") == "en"
    assert detect_language("el sistema procesa documentos en varios idiomas y genera") == "es"


def test_chunking_overlap_and_structure():
    text = "\n\n".join(f"Paragraph number {i} with some filler words here." for i in range(40))
    doc = Document(doc_id="d1", source="x", file_type="text", text=text, checksum="c")
    chunks = chunk_document(doc, target_chars=200, overlap_chars=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)  # oversize safety
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_deduplicate_removes_exact_and_near():
    base = "the retrieval system indexes documents and returns passages"
    chunks = [
        Chunk(chunk_id="1", doc_id="d", source="s", index=0, text=base),
        Chunk(chunk_id="2", doc_id="d", source="s", index=1, text=base),  # exact dup
        Chunk(chunk_id="3", doc_id="d", source="s", index=2, text=base + " extra"),  # near dup
        Chunk(chunk_id="4", doc_id="d", source="s", index=3, text="a completely different sentence"),
    ]
    kept = deduplicate(chunks, near_threshold=0.8)
    kept_text = [c.text for c in kept]
    assert base in kept_text
    assert "a completely different sentence" in kept_text
    assert len(kept) == 2


def test_hash_embedder_is_deterministic_and_normalized():
    emb = get_embedder("hash", dim=64)
    v1 = emb.embed(["hello world"])[0]
    v2 = emb.embed(["hello world"])[0]
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_end_to_end_ingest_search_and_sync(tmp_path):
    source = tmp_path / "src"
    store = tmp_path / "idx"
    write(str(source / "a.md"), "# Title A\n\nAlpha content about vectors and search.")
    write(str(source / "b.txt"), "Beta notes about incremental synchronization of the index.")

    pipe = Pipeline(str(store), embedder=get_embedder("hash", dim=128))
    report = pipe.ingest(str(source))
    assert report.added == 2
    assert report.chunks_written >= 2

    hits = pipe.search("incremental synchronization", top_k=3)
    assert hits
    assert any("synchronization" in h.text.lower() for h in hits)
    pipe.close()

    # --- update one file, delete another ---
    write(str(source / "a.md"), "# Title A\n\nAlpha content UPDATED with new material.")
    os.remove(str(source / "b.txt"))

    pipe = Pipeline(str(store), embedder=get_embedder("hash", dim=128))
    report2 = pipe.ingest(str(source))
    assert report2.updated == 1
    assert report2.deleted == 1

    status = pipe.status()
    # one active document remains (a.md); b.txt is marked deleted
    assert status["by_status"].get("indexed", {}).get("documents") == 1
    pipe.close()


def test_unchanged_files_are_not_reprocessed(tmp_path):
    source = tmp_path / "src"
    store = tmp_path / "idx"
    write(str(source / "a.md"), "# Stable\n\nThis content does not change between runs.")

    pipe = Pipeline(str(store), embedder=get_embedder("hash", dim=64))
    pipe.ingest(str(source))
    pipe.close()

    pipe = Pipeline(str(store), embedder=get_embedder("hash", dim=64))
    report = pipe.ingest(str(source))
    assert report.added == 0
    assert report.updated == 0
    assert report.deleted == 0
    pipe.close()
