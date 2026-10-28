"""Build the result database from the D4GC dataset.

The D4GC archive ships two parallel folders:

* ``Original/`` — the source documents (``*.pdf`` and ``*.html``)
* ``Parsed/``   — one ``*.md`` per original, already extracted to text

We feed the *parsed* markdown through the pipeline (normalize -> validate ->
language -> chunk -> dedup -> embed -> store) because it is clean, extracted
text and needs no PDF/HTML backend. For each parsed file we look up its matching
original so the metadata database records the *true* source path and file type
(pdf / html) rather than "markdown".

Usage:
    python scripts/ingest_d4gc.py \
        --data "D4GC data/extracted/D4GC data" \
        --store d4gc_index
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make the package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import detect, normalize, language as lang_mod  # noqa: E402
from pipeline.chunk import chunk_document  # noqa: E402
from pipeline.dedup import deduplicate  # noqa: E402
from pipeline.embeddings import get_embedder  # noqa: E402
from pipeline.extractors.markdown import MarkdownExtractor  # noqa: E402
from pipeline.models import Document, STATUS_INDEXED, STATUS_FAILED  # noqa: E402
from pipeline.runner import Pipeline  # noqa: E402
from pipeline.sync import file_checksum  # noqa: E402
import uuid  # noqa: E402


def build_original_index(original_dir: str) -> dict[str, str]:
    """Map document id -> original file path (keeps real type: pdf/html)."""
    index: dict[str, str] = {}
    if not os.path.isdir(original_dir):
        return index
    for name in os.listdir(original_dir):
        doc_id = os.path.splitext(name)[0]
        index[doc_id] = os.path.join(original_dir, name)
    return index


def ingest_d4gc(data_dir: str, store_dir: str, embedder_name: str, dim: int,
                limit: int | None) -> None:
    parsed_dir = os.path.join(data_dir, "Parsed")
    original_dir = os.path.join(data_dir, "Original")

    if not os.path.isdir(parsed_dir):
        raise SystemExit(f"Parsed directory not found: {parsed_dir}")

    originals = build_original_index(original_dir)

    embedder_kwargs = {"dim": dim} if embedder_name == "hash" else {}
    pipe = Pipeline(store_dir, embedder=get_embedder(embedder_name, **embedder_kwargs))
    md_extractor = MarkdownExtractor()

    parsed_files = sorted(f for f in os.listdir(parsed_dir) if f.endswith(".md"))
    if limit:
        parsed_files = parsed_files[:limit]

    total = len(parsed_files)
    indexed = 0
    failed = 0
    chunks_total = 0
    start = time.time()

    print(f"Processing {total} parsed documents from {parsed_dir}")

    for i, fname in enumerate(parsed_files, start=1):
        doc_id_name = os.path.splitext(fname)[0]
        parsed_path = os.path.join(parsed_dir, fname)

        # Attribute to the real original source + type when available.
        original_path = originals.get(doc_id_name)
        if original_path:
            source = original_path
            file_type = detect.detect_type(original_path)
            checksum = file_checksum(original_path)
        else:
            source = parsed_path
            file_type = detect.TYPE_MARKDOWN
            checksum = file_checksum(parsed_path)

        try:
            result = md_extractor.extract(parsed_path)
            text = normalize.normalize_text(result.text)
            normalize.validate_text(text)
            language = lang_mod.detect_language(text)

            doc = Document(
                doc_id=uuid.uuid5(uuid.NAMESPACE_URL, os.path.abspath(source)).hex,
                source=source,
                file_type=file_type,
                text=text,
                checksum=checksum,
                language=language,
                version=1,
                structure=result.structure,
            )

            chunks = chunk_document(
                doc, target_chars=pipe.target_chars, overlap_chars=pipe.overlap_chars
            )
            chunks = deduplicate(chunks, near_threshold=pipe.near_dedup_threshold)

            if chunks:
                vectors = pipe.embedder.embed([c.text for c in chunks])
                pipe.vectors.upsert(
                    [c.chunk_id for c in chunks],
                    vectors,
                    [c.text for c in chunks],
                    [c.to_metadata() for c in chunks],
                )

            pipe.db.upsert_document(
                doc_id=doc.doc_id, source=doc.source, file_type=doc.file_type,
                language=doc.language, checksum=doc.checksum, version=1,
                status=STATUS_INDEXED, chunk_count=len(chunks),
            )
            pipe.db.replace_chunks(doc.doc_id, chunks)
            indexed += 1
            chunks_total += len(chunks)
        except Exception as exc:  # record the failure, keep going
            failed += 1
            try:
                pipe.db.upsert_document(
                    doc_id=uuid.uuid5(uuid.NAMESPACE_URL, os.path.abspath(source)).hex,
                    source=source, file_type=file_type, language=None,
                    checksum=checksum, version=1, status=STATUS_FAILED, chunk_count=0,
                )
            except Exception:
                pass
            print(f"  [FAIL] {fname}: {type(exc).__name__}: {exc}")

        if i % 50 == 0 or i == total:
            elapsed = time.time() - start
            print(f"  {i}/{total} docs | {chunks_total} chunks | "
                  f"{failed} failed | {elapsed:.1f}s")

    pipe.close()
    print("\nDone.")
    print(f"  indexed: {indexed}")
    print(f"  failed:  {failed}")
    print(f"  chunks:  {chunks_total}")
    print(f"  store:   {store_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest the D4GC dataset into a result database")
    p.add_argument("--data", required=True,
                   help="path to the extracted 'D4GC data' dir (contains Original/ and Parsed/)")
    p.add_argument("--store", default="d4gc_index", help="output index/store directory")
    p.add_argument("--embedder", default="hash", choices=["hash", "sentence-transformers", "st"])
    p.add_argument("--dim", type=int, default=256, help="hash embedder dimension")
    p.add_argument("--limit", type=int, default=None, help="process only the first N docs (debug)")
    args = p.parse_args(argv)
    ingest_d4gc(args.data, args.store, args.embedder, args.dim, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
