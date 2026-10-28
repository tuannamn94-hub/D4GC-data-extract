"""Pipeline orchestration.

Wires detection -> extraction -> normalization/validation -> language ->
chunking -> dedup -> embedding -> storage, and drives incremental sync.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

from . import detect, normalize, language as lang_mod
from .chunk import chunk_document
from .dedup import deduplicate
from .embeddings import Embedder, get_embedder
from .extractors import get_extractor, PdfExtractionUnavailable
from .metadata_db import MetadataDB
from .models import Document, STATUS_INDEXED, STATUS_FAILED
from .sync import build_plan, file_checksum, SyncPlan
from .vector_store import VectorStore, SearchHit


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (source, error)
    chunks_written: int = 0
    chunks_removed: int = 0

    def as_dict(self) -> dict:
        return {
            "added": self.added,
            "updated": self.updated,
            "deleted": self.deleted,
            "failed": self.failed,
            "chunks_written": self.chunks_written,
            "chunks_removed": self.chunks_removed,
        }


class Pipeline:
    def __init__(
        self,
        store_dir: str,
        *,
        embedder: Embedder | None = None,
        target_chars: int = 1000,
        overlap_chars: int = 150,
        near_dedup_threshold: float = 0.9,
    ):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self.db = MetadataDB(os.path.join(store_dir, "metadata.db"))
        self.vectors = VectorStore(os.path.join(store_dir, "vectors"))
        self.embedder = embedder or get_embedder("hash")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars
        self.near_dedup_threshold = near_dedup_threshold

    def close(self) -> None:
        self.db.close()

    # -- single-document processing ----------------------------------------
    def process_file(self, path: str, version: int) -> Document:
        """Run extraction through language detection for one file."""
        file_type = detect.detect_type(path)
        extractor = get_extractor(file_type)
        if extractor is None:
            raise ValueError(f"unsupported file type '{file_type}' for {path}")

        result = extractor.extract(path)
        text = normalize.normalize_text(result.text)
        normalize.validate_text(text)
        language = lang_mod.detect_language(text)
        checksum = file_checksum(path)
        doc_id = uuid.uuid5(uuid.NAMESPACE_URL, os.path.abspath(path)).hex

        return Document(
            doc_id=doc_id,
            source=path,
            file_type=file_type,
            text=text,
            checksum=checksum,
            language=language,
            version=version,
            structure=result.structure,
        )

    def _index_document(self, doc: Document, report: IngestReport, *, is_update: bool) -> None:
        # Remove stale vectors/chunks on update before re-adding.
        if is_update:
            old_ids = self.db.chunk_ids_for(doc.doc_id)
            report.chunks_removed += self.vectors.delete(old_ids)

        chunks = chunk_document(
            doc, target_chars=self.target_chars, overlap_chars=self.overlap_chars
        )
        chunks = deduplicate(chunks, near_threshold=self.near_dedup_threshold)

        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            self.vectors.upsert(
                [c.chunk_id for c in chunks],
                vectors,
                [c.text for c in chunks],
                [c.to_metadata() for c in chunks],
            )

        self.db.upsert_document(
            doc_id=doc.doc_id,
            source=doc.source,
            file_type=doc.file_type,
            language=doc.language,
            checksum=doc.checksum,
            version=doc.version,
            status=STATUS_INDEXED,
            chunk_count=len(chunks),
        )
        self.db.replace_chunks(doc.doc_id, chunks)
        report.chunks_written += len(chunks)

    # -- incremental ingest -------------------------------------------------
    def ingest(self, source_dir: str, extensions: set[str] | None = None) -> IngestReport:
        plan: SyncPlan = build_plan(source_dir, self.db, extensions)
        report = IngestReport()

        for path in plan.added:
            self._safe_index(path, version=1, is_update=False, report=report)
            if (path, ) not in report.failed:
                report.added += 1

        for path in plan.updated:
            row = self.db.get_document(path)
            next_version = (row["version"] + 1) if row else 1
            self._safe_index(path, version=next_version, is_update=True, report=report)
            report.updated += 1

        for source in plan.deleted:
            doc_id = self.db.get_document(source)
            if doc_id:
                removed = self.vectors.delete(self.db.chunk_ids_for(doc_id["doc_id"]))
                report.chunks_removed += removed
            self.db.mark_deleted(source)
            report.deleted += 1

        # Correct the added counter for any failures recorded during add.
        report.added -= sum(1 for src, _ in report.failed if self._was_add(src, plan))
        return report

    def _safe_index(self, path: str, *, version: int, is_update: bool, report: IngestReport) -> None:
        try:
            doc = self.process_file(path, version=version)
            self._index_document(doc, report, is_update=is_update)
        except PdfExtractionUnavailable as exc:
            report.failed.append((path, str(exc)))
            self._record_failure(path)
        except (normalize.ValidationError, ValueError) as exc:
            report.failed.append((path, str(exc)))
            self._record_failure(path)
        except Exception as exc:  # pragma: no cover - defensive
            report.failed.append((path, f"{type(exc).__name__}: {exc}"))
            self._record_failure(path)

    def _record_failure(self, path: str) -> None:
        try:
            self.db.upsert_document(
                doc_id=uuid.uuid5(uuid.NAMESPACE_URL, os.path.abspath(path)).hex,
                source=path,
                file_type=detect.detect_type(path),
                language=None,
                checksum=file_checksum(path),
                version=1,
                status=STATUS_FAILED,
                chunk_count=0,
            )
        except Exception:
            pass

    @staticmethod
    def _was_add(src: str, plan: SyncPlan) -> bool:
        return src in plan.added

    # -- query --------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        qvec = self.embedder.embed([query])[0]
        return self.vectors.search(qvec, top_k=top_k)

    def status(self) -> dict:
        stats = self.db.stats()
        stats["vector_count"] = len(self.vectors)
        stats["embedder"] = self.embedder.name
        stats["embedding_dim"] = self.embedder.dim
        return stats
