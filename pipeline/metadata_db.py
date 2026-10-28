"""SQLite-backed metadata database.

Tracks one row per document with id, source, language, version, checksum and
processing status, plus a chunk table linking chunks to documents. SQLite keeps
the repo clone-and-run with no external service.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Iterable

from .models import STATUS_INDEXED, STATUS_DELETED

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    source      TEXT NOT NULL UNIQUE,
    file_type   TEXT,
    language    TEXT,
    version     INTEGER NOT NULL DEFAULT 1,
    checksum    TEXT NOT NULL,
    status      TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    checksum   TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class MetadataDB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MetadataDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- document lookups ---------------------------------------------------
    def get_document(self, source: str) -> sqlite3.Row | None:
        cur = self._conn.execute("SELECT * FROM documents WHERE source = ?", (source,))
        return cur.fetchone()

    def all_sources(self, include_deleted: bool = False) -> dict[str, sqlite3.Row]:
        q = "SELECT * FROM documents"
        if not include_deleted:
            q += " WHERE status != ?"
            rows = self._conn.execute(q, (STATUS_DELETED,)).fetchall()
        else:
            rows = self._conn.execute(q).fetchall()
        return {r["source"]: r for r in rows}

    # -- upserts ------------------------------------------------------------
    def upsert_document(
        self,
        *,
        doc_id: str,
        source: str,
        file_type: str,
        language: str | None,
        checksum: str,
        version: int,
        status: str = STATUS_INDEXED,
        chunk_count: int = 0,
    ) -> None:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO documents
                (doc_id, source, file_type, language, version, checksum,
                 status, chunk_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                doc_id=excluded.doc_id,
                file_type=excluded.file_type,
                language=excluded.language,
                version=excluded.version,
                checksum=excluded.checksum,
                status=excluded.status,
                chunk_count=excluded.chunk_count,
                updated_at=excluded.updated_at
            """,
            (doc_id, source, file_type, language, version, checksum,
             status, chunk_count, now, now),
        )
        self._conn.commit()

    def replace_chunks(self, doc_id: str, chunks: Iterable[Any]) -> None:
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, idx, checksum) VALUES (?, ?, ?, ?)",
            [(c.chunk_id, c.doc_id, c.index, c.checksum) for c in chunks],
        )
        self._conn.commit()

    def chunk_ids_for(self, doc_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    def mark_deleted(self, source: str) -> str | None:
        row = self.get_document(source)
        if not row:
            return None
        doc_id = row["doc_id"]
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.execute(
            "UPDATE documents SET status=?, chunk_count=0, updated_at=? WHERE doc_id=?",
            (STATUS_DELETED, time.time(), doc_id),
        )
        self._conn.commit()
        return doc_id

    def set_status(self, source: str, status: str) -> None:
        self._conn.execute(
            "UPDATE documents SET status=?, updated_at=? WHERE source=?",
            (status, time.time(), source),
        )
        self._conn.commit()

    # -- reporting ----------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) n, COALESCE(SUM(chunk_count),0) chunks "
            "FROM documents GROUP BY status"
        )
        by_status = {r["status"]: {"documents": r["n"], "chunks": r["chunks"]} for r in cur}
        total_docs = sum(v["documents"] for v in by_status.values())
        total_chunks = sum(v["chunks"] for v in by_status.values())
        langs = self._conn.execute(
            "SELECT language, COUNT(*) n FROM documents "
            "WHERE status != ? GROUP BY language", (STATUS_DELETED,)
        ).fetchall()
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "by_status": by_status,
            "by_language": {r["language"]: r["n"] for r in langs},
        }
