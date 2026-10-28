# doc-pipeline

A modular document preprocessing pipeline for building a retrieval index (RAG).

It ingests mixed document types (PDF, HTML, Markdown, plain text, and more),
extracts and normalizes their content, detects language, preserves structure
(headings, page numbers, tables), removes duplicates, splits content into
chunks, generates multilingual embeddings, and stores everything in a vector
store alongside a metadata database. It also performs **incremental sync**:
new, updated, and deleted source files are detected on each run so the index
stays consistent with the source data.

## Pipeline stages

```
ingest -> detect type -> extract -> normalize + validate -> detect language
  -> preserve structure -> deduplicate -> chunk -> embed -> store
                                                             |
                                        change detection (new/updated/deleted)
```

| Stage            | Module                         | What it does                                           |
|------------------|--------------------------------|--------------------------------------------------------|
| Type detection   | `pipeline/detect.py`           | Identifies file type by extension + content sniffing   |
| Extraction       | `pipeline/extractors/`         | Pluggable parser per file type                         |
| Normalization    | `pipeline/normalize.py`        | Unicode normalization, whitespace, validation          |
| Language          | `pipeline/language.py`         | Detects the dominant language of the text              |
| Structure        | `pipeline/structure.py`        | Captures headings, page numbers, tables                |
| Deduplication    | `pipeline/dedup.py`            | Drops exact and near-duplicate chunks                  |
| Chunking         | `pipeline/chunk.py`            | Splits into overlapping, structure-aware chunks        |
| Embeddings       | `pipeline/embeddings.py`       | Generates multilingual vectors (pluggable provider)    |
| Metadata DB      | `pipeline/metadata_db.py`      | SQLite: id, source, language, version, checksum, status|
| Vector store     | `pipeline/vector_store.py`     | Stores vectors + metadata, supports similarity search  |
| Change detection | `pipeline/sync.py`             | Diffs source dir against DB (added/updated/deleted)    |
| Orchestration    | `pipeline/runner.py`           | Wires the stages together                              |

## Requirements

- Python 3.9+
- No external services required to run the defaults. The repo works offline
  out of the box using lightweight built-in implementations.

Optional, higher-quality backends (see `requirements-optional.txt`):

- `pypdf` for real PDF text extraction
- `beautifulsoup4` + `lxml` for robust HTML parsing
- `sentence-transformers` for real multilingual embeddings
- `langdetect` for stronger language detection

The pipeline **degrades gracefully**: if an optional library is missing, a
built-in fallback is used so a fresh clone always runs.

## Quick start

```bash
git clone <your-repo-url> doc-pipeline
cd doc-pipeline

# (optional) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# core deps only (optional extras for better quality)
pip install -r requirements-optional.txt   # optional

# run against the bundled sample documents
python -m pipeline.cli ingest --source sample_docs --store .index

# search the index
python -m pipeline.cli search "how does chunking work" --store .index

# show pipeline/index status
python -m pipeline.cli status --store .index
```

Running `ingest` again after editing, adding, or deleting files in the source
directory will sync the index incrementally.

## Project layout

```
doc-pipeline/
├── pipeline/
│   ├── __init__.py
│   ├── cli.py              # command-line entry point
│   ├── runner.py           # orchestrates the pipeline
│   ├── detect.py           # file-type detection
│   ├── normalize.py        # text normalization + validation
│   ├── language.py         # language detection
│   ├── structure.py        # structure extraction (headings, pages, tables)
│   ├── dedup.py            # deduplication
│   ├── chunk.py            # chunking
│   ├── embeddings.py       # embedding providers
│   ├── metadata_db.py      # SQLite metadata database
│   ├── vector_store.py     # vector storage + search
│   ├── sync.py             # change detection
│   ├── models.py           # shared dataclasses
│   └── extractors/
│       ├── __init__.py     # extractor registry
│       ├── base.py
│       ├── text.py
│       ├── markdown.py
│       ├── html.py
│       └── pdf.py
├── sample_docs/            # a few documents to try the pipeline on
├── tests/
├── requirements-optional.txt
├── .gitignore
└── README.md
```

## How change detection works

Each source file is hashed (SHA-256). The metadata DB records the checksum and
a monotonically increasing version per document. On each run `sync` compares the
current source directory to the DB:

- **New**: present on disk, absent in DB -> extracted and indexed.
- **Updated**: checksum changed -> old chunks/vectors removed, re-indexed,
  version incremented.
- **Deleted**: present in DB, absent on disk -> chunks/vectors purged, status
  set to `deleted`.

## Swapping in production backends

- **Embeddings**: set `--embedder sentence-transformers` (requires
  `sentence-transformers`). The default `hash` embedder is deterministic and
  offline, useful for tests and demos but not for semantic quality.
- **Vector DB**: `pipeline/vector_store.py` implements a small local store.
  The `VectorStore` interface is intentionally narrow (`upsert`, `delete`,
  `search`) so it can be backed by FAISS, Qdrant, pgvector, etc.

## License

MIT — see `LICENSE`.
