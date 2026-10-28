"""Command-line interface for the document pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from .embeddings import get_embedder
from .runner import Pipeline


def _build_pipeline(args) -> Pipeline:
    embedder_kwargs = {}
    if args.embedder == "hash" and getattr(args, "dim", None):
        embedder_kwargs["dim"] = args.dim
    embedder = get_embedder(args.embedder, **embedder_kwargs)
    return Pipeline(
        args.store,
        embedder=embedder,
        target_chars=args.chunk_size,
        overlap_chars=args.overlap,
        near_dedup_threshold=args.dedup_threshold,
    )


def cmd_ingest(args) -> int:
    pipe = _build_pipeline(args)
    try:
        exts = set(args.ext) if args.ext else None
        report = pipe.ingest(args.source, extensions=exts)
    finally:
        pipe.close()

    print("Ingest complete:")
    print(f"  added:          {report.added}")
    print(f"  updated:        {report.updated}")
    print(f"  deleted:        {report.deleted}")
    print(f"  chunks written: {report.chunks_written}")
    print(f"  chunks removed: {report.chunks_removed}")
    if report.failed:
        print(f"  failed:         {len(report.failed)}")
        for src, err in report.failed:
            print(f"    - {src}: {err}")
    return 0


def cmd_search(args) -> int:
    pipe = _build_pipeline(args)
    try:
        hits = pipe.search(args.query, top_k=args.top_k)
    finally:
        pipe.close()

    if args.json:
        print(json.dumps(
            [{"chunk_id": h.chunk_id, "score": h.score,
              "text": h.text, "metadata": h.metadata} for h in hits],
            indent=2, ensure_ascii=False,
        ))
        return 0

    if not hits:
        print("No results. Has the index been built? Run 'ingest' first.")
        return 0

    for rank, hit in enumerate(hits, start=1):
        src = hit.metadata.get("source", "?")
        page = hit.metadata.get("structure", {}).get("page")
        heading = hit.metadata.get("structure", {}).get("heading")
        loc = f" p.{page}" if page else ""
        head = f" | {heading}" if heading else ""
        snippet = hit.text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        print(f"#{rank}  score={hit.score:.4f}  {src}{loc}{head}")
        print(f"    {snippet}\n")
    return 0


def cmd_status(args) -> int:
    pipe = _build_pipeline(args)
    try:
        st = pipe.status()
    finally:
        pipe.close()
    print(json.dumps(st, indent=2, ensure_ascii=False))
    return 0


def _add_global_options(parser: argparse.ArgumentParser) -> None:
    """Shared options, attached to both the top parser and each subparser so
    they may appear before or after the subcommand name."""
    parser.add_argument("--store", default=".index",
                        help="index/store directory (default: .index)")
    parser.add_argument("--embedder", default="hash",
                        choices=["hash", "sentence-transformers", "st"],
                        help="embedding backend (default: hash)")
    parser.add_argument("--dim", type=int, default=256, help="hash embedder dimension")
    parser.add_argument("--chunk-size", type=int, default=1000, help="target chunk size in chars")
    parser.add_argument("--overlap", type=int, default=150, help="chunk overlap in chars")
    parser.add_argument("--dedup-threshold", type=float, default=0.9,
                        help="near-duplicate Jaccard threshold (1.0 disables)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline", description="Document preprocessing pipeline")
    _add_global_options(p)

    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="index a source directory (incremental)")
    _add_global_options(pi)
    pi.add_argument("--source", required=True, help="directory of source documents")
    pi.add_argument("--ext", nargs="*", help="limit to these extensions, e.g. --ext .md .pdf")
    pi.set_defaults(func=cmd_ingest)

    ps = sub.add_parser("search", help="search the index")
    _add_global_options(ps)
    ps.add_argument("query", help="query text")
    ps.add_argument("--top-k", type=int, default=5)
    ps.add_argument("--json", action="store_true", help="output JSON")
    ps.set_defaults(func=cmd_search)

    pst = sub.add_parser("status", help="show index/pipeline status")
    _add_global_options(pst)
    pst.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
