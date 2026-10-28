"""Change detection between a source directory and the metadata DB."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

from .metadata_db import MetadataDB


@dataclass
class SyncPlan:
    added: list[str] = field(default_factory=list)      # sources to index fresh
    updated: list[str] = field(default_factory=list)    # sources whose checksum changed
    deleted: list[str] = field(default_factory=list)    # sources gone from disk
    unchanged: list[str] = field(default_factory=list)  # nothing to do

    def is_empty(self) -> bool:
        return not (self.added or self.updated or self.deleted)

    def summary(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "updated": len(self.updated),
            "deleted": len(self.deleted),
            "unchanged": len(self.unchanged),
        }


def file_checksum(path: str, chunk_size: int = 65536) -> str:
    """SHA-256 of the raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def scan_source(source_dir: str, extensions: set[str] | None = None) -> list[str]:
    """Return a sorted list of file paths under ``source_dir``."""
    found: list[str] = []
    for root, _dirs, files in os.walk(source_dir):
        for name in files:
            if name.startswith("."):
                continue
            if extensions and os.path.splitext(name)[1].lower() not in extensions:
                continue
            found.append(os.path.join(root, name))
    return sorted(found)


def build_plan(source_dir: str, db: MetadataDB, extensions: set[str] | None = None) -> SyncPlan:
    """Compare the source directory with the DB and produce a sync plan."""
    plan = SyncPlan()
    on_disk = scan_source(source_dir, extensions)
    known = db.all_sources(include_deleted=False)
    seen: set[str] = set()

    for path in on_disk:
        seen.add(path)
        row = known.get(path)
        if row is None:
            plan.added.append(path)
            continue
        if file_checksum(path) != row["checksum"]:
            plan.updated.append(path)
        else:
            plan.unchanged.append(path)

    # Anything known but no longer on disk is a deletion.
    for source in known:
        if source not in seen:
            plan.deleted.append(source)

    return plan
