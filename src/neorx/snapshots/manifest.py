"""
What a snapshot records about itself.

A release identifier alone does not identify a derived extract. Two
extracts of OpenTargets 18.06 produced by different extractor code are
different inputs to every number computed downstream, so the extractor
version is recorded beside the release id, and the digest is taken over
the extract actually written rather than the upstream file.

Keyed by (source, release), so re-running an extraction replaces its entry
instead of appending a second one.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import tomli_w

__all__ = ["SnapshotEntry", "digest_file", "read_manifest", "write_entry"]


@dataclass(frozen=True)
class SnapshotEntry:
    """One derived extract, and everything needed to identify it."""

    source: str
    release: str
    url: str
    sha256: str
    extractor_version: int
    rows: int
    synthesised_consensus: bool = False


def digest_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large extract does not load."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(manifest_path: Path) -> dict[tuple[str, str], SnapshotEntry]:
    """Read every entry, keyed by (source, release). Absent file: no entries."""
    path = Path(manifest_path)
    if not path.exists():
        return {}

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries: dict[tuple[str, str], SnapshotEntry] = {}
    for source, releases in (data.get("snapshot") or {}).items():
        for release, fields in releases.items():
            entry = SnapshotEntry(source=source, release=release, **fields)
            entries[(source, release)] = entry
    return entries


def write_entry(manifest_path: Path, entry: SnapshotEntry) -> None:
    """Insert or replace one entry, leaving every other entry untouched."""
    path = Path(manifest_path)
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    fields = asdict(entry)
    source = fields.pop("source")
    release = fields.pop("release")

    data.setdefault("snapshot", {}).setdefault(source, {})[release] = fields
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
