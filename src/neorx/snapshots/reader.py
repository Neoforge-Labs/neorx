"""
Reading derived extracts back off disk.

This layer knows the on-disk layout and nothing else. By the time a frame
reaches it, which upstream era produced it is invisible -- that is the
whole point of extracting to a canonical schema.

A missing snapshot raises rather than returning an empty frame. An empty
frame reads downstream as "this release recorded no evidence for that
disease", which is a finding; a missing extract is a configuration
mistake, and the two must not be confusable.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

__all__ = ["SnapshotMissingError", "SnapshotStore"]


class SnapshotMissingError(FileNotFoundError):
    """No extract exists for a requested source and release."""


class SnapshotStore:
    """Derived extracts under a root directory.

    Layout::

        <root>/opentargets/<release>/associations.parquet
        <root>/omnipath/<release>/interactions.parquet
        <root>/manifest.toml
    """

    _FILES = {
        "opentargets": "associations.parquet",
        "omnipath": "interactions.parquet",
    }

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, source: str, release: str) -> Path:
        return self.root / source / release / self._FILES[source]

    def has(self, source: str, release: str) -> bool:
        """Whether an extract exists, without raising."""
        if source not in self._FILES:
            return False
        return self._path(source, release).exists()

    def _read(self, source: str, release: str) -> pl.DataFrame:
        path = self._path(source, release)
        if not path.exists():
            raise SnapshotMissingError(
                f"No {source} snapshot for release {release!r} at {path}. "
                f"Build it with `neorx snapshot build {source} {release}`. "
                f"A dated run will not fall back to live data."
            )
        return pl.read_parquet(path)

    def associations(self, release: str) -> pl.DataFrame:
        """Canonical association rows for an OpenTargets release."""
        return self._read("opentargets", release)

    def interactions(self, release: str) -> pl.DataFrame:
        """Canonical interaction rows for an OmniPath archive date."""
        return self._read("omnipath", release)
