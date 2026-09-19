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

from collections.abc import Callable
from pathlib import Path

import polars as pl

__all__ = ["SNAPSHOTS_DIR", "SnapshotMissingError", "SnapshotStore"]

# This repository's snapshot store. Anchored to the repo root rather than
# the cwd, so `neorx snapshot build` writes where the experiments read: a
# relative default meant a build run from a subdirectory produced a store
# nothing would read, which is the write-side half of the defect that let
# a run cite a manifest describing a different store.
SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "snapshots"


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

    def __init__(
        self,
        root: Path,
        *,
        on_read: Callable[[str, str, Path], None] | None,
    ) -> None:
        """``on_read(source, release, path)`` runs after every extract read.

        It is how a run record learns what a run actually read, so it can
        cite exactly that -- from this store's own manifest, rather than
        from a second path that might name a different store.

        Required, with no default, deliberately. When it defaulted to
        ``None`` an experiment could read one extract through
        ``RunRecord.snapshot_store`` and another through a store it built
        itself, and finish `complete` and `citable` while citing only the
        first: the runner's guard fires when NO read was tracked, not when
        some were missed. It cannot see a store it was never told about,
        so the construction has to say. ``on_read=None`` is still
        available -- for the CLI and for unit tests of the reader -- but
        it is now written down rather than inherited.
        """
        self.root = Path(root)
        self._on_read = on_read

    @property
    def manifest_path(self) -> Path:
        """The manifest describing the extracts under this root.

        Derived from the root rather than configured beside it, so the
        extracts a run reads and the manifest it cites cannot come from two
        different places.
        """
        return self.root / "manifest.toml"

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
        frame = pl.read_parquet(path)
        if self._on_read is not None:
            # The path too: a citation has to be checked against the bytes
            # that were actually read, not just against the manifest.
            self._on_read(source, release, path)
        return frame

    def associations(self, release: str) -> pl.DataFrame:
        """Canonical association rows for an OpenTargets release."""
        return self._read("opentargets", release)

    def interactions(self, release: str) -> pl.DataFrame:
        """Canonical interaction rows for an OmniPath archive date."""
        return self._read("omnipath", release)
