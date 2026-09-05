"""
Choosing between a live client and a snapshot reader.

With no date, every source resolves to the live client it always used and
behaviour is unchanged -- interactive single-disease use is unaffected by
this whole sub-project.

With a date, the sources feeding the causal subgraph resolve to snapshot
readers, and a source with no snapshot at that date is refused by name.
The refusal matters more than it looks: a dated run that quietly mixed in
current data would produce exactly the anachronistic graph this
sub-project exists to prevent, and would do it invisibly.

Sources that contribute only associational edges stay live even on a dated
run. Identification filters those edges out before it runs, so pinning
them would buy rigour against a leak that does not exist.
"""

from __future__ import annotations

from typing import Callable, Protocol

import polars as pl

from neorx.core.sources.snapshot_sources import snapshot_reader

__all__ = ["SourceResolver", "UnpinnedSourceError"]

# Sources whose data reaches the causal subgraph, and therefore must be
# pinned on a dated run. Everything else contributes edges that
# graph_semantics filters out before identification.
PINNED_SOURCES = frozenset({"opentargets", "omnipath"})


class UnpinnedSourceError(RuntimeError):
    """A dated run asked for a source that has no snapshot at that date."""


class _Store(Protocol):
    def has(self, source: str, release: str) -> bool: ...

    def associations(self, release: str) -> pl.DataFrame: ...

    def interactions(self, release: str) -> pl.DataFrame: ...


class SourceResolver:
    """Maps a source name plus an optional date to a reader.

    ``release_for(source, as_of)`` maps a date to a release identifier, so
    the date-to-release table lives in one place rather than at every call
    site.
    """

    def __init__(
        self,
        store: _Store,
        release_for: Callable[[str, str], str],
        live: dict[str, Callable[..., object]],
    ) -> None:
        self._store = store
        self._release_for = release_for
        self._live = live

    def resolve(self, source: str, as_of: str | None) -> Callable[..., object]:
        """Return the reader for this source at this date.

        Undated, or unpinned, this is the live client itself. Dated and
        pinned, it is a snapshot reader bound to the release, returning
        the same ``tuple[list[GraphNode], list[GraphEdge]]`` the live
        client returns -- so the caller assembles a dated graph through
        the same code path as a live one and cannot accidentally hold a
        reader it does not know how to call.

        Raises ``UnpinnedSourceError`` if the date requires a snapshot that
        does not exist. It does not fall back to the live client.
        """
        if as_of is None or source not in PINNED_SOURCES:
            return self._live[source]

        release = self._release_for(source, as_of)
        if not self._store.has(source, release):
            raise UnpinnedSourceError(
                f"A run dated {as_of} needs the {source} snapshot for release "
                f"{release!r}, which has not been built. Build it with "
                f"`neorx snapshot build {source} {release}`. This will not "
                f"fall back to live data: doing so would silently mix current "
                f"evidence into a dated graph."
            )

        return snapshot_reader(source, self._store, release)
