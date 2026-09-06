"""Build a disease graph as of a pinned date, and record what that cost.

This is the first production path that can run a dated build. Until it
existed, ``SourceResolver`` took its date-to-release mapping as an injected
callable whose only implementations were lambdas inside tests: the whole
pinning apparatus was reachable from the test suite and from nowhere else.

Three things are recorded per disease, and each closes a success criterion
the sub-project had left open.

**Which extracts produced the numbers.** The experiment declares
``reads_snapshots``, so the runner cites every pinned extract -- release,
digest, extractor version, and whether its consensus direction was
synthesised -- in ``env.json``.

**What an unpinned source was not allowed to contribute.** The spec asks
for exclusions to be *recorded with the source that contributed them*,
not counted, on the reasoning that a silent filter is how frame leakage
creeps back after a refactor. A run that suddenly excludes four hundred
genes where it used to exclude twelve has had something change upstream,
and that must be legible in the record rather than absorbed into a log
line nobody reads.

**How long it took.** Building from extracts should beat the live path by
roughly an order of magnitude -- the live path makes eight sequential API
round trips per disease. That is a claim, and a claim in this project
needs a measurement behind it, so the timing is recorded rather than
asserted. Set ``COMPARE_AGAINST_LIVE`` to measure the ratio; it is off by
default because a live build takes minutes per disease and hits eight
external services.
"""

from __future__ import annotations

import time
from pathlib import Path

from neorx.core.graph.graph_builder import build_disease_graph
from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.releases import TIME_POINTS, release_for
from neorx.snapshots.resolver import SourceResolver

# Matches the default `--root` of `neorx snapshot build`.
STORE_ROOT = Path("snapshots")

# Diseases to build, as (name, EFO id). A dated build requires the id:
# the extract is keyed on it and carries no disease names, and resolving a
# name through the OpenTargets API would be a live call inside a dated
# build. These are three of the original seven-disease benchmark, kept so
# the new numbers can be set beside the old ones -- with every caveat in
# the corrigendum audit attached.
DISEASES: tuple[tuple[str, str], ...] = (
    ("HIV infection", "EFO_0000764"),
    ("Alzheimer disease", "MONDO_0004975"),
    ("type 2 diabetes mellitus", "MONDO_0005148"),
)

# Off by default: a live build is minutes per disease across eight
# external services. Turn it on to record the ratio criterion 8 asks for.
COMPARE_AGAINST_LIVE = False


def dated_resolver(store: SnapshotStore) -> SourceResolver:
    """A resolver over the pinned release table.

    The live mapping is empty on purpose. A dated build must not be able
    to reach a live client for a pinned source even by accident, and an
    empty mapping turns that into a ``KeyError`` at the point of the
    mistake rather than a graph that silently mixes eras.
    """
    return SourceResolver(store=store, release_for=release_for, live={})


def build_row(
    store: SnapshotStore,
    disease: str,
    disease_id: str,
    as_of: str,
) -> dict:
    """One dated build, timed, with its exclusions.

    Pure enough to unit-test: everything it needs arrives as an argument,
    and it returns a row rather than writing one.
    """
    started = time.perf_counter()
    graph = build_disease_graph(
        disease,
        as_of=as_of,
        disease_id=disease_id,
        resolver=dated_resolver(store),
        use_cache=False,
    )
    snapshot_seconds = time.perf_counter() - started

    live_seconds = None
    if COMPARE_AGAINST_LIVE:
        started = time.perf_counter()
        build_disease_graph(disease, use_cache=False)
        live_seconds = time.perf_counter() - started

    point = TIME_POINTS[as_of]
    return {
        "disease": disease,
        "disease_id": disease_id,
        "as_of": as_of,
        "opentargets_release": point["opentargets"],
        "omnipath_release": point["omnipath"],
        # Recorded per row because it is not what the time point's name
        # implies: the 2025-06 point's regulatory data is from 2023-07-28,
        # OmniPath having published no refresh in between. A reader
        # comparing time points needs that in the same table as the
        # numbers, not in a footnote.
        "omnipath_data_as_of": point["omnipath_data_as_of"],
        "n_nodes": len(graph.nodes),
        "n_genes": graph.n_genes,
        "n_edges": len(graph.edges),
        "n_frame_exclusions": len(graph.frame_exclusions),
        # The exclusions themselves, not a count. A count cannot tell you
        # WHICH genes stopped arriving or from which source, which is the
        # whole diagnostic value.
        "frame_exclusions": graph.frame_exclusions,
        "snapshot_seconds": round(snapshot_seconds, 3),
        "live_seconds": None if live_seconds is None else round(live_seconds, 3),
        "speedup": (
            None
            if live_seconds is None or snapshot_seconds == 0
            else round(live_seconds / snapshot_seconds, 1)
        ),
    }


@experiment(
    name="dated-build",
    reads_snapshots=True,
    volatile_fields=("snapshot_seconds", "live_seconds", "speedup"),
    help="Build disease graphs from pinned releases; record exclusions and timing.",
)
def dated_build(record: RunRecord) -> None:
    """Build every disease at every pinned time point.

    One row per (disease, time point), appended as each completes, so a
    release whose extract has not been built yet costs that row rather
    than the run -- the same reason sub-project 2 writes rows
    incrementally.
    """
    store = SnapshotStore(STORE_ROOT)
    for as_of in sorted(TIME_POINTS):
        for disease, disease_id in DISEASES:
            record.append_row(build_row(store, disease, disease_id, as_of))
