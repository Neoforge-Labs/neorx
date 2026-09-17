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

import polars as pl

from neorx.core.graph.graph_builder import build_disease_graph
from neorx.experiments.record import SNAPSHOTS_DIR, RunRecord
from neorx.experiments.registry import experiment
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.releases import TIME_POINTS, release_for
from neorx.snapshots.resolver import SourceResolver

# The repository's snapshot store, anchored to the repo root rather than
# the cwd. A relative path let a run launched from a subdirectory read one
# store while its record cited another's manifest.
STORE_ROOT = SNAPSHOTS_DIR


class UnknownDiseaseForRelease(KeyError):
    """A disease id the pinned release does not carry.

    Raised rather than recorded as zeros: an empty result reads as a
    finding about the disease, and this is a fact about the id.
    """


# Diseases to build, as (name, candidate ids). A dated build requires an
# id: the extract is keyed on one and carries no disease names, and
# resolving a name through the OpenTargets API would be a live call inside
# a dated build.
#
# One id per disease does not work, because OpenTargets RE-KEYS diseases
# between releases. Checked against the live API (release 26.06):
# EFO_0000764 -- the id this sub-project used throughout -- no longer
# resolves at all, and HIV is now MONDO_0005109, whose dbXRefs list
# "EFO:0000764". So the two are the same disease under different keys, and
# which key a given release uses is a property of that release.
#
# Rather than guess per era, `build_row` resolves the id against the
# release's own extract: exactly one candidate present is the id, none is
# a refusal, and more than one is an ambiguity it also refuses. The pinned
# data is the authority, so no era-to-id table can go stale.
#
# Only HIV ships configured. Alzheimer's and type 2 diabetes are held back
# deliberately: their MONDO ids (MONDO_0004975, MONDO_0005148) are
# verified in 26.06, but neither lists an EFO cross-reference, so their
# pre-MONDO keys cannot be established from the API and inventing one
# risks matching a DIFFERENT disease that happens to hold that id in an
# older release. Add them in sub-project 5, taking the ids from the
# corpus-census rows for each built extract, which list the disease ids
# each release actually carries.
DISEASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("HIV infection", ("MONDO_0005109", "EFO_0000764")),
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


def resolve_disease_id(
    associations: pl.DataFrame,
    disease: str,
    candidates: tuple[str, ...],
    release: str,
) -> str:
    """Which of ``candidates`` this release actually keys ``disease`` by.

    An id the release does not carry yields an empty frame, and an empty
    frame is indistinguishable in a results table from "this release
    recorded no genetic evidence for this disease" -- one is a
    configuration mistake and the other is a finding. So the release's own
    extract decides, and both failures are loud: none present is a wrong
    configuration, and more than one present cannot be resolved here
    because either could be the disease meant.
    """
    present = [
        c for c in candidates
        if associations.filter(pl.col("disease_id") == c).height > 0
    ]
    if len(present) == 1:
        return present[0]
    if not present:
        raise UnknownDiseaseForRelease(
            f"none of {list(candidates)} appears in the OpenTargets "
            f"{release} extract, so a build for {disease!r} would record "
            f"zeros indistinguishable from an absence of evidence. Take the "
            f"id this release uses from its corpus-census row, which lists "
            f"the disease ids the extract carries."
        )
    raise UnknownDiseaseForRelease(
        f"{present} all appear in the OpenTargets {release} extract for "
        f"{disease!r}. They cannot be merged here: each may carry different "
        f"evidence, and picking one would be a silent choice about which "
        f"population the numbers describe."
    )


def build_row(
    store: SnapshotStore,
    disease: str,
    candidates: tuple[str, ...],
    as_of: str,
) -> dict:
    """One dated build, timed, with its exclusions.

    Pure enough to unit-test: everything it needs arrives as an argument,
    and it returns a row rather than writing one.
    """
    release = release_for("opentargets", as_of)
    disease_id = resolve_disease_id(
        store.associations(release), disease, candidates, release
    )

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
    run that fails partway keeps the rows it already wrote.

    It does NOT survive a missing extract: reading it raises
    ``SnapshotMissingError`` (or ``UnpinnedSourceError``, whichever the
    build reaches first) and the run finalises `failed` with the rows so
    far. That is deliberate -- catching it here to keep going would be
    swallowing the one error that says a dated build could not be pinned.
    Build the extracts first.
    """
    store = record.snapshot_store(STORE_ROOT)
    for as_of in sorted(TIME_POINTS):
        for disease, candidates in DISEASES:
            record.append_row(build_row(store, disease, candidates, as_of))
