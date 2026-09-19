"""Choosing between live and snapshot, and refusing in between.

A dated run that quietly mixes in current data produces exactly the
anachronistic graph this sub-project exists to prevent, and does so
invisibly. So a source with no snapshot at a requested date is a named
refusal, never a fallback.
"""

import polars as pl
import pytest

from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.resolver import SourceResolver, UnpinnedSourceError
from neorx.snapshots.schema import ASSOCIATION_COLUMNS, INTERACTION_COLUMNS


class _FakeStore:
    def __init__(self, present):
        self._present = present

    def has(self, source, release):
        return (source, release) in self._present


RELEASES = {
    "2018-06": {"opentargets": "18.06", "omnipath": "20180614"},
    "2025-06": {"opentargets": "25.06", "omnipath": "20250813"},
}


def _resolver(present=(("opentargets", "18.06"), ("omnipath", "20180614"))):
    return SourceResolver(
        store=_FakeStore(set(present)),
        release_for=lambda source, as_of: RELEASES[as_of][source],
        live={"opentargets": lambda: "LIVE_OT", "omnipath": lambda: "LIVE_OMNI",
              "string": lambda: "LIVE_STRING"},
    )


def test_no_date_returns_the_live_client():
    assert _resolver().resolve("opentargets", as_of=None)() == "LIVE_OT"


def test_an_unpinned_source_stays_live_even_on_a_dated_run():
    # STRING contributes only associational edges, which identification
    # filters out, so pinning it would buy rigour against a leak that does
    # not exist.
    assert _resolver().resolve("string", as_of="2018-06")() == "LIVE_STRING"


def _store_with_an_extract(root):
    """A real on-disk store, because a resolved reader really reads it."""
    ot = root / "opentargets" / "18.06"
    ot.mkdir(parents=True)
    pl.DataFrame(
        {
            "target_id": ["ENSG0000001"],
            "target_symbol": ["CCR5"],
            "disease_id": ["EFO_0000764"],
            "datatype": ["genetic_association"],
            "score": [0.75],
        },
        schema=ASSOCIATION_COLUMNS,
    ).write_parquet(ot / "associations.parquet")

    op = root / "omnipath" / "20180614"
    op.mkdir(parents=True)
    pl.DataFrame(
        {
            "source_symbol": ["CCR5"],
            "target_symbol": ["CXCR4"],
            "is_directed": [True],
            "consensus_direction": [True],
            "is_stimulation": [True],
            "is_inhibition": [False],
            "primary_sources": ["SIGNOR"],
            "references": ["SIGNOR:12345"],
        },
        schema=INTERACTION_COLUMNS,
    ).write_parquet(op / "interactions.parquet")

    return SnapshotStore(root, on_read=None)


def _snapshot_resolver(root):
    return SourceResolver(
        store=_store_with_an_extract(root),
        release_for=lambda source, as_of: RELEASES[as_of][source],
        live={"opentargets": lambda *a, **k: "LIVE_OT",
              "omnipath": lambda *a, **k: "LIVE_OMNI"},
    )


def test_a_dated_run_reads_the_snapshot_rather_than_the_live_client(tmp_path):
    # The reader must be the snapshot's, and it must be a real reader:
    # returning a description of the snapshot rather than its contents is
    # how a dated build ended up made entirely of live data before.
    reader = _snapshot_resolver(tmp_path).resolve("opentargets", as_of="2018-06")
    nodes, edges = reader("HIV infection", disease_id="EFO_0000764", max_results=5)
    assert [n.name for n in nodes] == ["CCR5"]
    assert nodes[0].score == pytest.approx(0.75)
    assert [(e.source_id, e.target_id) for e in edges] == [
        ("gene:CCR5", "disease:hiv_infection")
    ]


def test_a_dated_omnipath_reader_returns_edges_from_the_snapshot(tmp_path):
    reader = _snapshot_resolver(tmp_path).resolve("omnipath", as_of="2018-06")
    nodes, edges = reader(["CCR5", "CXCR4"])
    assert nodes == []
    assert [(e.source_id, e.target_id) for e in edges] == [
        ("gene:CCR5", "gene:CXCR4")
    ]
    assert edges[0].primary_sources == ["SIGNOR"]


def test_a_missing_snapshot_refuses_rather_than_falling_back():
    with pytest.raises(UnpinnedSourceError):
        _resolver().resolve("opentargets", as_of="2025-06")


def test_the_refusal_names_the_source_and_the_release():
    with pytest.raises(UnpinnedSourceError, match="opentargets"):
        _resolver().resolve("opentargets", as_of="2025-06")
    with pytest.raises(UnpinnedSourceError, match="25.06"):
        _resolver().resolve("opentargets", as_of="2025-06")


def test_the_refusal_says_it_will_not_fall_back():
    # The message is the mechanism: someone hitting this must not conclude
    # that omitting the date is the fix.
    with pytest.raises(UnpinnedSourceError, match="not fall back"):
        _resolver().resolve("omnipath", as_of="2025-06")
