"""The first production path that can run a dated build.

Before this, `SourceResolver` took its date-to-release mapping as an
injected callable whose only implementations were lambdas inside tests --
the pinning apparatus was reachable from the suite and from nowhere else,
so three of the spec's success criteria could not be satisfied by any code
path a person could invoke.
"""

import json

import polars as pl
import pytest

from neorx.core.graph.models import GraphNode, NodeType
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.resolver import UnpinnedSourceError
from neorx.snapshots.schema import (
    ASSOCIATION_COLUMNS,
    INTERACTION_COLUMNS,
)

DISEASE_ID = "EFO_0000764"


def _store(tmp_path):
    snap = tmp_path / "snapshots"
    assoc = snap / "opentargets" / "18.06"
    assoc.mkdir(parents=True)
    pl.DataFrame(
        {
            "target_id": ["E1", "E2"],
            "target_symbol": ["CCR5", "CXCR4"],
            "disease_id": [DISEASE_ID] * 2,
            "datatype": ["genetic_association"] * 2,
            "score": [0.75, 0.40],
        },
        schema=ASSOCIATION_COLUMNS,
    ).write_parquet(assoc / "associations.parquet")

    inter = snap / "omnipath" / "20180614-20181114"
    inter.mkdir(parents=True)
    pl.DataFrame(
        {
            "source_symbol": ["CCR5"],
            "target_symbol": ["CXCR4"],
            "is_directed": [True],
            "consensus_direction": [True],
            "is_stimulation": [True],
            "is_inhibition": [False],
            "primary_sources": ["SIGNOR"],
            "references": ["1"],
        },
        schema=INTERACTION_COLUMNS,
    ).write_parquet(inter / "interactions.parquet")
    return SnapshotStore(snap)


def _stub_live(monkeypatch, nodes=()):
    import neorx.core.graph.graph_builder as gb

    monkeypatch.setattr(gb, "query_monarch", lambda *_a, **_kw: (list(nodes), []))
    for name in (
        "query_chembl",
        "query_kegg_pathways",
        "query_reactome_pathways",
        "query_string_interactions",
    ):
        monkeypatch.setattr(gb, name, lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(gb, "query_uniprot", lambda *_a, **_kw: {})
    monkeypatch.setattr(gb, "query_pdb_structures", lambda *_a, **_kw: {})


def test_a_dated_build_runs_from_the_pinned_release_table(tmp_path, monkeypatch):
    from experiments.dated_build import build_row

    _stub_live(monkeypatch)
    row = build_row(_store(tmp_path), "HIV infection", DISEASE_ID, "2018-06")

    # The releases come from the table, not from the caller.
    assert row["opentargets_release"] == "18.06"
    assert row["omnipath_release"] == "20180614-20181114"
    assert row["n_genes"] == 2


def test_the_row_carries_the_omnipath_staleness_beside_the_numbers(
    tmp_path, monkeypatch
):
    """The 2025 time point is not what its name implies.

    OmniPath published no refresh between 2023-07-28 and 2025-08-13, so
    the dump covering 2025-06 is 23 months older than the OpenTargets
    release beside it. A reader comparing time points needs that in the
    same table as the numbers, not in a footnote they may not reach.
    """
    from experiments.dated_build import build_row

    _stub_live(monkeypatch)
    row = build_row(_store(tmp_path), "HIV infection", DISEASE_ID, "2018-06")
    assert row["omnipath_data_as_of"] == "2018-06-14"


def test_exclusions_are_recorded_with_their_source_not_counted(
    tmp_path, monkeypatch
):
    """Spec layer three: the record names what was excluded and by whom.

    A count cannot say WHICH genes stopped arriving or from which source,
    which is the entire diagnostic value -- a run that suddenly excludes
    four hundred genes where it used to exclude twelve has had something
    change upstream.
    """
    from experiments.dated_build import build_row

    _stub_live(
        monkeypatch,
        nodes=[
            GraphNode(
                node_id="gene:OFFFRAME",
                name="OFFFRAME",
                node_type=NodeType.GENE,
                source="Monarch",
                score=0.99,
            ),
            GraphNode(
                node_id="pathogen:plasmodium_falciparum:CCR5",
                name="CCR5",
                node_type=NodeType.PATHOGEN_GENE,
                source="ChEMBL",
                score=0.9,
            ),
        ],
    )
    row = build_row(_store(tmp_path), "HIV infection", DISEASE_ID, "2018-06")

    by_reason = {e["reason"]: e for e in row["frame_exclusions"]}
    assert by_reason["off_frame"]["symbol"] == "OFFFRAME"
    assert by_reason["off_frame"]["source"] == "Monarch"
    assert (
        by_reason["excluded_by_type"]["node_id"]
        == "pathogen:plasmodium_falciparum:CCR5"
    )
    assert row["n_frame_exclusions"] == len(row["frame_exclusions"])
    # And the row is serialisable, since it is written to rows.jsonl.
    json.dumps(row)


def test_the_timing_is_recorded_and_the_ratio_is_absent_without_a_live_run(
    tmp_path, monkeypatch
):
    """A claim needs a measurement; an unmeasured ratio stays None.

    Criterion 8 wants the snapshot path shown to beat the live path by an
    order of magnitude. The mechanism records both, but `speedup` is None
    until a live build actually runs -- rather than a plausible constant
    standing in for the measurement, which is the defect class this
    project gates against.
    """
    from experiments.dated_build import build_row

    _stub_live(monkeypatch)
    row = build_row(_store(tmp_path), "HIV infection", DISEASE_ID, "2018-06")
    assert row["snapshot_seconds"] >= 0.0
    assert row["live_seconds"] is None
    assert row["speedup"] is None


def test_a_dated_build_cannot_reach_a_live_client_for_a_pinned_source(tmp_path):
    """The resolver's live mapping is empty on purpose.

    A dated run that fell back to a live client for OpenTargets or
    OmniPath would rebuild the anachronism invisibly, so the mapping that
    would let it is not populated at all.
    """
    from experiments.dated_build import dated_resolver

    resolver = dated_resolver(_store(tmp_path))
    # A pinned source with a snapshot resolves to a reader.
    assert resolver.resolve("opentargets", "2018-06") is not None
    # A pinned source with no snapshot refuses rather than falling back.
    with pytest.raises(UnpinnedSourceError):
        resolver.resolve("opentargets", "2021-11")
    # And an unpinned source has nowhere to go, by construction.
    with pytest.raises(KeyError):
        resolver.resolve("string", "2018-06")


def test_the_experiment_declares_that_it_reads_snapshots(tmp_path):
    """Without the declaration its env.json would cite no inputs at all.

    That is the provenance failure this project already found in four
    published papers: a number with no traceable derivation.
    """
    import experiments  # noqa: F401  -- registers the definitions
    from neorx.experiments.registry import get_experiment

    assert get_experiment("dated-build").reads_snapshots is True
    assert get_experiment("corpus-census").reads_snapshots is True
