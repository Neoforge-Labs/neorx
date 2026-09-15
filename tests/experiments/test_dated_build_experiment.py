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


# ── The links that had no test ──────────────────────────────────────
#
# Each of these was a mutation that survived the whole suite: correct
# code with nothing asserting the connection that makes it matter. That
# is the recurring pattern on this branch, so they are pinned directly.


def test_running_a_reads_snapshots_experiment_actually_cites_extracts(
    tmp_path, monkeypatch
):
    """The wiring, not the flag.

    Replacing the runner's `snapshot_manifest=... if defn.reads_snapshots`
    with `None` left every test green: one test asserted the flag was
    True, another asserted RunRecord.create works when handed a manifest,
    and nothing asserted that running the experiment connects them.
    """
    import json

    import experiments.dated_build as db
    from neorx.experiments.registry import run_experiment
    from neorx.snapshots.manifest import SnapshotEntry, write_entry

    store = _store(tmp_path)
    manifest = store.root / "manifest.toml"
    write_entry(
        manifest,
        SnapshotEntry(
            source="opentargets",
            release="18.06",
            url="https://example.invalid/18.06",
            sha256="a" * 64,
            extractor_version=2,
            rows=2,
        ),
    )
    _stub_live(monkeypatch)
    monkeypatch.setattr(db, "STORE_ROOT", store.root)
    monkeypatch.setattr(db, "DISEASES", (("HIV infection", DISEASE_ID),))
    monkeypatch.setattr(
        db, "TIME_POINTS", {"2018-06": db.TIME_POINTS["2018-06"]}
    )

    record = run_experiment(
        "dated-build", runs_dir=tmp_path / "runs", snapshot_manifest=manifest
    )
    env = json.loads((record.path / "env.json").read_text())
    assert env["snapshots"]["opentargets/18.06"]["sha256"] == "a" * 64


def test_a_run_that_cites_nothing_is_refused_rather_than_completed(
    tmp_path, monkeypatch
):
    """Extracts present, manifest absent: the defect the flag was meant to fix.

    Before this, such a run read the extracts, wrote "snapshots": {}, and
    finalised `complete` -- a number with no traceable derivation, which
    is what this project found in four published papers.
    """
    import experiments.dated_build as db
    from neorx.experiments.record import ProvenanceError
    from neorx.experiments.registry import run_experiment

    store = _store(tmp_path)  # extracts exist, no manifest written
    _stub_live(monkeypatch)
    monkeypatch.setattr(db, "STORE_ROOT", store.root)

    with pytest.raises(ProvenanceError) as excinfo:
        run_experiment(
            "dated-build",
            runs_dir=tmp_path / "runs",
            snapshot_manifest=store.root / "manifest.toml",
        )
    assert "reads_snapshots" in str(excinfo.value)


def test_withheld_material_is_recorded_per_source(tmp_path, monkeypatch):
    """The reason the new rule needs, and the one with no coverage.

    `off_frame` and `excluded_by_type` were asserted; `withheld` -- what
    an unpinned source offered that WAS in the frame and was refused
    anyway -- was not, and deleting the loop that records it left every
    test green. On a real build it is the largest category.
    """
    from experiments.dated_build import build_row

    _stub_live(
        monkeypatch,
        nodes=[
            GraphNode(
                node_id="gene:CCR5",  # in the frame, spelled as the release does
                name="CCR5",
                node_type=NodeType.GENE,
                source="Monarch",
                score=0.99,
            )
        ],
    )
    row = build_row(_store(tmp_path), "HIV infection", DISEASE_ID, "2018-06")

    withheld = [e for e in row["frame_exclusions"] if e["reason"] == "withheld"]
    assert withheld, "in-frame material refused anyway must still be recorded"
    assert withheld[0]["source"] == "Monarch"
    assert int(withheld[0]["n_nodes"]) == 1


def test_a_disease_id_the_release_does_not_carry_is_refused(
    tmp_path, monkeypatch
):
    """Zeros read as a finding; this is a fact about the id.

    OpenTargets keys diseases by EFO in the 18.06 era and by MONDO from
    21.x, so pairing one id with every time point silently recorded
    n_genes: 0 rows indistinguishable from "this release recorded no
    genetic evidence".
    """
    from experiments.dated_build import UnknownDiseaseForRelease, build_row

    _stub_live(monkeypatch)
    with pytest.raises(UnknownDiseaseForRelease) as excinfo:
        build_row(
            _store(tmp_path), "Alzheimer disease", "MONDO_0004975", "2018-06"
        )
    message = str(excinfo.value)
    assert "MONDO_0004975" in message
    assert "18.06" in message
