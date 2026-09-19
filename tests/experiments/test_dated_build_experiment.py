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
    row = build_row(_store(tmp_path), "HIV infection", (DISEASE_ID,), "2018-06")

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
    row = build_row(_store(tmp_path), "HIV infection", (DISEASE_ID,), "2018-06")
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
    row = build_row(_store(tmp_path), "HIV infection", (DISEASE_ID,), "2018-06")

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
    row = build_row(_store(tmp_path), "HIV infection", (DISEASE_ID,), "2018-06")
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


_FILES = {"opentargets": "associations.parquet", "omnipath": "interactions.parquet"}


def _entry(source, release, digest, store=None):
    """A manifest entry.

    When the store holds the extract, the digest is taken from those bytes:
    a citation is checked against what the run actually read, so a literal
    stands in only for an extract this store does not have.
    """
    from neorx.snapshots.manifest import SnapshotEntry, digest_file

    sha = digest * 64
    if store is not None:
        path = store.root / source / release / _FILES[source]
        if path.exists():
            sha = digest_file(path)
    return SnapshotEntry(
        source=source,
        release=release,
        url=f"https://example.invalid/{source}/{release}",
        sha256=sha,
        extractor_version=2,
        rows=2,
    )


def _run_dated(tmp_path, monkeypatch, store):
    import experiments.dated_build as db
    from neorx.experiments.registry import run_experiment

    _stub_live(monkeypatch)
    monkeypatch.setattr(db, "STORE_ROOT", store.root)
    monkeypatch.setattr(db, "DISEASES", (("HIV infection", (DISEASE_ID,)),))
    monkeypatch.setattr(db, "TIME_POINTS", {"2018-06": db.TIME_POINTS["2018-06"]})
    return run_experiment("dated-build", runs_dir=tmp_path / "runs")


def test_running_the_experiment_cites_exactly_what_it_read(tmp_path, monkeypatch):
    """The wiring, end to end, through the real runner.

    A build reads the OpenTargets extract AND the OmniPath extract, so
    both must be cited -- and nothing else, even though the manifest also
    describes a 25.06 extract the run never opened.
    """
    import json

    from neorx.snapshots.manifest import write_entry

    store = _store(tmp_path)
    for entry in (
        _entry("opentargets", "18.06", "a", store),
        _entry("omnipath", "20180614-20181114", "b", store),
        _entry("opentargets", "25.06", "c", store),
    ):
        write_entry(store.manifest_path, entry)

    record = _run_dated(tmp_path, monkeypatch, store)
    env = json.loads((record.path / "env.json").read_text())
    assert set(env["snapshots"]) == {
        "opentargets/18.06",
        "omnipath/20180614-20181114",
    }
    from neorx.snapshots.manifest import digest_file

    assert env["snapshots"]["opentargets/18.06"]["sha256"] == digest_file(
        store.root / "opentargets" / "18.06" / "associations.parquet"
    )


def test_a_run_whose_extracts_have_no_manifest_is_refused(tmp_path, monkeypatch):
    """Extracts present, manifest absent.

    Such a run once read the extracts, wrote "snapshots": {}, and
    finalised `complete` -- a number with no traceable derivation, which
    is what this project found in four published papers.
    """
    from neorx.experiments.record import ProvenanceError

    store = _store(tmp_path)  # extracts exist, no manifest written
    with pytest.raises(ProvenanceError) as excinfo:
        _run_dated(tmp_path, monkeypatch, store)
    assert "opentargets/18.06" in str(excinfo.value)


def test_an_unrelated_manifest_entry_does_not_satisfy_the_citation(
    tmp_path, monkeypatch
):
    """The check once accepted ANY citation.

    Executed by a reviewer: a manifest describing only a 25.06 extract let
    a run over 18.06 complete, marked complete, citing the wrong release.
    """
    from neorx.experiments.record import ProvenanceError
    from neorx.snapshots.manifest import write_entry

    store = _store(tmp_path)
    write_entry(store.manifest_path, _entry("opentargets", "25.06", "c", store))
    with pytest.raises(ProvenanceError):
        _run_dated(tmp_path, monkeypatch, store)


def test_a_partly_described_store_is_refused(tmp_path, monkeypatch):
    # OpenTargets is described, OmniPath is not. The run read both, so the
    # numbers depend on an extract it cannot identify.
    from neorx.experiments.record import ProvenanceError
    from neorx.snapshots.manifest import write_entry

    store = _store(tmp_path)
    write_entry(store.manifest_path, _entry("opentargets", "18.06", "a", store))
    with pytest.raises(ProvenanceError) as excinfo:
        _run_dated(tmp_path, monkeypatch, store)
    assert "omnipath/20180614-20181114" in str(excinfo.value)


def test_a_refused_run_is_recorded_as_failed(tmp_path, monkeypatch):
    import json

    from neorx.experiments.record import ProvenanceError

    store = _store(tmp_path)
    with pytest.raises(ProvenanceError):
        _run_dated(tmp_path, monkeypatch, store)
    (run_dir,) = (tmp_path / "runs").iterdir()
    summary = json.loads((run_dir / "record.json").read_text())
    assert summary["status"] == "failed"
    assert summary["citable"] is False


def test_declaring_reads_snapshots_without_reading_through_the_record_is_refused(
    tmp_path,
):
    """A store opened some other way reads nothing the record can cite.

    `reads_snapshots` is how the runner knows to expect reads. An
    experiment that declares it and records none has opened its store
    directly, so everything it read is invisible to the citation.
    """
    from neorx.experiments.record import ProvenanceError
    from neorx.experiments.registry import experiment, run_experiment

    store = _store(tmp_path)

    @experiment(name="bypasses-the-record", reads_snapshots=True)
    def _bypass(record):
        record.append_row({"n": store.associations("18.06").height})

    with pytest.raises(ProvenanceError) as excinfo:
        run_experiment("bypasses-the-record", runs_dir=tmp_path / "runs")
    assert "snapshot_store" in str(excinfo.value)


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
    row = build_row(_store(tmp_path), "HIV infection", (DISEASE_ID,), "2018-06")

    withheld = [e for e in row["frame_exclusions"] if e["reason"] == "withheld"]
    assert withheld, "in-frame material refused anyway must still be recorded"
    assert withheld[0]["source"] == "Monarch"
    assert int(withheld[0]["n_nodes"]) == 1


def test_a_disease_id_the_release_does_not_carry_is_refused(
    tmp_path, monkeypatch
):
    """Zeros read as a finding; this is a fact about the id.

    OpenTargets re-keys diseases between releases -- in 26.06,
    EFO_0000764 no longer resolves and HIV is MONDO_0005109 -- so an id
    that is right for one release can be absent from another, and the
    build would silently record n_genes: 0.
    """
    from experiments.dated_build import UnknownDiseaseForRelease, build_row

    _stub_live(monkeypatch)
    with pytest.raises(UnknownDiseaseForRelease) as excinfo:
        build_row(
            _store(tmp_path), "Alzheimer disease", ("MONDO_0004975",), "2018-06"
        )
    message = str(excinfo.value)
    assert "MONDO_0004975" in message
    assert "18.06" in message


def test_the_release_decides_which_candidate_id_is_used(tmp_path, monkeypatch):
    # Both keys HIV has held are offered; the extract carries one of them,
    # and that is the id recorded -- whichever order they were listed in.
    from experiments.dated_build import build_row

    _stub_live(monkeypatch)
    row = build_row(
        _store(tmp_path), "HIV infection", ("MONDO_0005109", DISEASE_ID), "2018-06"
    )
    assert row["disease_id"] == DISEASE_ID
    assert row["n_genes"] == 2


def test_two_candidates_both_present_is_refused_as_ambiguous():
    """Either could be the disease meant, so neither is picked silently."""
    from experiments.dated_build import UnknownDiseaseForRelease, resolve_disease_id

    associations = pl.DataFrame(
        {
            "target_id": ["E1", "E2"],
            "target_symbol": ["A", "B"],
            "disease_id": ["EFO_0000764", "MONDO_0005109"],
            "datatype": ["genetic_association"] * 2,
            "score": [0.5, 0.5],
        },
        schema=ASSOCIATION_COLUMNS,
    )
    with pytest.raises(UnknownDiseaseForRelease) as excinfo:
        resolve_disease_id(
            associations, "HIV infection", ("EFO_0000764", "MONDO_0005109"), "18.06"
        )
    assert "cannot be merged" in str(excinfo.value)


def test_the_shipped_disease_table_only_offers_ids_that_name_one_disease():
    """HIV's two keys are verified to be the same disease.

    In the live 26.06 release MONDO_0005109's dbXRefs list EFO:0000764.
    Alzheimer's and type 2 diabetes list no EFO cross-reference, so their
    pre-MONDO keys could not be established and they are not shipped --
    an invented id risks matching a different disease in an older release.
    """
    from experiments.dated_build import DISEASES

    assert DISEASES == (("HIV infection", ("MONDO_0005109", "EFO_0000764")),)


def test_the_snapshot_store_is_anchored_to_the_repository_not_the_cwd():
    """A cwd-relative store path is how a run cited bytes it did not read.

    Executed by a reviewer: launched from a subdirectory holding its own
    store, a run read that store's extracts and cited the repo-root
    manifest's digest. Reading through the record now takes the manifest
    from the store actually read, but the store both experiments open must
    still be the repository's, whatever directory the command ran in.
    """
    from pathlib import Path

    import experiments.corpus_census as cc
    import experiments.dated_build as db
    from neorx.experiments.record import RUNS_DIR, SNAPSHOTS_DIR

    assert SNAPSHOTS_DIR.is_absolute()
    # Same anchor as the run records themselves.
    assert SNAPSHOTS_DIR.parent == RUNS_DIR.parent
    repo_root = Path(db.__file__).resolve().parents[1]
    assert repo_root / "snapshots" == SNAPSHOTS_DIR
    assert db.STORE_ROOT == SNAPSHOTS_DIR
    assert cc.STORE_ROOT == SNAPSHOTS_DIR


def test_a_run_that_fails_midway_still_cites_what_it_read(tmp_path, monkeypatch):
    """Partial rows need their provenance as much as complete ones.

    Deleting the settlement call from the runner's failure path left the
    whole suite green: a failed run kept its partial rows with
    `"snapshots": {}` and no record of what produced them.
    """
    import json

    import experiments.dated_build as db
    from neorx.experiments.registry import experiment, run_experiment
    from neorx.snapshots.manifest import SnapshotEntry, digest_file, write_entry

    store = _store(tmp_path)
    extract = store.root / "opentargets" / "18.06" / "associations.parquet"
    write_entry(
        store.manifest_path,
        SnapshotEntry(
            source="opentargets",
            release="18.06",
            url="https://example.invalid/18.06",
            sha256=digest_file(extract),
            extractor_version=2,
            rows=2,
        ),
    )
    monkeypatch.setattr(db, "STORE_ROOT", store.root)

    @experiment(name="reads-then-raises", reads_snapshots=True)
    def _reads_then_raises(record):
        rows = record.snapshot_store(store.root).associations("18.06").height
        record.append_row({"n": rows})
        raise RuntimeError("upstream blew up")

    with pytest.raises(RuntimeError, match="upstream blew up"):
        run_experiment("reads-then-raises", runs_dir=tmp_path / "runs")

    (run_dir,) = (tmp_path / "runs").iterdir()
    env = json.loads((run_dir / "env.json").read_text())
    assert set(env["snapshots"]) == {"opentargets/18.06"}
    summary = json.loads((run_dir / "record.json").read_text())
    assert summary["status"] == "failed"


def test_a_settlement_fault_does_not_cost_the_run_record(tmp_path, monkeypatch):
    """A second fault must not hide the first.

    Without this, a malformed manifest.toml raised inside the failure path,
    finalise() never ran, no record.json was written, and a failed run read
    back later as merely incomplete -- while the operator saw the manifest
    error instead of the one that stopped the run.
    """
    import json

    from neorx.experiments.registry import experiment, run_experiment

    store = _store(tmp_path)
    store.manifest_path.write_text("this is not valid toml {{{", encoding="utf-8")

    @experiment(name="raises-with-bad-manifest", reads_snapshots=True)
    def _raises(record):
        record.snapshot_store(store.root).associations("18.06")
        raise RuntimeError("the real failure")

    with pytest.raises(RuntimeError, match="the real failure"):
        run_experiment("raises-with-bad-manifest", runs_dir=tmp_path / "runs")

    (run_dir,) = (tmp_path / "runs").iterdir()
    summary = json.loads((run_dir / "record.json").read_text())
    assert summary["status"] == "failed"
    env = json.loads((run_dir / "env.json").read_text())
    # And the settlement fault is recorded rather than lost.
    assert "snapshots_settlement_error" in env
