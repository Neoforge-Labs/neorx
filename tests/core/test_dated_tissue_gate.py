"""The tissue gate is an unpinned source, and a dated build must not use it.

The eighth leak channel found on this path, and the first that reached a
reported number *after* the graph was built -- so no fingerprint of a
`DiseaseGraph` could see it.

`TissueFilter` queries the Human Protein Atlas live
(`proteinatlas.org/search/<gene>?format=json`). `scoring.classify_target`
treats its answer as a boolean gate: a gene that fails it is demoted to
CORRELATIONAL whatever its confidence. So on a graph built from a 2018
release, today's expression data decided `classification`,
`is_causal_target` and `tissue_relevant` -- which is exactly what "a dated
build takes nothing from an unpinned source" forbids.

Skipping the gate is not the same as passing it, and a dated run is
therefore more permissive here than a live one. That difference is
recorded on every result rather than left to be inferred from a
`tissue_relevant=True` that looks indistinguishable from a pass.
"""

from unittest.mock import patch

import pytest

from neorx.core.causal.identifier import evaluate_all_targets
from neorx.core.graph.models import (
    DiseaseGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)


def _graph(as_of=None):
    return DiseaseGraph(
        disease_name="HIV infection",
        disease_id="EFO_0000764",
        as_of=as_of,
        nodes=[
            GraphNode(
                node_id="disease:hiv_infection",
                name="HIV infection",
                node_type=NodeType.DISEASE,
                source="NeoRx",
                score=1.0,
            ),
            GraphNode(
                node_id="gene:CCR5",
                name="CCR5",
                node_type=NodeType.GENE,
                source="Open Targets",
                score=0.75,
                metadata={"datatype_scores": {"genetic_association": 0.75}},
            ),
        ],
        edges=[
            GraphEdge(
                source_id="gene:CCR5",
                target_id="disease:hiv_infection",
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=0.75,
                source_db="Open Targets",
                evidence="OT genetic",
                evidence_class="genetic_association",
                primary_sources=["Open Targets"],
            ),
        ],
    )


# A tissue answer hostile enough to flip the gate: the gene is expressed
# only somewhere unrelated to the disease.
_FAILS_GATE = (False, 0.0, "expressed only in testis")


def test_a_dated_build_does_not_query_the_tissue_atlas_at_all():
    """The strongest form: the unpinned source is never reached.

    Asserted on the call rather than on the result, because a result that
    happens to match could still have been produced by a live call whose
    answer agreed.
    """
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant"
    ) as probe:
        probe.return_value = _FAILS_GATE
        evaluate_all_targets(_graph(as_of="2018-06"))
    probe.assert_not_called()


def test_an_undated_build_still_applies_the_gate():
    # The live path is unchanged: this is about dated provenance, not
    # about abandoning the tissue gate.
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant"
    ) as probe:
        probe.return_value = _FAILS_GATE
        results = evaluate_all_targets(_graph())
    probe.assert_called()
    assert results[0].tissue_relevant is False


def test_todays_tissue_data_cannot_change_a_dated_classification():
    """Criterion 11, for a field computed after the graph is built.

    Same snapshot, same graph; the only difference is what the Human
    Protein Atlas says today. Before this fix the two runs disagreed on
    `classification`, `is_causal_target` and `tissue_relevant`.
    """
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant",
        return_value=_FAILS_GATE,
    ):
        hostile = evaluate_all_targets(_graph(as_of="2018-06"))
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant",
        return_value=(True, 0.9, "expressed in blood"),
    ):
        friendly = evaluate_all_targets(_graph(as_of="2018-06"))

    def _fingerprint(results):
        return [
            (
                r.protein_id,
                r.classification,
                r.is_causal_target,
                r.tissue_relevant,
                round(r.causal_confidence, 9),
            )
            for r in results
        ]

    assert _fingerprint(hostile) == _fingerprint(friendly)


def test_the_skipped_gate_is_recorded_not_silent():
    """`tissue_relevant=True` alone would read as a pass.

    A reader comparing a dated classification with a live one needs to
    know the gate never ran, and needs it on the result rather than in a
    log line nobody reads back.
    """
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant",
        return_value=_FAILS_GATE,
    ):
        results = evaluate_all_targets(_graph(as_of="2018-06"))

    explanation = results[0].tissue_explanation
    assert "2018-06" in explanation
    assert "not applied" in explanation.lower()
    # And it must say why, so the asymmetry is not mistaken for a pass.
    assert "not the same as passed" in explanation.lower()


@pytest.mark.parametrize("as_of", ["2018-06", "2021-11", "2025-06"])
def test_every_time_point_skips_it(as_of):
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant"
    ) as probe:
        probe.return_value = _FAILS_GATE
        evaluate_all_targets(_graph(as_of=as_of))
    probe.assert_not_called()


# ── The two halves of the sources_queried fix ───────────────────────
#
# Removing either half alone was invisible to the whole suite, because
# each masks the other: skipping the UniProt/PDB fetches means nothing is
# appended to filter, and the filter means an append would be removed.
# A guard pair that only works together still needs each half pinned, or
# a later change removes one and nothing says so.


def _dated_graph_probe(monkeypatch, tmp_path, loud):
    """Build a dated graph with live sources loud or silent."""
    import polars as pl

    import neorx.core.graph.graph_builder as gb
    from neorx.core.cache import FileCache
    from neorx.snapshots.reader import SnapshotStore
    from neorx.snapshots.resolver import SourceResolver
    from neorx.snapshots.schema import ASSOCIATION_COLUMNS, INTERACTION_COLUMNS

    snap = tmp_path / ("loud" if loud else "silent") / "snapshots"
    d = snap / "opentargets" / "18.06"
    d.mkdir(parents=True)
    pl.DataFrame(
        {
            "target_id": ["E1"],
            "target_symbol": ["CCR5"],
            "disease_id": ["EFO_1"],
            "datatype": ["genetic_association"],
            "score": [0.7],
        },
        schema=ASSOCIATION_COLUMNS,
    ).write_parquet(d / "associations.parquet")
    d2 = snap / "omnipath" / "18.06"
    d2.mkdir(parents=True)
    pl.DataFrame(
        {
            "source_symbol": ["CCR5"],
            "target_symbol": ["CCR5"],
            "is_directed": [True],
            "consensus_direction": [True],
            "is_stimulation": [True],
            "is_inhibition": [False],
            "primary_sources": ["S"],
            "references": ["1"],
        },
        schema=INTERACTION_COLUMNS,
    ).write_parquet(d2 / "interactions.parquet")

    uniprot = (
        {"CCR5": {"uniprot_id": "P51681", "pdb_ids": ["5UIW"], "is_druggable": True}}
        if loud
        else {}
    )
    calls = {"uniprot": 0, "pdb": 0}

    def _uniprot(*_a, **_kw):
        calls["uniprot"] += 1
        return uniprot

    def _pdb(*_a, **_kw):
        calls["pdb"] += 1
        return {"CCR5": [{"pdb_id": "5UIW"}]} if loud else {}

    for name in (
        "query_monarch",
        "query_chembl",
        "query_kegg_pathways",
        "query_reactome_pathways",
        "query_string_interactions",
    ):
        monkeypatch.setattr(gb, name, lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(gb, "query_uniprot", _uniprot)
    monkeypatch.setattr(gb, "query_pdb_structures", _pdb)
    monkeypatch.setattr(
        gb, "get_cache", lambda: FileCache(cache_dir=tmp_path / "c" / str(loud))
    )
    resolver = SourceResolver(
        store=SnapshotStore(snap), release_for=lambda _s, _a: "18.06", live={}
    )
    graph = gb.build_disease_graph(
        "d", as_of="2018-06", disease_id="EFO_1", resolver=resolver
    )
    return graph, calls


def test_a_dated_build_does_not_query_uniprot_or_pdb(tmp_path, monkeypatch):
    """Half one: the fetches do not happen.

    Every node on a dated build is pinned and takes no enrichment, so
    these calls fetched data that was discarded -- and made a dated build
    depend on two live services being reachable. A 2018 graph must not
    stop reproducing because the PDB is down in 2029.
    """
    _graph, calls = _dated_graph_probe(monkeypatch, tmp_path, loud=True)
    assert calls == {"uniprot": 0, "pdb": 0}


def test_an_undated_build_still_queries_them(tmp_path, monkeypatch):
    import neorx.core.graph.graph_builder as gb
    from neorx.core.cache import FileCache

    calls = {"n": 0}

    def _uniprot(*_a, **_kw):
        calls["n"] += 1
        return {"CCR5": {"uniprot_id": "P51681"}}

    for name in (
        "query_monarch",
        "query_chembl",
        "query_kegg_pathways",
        "query_reactome_pathways",
        "query_string_interactions",
        "query_omnipath",
    ):
        monkeypatch.setattr(gb, name, lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(gb, "query_open_targets", lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(gb, "query_uniprot", _uniprot)
    monkeypatch.setattr(gb, "query_pdb_structures", lambda *_a, **_kw: {})
    monkeypatch.setattr(
        gb, "get_cache", lambda: FileCache(cache_dir=tmp_path / "undated")
    )
    gb.build_disease_graph("d")
    assert calls["n"] == 1


def test_sources_queried_lists_only_what_contributed(tmp_path, monkeypatch):
    """Half two: the list names contributors, not everything dialled.

    It is persisted and rendered as "Sources" in the report, the API and
    the CLI, so a dated graph listing eight databases when two built it
    is a claim to a reader, not an internal detail.
    """
    graph, _calls = _dated_graph_probe(monkeypatch, tmp_path, loud=True)
    assert graph.sources_queried == ["OpenTargets", "OmniPath"]
