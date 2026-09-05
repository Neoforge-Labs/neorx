"""A dated build end to end.

Two invariants, and they are different.

The first is about the *frame*: a gene injected into a dated run's
assembled graph, from a source that bypasses gene-list restriction, never
reaches the candidate list.

The second is about the *data*, and is the one this file exists for. A
dated build must be made of the pinned release. Restricting which genes
may be evaluated while every score attached to them is fetched live
produces a graph that looks correctly dated and is not: for temporal
holdout the current association score encodes the outcome being
predicted. So the tests below stub the live Open Targets and OmniPath
clients to emit sentinel genes and assert those sentinels are absent,
and build the snapshot as real parquet on disk rather than mocking
``SnapshotStore`` -- the previous version of this defect survived a
green suite precisely because the snapshot layer was never executed.
"""

import polars as pl
import pytest

from neorx.core.causal.identifier import identify_causal_targets
from neorx.core.graph.models import (
    DiseaseGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)
from neorx.snapshots.schema import ASSOCIATION_COLUMNS, INTERACTION_COLUMNS

DISEASE = "HIV infection"
DISEASE_ID = "EFO_0000764"
RELEASE = "18.06"

# Genes that exist only in the snapshot, and genes that exist only in the
# live clients. A dated build must contain the first and none of the
# second.
SNAPSHOT_GENES = ("CCR5", "CXCR4")
LIVE_ONLY_GENE = "TODAYONLY"


def _graph_with_an_off_frame_gene():
    nodes = [
        GraphNode(node_id="gene:PIK3CA", name="PIK3CA", node_type=NodeType.GENE,
                  source="Open Targets", score=0.9),
        GraphNode(node_id="gene:LATER", name="LATER", node_type=NodeType.GENE,
                  source="Monarch", score=0.9),
        GraphNode(node_id="disease:d", name="d", node_type=NodeType.DISEASE,
                  source="NeoRx", score=1.0),
    ]
    edges = [
        GraphEdge(source_id=n.node_id, target_id="disease:d",
                  edge_type=EdgeType.ASSOCIATED_WITH, weight=0.9,
                  source_db="Open Targets", evidence_class="genetic_association")
        for n in nodes[:2]
    ]
    return DiseaseGraph(disease_name="d", disease_id="disease:d",
                        nodes=nodes, edges=edges)


def test_an_off_frame_gene_never_reaches_the_results():
    results = identify_causal_targets(
        _graph_with_an_off_frame_gene(), top_n=10,
        frame=frozenset({"PIK3CA"}),
    )
    assert all(r.gene_name != "LATER" for r in results)


def test_without_a_frame_the_same_gene_is_evaluated():
    results = identify_causal_targets(_graph_with_an_off_frame_gene(), top_n=10)
    assert any(r.gene_name == "LATER" for r in results)


def test_a_dated_build_refuses_when_the_snapshot_is_missing(tmp_path):
    from neorx.core.graph.graph_builder import build_disease_graph
    from neorx.snapshots.reader import SnapshotStore
    from neorx.snapshots.resolver import SourceResolver, UnpinnedSourceError

    resolver = SourceResolver(
        store=SnapshotStore(tmp_path),
        release_for=lambda source, as_of: RELEASE,
        live={},
    )
    with pytest.raises(UnpinnedSourceError):
        build_disease_graph(
            DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
        )


# ── A real snapshot on disk ─────────────────────────────────────────


def _write_snapshot(root):
    """Write real parquet extracts, in the canonical schema.

    Not a mocked ``SnapshotStore``: the point of these tests is that the
    reading actually happens, and a mock would assert only that this test
    file agrees with itself.
    """
    associations = root / "opentargets" / RELEASE
    associations.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "target_id": ["ENSG0000001", "ENSG0000001", "ENSG0000002",
                          "ENSG0000003"],
            "target_symbol": ["CCR5", "CCR5", "CXCR4", "LITONLY"],
            "disease_id": [DISEASE_ID] * 4,
            # CCR5 carries both a genetic and a non-genetic datatype, so
            # the breakdown has something to preserve and the node score
            # has something to pick out of it.
            "datatype": ["genetic_association", "known_drug",
                         "somatic_mutation", "literature"],
            "score": [0.75, 0.90, 0.40, 0.99],
        },
        schema=ASSOCIATION_COLUMNS,
    ).write_parquet(associations / "associations.parquet")

    interactions = root / "omnipath" / RELEASE
    interactions.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "source_symbol": ["CCR5"],
            "target_symbol": ["CXCR4"],
            "is_directed": [True],
            "consensus_direction": [True],
            "is_stimulation": [False],
            "is_inhibition": [True],
            "primary_sources": ["SIGNOR;TRRUST"],
            "references": ["SIGNOR:11111;TRRUST:22222"],
        },
        schema=INTERACTION_COLUMNS,
    ).write_parquet(interactions / "interactions.parquet")


def _make_resolver(root):
    """A resolver over a real snapshot store."""
    from neorx.snapshots.reader import SnapshotStore
    from neorx.snapshots.resolver import SourceResolver

    _write_snapshot(root)
    return SourceResolver(
        store=SnapshotStore(root),
        release_for=lambda source, as_of: RELEASE,
        live={},
    )


def _stub_sources(monkeypatch):
    """Replace every network-backed source call with an instant stub.

    Open Targets and OmniPath emit a sentinel gene that exists nowhere in
    the snapshot, so a dated build that reached them is detectable in the
    graph itself and not only in a call counter. Returns the counters.
    """
    import neorx.core.graph.graph_builder as graph_builder
    import neorx.core.sources.open_targets as open_targets_module

    calls = {"n": 0, "open_targets": 0, "omnipath": 0}

    def _empty_pair(*_args, **_kwargs):
        calls["n"] += 1
        return [], []

    def _empty_dict(*_args, **_kwargs):
        calls["n"] += 1
        return {}

    def _live_open_targets(*_args, **_kwargs):
        calls["n"] += 1
        calls["open_targets"] += 1
        node = GraphNode(
            node_id=f"gene:{LIVE_ONLY_GENE}", name=LIVE_ONLY_GENE,
            node_type=NodeType.GENE, source="Open Targets", score=1.0,
        )
        edge = GraphEdge(
            source_id=node.node_id,
            target_id=f"disease:{DISEASE.lower().replace(' ', '_')}",
            edge_type=EdgeType.ASSOCIATED_WITH, weight=1.0,
            source_db="Open Targets", evidence_class="genetic_association",
        )
        return [node], [edge]

    def _live_omnipath(*_args, **_kwargs):
        calls["n"] += 1
        calls["omnipath"] += 1
        return [], [GraphEdge(
            source_id=f"gene:{LIVE_ONLY_GENE}", target_id="gene:CCR5",
            edge_type=EdgeType.ACTIVATES, weight=0.9, source_db="OmniPath",
            evidence_class="regulatory", sign=1,
        )]

    monkeypatch.setattr(graph_builder, "query_monarch", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_open_targets", _live_open_targets)
    monkeypatch.setattr(graph_builder, "query_chembl", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_kegg_pathways", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_reactome_pathways", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_string_interactions", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_omnipath", _live_omnipath)
    monkeypatch.setattr(graph_builder, "query_uniprot", _empty_dict)
    monkeypatch.setattr(graph_builder, "query_pdb_structures", _empty_dict)
    monkeypatch.setattr(
        open_targets_module, "resolve_disease_id", lambda *_a, **_kw: None
    )
    return calls


def _fresh_cache(tmp_path, monkeypatch):
    import neorx.core.graph.graph_builder as graph_builder
    from neorx.core.cache import FileCache

    cache = FileCache(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(graph_builder, "get_cache", lambda: cache)
    return cache


def _dated_build(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")
    graph = build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    return graph, calls


# ── The data a dated build is made of ───────────────────────────────


def test_a_dated_build_calls_neither_live_opentargets_nor_live_omnipath(
    tmp_path, monkeypatch
):
    _graph, calls = _dated_build(tmp_path, monkeypatch)
    assert calls["open_targets"] == 0
    assert calls["omnipath"] == 0
    # The unpinned sources are still live -- that is the design, not an
    # oversight -- so the build did do work.
    assert calls["n"] > 0


def test_a_gene_present_only_in_the_snapshot_reaches_the_graph(
    tmp_path, monkeypatch
):
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    names = {n.name for n in graph.nodes}
    for gene in SNAPSHOT_GENES:
        assert gene in names


def test_a_gene_present_only_in_the_live_client_does_not(tmp_path, monkeypatch):
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    assert LIVE_ONLY_GENE not in {n.name for n in graph.nodes}
    assert all(
        LIVE_ONLY_GENE not in edge.source_id and LIVE_ONLY_GENE not in edge.target_id
        for edge in graph.edges
    )


def test_the_node_score_is_the_releases_genetic_score_not_an_overall_score(
    tmp_path, monkeypatch
):
    # CCR5's rows are genetic_association 0.75 and known_drug 0.90. The
    # score must be 0.75: the genetic evidence, which is a value in the
    # extract -- not 0.90, not a sum, not a harmonic mean of the two.
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    ccr5 = next(n for n in graph.nodes if n.name == "CCR5")
    assert ccr5.score == pytest.approx(0.75)
    assert ccr5.metadata["datatype_scores"] == {
        "genetic_association": 0.75, "known_drug": 0.90,
    }
    assert ccr5.metadata["has_known_drug"] is True


def test_a_target_without_genetic_evidence_is_not_given_an_invented_score(
    tmp_path, monkeypatch
):
    # LITONLY has a literature score of 0.99 and nothing genetic. It gets
    # no node rather than a stand-in score, and so is not a candidate.
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    assert "LITONLY" not in {n.name for n in graph.nodes}


def test_the_dated_omnipath_edge_comes_from_the_snapshot(tmp_path, monkeypatch):
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    regulatory = [e for e in graph.edges if e.source_db == "OmniPath"]
    assert [(e.source_id, e.target_id) for e in regulatory] == [
        ("gene:CCR5", "gene:CXCR4")
    ]
    assert regulatory[0].edge_type == EdgeType.INHIBITS
    assert regulatory[0].sign == -1
    assert regulatory[0].primary_sources == ["SIGNOR", "TRRUST"]
    assert regulatory[0].pmids == ["11111", "22222"]


def test_the_graph_records_the_date_it_was_built_as_of(tmp_path, monkeypatch):
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    assert graph.as_of == "2018-06"
    assert graph.disease_id == DISEASE_ID


def test_an_undated_build_records_no_as_of(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    graph = build_disease_graph(DISEASE)
    assert graph.as_of is None
    # And the live client's gene is present, because nothing was pinned.
    assert LIVE_ONLY_GENE in {n.name for n in graph.nodes}


def test_a_dated_build_refuses_without_a_disease_id(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")
    with pytest.raises(ValueError, match="disease_id"):
        build_disease_graph(DISEASE, as_of="2018-06", resolver=resolver)


# ── Cache key must include `as_of` ──────────────────────────────────
#
# The resolver check runs before the cache lookup and correctly fails
# fast when a snapshot is missing. But when the snapshot exists,
# resolution passes -- and before dd52a7b, the cache key was built from
# disease/max_genes/string_min_score alone, so a dated call could hit a
# cache entry written by a live (undated) call of the same disease and
# silently return the live graph. That is the exact anachronism this
# sub-project exists to prevent, so it must be exercised directly rather
# than trusted to the resolver check alone.


def test_a_dated_build_does_not_return_a_cached_live_graph(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")

    live = build_disease_graph(DISEASE)
    after_live = calls["n"]
    assert after_live > 0
    assert LIVE_ONLY_GENE in {n.name for n in live.nodes}

    dated = build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    # A cache hit on the live entry would leave the call count unchanged
    # and hand back the live graph.
    assert calls["n"] > after_live
    assert dated.as_of == "2018-06"
    assert LIVE_ONLY_GENE not in {n.name for n in dated.nodes}


def test_two_dated_builds_with_different_as_of_do_not_share_a_cache_entry(
    tmp_path, monkeypatch
):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")

    build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    after_first = calls["n"]
    assert after_first > 0

    build_disease_graph(
        DISEASE, as_of="2020-01", resolver=resolver, disease_id=DISEASE_ID,
    )
    assert calls["n"] > after_first


def test_an_undated_build_still_hits_its_own_cache(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)

    build_disease_graph(DISEASE)
    after_first = calls["n"]
    assert after_first > 0

    build_disease_graph(DISEASE)
    # Same disease, same params, no as_of -- must be a cache hit.
    assert calls["n"] == after_first


def test_a_dated_build_hits_its_own_cache(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")

    first = build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    after_first = calls["n"]

    second = build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    assert calls["n"] == after_first
    # A cached dated graph must still say it is dated.
    assert second.as_of == first.as_of == "2018-06"
