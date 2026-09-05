"""A dated build end to end.

The invariant a reviewer would ask for: a gene injected into a dated run's
assembled graph, from a source that bypasses gene-list restriction, never
reaches the candidate list -- and its exclusion is recorded.
"""

import pytest

from neorx.core.causal.identifier import identify_causal_targets
from neorx.core.graph.models import (
    DiseaseGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)


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
        release_for=lambda source, as_of: "18.06",
        live={},
    )
    with pytest.raises(UnpinnedSourceError):
        build_disease_graph("HIV infection", as_of="2018-06", resolver=resolver)


# ── Cache key must include `as_of` ──────────────────────────────────
#
# The resolver check above runs before the cache lookup and correctly
# fails fast when a snapshot is missing. But when the snapshot exists,
# resolution passes -- and before this fix, the cache key was built from
# disease/max_genes/string_min_score alone, so a dated call could hit a
# cache entry written by a live (undated) call of the same disease and
# silently return the live graph. That is the exact anachronism this
# sub-project exists to prevent, so it must be exercised directly rather
# than trusted to the resolver check alone.


def _make_resolver(root):
    """A resolver whose pinned-source snapshots exist, so resolve() passes."""
    from neorx.snapshots.reader import SnapshotStore
    from neorx.snapshots.resolver import SourceResolver

    for source, filename in (
        ("opentargets", "associations.parquet"),
        ("omnipath", "interactions.parquet"),
    ):
        path = root / source / "18.06" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    return SourceResolver(
        store=SnapshotStore(root),
        release_for=lambda source, as_of: "18.06",
        live={},
    )


def _stub_sources(monkeypatch):
    """Replace every network-backed source call with an instant stub.

    Returns a call counter so a test can tell a fresh build (the stubs
    ran) apart from a cache hit (they didn't).
    """
    import neorx.core.graph.graph_builder as graph_builder
    import neorx.core.sources.open_targets as open_targets_module

    calls = {"n": 0}

    def _empty_pair(*_args, **_kwargs):
        calls["n"] += 1
        return [], []

    def _empty_dict(*_args, **_kwargs):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(graph_builder, "query_monarch", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_open_targets", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_chembl", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_kegg_pathways", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_reactome_pathways", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_string_interactions", _empty_pair)
    monkeypatch.setattr(graph_builder, "query_omnipath", _empty_pair)
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


def test_a_dated_build_does_not_return_a_cached_live_graph(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")

    build_disease_graph("HIV infection")
    after_live = calls["n"]
    assert after_live > 0

    build_disease_graph("HIV infection", as_of="2018-06", resolver=resolver)
    # A cache hit on the live entry would leave the call count unchanged.
    assert calls["n"] > after_live


def test_two_dated_builds_with_different_as_of_do_not_share_a_cache_entry(
    tmp_path, monkeypatch
):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")

    build_disease_graph("HIV infection", as_of="2018-06", resolver=resolver)
    after_first = calls["n"]
    assert after_first > 0

    build_disease_graph("HIV infection", as_of="2020-01", resolver=resolver)
    assert calls["n"] > after_first


def test_an_undated_build_still_hits_its_own_cache(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)

    build_disease_graph("HIV infection")
    after_first = calls["n"]
    assert after_first > 0

    build_disease_graph("HIV infection")
    # Same disease, same params, no as_of -- must be a cache hit.
    assert calls["n"] == after_first
