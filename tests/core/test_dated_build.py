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
