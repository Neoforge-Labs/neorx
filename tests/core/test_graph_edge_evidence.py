"""
GraphEdge carries evidence class, sign, and primary sources to NetworkX.

These three tests are about NeoRx's own graph model, not about causal
admissibility. They shared a file with the admissibility tests until the
rules themselves moved to the ``causalphylo`` package; those tests went
with them, and what the admissibility rules read off an edge --
``edge_type`` and ``evidence_class`` -- is set here.
"""

from neorx.core.graph.graph_builder import disease_graph_to_networkx
from neorx.core.graph.models import (
    DiseaseGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)


def _graph_with_edge(**edge_kwargs) -> DiseaseGraph:
    return DiseaseGraph(
        disease_name="test",
        nodes=[
            GraphNode(node_id="gene:A", name="A", node_type=NodeType.GENE,
                      source="X", score=0.9),
            GraphNode(node_id="disease:test", name="test",
                      node_type=NodeType.DISEASE, source="X", score=1.0),
        ],
        edges=[GraphEdge(source_id="gene:A", target_id="disease:test",
                         **edge_kwargs)],
    )


def test_graph_edge_defaults_are_unclassified_unsigned_and_unsourced():
    edge = GraphEdge(
        source_id="gene:A", target_id="disease:test",
        edge_type=EdgeType.ASSOCIATED_WITH,
    )
    assert edge.evidence_class == ""
    assert edge.sign == 0
    assert edge.primary_sources == []


def test_evidence_class_sign_and_primary_sources_reach_the_networkx_graph():
    graph = _graph_with_edge(
        edge_type=EdgeType.ASSOCIATED_WITH,
        weight=0.7,
        source_db="Open Targets",
        evidence_class="genetic_association",
        sign=-1,
        primary_sources=["SIGNOR", "TRRUST"],
    )
    G = disease_graph_to_networkx(graph)
    attrs = G.edges["gene:A", "disease:test"]
    assert attrs["evidence_class"] == "genetic_association"
    assert attrs["sign"] == -1
    assert attrs["primary_sources"] == ["SIGNOR", "TRRUST"]


def test_sign_rejects_values_outside_minus_one_to_one():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GraphEdge(
            source_id="gene:A", target_id="disease:test",
            edge_type=EdgeType.ASSOCIATED_WITH, sign=2,
        )
