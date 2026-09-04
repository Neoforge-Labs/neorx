from neorx.core.graph.models import (
    DiseaseGraph, GraphEdge, GraphNode, EdgeType, NodeType,
)
from neorx.core.graph.graph_builder import disease_graph_to_networkx


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


import networkx as nx
from neorx.core.causal.graph_semantics import (
    acyclic_core,
    causal_subgraph,
    cyclic_components,
    is_causal_admissible,
)


def test_regulatory_gene_gene_edge_is_admissible():
    assert is_causal_admissible(
        {"edge_type": "activates", "evidence_class": "regulatory"}
    )


def test_gene_disease_edge_is_admissible_only_with_genetic_evidence():
    genetic = {"edge_type": "associated_with",
               "evidence_class": "genetic_association"}
    somatic = {"edge_type": "associated_with",
               "evidence_class": "somatic_mutation"}
    literature = {"edge_type": "associated_with",
                  "evidence_class": "literature"}
    assert is_causal_admissible(genetic)
    assert is_causal_admissible(somatic)
    assert not is_causal_admissible(literature)


def test_unclassified_association_is_not_admissible():
    assert not is_causal_admissible(
        {"edge_type": "associated_with", "evidence_class": ""}
    )


def test_string_interaction_is_never_admissible():
    # STRING data is undirected; its arrow direction is an artefact of
    # which protein landed in the first response column.
    assert not is_causal_admissible(
        {"edge_type": "interacts_with", "evidence_class": "regulatory"}
    )


def test_pathway_membership_is_never_admissible():
    assert not is_causal_admissible(
        {"edge_type": "participates_in", "evidence_class": ""}
    )


def test_causal_subgraph_keeps_admissible_edges_and_drops_the_rest():
    G = nx.DiGraph()
    G.add_edge("gene:U", "gene:X", edge_type="activates",
               evidence_class="regulatory")
    G.add_edge("gene:X", "disease:d", edge_type="associated_with",
               evidence_class="genetic_association")
    G.add_edge("gene:P", "disease:d", edge_type="associated_with",
               evidence_class="literature")
    G.add_edge("gene:X", "pathway:1", edge_type="participates_in",
               evidence_class="")

    sub = causal_subgraph(G)

    assert set(sub.edges()) == {("gene:U", "gene:X"), ("gene:X", "disease:d")}
    assert "pathway:1" not in sub
    assert "gene:P" not in sub


def test_causal_subgraph_preserves_edge_attributes():
    G = nx.DiGraph()
    G.add_edge("gene:U", "gene:X", edge_type="inhibits",
               evidence_class="regulatory", sign=-1,
               primary_sources=["SIGNOR"], weight=0.8)
    sub = causal_subgraph(G)
    assert sub.edges["gene:U", "gene:X"]["sign"] == -1
    assert sub.edges["gene:U", "gene:X"]["primary_sources"] == ["SIGNOR"]


def test_cyclic_components_reports_only_nontrivial_sccs():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "a")
    G.add_edge("b", "c")
    assert cyclic_components(G) == [frozenset({"a", "b"})]


def test_acyclic_core_removes_cyclic_nodes_and_names_them():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "a")
    G.add_edge("b", "c")
    G.add_edge("c", "d")

    core = acyclic_core(G)

    assert core.excluded == frozenset({"a", "b"})
    assert nx.is_directed_acyclic_graph(core.dag)
    assert set(core.dag.nodes()) == {"c", "d"}


def test_acyclic_core_of_a_dag_excludes_nothing():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    core = acyclic_core(G)
    assert core.excluded == frozenset()
    assert set(core.dag.nodes()) == {"a", "b"}


def test_self_loop_node_is_reported_as_cyclic():
    """A self-loop node should be detected as cyclic."""
    G = nx.DiGraph()
    G.add_edge("a", "a", edge_type="activates", evidence_class="regulatory")
    G.add_edge("a", "b", edge_type="activates", evidence_class="regulatory")

    cyclic = cyclic_components(G)
    assert cyclic == [frozenset({"a"})]


def test_acyclic_core_excludes_self_loop_node_and_returns_dag():
    """A self-loop node should be excluded and result in a valid DAG."""
    G = nx.DiGraph()
    G.add_edge("a", "a", edge_type="activates", evidence_class="regulatory")
    G.add_edge("a", "b", edge_type="activates", evidence_class="regulatory")

    core = acyclic_core(G)

    assert core.excluded == frozenset({"a"})
    assert nx.is_directed_acyclic_graph(core.dag)
    assert set(core.dag.nodes()) == {"b"}
    assert list(core.dag.edges()) == []


def test_empty_graph_through_acyclic_core():
    """An empty graph should behave sanely through acyclic_core."""
    G = nx.DiGraph()

    core = acyclic_core(G)

    assert core.excluded == frozenset()
    assert nx.is_directed_acyclic_graph(core.dag)
    assert len(core.dag) == 0


def test_empty_graph_through_causal_subgraph():
    """An empty graph should behave sanely through causal_subgraph."""
    G = nx.DiGraph()

    sub = causal_subgraph(G)

    assert len(sub) == 0
    assert list(sub.edges()) == []


def test_graph_with_no_admissible_edges_through_causal_subgraph():
    """A graph with only non-admissible edges should filter to empty."""
    G = nx.DiGraph()
    G.add_edge("gene:X", "pathway:1", edge_type="participates_in",
               evidence_class="")
    G.add_edge("protein:Y", "protein:Z", edge_type="interacts_with",
               evidence_class="")

    sub = causal_subgraph(G)

    assert len(sub) == 0
    assert list(sub.edges()) == []


def test_graph_with_no_admissible_edges_through_acyclic_core():
    """A graph with only non-admissible edges should be empty after filtering."""
    G = nx.DiGraph()
    G.add_edge("gene:X", "pathway:1", edge_type="participates_in",
               evidence_class="")

    sub = causal_subgraph(G)
    core = acyclic_core(sub)

    assert core.excluded == frozenset()
    assert nx.is_directed_acyclic_graph(core.dag)
    assert len(core.dag) == 0
