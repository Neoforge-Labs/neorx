"""
Tests for neorx.core.graph.graph_builder
======================================
"""

import pytest
from neorx.core.graph.graph_builder import (
    build_disease_graph,
    disease_graph_to_networkx,
    _extract_gene_symbols,
    _merge_nodes,
)
from neorx.core.graph.models import (
    DiseaseGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)


class TestBuildDiseaseGraph:
    """Test the full graph construction pipeline.

    Uses the session-scoped ``hiv_graph`` fixture from conftest.py
    so the graph is built ONCE across the entire test session.
    """

    def test_build_hiv_graph(self, hiv_graph):
        """Build a graph for HIV — our validation case."""
        assert isinstance(hiv_graph, DiseaseGraph)
        assert hiv_graph.disease_name == "HIV"
        assert len(hiv_graph.nodes) > 0
        assert len(hiv_graph.edges) > 0
        assert hiv_graph.n_genes > 0 or hiv_graph.n_proteins > 0

    def test_graph_has_disease_node(self, hiv_graph):
        """The graph must contain a disease outcome node."""
        disease_nodes = [
            n for n in hiv_graph.nodes if n.node_type == NodeType.DISEASE
        ]
        assert len(disease_nodes) >= 1
        assert "hiv" in disease_nodes[0].node_id.lower()

    def test_graph_sources_queried(self, hiv_graph):
        """All expected data sources should be queried."""
        assert "Monarch" in hiv_graph.sources_queried
        assert "OpenTargets" in hiv_graph.sources_queried
        assert "KEGG" in hiv_graph.sources_queried
        assert "Reactome" in hiv_graph.sources_queried

    def test_graph_has_pathways(self, hiv_graph):
        """The graph should include pathway nodes."""
        assert hiv_graph.n_pathways > 0

    def test_graph_edges_reference_existing_nodes(self, hiv_graph):
        """All edge endpoints must exist as nodes."""
        node_ids = {n.node_id for n in hiv_graph.nodes}
        for edge in hiv_graph.edges:
            assert edge.source_id in node_ids, f"Missing source: {edge.source_id}"
            assert edge.target_id in node_ids, f"Missing target: {edge.target_id}"

    def test_build_diabetes_graph(self):
        """Ensure it works for multiple diseases (separate API calls)."""
        graph = build_disease_graph("Type 2 Diabetes", allow_mocks=True)
        assert graph.disease_name == "Type 2 Diabetes"
        assert len(graph.nodes) > 0


class TestDiseaseGraphToNetworkx:
    """Test NetworkX conversion."""

    def test_conversion_preserves_nodes(self, hiv_graph):
        G = disease_graph_to_networkx(hiv_graph)
        assert len(G.nodes) == len(hiv_graph.nodes)

    def test_conversion_preserves_every_unique_edge_pair(self, hiv_graph):
        # NOT a count comparison. A DiGraph holds one edge per node pair,
        # and the assembled graph legitimately contains several edges for
        # the same pair (STRING and OmniPath both describe protein
        # relationships). Counts therefore differ whenever sources
        # overlap; what must hold is that no PAIR is lost. See
        # TestEdgeCollisionResolution for which edge survives.
        G = disease_graph_to_networkx(hiv_graph)
        assert set(G.edges()) == {
            (e.source_id, e.target_id) for e in hiv_graph.edges
        }

    def test_node_attributes(self, hiv_graph):
        G = disease_graph_to_networkx(hiv_graph)
        for node_id, data in G.nodes(data=True):
            assert "name" in data
            assert "node_type" in data
            assert "score" in data


class TestExtractGeneSymbols:
    """Test gene symbol extraction."""

    def test_extracts_genes(self):
        nodes = [
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE),
            GraphNode(node_id="gene:BRCA1", name="BRCA1", node_type=NodeType.GENE),
            GraphNode(node_id="pathway:P1", name="PI3K", node_type=NodeType.PATHWAY),
        ]
        symbols = _extract_gene_symbols(nodes)
        assert "TP53" in symbols
        assert "BRCA1" in symbols
        assert len(symbols) == 2  # Pathway excluded

    def test_deduplication(self):
        nodes = [
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE),
            GraphNode(node_id="gene:TP53_2", name="tp53", node_type=NodeType.GENE),
        ]
        symbols = _extract_gene_symbols(nodes)
        assert len(symbols) == 1


class TestMergeNodes:
    """Test node merging logic."""

    def test_merge_by_id(self):
        nodes = [
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE,
                      source="Monarch", score=0.8),
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE,
                      source="OpenTargets", score=0.9),
        ]
        edges = []
        merged_n, merged_e = _merge_nodes(nodes, edges)
        assert len(merged_n) == 1
        assert merged_n[0].score == 0.9  # Kept highest

    def test_merge_preserves_uniprot(self):
        nodes = [
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE,
                      uniprot_id=None),
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE,
                      uniprot_id="P04637"),
        ]
        merged_n, _ = _merge_nodes(nodes, [])
        assert merged_n[0].uniprot_id == "P04637"

    def test_edge_deduplication(self):
        edges = [
            GraphEdge(source_id="gene:A", target_id="gene:B",
                      edge_type=EdgeType.INTERACTS_WITH),
            GraphEdge(source_id="gene:A", target_id="gene:B",
                      edge_type=EdgeType.INTERACTS_WITH),
        ]
        _, merged_e = _merge_nodes([], edges)
        assert len(merged_e) == 1

    def test_different_edge_types_preserved(self):
        edges = [
            GraphEdge(source_id="gene:A", target_id="gene:B",
                      edge_type=EdgeType.INTERACTS_WITH),
            GraphEdge(source_id="gene:A", target_id="gene:B",
                      edge_type=EdgeType.ACTIVATES),
        ]
        _, merged_e = _merge_nodes([], edges)
        assert len(merged_e) == 2


class TestEdgeCollisionResolution:
    """A DiGraph cannot hold two edges between the same pair.

    STRING and OmniPath both report protein relationships, so the same
    gene pair frequently arrives twice: once as an undirected
    ``interacts_with`` (not causal-admissible) and once as a directed,
    signed regulatory edge (admissible). Only one survives the
    conversion. Which one must be decided by an explicit rule, not by
    the order the graph builder happens to append them in -- an
    associational edge silently overwriting a causal one removes an
    arrow from the causal subgraph, lowers the identifiability rate,
    and looks like a finding rather than a bug.
    """

    @staticmethod
    def _pair(*edges):
        from neorx.core.graph.models import DiseaseGraph, GraphNode, NodeType

        nodes = [
            GraphNode(node_id=f"gene:{n}", name=n, node_type=NodeType.GENE,
                      source="test", score=0.9)
            for n in ("A", "B")
        ]
        return DiseaseGraph(disease_name="t", nodes=nodes, edges=list(edges))

    @staticmethod
    def _string_edge(weight=0.9):
        from neorx.core.graph.models import EdgeType, GraphEdge
        return GraphEdge(
            source_id="gene:A", target_id="gene:B",
            edge_type=EdgeType.INTERACTS_WITH, weight=weight,
            source_db="STRING", primary_sources=["STRING"],
        )

    @staticmethod
    def _omnipath_edge(weight=0.6):
        from neorx.core.graph.models import EdgeType, GraphEdge
        return GraphEdge(
            source_id="gene:A", target_id="gene:B",
            edge_type=EdgeType.ACTIVATES, weight=weight,
            source_db="OmniPath", evidence_class="regulatory", sign=1,
            primary_sources=["SIGNOR"],
        )

    def test_causal_edge_survives_regardless_of_insertion_order(self):
        from neorx.core.causal.graph_semantics import is_causal_admissible

        for label, edges in (
            ("string first", (self._string_edge(), self._omnipath_edge())),
            ("omnipath first", (self._omnipath_edge(), self._string_edge())),
        ):
            G = disease_graph_to_networkx(self._pair(*edges))
            attrs = G.edges["gene:A", "gene:B"]
            assert is_causal_admissible(attrs), (
                f"{label}: an associational edge overwrote a causal one"
            )
            assert attrs["edge_type"] == "activates", label

    def test_causal_edge_wins_even_when_the_associational_one_weighs_more(self):
        # STRING's weight is higher, but weight never outranks admissibility.
        from neorx.core.causal.graph_semantics import is_causal_admissible

        G = disease_graph_to_networkx(
            self._pair(self._omnipath_edge(weight=0.2), self._string_edge(weight=1.0))
        )
        assert is_causal_admissible(G.edges["gene:A", "gene:B"])

    def test_colliding_edges_merge_their_primary_sources(self):
        # Corroboration counts distinct primary evidence; a collision must
        # not discard the losing edge's provenance.
        G = disease_graph_to_networkx(
            self._pair(self._string_edge(), self._omnipath_edge())
        )
        assert set(G.edges["gene:A", "gene:B"]["primary_sources"]) == {"STRING", "SIGNOR"}

    def test_among_equally_admissible_edges_the_heavier_one_wins(self):
        from neorx.core.graph.models import EdgeType, GraphEdge

        light = GraphEdge(
            source_id="gene:A", target_id="gene:B", edge_type=EdgeType.ACTIVATES,
            weight=0.3, source_db="OmniPath", evidence_class="regulatory", sign=1,
        )
        heavy = GraphEdge(
            source_id="gene:A", target_id="gene:B", edge_type=EdgeType.INHIBITS,
            weight=0.95, source_db="OmniPath", evidence_class="regulatory", sign=-1,
        )
        G = disease_graph_to_networkx(self._pair(light, heavy))
        assert G.edges["gene:A", "gene:B"]["weight"] == 0.95
        assert G.edges["gene:A", "gene:B"]["sign"] == -1

    def test_every_unique_pair_survives_the_conversion(self, hiv_graph):
        # The real contract, replacing the old count-equality assertion:
        # the conversion deduplicates by node pair, so edge COUNTS may
        # differ, but no pair may be lost.
        G = disease_graph_to_networkx(hiv_graph)
        pairs = {(e.source_id, e.target_id) for e in hiv_graph.edges}
        assert set(G.edges()) == pairs
