"""
Counting the evidence behind a candidate target.

How many pathways mention it, how many proteins it interacts with, how
many independent databases report it, and how strongly the graph's edge
weights connect it to the disease. None of this is causal inference --
it is the bookkeeping that ranks candidates once identification has said
which claims are admissible.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from neorx.core.graph.models import DiseaseGraph


def collect_source_scores(
    graph: DiseaseGraph, gene_name: str,
) -> dict[str, float]:
    """Collect per-source association scores for a gene."""
    scores: dict[str, float] = {}
    for node in graph.nodes:
        if node.name.upper() == gene_name.upper():
            if node.source:
                for src in node.source.split(", "):
                    scores[src] = max(scores.get(src, 0.0), node.score)
    return scores


def count_pathway_connections(G: nx.DiGraph, node_id: str) -> int:
    """Count how many pathways this gene participates in."""
    count = 0
    if not G.has_node(node_id):
        return 0
    for _, target, data in G.edges(node_id, data=True):
        if data.get("edge_type") == "participates_in":
            count += 1
    # Also check incoming
    for source, _, data in G.in_edges(node_id, data=True):
        if data.get("edge_type") == "participates_in":
            count += 1
    return count


def count_protein_interactions(G: nx.DiGraph, node_id: str) -> int:
    """Count protein–protein interactions for this node."""
    count = 0
    if not G.has_node(node_id):
        return 0
    for _, _, data in G.edges(node_id, data=True):
        if data.get("edge_type") == "interacts_with":
            count += 1
    for _, _, data in G.in_edges(node_id, data=True):
        if data.get("edge_type") == "interacts_with":
            count += 1
    return count


def count_evidence_streams(
    source_scores: dict[str, float],
    n_pathways: int,
    n_interactions: int,
    node_data: dict[str, Any],
) -> int:
    """Count independent evidence streams supporting a target.

    Evidence streams:
    1. Gene-disease association databases (Monarch, OpenTargets)
    2. Pathway membership (KEGG, Reactome)
    3. Protein-protein interactions (STRING)
    4. Structural data (PDB)
    5. Druggability / functional annotation (UniProt)
    6. Drug evidence (ChEMBL — validated drug targets)

    Each counts as ONE stream even if multiple sources within
    the category confirm it (e.g. both KEGG and Reactome = 1
    pathway stream, not 2).
    """
    streams = 0

    # Stream 1: Gene-disease association databases
    assoc_sources = {"Monarch", "OpenTargets"}
    if any(s in source_scores for s in assoc_sources):
        streams += 1

    # Stream 2: Pathway membership
    if n_pathways > 0:
        streams += 1

    # Stream 3: Protein interactions
    if n_interactions > 0:
        streams += 1

    # Stream 4: 3D structural data
    pdb_ids = node_data.get("pdb_ids", [])
    if pdb_ids:
        streams += 1

    # Stream 5: Functional annotation / druggability
    metadata = node_data.get("metadata", {})
    if metadata.get("is_druggable") or metadata.get("go_terms"):
        streams += 1

    # Stream 6: ChEMBL drug evidence
    if metadata.get("chembl_drug_evidence_score") or "ChEMBL" in source_scores:
        streams += 1

    return streams


def compute_path_strength(
    G: nx.DiGraph, source: str, target: str,
) -> float:
    """Compute causal path strength from edge weights.

    Shorter paths with higher-weight edges indicate stronger
    causal mechanisms.
    """
    # Try directed path first
    try:
        path = nx.shortest_path(G, source, target)
        if len(path) < 2:
            return 0.0
        # Product of edge weights along path
        strength = 1.0
        for i in range(len(path) - 1):
            edata = G.get_edge_data(path[i], path[i + 1], {})
            strength *= edata.get("weight", 0.5)
        # Discount for path length (shorter = stronger)
        strength *= 1.0 / len(path)
        return strength
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        pass

    # Undirected fallback (weaker evidence)
    try:
        G_u = G.to_undirected()
        path = nx.shortest_path(G_u, source, target)
        strength = 0.5 / len(path)  # Halved for undirected
        return strength
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return 0.1  # Minimal baseline


def corroboration_factor(G: nx.DiGraph, node: str) -> float:
    """Multi-source corroboration, counted over primary evidence.

    An aggregator is not a source. OmniPath re-reports interactions
    STRING also reports, so counting ``source_db`` values would let one
    curated interaction inflate the factor twice. Where an edge names its
    ``primary_sources``, those are counted instead of the aggregator.
    """
    primary: set[str] = set()

    for _, _, attrs in list(G.edges(node, data=True)) + list(
        G.in_edges(node, data=True)
    ):
        named = attrs.get("primary_sources") or []
        if named:
            primary.update(named)
        elif attrs.get("source_db"):
            primary.add(attrs["source_db"])

    return min(1.5, 1.0 + len(primary) * 0.1)
