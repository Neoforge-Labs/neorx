"""
Counting the evidence behind a candidate target.

How many pathways mention it, how many proteins it interacts with, how
many independent databases report it, and how strongly the graph's edge
weights connect it to the disease. None of this is causal inference --
it is the bookkeeping that ranks candidates once identification has said
which claims are admissible.
"""

from __future__ import annotations

import os
from typing import Any

import networkx as nx
import numpy as np

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


def _estimate_evidence_score(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> float:
    """Estimate causal effect via multi-source evidence triangulation.

    Instead of fabricating synthetic data (which is scientifically
    circular), we compute the causal effect from the knowledge
    graph itself:

    1. **Path strength**: Product of edge weights along the
       shortest causal path.  Stronger edges (higher evidence)
       yield stronger effects.

    2. **Adjustment set strength**: Whether an adjustment set was
       identified by ``find_adjustment_set``. Heuristic weight, not a
       d-separation result; identification is done there, not here.

    3. **Topological importance**: Betweenness centrality —
       central nodes have broader causal influence.

    4. **Multi-source corroboration**: How many independent
       databases confirm this treatment → outcome link?

    For production with real patient data (GEO, TCGA, UK Biobank),
    replace this with DoWhy on observed expression/genotype data.

    Returns
    -------
    float
        A weighted multi-source evidence score. This is not a causal
        effect size: no interventional or patient-level data enters it.
        It ranks candidates; it does not estimate a magnitude.
    """
    score = G.nodes[treatment].get("score", 0.0) if G.has_node(treatment) else 0.0

    # 1. Path-based strength
    path_strength = compute_path_strength(G, treatment, outcome)

    # 2. Adjustment set weight
    # Heuristic weight distinguishing targets with an adjustment set (1.0) from
    # those without (0.8). NOT a d-separation result; identification is decided
    # by neorx.core.causal.backdoor.find_adjustment_set.
    dsep_factor = 1.0 if adjustment_set else 0.8

    # 3. Topological importance (betweenness centrality)
    try:
        centrality = nx.betweenness_centrality(G)
        cent_score = centrality.get(treatment, 0.0)
    except Exception:
        cent_score = 0.0

    # 4. Multi-source corroboration
    source_factor = corroboration_factor(G, treatment)

    # Direct causal edge bonus
    direct_causal = False
    for _, tgt, edata in G.edges(treatment, data=True):
        if edata.get("edge_type") == "causes":
            direct_causal = True
            break

    evidence = score * path_strength * dsep_factor * (1.0 + cent_score) * source_factor
    if direct_causal:
        evidence *= 1.5

    return evidence


def _sensitivity_analysis(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    original_effect: float,
) -> float:
    """Leave-one-source-out sensitivity analysis.

    For each data source, we remove its edges and recompute
    the path strength.  If the effect is stable across source
    removals, the finding is robust.

    Additional checks:
    - Directed path existence (causal mechanism)
    - Multi-source connectivity (convergent evidence)

    Returns
    -------
    float
        Robustness score 0–1.  Higher = more robust.
    """
    if not G.has_node(treatment):
        return 0.0

    seed = int(os.environ.get("NEORX_SEED", "42"))
    rng = np.random.default_rng(hash(treatment) % (2**31) + seed)
    robustness_scores: list[float] = []

    # ── Test 1: Leave-one-source-out stability ──────────────

    # Collect all source databases for edges touching treatment
    all_edges = list(G.edges(treatment, data=True)) + list(G.in_edges(treatment, data=True))
    sources = {e[2].get("source_db", "") for e in all_edges if e[2].get("source_db")}

    if len(sources) > 1:
        source_effects: list[float] = []
        for excluded in sources:
            G_reduced = G.copy()
            to_remove = [
                (u, v) for u, v, d in G_reduced.edges(data=True)
                if d.get("source_db", "") == excluded
            ]
            G_reduced.remove_edges_from(to_remove)
            try:
                ps = compute_path_strength(G_reduced, treatment, outcome)
                node_score = G_reduced.nodes[treatment].get("score", 0.0) if G_reduced.has_node(treatment) else 0.0
                source_effects.append(node_score * ps)
            except Exception:
                source_effects.append(0.0)

        if source_effects:
            mean_eff = float(np.mean(source_effects))
            std_eff = float(np.std(source_effects))
            cv = std_eff / (abs(mean_eff) + 1e-8) if mean_eff != 0 else 1.0
            source_stability = max(0.0, min(1.0, 1.0 - cv))
        else:
            source_stability = 0.3
    else:
        source_stability = 0.3  # Single source → low robustness

    robustness_scores.append(source_stability)

    # ── Test 2: Directed path existence ─────────────────────

    try:
        has_directed = nx.has_path(G, treatment, outcome)
    except nx.NodeNotFound:
        has_directed = False

    if has_directed:
        try:
            path_len = nx.shortest_path_length(G, treatment, outcome)
            path_robustness = min(1.0, 1.0 / path_len)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_robustness = 0.0
    else:
        try:
            has_undir = nx.has_path(G.to_undirected(), treatment, outcome)
            path_robustness = 0.3 if has_undir else 0.0
        except nx.NodeNotFound:
            path_robustness = 0.0

    robustness_scores.append(min(1.0, path_robustness))

    # ── Test 3: Multi-source connectivity ───────────────────

    in_degree = G.in_degree(treatment)
    out_degree = G.out_degree(treatment)
    connectivity_robustness = min(1.0, (in_degree + out_degree) / 10.0)
    robustness_scores.append(connectivity_robustness)

    return float(np.mean(robustness_scores))
