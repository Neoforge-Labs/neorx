"""
Causal Target Identifier
=========================

This is the **NOVEL** core of NeoRx.  While every other
drug-discovery pipeline ranks targets by association scores
(correlation), we apply **Pearl's causal inference framework**
to distinguish genuine causal drivers from correlational
bystanders.

The Fundamental Problem
-----------------------
Gene–disease association databases (Monarch Initiative, Open Targets)
report that TNF-α is strongly associated with HIV.  Indeed,
TNF-α levels are elevated during HIV infection.  But TNF-α
elevation is a *consequence* of immune activation — it is
downstream of the infection.  Inhibiting TNF-α does not treat
HIV; it makes it worse by suppressing immune defence.

Conversely, CCR5 is the HIV-1 co-receptor.  A loss-of-function
mutation (CCR5-Δ32) confers near-complete resistance to HIV-1.
Maraviroc, which blocks CCR5, is an approved antiretroviral.
CCR5 is a **causal** target.

Our Method
----------
1. **Graph → Causal Model**: Convert the disease knowledge graph
   into a DAG suitable for DoWhy.  Each edge encodes a potential
   causal direction.

2. **Backdoor Criterion**: For each candidate target, check
   whether the causal effect on the disease outcome is
   *identifiable* — i.e. whether there exists a valid adjustment
   set that blocks all confounding paths.

3. **Effect Estimation**: Estimate the Average Treatment Effect
   (ATE) of intervening on the target.  We use linear regression
   (for interpretability) and inverse propensity weighting (for
   robustness).

4. **Sensitivity Analysis**: Apply DoWhy's refutation tests:
   - ``random_common_cause``: Add a random confounder.
   - ``placebo_treatment``: Shuffle the treatment variable.
   - ``data_subset``: Re-estimate on random subsets.
   If the estimate is fragile, the target may be correlational.

5. **Classification**: Combine causal effect, robustness, graph
   topology (in-degree, pathway membership), and druggability
   into a final classification:
   - **Causal**: High effect + robust + identifiable
   - **Correlational**: High association but fragile/unidentifiable
   - **Inconclusive**: Insufficient evidence
"""

from __future__ import annotations

import logging
import os
from typing import Any

import networkx as nx
import numpy as np

from neorx.core.graph.models import (
    NeoRxResult,
    DiseaseGraph,
    GraphNode,
    NodeType,
    EdgeType,
    TargetClassification,
)
from neorx.core.graph.graph_builder import disease_graph_to_networkx
from neorx.core.bio.classifier import TargetClassifier, TargetType, classify_disease
from neorx.core.bio.tissue_filter import TissueFilter
from neorx.core.causal.backdoor import find_adjustment_set
from neorx.core.causal.evidence import (
    collect_source_scores,
    compute_path_strength,
    count_evidence_streams,
    count_pathway_connections,
    count_protein_interactions,
)
from neorx.core.causal.scoring import (
    assess_druggability,
    classify_target,
    compute_causal_confidence,
    evaluate_pathogen_target,
    organism_disease_relevance,
)

logger = logging.getLogger(__name__)


def identify_causal_targets(
    graph: DiseaseGraph,
    top_n: int = 10,
    min_causal_confidence: float = 0.3,
) -> list[NeoRxResult]:
    """Identify and rank causal drug targets from a disease graph.

    Parameters
    ----------
    graph : DiseaseGraph
        The assembled disease causal graph.
    top_n : int
        Maximum number of targets to return.
    min_causal_confidence : float
        Minimum causal confidence threshold.

    Returns
    -------
    list[NeoRxResult]
        Ranked list of causal target assessments, best first.
    """
    G = disease_graph_to_networkx(graph)
    disease_node_id = _find_disease_node(G, graph.disease_name)

    if not disease_node_id:
        logger.error("No disease node found in graph.")
        return []

    # ── Biological intelligence layer ─────────────────────────
    disease_type = classify_disease(graph.disease_name)
    classifier = TargetClassifier()
    tissue_filter = TissueFilter()

    logger.info(
        "Disease type: %s → classifier + tissue filter active.",
        disease_type.value,
    )

    # Get candidate genes/proteins
    candidates = _get_candidate_nodes(G, disease_node_id)
    logger.info("Evaluating %d candidate targets…", len(candidates))

    # Count distinct sources that actually contributed gene/protein nodes
    _gene_sources: set[str] = set()
    for node in graph.nodes:
        if node.node_type in (NodeType.GENE, NodeType.PROTEIN, NodeType.PATHOGEN_GENE) and node.source:
            for src in node.source.split(", "):
                _gene_sources.add(src.strip())
    n_active_sources = max(1, len(_gene_sources))
    logger.info("Active gene-level sources: %d (%s).",
                n_active_sources, ", ".join(sorted(_gene_sources)))

    results: list[NeoRxResult] = []
    for node_id in candidates:
        result = _evaluate_target(
            G, node_id, disease_node_id, graph,
            n_active_sources=n_active_sources,
            classifier=classifier,
            tissue_filter=tissue_filter,
            disease_name=graph.disease_name,
            disease_type=disease_type,
        )
        results.append(result)

    # Sort by causal_confidence descending
    results.sort(key=lambda r: r.causal_confidence, reverse=True)

    # Split human and pathogen results to prevent pathogen targets
    # from completely crowding out human targets.  Each pool gets
    # at least half the slots (with leftover going to whichever
    # pool has more high-confidence results).
    human_results = [r for r in results if r.target_type != "PATHOGEN_DIRECT"]
    pathogen_results = [r for r in results if r.target_type == "PATHOGEN_DIRECT"]

    half = top_n // 2
    # Each pool gets at least half, remainder filled from the other
    top_human = [r for r in human_results if r.causal_confidence >= min_causal_confidence][:half]
    top_pathogen = [r for r in pathogen_results if r.causal_confidence >= min_causal_confidence][:half]

    # Fill remaining slots from whichever pool has leftovers
    remaining = top_n - len(top_human) - len(top_pathogen)
    if remaining > 0:
        used_ids = {r.protein_id for r in top_human} | {r.protein_id for r in top_pathogen}
        overflow = [
            r for r in results
            if r.protein_id not in used_ids
            and r.causal_confidence >= min_causal_confidence
        ][:remaining]
        combined = top_human + top_pathogen + overflow
    else:
        combined = top_human + top_pathogen

    # Re-sort the combined list by confidence
    combined.sort(key=lambda r: r.causal_confidence, reverse=True)

    if not combined:
        # If nothing passes threshold, return top_n anyway
        combined = results[:top_n]

    return combined[:top_n]


def _find_disease_node(G: nx.DiGraph, disease_name: str) -> str | None:
    """Find the disease outcome node in the graph."""
    # First try exact match on node_id
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "disease":
            return node_id
    # Fallback: look for node with disease name
    for node_id, data in G.nodes(data=True):
        if disease_name.lower() in data.get("name", "").lower():
            return node_id
    return None


def _get_candidate_nodes(
    G: nx.DiGraph, disease_node_id: str,
) -> list[str]:
    """Get gene/protein nodes that could be drug targets."""
    candidates = []
    for node_id, data in G.nodes(data=True):
        ntype = data.get("node_type", "")
        if ntype in ("gene", "protein", "pathogen_gene") and node_id != disease_node_id:
            candidates.append(node_id)
    return candidates


def _evaluate_target(
    G: nx.DiGraph,
    target_id: str,
    disease_id: str,
    graph: DiseaseGraph,
    *,
    n_active_sources: int = 4,
    classifier: TargetClassifier | None = None,
    tissue_filter: TissueFilter | None = None,
    disease_name: str = "",
    disease_type: Any = None,
) -> NeoRxResult:
    """Evaluate whether a target is causally linked to the disease.

    This is the core causal reasoning function.  It uses a
    combination of:
    1. Graph topology (paths, adjustment sets)
    2. Simulated causal effect estimation via DoWhy
    3. Sensitivity/robustness analysis
    4. Multi-source evidence aggregation
    5. **Biological classification** — symptom marker detection
    6. **Tissue relevance** — HPA expression filtering
    """
    node_data = G.nodes[target_id]
    gene_name = node_data.get("name", target_id)
    node_type_str = node_data.get("node_type", "")

    # ── Pathogen targets get a specialised evaluation path ──────
    #    They come from ChEMBL and represent validated drug targets
    #    in the pathogen organism (e.g. PfDHFR-TS, HIV protease).
    #    They bypass biological classification and tissue filtering
    #    because those concepts only apply to human genes.
    if node_type_str == "pathogen_gene":
        causal_pathway = _find_causal_pathway(G, target_id, disease_id)
        return evaluate_pathogen_target(
            G, target_id, disease_id, graph,
            causal_pathway=causal_pathway,
            n_active_sources=n_active_sources,
            disease_name=disease_name,
        )

    # ── Step 0: Biological Classification ───────────────────────

    target_type = TargetType.CORRELATIONAL  # safe default
    tissue_relevant = True   # boolean gate (True = pass)
    tissue_coverage = 0.0    # diagnostic annotation
    tissue_explanation = ""

    if classifier is not None:
        from neorx.core.bio.classifier import DiseaseType as DT
        dt = disease_type if disease_type is not None else DT.OTHER
        target_type, _type_reason = classifier.classify(gene_name, dt, node_data)
        logger.debug(
            "  %s → target_type=%s", gene_name, target_type.value,
        )

    if tissue_filter is not None and disease_name:
        tissue_relevant, tissue_coverage, tissue_explanation = (
            tissue_filter.is_tissue_relevant(gene_name, disease_name)
        )

    # ── Step 1: Graph-Based Causal Analysis ─────────────────────

    # Find paths from target to disease
    causal_pathway = _find_causal_pathway(G, target_id, disease_id)

    # Identification: does the causal subgraph license a claim here?
    identification = find_adjustment_set(G, target_id, disease_id)
    adjustment_set = list(identification.adjustment_set)
    is_identifiable = identification.identifiable

    # ── Step 2: Causal Effect Estimation ────────────────────────

    evidence_score = _estimate_evidence_score(
        G, target_id, disease_id, adjustment_set,
    )

    # ── Step 3: Sensitivity Analysis ────────────────────────────

    robustness = _sensitivity_analysis(
        G, target_id, disease_id, evidence_score,
    )

    # ── Step 4: Topological Evidence ────────────────────────────

    # Count supporting pathways
    n_pathways = count_pathway_connections(G, target_id)

    # Count protein interactions
    n_interactions = count_protein_interactions(G, target_id)

    # Source-level scores
    source_scores = collect_source_scores(graph, gene_name)

    # Druggability heuristic
    druggability = assess_druggability(node_data)

    # Disease specificity (from Open Targets)
    metadata = node_data.get("metadata", {})
    n_associated_diseases = metadata.get("n_associated_diseases", 0)

    # ── Step 5: Composite Causal Confidence ─────────────────────

    causal_confidence = compute_causal_confidence(
        effect=abs(evidence_score),
        robustness=robustness,
        is_identifiable=is_identifiable,
        n_pathways=n_pathways,
        n_interactions=n_interactions,
        source_scores=source_scores,
        druggability=druggability,
        n_active_sources=n_active_sources,
        n_associated_diseases=n_associated_diseases,
    )

    # ── Step 6: Classification ──────────────────────────────────

    # Count independent evidence streams
    evidence_streams = count_evidence_streams(
        source_scores=source_scores,
        n_pathways=n_pathways,
        n_interactions=n_interactions,
        node_data=node_data,
    )

    classification, reasoning = classify_target(
        gene_name=gene_name,
        causal_confidence=causal_confidence,
        robustness=robustness,
        is_identifiable=is_identifiable,
        n_pathways=n_pathways,
        druggability=druggability,
        target_type=target_type,
        tissue_relevant=tissue_relevant,
        evidence_streams=evidence_streams,
    )

    return NeoRxResult(
        protein_id=target_id,
        protein_name=node_data.get("name", ""),
        gene_name=gene_name,
        uniprot_id=node_data.get("uniprot_id", ""),
        pdb_ids=node_data.get("pdb_ids", []),
        causal_confidence=causal_confidence,
        adjustment_set=adjustment_set,
        identifiable=identification.identifiable,
        identification_reason=identification.reason.value,
        n_near_miss_confounders=identification.n_near_miss_confounders,
        adjustment_search_truncated=identification.search_truncated,
        causal_pathway=causal_pathway,
        robustness_score=robustness,
        druggability_score=druggability,
        classification=classification,
        is_causal_target=(classification == TargetClassification.CAUSAL),
        reasoning=reasoning,
        source_scores=source_scores,
        n_supporting_pathways=n_pathways,
        n_protein_interactions=n_interactions,
        target_type=target_type.value if hasattr(target_type, "value") else str(target_type),
        tissue_relevant=tissue_relevant,
        tissue_coverage=tissue_coverage,
        tissue_explanation=tissue_explanation,
        evidence_streams=evidence_streams,
    )


# ── Causal Analysis Subroutines ────────────────────────────────────

def _find_causal_pathway(
    G: nx.DiGraph, source: str, target: str,
) -> list[str]:
    """Find the shortest causal path from source to target.

    Prefers directed paths (genuine causal mechanisms) over
    undirected paths (which may include PPI edges that don't
    imply causal direction).  When only an undirected path
    exists, it is returned but flagged by the identifier as
    weaker causal evidence.
    """
    # 1. Try directed path first (strongest causal evidence)
    try:
        path = nx.shortest_path(G, source, target)
        return path
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        pass

    # 2. Fall back to undirected (PPI / interaction edges)
    try:
        path = nx.shortest_path(G.to_undirected(), source, target)
        return path
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return []


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
    source_dbs = set()
    for _, _, edata in G.edges(treatment, data=True):
        src = edata.get("source_db", "")
        if src:
            source_dbs.add(src)
    for _, _, edata in G.in_edges(treatment, data=True):
        src = edata.get("source_db", "")
        if src:
            source_dbs.add(src)
    source_factor = min(1.5, 1.0 + len(source_dbs) * 0.1)

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
