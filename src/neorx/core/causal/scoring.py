"""
Turning evidence into a ranked, labelled target.

Druggability, the combined causal confidence, and the final
causal-versus-correlational classification. These are weighted
heuristics over the evidence counts, and they are named as such: nothing
here estimates a causal effect.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from neorx.core.graph.models import DiseaseGraph, NeoRxResult, TargetClassification
from neorx.core.bio.classifier import TargetType


def assess_druggability(node_data: dict[str, Any]) -> float:
    """Score druggability based on available evidence.

    Uses Open Targets tractability data when available,
    structural information, and protein family heuristics.
    """
    score = 0.3  # Base

    pdb_ids = node_data.get("pdb_ids", [])
    if pdb_ids:
        score += 0.2  # Has 3D structure

    uniprot_id = node_data.get("uniprot_id", "")
    if uniprot_id:
        score += 0.1  # Well-characterised protein

    # Open Targets tractability data (propagated from graph_builder)
    metadata = node_data.get("metadata", {})
    tractability = metadata.get("tractability", [])
    if tractability:
        for entry in tractability:
            if isinstance(entry, dict) and entry.get("value"):
                score += 0.15
                break  # At least one modality is tractable

    # UniProt druggability flag
    if metadata.get("is_druggable"):
        score += 0.15

    # Protein family heuristic from description text
    description = node_data.get("description", "").lower()
    druggable_keywords = [
        "receptor", "kinase", "protease", "enzyme", "channel",
        "transporter", "gpcr", "nuclear receptor",
    ]
    if any(kw in description for kw in druggable_keywords):
        score += 0.15

    return min(1.0, score)


def compute_causal_confidence(
    effect: float,
    robustness: float,
    is_identifiable: bool,
    n_pathways: int,
    n_interactions: int,
    source_scores: dict[str, float],
    druggability: float,
    n_active_sources: int = 4,
    n_associated_diseases: int = 0,
) -> float:
    """Compute composite causal confidence score.

    Weights:
    - Causal effect magnitude: 30%
    - Robustness (sensitivity analysis): 25%
    - Identifiability (backdoor criterion): 15%
    - Multi-source consensus: 10%
    - Disease specificity: 10%
    - Druggability: 10%

    Disease specificity replaces the former "network centrality"
    weight.  Hub genes (TP53, AKT1) are associated with thousands
    of diseases — they are generic, not specific.  Specificity
    rewards targets that are uniquely linked to the disease under
    study, penalising promiscuous hubs.

    Formula: specificity = 1 / log2(n_diseases + 2)
    - Gene linked to 1 disease:   specificity = 1.0
    - Gene linked to 10 diseases: specificity = 0.29
    - Gene linked to 100:         specificity = 0.15
    - Gene linked to 1000:        specificity = 0.10
    """
    # Normalise effect to 0-1
    effect_norm = min(1.0, effect)

    # Multi-source consensus: more sources = higher
    n_sources = len(source_scores)
    avg_source_score = float(np.mean(list(source_scores.values()))) if source_scores else 0.0
    consensus = min(1.0, (n_sources / max(1, n_active_sources)) * avg_source_score)

    # Disease specificity — replaces network centrality
    if n_associated_diseases > 0:
        specificity = 1.0 / np.log2(n_associated_diseases + 2)
    else:
        # No data → neutral (0.5), neither reward nor penalise
        specificity = 0.5

    confidence = (
        0.30 * effect_norm
        + 0.25 * robustness
        + 0.15 * (1.0 if is_identifiable else 0.0)
        + 0.10 * consensus
        + 0.10 * min(1.0, specificity)
        + 0.10 * druggability
    )

    return round(min(1.0, max(0.0, confidence)), 4)


def classify_target(
    gene_name: str,
    causal_confidence: float,
    robustness: float,
    is_identifiable: bool,
    n_pathways: int,
    druggability: float,
    target_type: TargetType = TargetType.CORRELATIONAL,
    tissue_relevant: bool = True,
    evidence_streams: int = 0,
) -> tuple[TargetClassification, str]:
    """Classify a target as causal, correlational, or inconclusive.

    Classification rules:

    **Automatic demotion** (overrides confidence scores):
    - HOST_SYMPTOM targets → always CORRELATIONAL
    - tissue_relevant=False → always CORRELATIONAL

    **Tissue gate** (boolean, not a modifier):
    - tissue_relevant is True/False from the tissue filter.
    - True = gene is expressed in a disease-relevant tissue
      (or expression unknown → pass).
    - False = gene is only expressed in irrelevant tissues
      → demoted to CORRELATIONAL regardless of confidence.
    - The gate NEVER modifies causal_confidence.  Confidence
      stays pure — it measures causal evidence quality, not
      tissue expression.

    **Evidence triangulation** (for CAUSAL status):
    - Must have ≥2 independent evidence streams
    - causal_confidence ≥ 0.6 AND robust AND identifiable

    **Standard rules**:
    - Correlational: confidence < 0.4 OR not robust
    - Inconclusive: everything in between
    """
    reasons = []

    # ── Biological overrides (before confidence check) ──────

    # 1. Symptom markers are NEVER causal drug targets
    if target_type == TargetType.HOST_SYMPTOM:
        classification = TargetClassification.CORRELATIONAL
        reasons.append(
            f"⚠ {gene_name} classified as HOST_SYMPTOM: this gene "
            f"encodes a receptor/channel associated with disease "
            f"symptoms (e.g. seizures, pain), not with the disease "
            f"mechanism itself. Targeting symptom markers does not "
            f"treat the underlying disease."
        )
        reasons.append(
            f"Confidence was {causal_confidence:.2f} but biological "
            f"classification overrides statistical score."
        )
        return classification, " ".join(reasons)

    # 2. Tissue gate — independent boolean criterion
    #    If tissue_relevant is False, the gene is expressed only
    #    in tissues unrelated to this disease.  Demote regardless
    #    of how strong the statistical evidence looks.
    if not tissue_relevant:
        classification = TargetClassification.CORRELATIONAL
        reasons.append(
            f"⚠ {gene_name} FAILED tissue gate: expressed only in "
            f"tissues not relevant to this disease. "
            f"Confidence was {causal_confidence:.2f} but tissue "
            f"expression does not support this target."
        )
        return classification, " ".join(reasons)

    # ── Standard classification with evidence triangulation ──

    # When evidence_streams is explicitly provided (>0), enforce
    # the triangulation requirement.  When not provided (legacy
    # callers using default of 0), fall back to the original
    # thresholds for backward compatibility.
    triangulation_ok = evidence_streams >= 2 or evidence_streams == 0

    if (
        causal_confidence >= 0.6
        and robustness >= 0.4
        and is_identifiable
        and triangulation_ok
    ):
        classification = TargetClassification.CAUSAL
        reasons.append(
            f"{gene_name} has causal confidence {causal_confidence:.2f}, "
            f"supported by {n_pathways} pathway(s), robustness score "
            f"{robustness:.2f}, and {evidence_streams} independent "
            f"evidence stream(s). Tissue gate: PASS."
        )
        if target_type in (TargetType.PATHOGEN_DIRECT, TargetType.HOST_INVASION):
            reasons.append(
                f"Target type {target_type.value}: this is a direct "
                f"disease-mechanism target."
            )
        if druggability >= 0.5:
            reasons.append(
                f"Druggability score {druggability:.2f} indicates tractable "
                f"target with known 3D structures."
            )
        reasons.append(
            "Backdoor criterion satisfied: causal effect is identifiable "
            "after adjusting for confounders."
        )

    elif causal_confidence < 0.4 or robustness < 0.3:
        classification = TargetClassification.CORRELATIONAL
        reasons.append(
            f"{gene_name} is likely correlational (confidence={causal_confidence:.2f}, "
            f"robustness={robustness:.2f})."
        )
        if not is_identifiable:
            reasons.append(
                "No valid causal path identified — association may be "
                "due to confounding."
            )
        reasons.append(
            "Sensitivity analysis suggests the association is fragile "
            "and may not survive intervention."
        )

    elif evidence_streams > 0 and evidence_streams < 2 and causal_confidence >= 0.6:
        # High confidence but insufficient independent evidence
        classification = TargetClassification.INCONCLUSIVE
        reasons.append(
            f"{gene_name} has confidence {causal_confidence:.2f} but only "
            f"{evidence_streams} evidence stream(s). ≥2 required for CAUSAL."
        )

    else:
        classification = TargetClassification.INCONCLUSIVE
        reasons.append(
            f"{gene_name} has moderate evidence (confidence={causal_confidence:.2f}) "
            f"but insufficient data for definitive classification."
        )

    return classification, " ".join(reasons)


# ── Organism–Disease Relevance ─────────────────────────────────────

# Keyword mapping: which organisms are the primary pathogens
# for each disease.  This is basic epidemiology, not target curation.
_DISEASE_ORGANISMS: dict[str, list[str]] = {
    "malaria": ["plasmodium", "falciparum", "vivax", "malariae", "ovale", "knowlesi"],
    "hiv": ["immunodeficiency", "hiv"],
    "ebola": ["ebola", "ebolavirus"],
    "tuberculosis": ["tuberculosis", "mycobacterium"],
    "hepatitis": ["hepatitis"],
    "covid": ["sars", "coronavirus"],
    "influenza": ["influenza"],
    "dengue": ["dengue"],
    "zika": ["zika"],
    "cholera": ["vibrio", "cholera"],
    "typhoid": ["salmonella", "typhi"],
    "leprosy": ["leprae", "leprosy"],
    "chagas": ["trypanosoma", "cruzi"],
    "sleeping sickness": ["trypanosoma", "brucei"],
    "leishmaniasis": ["leishmania"],
}


def organism_disease_relevance(
    organism: str, disease_name: str,
) -> float:
    """Score how relevant a pathogen organism is to a disease.

    Returns 1.0 if the organism matches the primary pathogen,
    0.3 for generic/unrelated organisms, enabling the confidence
    formula to demote off-target pathogens.

    For non-infectious diseases (cancer, neurological, metabolic),
    ALL pathogen targets get 0.0 — these diseases have no
    causative pathogen, so any pathogen target in ChEMBL is from
    co-prescribed medications (e.g. antibiotics for Alzheimer's
    patients) and should not be ranked.
    """
    if not organism or not disease_name:
        return 0.0

    org_lower = organism.lower()
    disease_lower = disease_name.lower()

    # Check explicit mapping first
    for disease_key, keywords in _DISEASE_ORGANISMS.items():
        if disease_key in disease_lower:
            for kw in keywords:
                if kw in org_lower:
                    return 1.0
            # Disease matched a mapping but organism didn't → off-target
            return 0.3

    # No explicit mapping → this disease has no known pathogen.
    # All pathogen targets are irrelevant (from co-prescribed
    # medications like antibiotics, not disease-specific drugs).
    return 0.0


def evaluate_pathogen_target(
    G: nx.DiGraph,
    target_id: str,
    disease_id: str,
    graph: DiseaseGraph,
    *,
    causal_pathway: list[str],
    n_active_sources: int = 4,
    disease_name: str = "",
) -> NeoRxResult:
    """Evaluate a pathogen target from ChEMBL.

    Pathogen targets (e.g. PfDHFR-TS, HIV-1 protease) are
    validated by existing drugs.  Their confidence is based on
    drug evidence rather than causal graph analysis:

    - Clinical phase = highest weight (Phase 4 approved drug
      = maximum confidence)
    - Number of drugs targeting this protein
    - Mechanism-of-action diversity

    They bypass:
    - Biological classifier (not host genes)
    - Tissue filter (pathogen proteins don't express in human tissue)
    - Causal path analysis (they're direct drug targets)

    ``causal_pathway`` is computed by the caller (``identifier.py`` still
    owns ``_find_causal_pathway``, which stays there for the tests that
    exercise it directly) and threaded through here rather than
    recomputed, so this function only needs the graph for node lookups.
    """
    node_data = G.nodes[target_id]
    gene_name = node_data.get("name", target_id)
    metadata = node_data.get("metadata", {})

    # Use graph.disease_name as fallback if disease_name not provided
    if not disease_name and graph:
        disease_name = graph.disease_name

    # Drug evidence score from ChEMBL (already computed)
    drug_score = metadata.get("chembl_drug_evidence_score", 0.5)
    clinical_phase = metadata.get("clinical_phase", 0)
    n_drugs = metadata.get("n_drugs", 0)
    drugs = metadata.get("drugs", [])
    moas = metadata.get("mechanisms_of_action", [])
    organism = metadata.get("pathogen_organism", "unknown pathogen")

    # ── Organism-disease relevance ──────────────────────────────
    #    ChEMBL returns ALL targets of drugs indicated for a disease,
    #    including co-infection antibiotics and anti-helminthics.
    #    A Phase 4 bacterial ribosome target shouldn't score as high
    #    as a Phase 4 P. falciparum DHFR for malaria.
    org_relevance = organism_disease_relevance(organism, disease_name)

    # Source scores — only ChEMBL for pathogen targets
    source_scores = {"ChEMBL": drug_score}

    # Druggability — pathogen drug targets are druggable by definition
    druggability = 1.0 if clinical_phase >= 3 else 0.8

    # Causal confidence for pathogen targets:
    # Based entirely on drug evidence (not graph topology)
    #   40% drug_score (phase + drug diversity + MOA diversity)
    #   25% druggability (always high for validated targets)
    #   15% identifiability (has path to disease? always yes)
    #   10% organism relevance (is this pathogen THE cause?)
    #   10% specificity (pathogen targets are highly specific)
    confidence = (
        0.40 * drug_score
        + 0.25 * druggability
        + 0.15 * 1.0  # always identifiable (known drug target)
        + 0.10 * org_relevance  # organism must match the disease
        + 0.10 * 0.9  # pathogen targets are disease-specific
    )
    confidence = round(min(1.0, max(0.0, confidence)), 4)

    # Robust if Phase 3+ with multiple drugs AND organism matches
    robustness = 0.0
    if clinical_phase >= 4:
        robustness = 0.9
    elif clinical_phase >= 3:
        robustness = 0.7
    elif clinical_phase >= 2:
        robustness = 0.5
    elif clinical_phase >= 1:
        robustness = 0.3
    if n_drugs >= 3:
        robustness = min(1.0, robustness + 0.1)

    # Penalise robustness for off-target organisms
    if org_relevance < 0.5:
        robustness *= 0.3  # heavy penalty — wrong organism

    # Evidence streams: ChEMBL drug evidence = 1 stream
    # Plus structural if PDB IDs exist
    evidence_streams = 1  # ChEMBL
    if node_data.get("pdb_ids"):
        evidence_streams += 1

    # Classification
    if confidence >= 0.6 and robustness >= 0.4:
        classification = TargetClassification.CAUSAL
        reasoning = (
            f"🦠 {gene_name} is a validated PATHOGEN drug target "
            f"({organism}). Phase {clinical_phase} with "
            f"{n_drugs} drug(s): {', '.join(drugs[:3])}. "
            f"MOA: {', '.join(moas[:2])}. "
            f"Drug evidence score: {drug_score:.2f}."
        )
    elif confidence >= 0.4:
        classification = TargetClassification.INCONCLUSIVE
        reasoning = (
            f"🦠 {gene_name} is a pathogen target ({organism}) "
            f"with Phase {clinical_phase} evidence. Confidence "
            f"{confidence:.2f} is moderate."
        )
    else:
        classification = TargetClassification.CORRELATIONAL
        reasoning = (
            f"🦠 {gene_name} ({organism}) has weak drug evidence "
            f"(Phase {clinical_phase}, confidence {confidence:.2f})."
        )

    return NeoRxResult(
        protein_id=target_id,
        protein_name=gene_name,
        gene_name=gene_name,
        uniprot_id=node_data.get("uniprot_id", ""),
        pdb_ids=node_data.get("pdb_ids", []),
        causal_confidence=confidence,
        adjustment_set=[],
        identifiable=False,
        identification_reason="no_causal_path",
        causal_pathway=causal_pathway,
        robustness_score=robustness,
        druggability_score=druggability,
        classification=classification,
        is_causal_target=(classification == TargetClassification.CAUSAL),
        reasoning=reasoning,
        source_scores=source_scores,
        n_supporting_pathways=0,
        n_protein_interactions=0,
        target_type="PATHOGEN_DIRECT",
        tissue_relevant=True,  # N/A for pathogen targets
        tissue_coverage=0.0,
        tissue_explanation="Pathogen target — tissue gate not applicable",
        evidence_streams=evidence_streams,
    )
