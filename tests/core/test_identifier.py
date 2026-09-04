"""
Tests for neorx.core.causal.identifier
====================================

The identifier is the NOVEL core of NeoRx.  These tests
verify that it correctly distinguishes causal targets from
correlational bystanders.

The gold-standard validation case: for HIV, CCR5 should be
classified as CAUSAL (it is the co-receptor — maraviroc works),
while TNF-α should be CORRELATIONAL (elevated in HIV but
inhibiting it worsens outcomes).
"""

import pytest
from neorx.core.causal.identifier import (
    identify_causal_targets,
    _find_disease_node,
    _find_causal_pathway,
    _sensitivity_analysis,
)
from neorx.core.causal.scoring import (
    compute_causal_confidence as _compute_causal_confidence,
    classify_target as _classify_target,
)
from neorx.core.graph.models import (
    NeoRxResult,
    TargetClassification,
    NodeType,
)


class TestIdentifyNeoRxs:
    """Integration tests for causal target identification.

    Uses the session-scoped ``hiv_graph`` fixture from conftest.py.
    """

    def test_returns_results(self, hiv_graph):
        targets = identify_causal_targets(hiv_graph, top_n=10)
        assert len(targets) > 0
        assert all(isinstance(t, NeoRxResult) for t in targets)

    def test_results_sorted_by_confidence(self, hiv_graph):
        targets = identify_causal_targets(hiv_graph, top_n=10)
        confs = [t.causal_confidence for t in targets]
        assert confs == sorted(confs, reverse=True)

    def test_ccr5_ranks_above_tnf(self, hiv_graph):
        """CCR5 (genuine causal target) should rank above TNF (correlational)."""
        targets = identify_causal_targets(hiv_graph, top_n=20)
        gene_names = [t.gene_name for t in targets]

        if "CCR5" in gene_names and "TNF" in gene_names:
            ccr5_idx = gene_names.index("CCR5")
            tnf_idx = gene_names.index("TNF")
            assert ccr5_idx < tnf_idx, (
                f"CCR5 should rank higher than TNF. "
                f"CCR5 at {ccr5_idx}, TNF at {tnf_idx}"
            )

    def test_has_classification(self, hiv_graph):
        targets = identify_causal_targets(hiv_graph, top_n=5)
        for t in targets:
            assert t.classification in (
                TargetClassification.CAUSAL,
                TargetClassification.CORRELATIONAL,
                TargetClassification.INCONCLUSIVE,
            )

    def test_causal_targets_have_reasoning(self, hiv_graph):
        targets = identify_causal_targets(hiv_graph, top_n=5)
        for t in targets:
            assert t.reasoning, f"{t.gene_name} missing reasoning"

    def test_top_n_respected(self, hiv_graph):
        targets = identify_causal_targets(hiv_graph, top_n=3)
        assert len(targets) <= 3


class TestFindDiseasePath:
    """Test causal pathway finding."""

    def test_finds_disease_node(self, hiv_networkx):
        node = _find_disease_node(hiv_networkx, "HIV")
        assert node is not None

    def test_finds_pathway_to_disease(self, hiv_networkx):
        disease_node = _find_disease_node(hiv_networkx, "HIV")
        # CCR5 should have a path to the disease
        path = _find_causal_pathway(hiv_networkx, "gene:CCR5", disease_node)
        assert len(path) > 0


class TestSensitivityAnalysis:
    """Test robustness estimation."""

    def test_returns_float_between_0_and_1(self, hiv_networkx):
        disease_node = _find_disease_node(hiv_networkx, "HIV")
        rob = _sensitivity_analysis(hiv_networkx, "gene:CCR5", disease_node, 0.5)
        assert 0.0 <= rob <= 1.0

    def test_nonzero_effect_gives_some_robustness(self, hiv_networkx):
        disease_node = _find_disease_node(hiv_networkx, "HIV")
        rob = _sensitivity_analysis(hiv_networkx, "gene:CCR5", disease_node, 0.8)
        assert rob > 0.0


class TestCausalConfidence:
    """Test composite causal confidence computation."""

    def test_high_evidence_gives_high_confidence(self):
        conf = _compute_causal_confidence(
            effect=0.8, robustness=0.9, is_identifiable=True,
            n_pathways=5, n_interactions=10,
            source_scores={"Monarch": 0.85, "OpenTargets": 0.9},
            druggability=0.8,
        )
        assert conf > 0.6

    def test_low_evidence_gives_low_confidence(self):
        conf = _compute_causal_confidence(
            effect=0.1, robustness=0.1, is_identifiable=False,
            n_pathways=0, n_interactions=0,
            source_scores={},
            druggability=0.1,
        )
        assert conf < 0.3

    def test_returns_between_0_and_1(self):
        conf = _compute_causal_confidence(
            effect=0.5, robustness=0.5, is_identifiable=True,
            n_pathways=2, n_interactions=3,
            source_scores={"Monarch": 0.5},
            druggability=0.5,
        )
        assert 0.0 <= conf <= 1.0


class TestClassifyTarget:
    """Test target classification logic."""

    def test_causal_classification(self):
        classification, reasoning = _classify_target(
            gene_name="CCR5",
            causal_confidence=0.8,
            robustness=0.7,
            is_identifiable=True,
            n_pathways=3,
            druggability=0.8,
        )
        assert classification == TargetClassification.CAUSAL
        assert "CCR5" in reasoning

    def test_correlational_classification(self):
        classification, reasoning = _classify_target(
            gene_name="TNF",
            causal_confidence=0.2,
            robustness=0.1,
            is_identifiable=False,
            n_pathways=1,
            druggability=0.5,
        )
        assert classification == TargetClassification.CORRELATIONAL
        assert "correlational" in reasoning.lower()

    def test_inconclusive_classification(self):
        classification, _ = _classify_target(
            gene_name="GENE_X",
            causal_confidence=0.5,
            robustness=0.4,
            is_identifiable=True,
            n_pathways=1,
            druggability=0.3,
        )
        assert classification == TargetClassification.INCONCLUSIVE


class TestIdentificationIsReported:
    """The verdict, not an assumption.

    Identifiability used to be `len(causal_pathway) > 0` -- the existence
    of any path -- while the adjustment set was computed and discarded.
    These tests pin the replacement.
    """

    def test_result_carries_the_identification_reason(self, hiv_graph):
        from neorx.core.causal.backdoor import IdentificationReason

        targets = identify_causal_targets(hiv_graph, top_n=5)
        assert targets
        valid = {r.value for r in IdentificationReason}
        for t in targets:
            assert t.identification_reason in valid

    def test_non_identifiable_targets_have_an_empty_adjustment_set(self, hiv_graph):
        targets = identify_causal_targets(hiv_graph, top_n=10)
        for t in targets:
            if not t.identifiable:
                assert t.adjustment_set == []

    def test_identifiability_no_longer_tracks_mere_path_existence(self, hiv_graph):
        # A literature-only graph has paths in the full graph but none in
        # the causal subgraph, so nothing may be identifiable.
        targets = identify_causal_targets(hiv_graph, top_n=10)
        for t in targets:
            if t.identification_reason == "no_causal_path":
                assert not t.identifiable

    def _pathogen_result(self, metadata: dict, disease: str = "malaria"):
        import networkx as nx

        from neorx.core.causal.scoring import evaluate_pathogen_target

        G = nx.DiGraph()
        G.add_node(
            "PATHOGEN:X",
            name="PfDHFR",
            node_type="pathogen_gene",
            metadata=metadata,
            pdb_ids=[],
        )
        return evaluate_pathogen_target(
            G, "PATHOGEN:X", "MONDO:1", None,
            causal_pathway=[], disease_name=disease,
        )

    def test_a_pathogen_target_is_not_scored_as_identifiable(self):
        """The score must agree with the verdict the same result reports.

        ``evaluate_pathogen_target`` reports identifiable=False /
        "no_causal_path" -- pathogen targets bypass causal path analysis,
        so the backdoor criterion is never attempted. Its confidence
        formula nonetheless awarded the full ``0.15 * 1.0``
        identifiability bonus, so the published failure taxonomy counted
        these as no_causal_path failures while the results table scored
        them as if identified. A node with *empty* metadata reached
        confidence 0.67, over the 0.6 CAUSAL threshold, on four constants.
        """
        result = self._pathogen_result({})

        assert result.identifiable is False
        assert result.identification_reason == "no_causal_path"

        # The remaining terms, with no identifiability bonus:
        # 0.40*drug_score(0.5) + 0.25*druggability(0.8)
        # + 0.10*org_relevance(0.3) + 0.10*specificity(0.9)
        assert result.causal_confidence == pytest.approx(0.52)
        assert result.causal_confidence < 0.6, (
            "an empty-metadata pathogen node must not clear the CAUSAL "
            "threshold on constants alone"
        )

    def test_pathogen_confidence_still_tracks_real_drug_evidence(self):
        """Removing the bonus lowers scores; it does not flatten them.

        A Phase 4 target of the organism that actually causes the disease
        still clears the threshold on evidence rather than on constants.
        """
        strong = self._pathogen_result(
            {
                "chembl_drug_evidence_score": 0.9,
                "clinical_phase": 4,
                "n_drugs": 3,
                "pathogen_organism": "Plasmodium falciparum",
            },
        )
        assert strong.causal_confidence == pytest.approx(0.80)
        assert strong.is_causal_target
        assert strong.causal_confidence > self._pathogen_result({}).causal_confidence


class TestCorroborationCountsPrimaryEvidence:
    def test_an_aggregator_does_not_double_count_a_shared_interaction(self):
        import networkx as nx
        from neorx.core.causal.evidence import corroboration_factor

        G = nx.DiGraph()
        # STRING and OmniPath both report the same underlying interaction;
        # OmniPath names STRING among its primary sources.
        G.add_edge("gene:A", "gene:B", source_db="STRING",
                   primary_sources=["STRING"])
        G.add_edge("gene:A", "gene:C", source_db="OmniPath",
                   primary_sources=["STRING"])

        # One distinct primary source, not two aggregators.
        assert corroboration_factor(G, "gene:A") == 1.1

    def test_distinct_primary_sources_each_count(self):
        import networkx as nx
        from neorx.core.causal.evidence import corroboration_factor

        G = nx.DiGraph()
        G.add_edge("gene:A", "gene:B", source_db="OmniPath",
                   primary_sources=["SIGNOR", "TRRUST"])
        assert corroboration_factor(G, "gene:A") == pytest.approx(1.2)

    def test_an_edge_without_primary_sources_falls_back_to_its_database(self):
        import networkx as nx
        from neorx.core.causal.evidence import corroboration_factor

        G = nx.DiGraph()
        G.add_edge("gene:A", "disease:d", source_db="Monarch",
                   primary_sources=[])
        assert corroboration_factor(G, "gene:A") == pytest.approx(1.1)
