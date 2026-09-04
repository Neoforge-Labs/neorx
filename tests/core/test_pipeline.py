"""
Tests for neorx.pipeline
==================================

Integration tests for the full pipeline and its sub-components.
Uses the session-scoped ``hiv_graph`` fixture so the graph is
built ONCE rather than once per test.
"""

import pytest
from neorx.core.pipeline import (
    run_pipeline,
    _fallback_molecules,
    _estimate_admet,
)
from neorx.core.graph.models import (
    PipelineResult,
    JobStatus,
)


class TestRunPipeline:
    """End-to-end pipeline tests.

    All HIV tests pass ``prebuilt_graph`` to avoid redundant API calls.
    """

    def test_pipeline_completes(self, hiv_graph):
        """The pipeline should complete without errors."""
        result = run_pipeline(
            "HIV",
            top_n_targets=3,
            candidates_per_target=10,
            generate_molecules=True,
            run_docking=False,  # Skip docking for speed
            generate_report=False,
            prebuilt_graph=hiv_graph,
        )
        assert isinstance(result, PipelineResult)
        assert result.job.status == JobStatus.COMPLETE

    def test_pipeline_has_graph(self, hiv_graph):
        result = run_pipeline(
            "HIV", top_n_targets=2, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        assert result.graph is not None
        assert len(result.graph.nodes) > 0

    def test_pipeline_has_causal_targets(self, hiv_graph):
        result = run_pipeline(
            "HIV", top_n_targets=5, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        assert len(result.causal_targets) > 0

    def test_pipeline_with_generation(self, hiv_graph):
        """Pipeline with molecule generation should produce candidates."""
        result = run_pipeline(
            "HIV", top_n_targets=2, candidates_per_target=10,
            generate_molecules=True, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        # May have candidates if any targets are causal
        if result.n_causal_targets > 0:
            assert len(result.scored_candidates) > 0

    def test_pipeline_disease_stored(self, hiv_graph):
        result = run_pipeline(
            "HIV", top_n_targets=2, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        assert result.disease == "HIV"

    def test_pipeline_job_id_set(self, hiv_graph):
        result = run_pipeline(
            "HIV", top_n_targets=2, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        assert result.job.job_id is not None
        assert len(result.job.job_id) > 0

    def test_pipeline_different_disease(self):
        """Separate disease — requires its own API calls."""
        result = run_pipeline(
            "Type 2 Diabetes", top_n_targets=2, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False,
        )
        assert result.disease == "Type 2 Diabetes"
        assert result.graph is not None


class TestFallbackMolecules:
    """Test the fallback molecule generator."""

    def test_returns_molecules(self):
        mols = _fallback_molecules("CCR5")
        assert len(mols) > 0
        assert all(isinstance(s, str) for s in mols)

    def test_same_scaffolds_for_any_gene(self):
        """Fallback molecules are now disease-agnostic."""
        mols_ccr5 = _fallback_molecules("CCR5")
        mols_unknown = _fallback_molecules("UNKNOWN_GENE")
        assert mols_ccr5 == mols_unknown
        assert len(mols_ccr5) == 10

    def test_unknown_gene_returns_general(self):
        mols = _fallback_molecules("UNKNOWN_GENE")
        assert len(mols) > 0  # At least general scaffolds


class TestEstimateAdmet:
    """Test ADMET estimation heuristic."""

    def test_good_properties(self):
        score = _estimate_admet(mw=300.0, logp=2.5, qed=0.7)
        assert score >= 0.7

    def test_bad_mw(self):
        score = _estimate_admet(mw=800.0, logp=2.5, qed=0.7)
        assert score < 0.9  # Penalised

    def test_bad_logp(self):
        score = _estimate_admet(mw=300.0, logp=8.0, qed=0.7)
        assert score < 0.9  # Penalised

    def test_none_values(self):
        score = _estimate_admet(mw=None, logp=None, qed=None)
        assert 0.0 <= score <= 1.0

    def test_clamped(self):
        score = _estimate_admet(mw=300.0, logp=2.5, qed=0.9)
        assert 0.0 <= score <= 1.0


class TestRLPipelineFailureIsReported:
    """A failed RL run must not report COMPLETE.

    ``run_rl_pipeline`` wrapped the RL loop in
    ``except Exception: logger.warning("RL loop failed (%s) — collecting
    partial results.")``. There are no partial results to collect --
    ``all_candidates`` is populated all-or-nothing by
    ``generate_candidates_with_rl`` -- so the job finished COMPLETE with
    zero candidates and a single warning line. That is what hid the
    AttributeError in the RL stage on every run for the project's entire
    history.
    """

    def _minimal_graph(self):
        from neorx.core.graph.models import (
            DiseaseGraph,
            EdgeType,
            GraphEdge,
            GraphNode,
            NodeType,
        )

        return DiseaseGraph(
            disease_name="testdisease",
            disease_id="MONDO:0000001",
            nodes=[
                GraphNode(
                    node_id="MONDO:0000001",
                    name="testdisease",
                    node_type=NodeType.DISEASE,
                    source="test",
                    score=1.0,
                ),
                GraphNode(
                    node_id="GENE:G00",
                    name="G00",
                    node_type=NodeType.GENE,
                    source="test",
                    score=0.5,
                ),
            ],
            edges=[
                GraphEdge(
                    source_id="GENE:G00",
                    target_id="MONDO:0000001",
                    edge_type=EdgeType.ASSOCIATED_WITH,
                    weight=0.6,
                    source_db="test",
                )
            ],
            sources_queried=["test"],
        )

    def test_a_raising_rl_loop_marks_the_job_failed(self, monkeypatch):
        import neorx.core.pipeline as pipeline_module
        from neorx.core.pipeline import run_rl_pipeline

        def boom(*args, **kwargs):
            raise AttributeError("'DrugDiscoveryEnv' object has no attribute '_target_states'")

        monkeypatch.setattr(pipeline_module, "generate_candidates_with_rl", boom)

        result = run_rl_pipeline(
            "testdisease",
            top_n_targets=1,
            n_episodes=1,
            max_steps_per_episode=10,
            prebuilt_graph=self._minimal_graph(),
        )

        assert result.job.status == JobStatus.FAILED
        assert result.job.status != JobStatus.COMPLETE
        assert "_target_states" in result.job.error

    def test_a_missing_causalbiorl_still_falls_back(self, monkeypatch):
        """The one recoverable failure keeps its fallback.

        CausalBioRL is optional and the linear flow is a real
        alternative that produces real candidates -- unlike the broad
        catch this replaced, which produced none.
        """
        import neorx.core.pipeline as pipeline_module
        from neorx.core.pipeline import run_rl_pipeline

        def no_module(*args, **kwargs):
            raise ImportError("No module named 'neorx.causalbiorl'")

        monkeypatch.setattr(pipeline_module, "generate_candidates_with_rl", no_module)
        monkeypatch.setattr(
            pipeline_module, "_generate_for_target", lambda target, n: [],
        )

        result = run_rl_pipeline(
            "testdisease",
            top_n_targets=1,
            n_episodes=1,
            max_steps_per_episode=10,
            prebuilt_graph=self._minimal_graph(),
        )

        assert result.job.status == JobStatus.COMPLETE
