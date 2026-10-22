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
    _estimate_admet,
    _generate_for_target,
    _GENERATION_TEMPERATURE,
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


class TestGenerateForTarget:
    """Candidates must come from the trained GenMol weights.

    ``_generate_for_target`` used to build a tokenizer from thirteen
    hard-coded SMILES and sample from a freshly constructed, untrained
    ``MolVAE`` -- ``load_pretrained()`` was never called. The body was
    wrapped in ``except Exception: logger.warning(...)`` followed by a
    list of curated scaffolds, and random weights usually decode to
    nothing sanitisable, so most runs silently returned those ten
    scaffolds and the rest returned noise (measured: ``["Cl"]`` at seed
    0, ``["O"]`` at seed 1). Downstream screening and docking could not
    distinguish any of the three outcomes.

    These tests pin the two halves of the fix: generation is the shipped
    model's output, and a missing model is a failure rather than a
    substitution.
    """

    @staticmethod
    def _target():
        from neorx.core.graph.models import NeoRxResult

        return NeoRxResult(
            protein_id="P04637",
            protein_name="Cellular tumor antigen p53",
            gene_name="TP53",
            uniprot_id="P04637",
            is_causal_target=True,
        )

    def test_uses_the_shipped_tokenizer(self, monkeypatch):
        """The tokenizer handed to ``generate`` is the shipped asset.

        The thirteen-SMILES vocabulary had 25 tokens against the
        asset's 34, lacked ten the trained model emits (``# + 4 5 6 8 9
        I P S``) and contained one it does not (``@@``). It also drove
        ``MolVAE(vocab_size=max(25, 64))`` -- a 4,151,872-parameter
        model where the checkpoint has 4,140,322, so the shipped
        weights could not have been loaded into it even by accident.
        """
        import importlib

        from neorx.genmol.data.tokenizer import SmilesTokenizer
        from neorx.genmol.pretrained import ASSET_DIR, TOKENIZER_NAME

        # ``neorx.genmol`` re-exports the ``generate`` *function*, which
        # shadows the submodule of the same name on the package, so
        # ``import neorx.genmol.generate as m`` binds the function.
        generate_module = importlib.import_module("neorx.genmol.generate")

        seen = {}

        def capture(model, tokenizer, **kwargs):
            seen["model"] = model
            seen["tokenizer"] = tokenizer
            seen["kwargs"] = kwargs
            return ["CCO"]

        monkeypatch.setattr(generate_module, "generate", capture)

        _generate_for_target(self._target(), 10)

        shipped = SmilesTokenizer.load(ASSET_DIR / TOKENIZER_NAME)
        assert seen["tokenizer"].vocab_size == shipped.vocab_size == 34
        assert seen["tokenizer"].token_to_idx == shipped.token_to_idx
        # The model was built for that vocabulary, not padded to 64.
        assert seen["model"].decoder.vocab_size == shipped.vocab_size
        assert sum(p.numel() for p in seen["model"].parameters()) == 4_140_322
        assert seen["kwargs"]["temperature"] == _GENERATION_TEMPERATURE
        assert seen["kwargs"]["validate"] is True

    def test_output_is_the_pretrained_models_output(self):
        """Seeded, the result equals generating from ``load_pretrained()``.

        This is the assertion the old path cannot satisfy: at seed 0 it
        returned the ten curated scaffolds, none of which the model
        produces.
        """
        import torch

        from neorx.core.pipeline import _filter_novel
        from neorx.genmol import load_pretrained
        from neorx.genmol.generate import generate

        torch.manual_seed(0)
        actual = _generate_for_target(self._target(), 30)

        # Mirror the call order exactly: ``load_pretrained()`` constructs a
        # ``MolVAE``, whose weight initialisation draws from the same global
        # RNG that sampling then uses.
        torch.manual_seed(0)
        model, tokenizer = load_pretrained()
        expected = _filter_novel(
            generate(
                model,
                tokenizer,
                n=30,
                temperature=_GENERATION_TEMPERATURE,
                validate=True,
            ),
            max_tanimoto=0.7,
        )

        assert actual == expected
        assert len(actual) > 0

    def test_a_missing_checkpoint_raises(self, monkeypatch):
        """``GenMolAssetError`` propagates; no curated substitute.

        The old code caught it, logged a warning and returned ten
        hard-coded molecules, which then went on to MolScreen and
        DockBot as though they were model output.
        """
        import neorx.genmol as genmol_module
        from neorx.genmol import GenMolAssetError

        def boom(*args, **kwargs):
            raise GenMolAssetError("no trained checkpoint found in /nowhere")

        monkeypatch.setattr(genmol_module, "load_pretrained", boom)

        with pytest.raises(GenMolAssetError, match="no trained checkpoint"):
            _generate_for_target(self._target(), 10)

    def test_no_sanitisable_molecule_raises(self, monkeypatch):
        """An empty decode fails loudly instead of returning nothing.

        Random weights decode to nothing sanitisable most of the time,
        and the old code's ``if smiles:`` then fell through to the
        curated list without even a warning. With the trained weights
        the measured rate is 0.951, so an empty result means the model
        and tokenizer disagree -- which must not surface as a COMPLETE
        job with zero candidates.
        """
        import importlib

        generate_module = importlib.import_module("neorx.genmol.generate")

        monkeypatch.setattr(
            generate_module, "generate", lambda *a, **k: [],
        )

        with pytest.raises(RuntimeError, match="no sanitisable molecule"):
            _generate_for_target(self._target(), 10)


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
