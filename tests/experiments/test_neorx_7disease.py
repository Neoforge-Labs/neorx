"""The seven-disease benchmark, recorded per disease as each completes."""

import pytest

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import get_experiment


def test_the_experiment_is_registered():
    import experiments  # noqa: F401

    assert get_experiment("neorx-7disease").name == "neorx-7disease"


def test_all_seven_diseases_are_configured():
    import experiments.neorx_7disease as mod

    assert len(mod.DISEASES) == 7
    lowered = [d.lower() for d in mod.DISEASES]
    for expected in ("hiv", "malaria", "ebola"):
        assert any(expected in d for d in lowered), expected


def test_a_disease_row_is_written_before_the_next_starts(tmp_path, monkeypatch):
    import experiments.neorx_7disease as mod

    calls: list[str] = []

    def fake_evaluate(disease):
        calls.append(disease)
        if len(calls) == 3:
            raise RuntimeError("simulated failure on the third disease")
        return {
            "found": 20,
            "causal": 7,
            "tp": ["CCR5"],
            "fp": [],
            "missed": [],
            "P": 0.4,
            "R": 0.75,
            "F1": 0.5,
            "grade": "B",
            "corr_P": 0.1,
            "corr_R": 0.5,
            "corr_F1": 0.17,
            "corr_fp": [],
            "demoted": [],
            "fp_ranks": {},
            "t_graph": 0.0,
            "t_identify": 1.0,
        }

    monkeypatch.setattr(mod, "_evaluate_disease", fake_evaluate)

    rec = RunRecord.create("neorx-7disease", runs_dir=tmp_path)
    with pytest.raises(RuntimeError):
        mod.neorx_7disease(rec)
    rec.finalise("failed")

    assert len(rec.rows()) == 2, "completed diseases must survive the failure"
    assert rec.rows()[0]["disease"] == mod.DISEASES[0]


def test_a_disease_absent_from_ground_truth_raises_instead_of_fabricating(monkeypatch):
    """_evaluate_disease must fail loudly, not record P=R=F1=0.0.

    A disease KnownTargetValidator cannot validate (report.validated is
    False) used to fall through to a hardcoded 0.0/0.0/0.0/"N/A" row --
    a fabricated metric recorded as if it were measured, inside the one
    subsystem built to make that impossible. It must raise instead.
    """
    import experiments.neorx_7disease as mod

    class _FakeTarget:
        gene_name = "SOMEGENE"
        is_causal_target = True

    class _FakeNode:
        node_id = "gene:SOMEGENE"
        score = 1.0

        class node_type:
            value = "gene"

    class _FakeGraph:
        nodes = [_FakeNode()]

    monkeypatch.setattr(
        "neorx.core.graph.graph_builder.build_disease_graph",
        lambda disease, use_cache=False: _FakeGraph(),
    )
    # ``_evaluate_disease`` evaluates every candidate and ranks that list
    # separately, so the identification summary can be computed over the
    # full population rather than the reported slice. Both halves are
    # stubbed here; neither is exercised before the raise under test.
    monkeypatch.setattr(
        "neorx.core.causal.identifier.evaluate_all_targets",
        lambda graph: [_FakeTarget()],
    )
    monkeypatch.setattr(
        "neorx.core.causal.identifier.rank_causal_targets",
        lambda results, top_n=20: list(results),
    )

    with pytest.raises(mod.UnvalidatableDiseaseError, match="not-a-real-disease"):
        mod._evaluate_disease("not-a-real-disease-xyz")


def test_an_unvalidatable_disease_fails_the_run_but_keeps_prior_rows(tmp_path, monkeypatch):
    import experiments.neorx_7disease as mod
    from neorx.experiments.registry import run_experiment

    good_row = {
        "found": 20,
        "causal": 7,
        "tp": ["CCR5"],
        "fp": [],
        "missed": [],
        "P": 0.4,
        "R": 0.75,
        "F1": 0.5,
        "grade": "B",
        "corr_P": 0.1,
        "corr_R": 0.5,
        "corr_F1": 0.17,
        "corr_fp": [],
        "demoted": [],
        "fp_ranks": {},
        "t_graph": 0.0,
        "t_identify": 1.0,
    }

    def fake_evaluate(disease):
        if disease == mod.DISEASES[0]:
            return good_row
        raise mod.UnvalidatableDiseaseError(f"{disease!r} has no ground truth")

    monkeypatch.setattr(mod, "_evaluate_disease", fake_evaluate)
    monkeypatch.setattr(mod, "DISEASES", mod.DISEASES[:2])

    with pytest.raises(mod.UnvalidatableDiseaseError):
        run_experiment("neorx-7disease", runs_dir=tmp_path)

    run_id = next((tmp_path).iterdir()).name
    from neorx.experiments.record import RunRecord

    rec = RunRecord.load(run_id, runs_dir=tmp_path)
    assert rec.status == "failed"
    assert len(rec.rows()) == 1, "the row completed before the failure must survive"


def test_every_row_carries_the_full_schema(tmp_path, monkeypatch):
    import experiments.neorx_7disease as mod

    monkeypatch.setattr(
        mod,
        "_evaluate_disease",
        lambda d: {
            "found": 20,
            "causal": 7,
            "tp": [],
            "fp": [],
            "missed": [],
            "P": 0.4,
            "R": 0.75,
            "F1": 0.5,
            "grade": "B",
            "corr_P": 0.1,
            "corr_R": 0.5,
            "corr_F1": 0.17,
            "corr_fp": [],
            "demoted": [],
            "fp_ranks": {},
            "t_graph": 0.0,
            "t_identify": 1.0,
        },
    )

    rec = RunRecord.create("neorx-7disease", runs_dir=tmp_path)
    mod.neorx_7disease(rec)

    required = {
        "disease",
        "found",
        "causal",
        "tp",
        "fp",
        "missed",
        "P",
        "R",
        "F1",
        "grade",
        "corr_P",
        "corr_R",
        "corr_F1",
        "corr_fp",
        "demoted",
        "fp_ranks",
        "t_graph",
        "t_identify",
    }
    for row in rec.rows():
        assert required <= set(row), sorted(required - set(row))
    assert len(rec.rows()) == 7


# ── Identifiability metrics ────────────────────────────────────────
#
# The non-trivial identifiability rate is what the rewritten manuscript
# leads with, and the failure breakdown is what makes a low rate legible
# rather than a bare zero. Both have to reach the run record.

from neorx.core.causal.backdoor import IdentificationReason


def test_identifiability_summary_counts_every_candidate():
    from experiments.neorx_7disease import summarise_identification

    class _R:
        def __init__(self, reason, identifiable, near_miss=0):
            self.identification_reason = reason
            self.identifiable = identifiable
            self.n_near_miss_confounders = near_miss

    results = [
        _R("identifiable_by_adjustment", True),
        _R("identifiable_trivially", True, near_miss=3),
        _R("no_causal_path", False),
        _R("cyclic_component", False),
    ]

    summary = summarise_identification(results)

    assert summary["n_candidates"] == 4
    assert sum(summary["reason_counts"].values()) == 4
    assert summary["n_identifiable_by_adjustment"] == 1
    assert summary["n_identifiable_trivially"] == 1


def test_nontrivial_rate_excludes_trivial_verdicts():
    from experiments.neorx_7disease import summarise_identification

    class _R:
        def __init__(self, reason, identifiable):
            self.identification_reason = reason
            self.identifiable = identifiable
            self.n_near_miss_confounders = 0

    results = [
        _R("identifiable_by_adjustment", True),
        _R("identifiable_trivially", True),
        _R("identifiable_trivially", True),
        _R("no_causal_path", False),
    ]

    summary = summarise_identification(results)

    assert summary["nontrivial_identifiability_rate"] == 0.25
    assert summary["cyclic_fraction"] == 0.0


def test_cyclic_fraction_is_reported():
    from experiments.neorx_7disease import summarise_identification

    class _R:
        def __init__(self, reason):
            self.identification_reason = reason
            self.identifiable = False
            self.n_near_miss_confounders = 0

    summary = summarise_identification(
        [_R("cyclic_component"), _R("cyclic_component"), _R("no_causal_path")]
    )
    assert summary["cyclic_fraction"] == pytest.approx(2 / 3)


def test_summary_of_no_candidates_is_zero_not_a_division_error():
    from experiments.neorx_7disease import summarise_identification

    summary = summarise_identification([])
    assert summary["n_candidates"] == 0
    assert summary["nontrivial_identifiability_rate"] == 0.0
    assert summary["cyclic_fraction"] == 0.0


def test_every_reason_appears_in_the_breakdown_even_at_zero():
    from experiments.neorx_7disease import summarise_identification

    summary = summarise_identification([])
    assert set(summary["reason_counts"]) == {
        r.value for r in IdentificationReason
    }


# ── The summary's population ───────────────────────────────────────
#
# The rate has to be computed over every candidate the identifier
# evaluated, not over the top-N it reports. Ranking is by
# ``causal_confidence``, whose formula adds a bonus for being
# identifiable, so the reported slice is a sample selected partly by the
# very property being measured -- the rate over it is inflated by
# construction, and its failure breakdown cannot sum to the number of
# candidates evaluated.


def _synthetic_disease_graph(n_candidates: int):
    """A disease graph with a known number of candidate targets.

    Built by hand rather than through ``build_disease_graph`` so the
    candidate count is exact and the test needs no network.
    """
    from neorx.core.graph.models import (
        DiseaseGraph,
        EdgeType,
        GraphEdge,
        GraphNode,
        NodeType,
    )

    disease_id = "MONDO:0000001"
    nodes = [
        GraphNode(
            node_id=disease_id,
            name="testdisease",
            node_type=NodeType.DISEASE,
            source="test",
            score=1.0,
        )
    ]
    edges = []
    for i in range(n_candidates):
        node_id = f"GENE:G{i:02d}"
        nodes.append(
            GraphNode(
                node_id=node_id,
                name=f"G{i:02d}",
                node_type=NodeType.GENE,
                source="test",
                score=0.5,
            )
        )
        edges.append(
            GraphEdge(
                source_id=node_id,
                target_id=disease_id,
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=0.6,
                source_db="test",
            )
        )

    return DiseaseGraph(
        disease_name="testdisease",
        disease_id=disease_id,
        nodes=nodes,
        edges=edges,
        sources_queried=["test"],
    )


def test_summary_covers_every_evaluated_candidate_not_the_reported_top_n():
    from experiments.neorx_7disease import TOP_N, summarise_identification
    from neorx.core.causal.identifier import (
        evaluate_all_targets,
        identify_causal_targets,
        rank_causal_targets,
    )

    n_candidates = TOP_N + 5
    graph = _synthetic_disease_graph(n_candidates)

    evaluations = evaluate_all_targets(graph)
    targets = rank_causal_targets(evaluations, top_n=TOP_N)

    # The seam exists and both sides are what they claim to be: every
    # candidate evaluated, and strictly fewer reported.
    assert len(evaluations) == n_candidates
    assert len(targets) == TOP_N < len(evaluations)

    # ``identify_causal_targets`` still returns exactly what it did --
    # the full list was added alongside it, not in place of it.
    assert [r.protein_id for r in identify_causal_targets(graph, top_n=TOP_N)] == [
        r.protein_id for r in targets
    ]

    summary = summarise_identification(evaluations)

    assert summary["n_candidates"] == len(evaluations)
    assert summary["n_candidates"] > TOP_N
    # Success criterion 10: the breakdown accounts for every candidate.
    assert sum(summary["reason_counts"].values()) == summary["n_candidates"]

    # And the top-N slice would have given a different, smaller
    # population -- which is why the experiment must not pass it.
    assert summarise_identification(targets)["n_candidates"] == TOP_N
