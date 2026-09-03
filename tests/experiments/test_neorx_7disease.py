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
        return {"found": 20, "causal": 7, "tp": ["CCR5"], "fp": [], "missed": [],
                "P": 0.4, "R": 0.75, "F1": 0.5, "grade": "B",
                "corr_P": 0.1, "corr_R": 0.5, "corr_F1": 0.17, "corr_fp": [],
                "demoted": [], "fp_ranks": {}, "t_graph": 0.0, "t_identify": 1.0}

    monkeypatch.setattr(mod, "_evaluate_disease", fake_evaluate)
    monkeypatch.setattr(mod, "_capture_enabled", lambda: False)

    rec = RunRecord.create("neorx-7disease", runs_dir=tmp_path)
    with pytest.raises(RuntimeError):
        mod.neorx_7disease(rec)
    rec.finalise("failed")

    assert len(rec.rows()) == 2, "completed diseases must survive the failure"
    assert rec.rows()[0]["disease"] == mod.DISEASES[0]


def test_every_row_carries_the_full_schema(tmp_path, monkeypatch):
    import experiments.neorx_7disease as mod

    monkeypatch.setattr(
        mod, "_evaluate_disease",
        lambda d: {"found": 20, "causal": 7, "tp": [], "fp": [], "missed": [],
                   "P": 0.4, "R": 0.75, "F1": 0.5, "grade": "B",
                   "corr_P": 0.1, "corr_R": 0.5, "corr_F1": 0.17, "corr_fp": [],
                   "demoted": [], "fp_ranks": {}, "t_graph": 0.0, "t_identify": 1.0},
    )
    monkeypatch.setattr(mod, "_capture_enabled", lambda: False)

    rec = RunRecord.create("neorx-7disease", runs_dir=tmp_path)
    mod.neorx_7disease(rec)

    required = {"disease", "found", "causal", "tp", "fp", "missed", "P", "R", "F1",
                "grade", "corr_P", "corr_R", "corr_F1", "corr_fp", "demoted",
                "fp_ranks", "t_graph", "t_identify"}
    for row in rec.rows():
        assert required <= set(row), sorted(required - set(row))
    assert len(rec.rows()) == 7
