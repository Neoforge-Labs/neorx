"""Figures render from run records. A literal in this file is a test failure."""

import json
from pathlib import Path

import pytest

from neorx.experiments.gates import find_hardcoded_metrics


def test_the_figure_module_contains_no_metric_literals():
    """The defect this whole sub-project exists to prevent."""
    findings = find_hardcoded_metrics(Path("experiments/figures.py"))
    assert findings == [], [f"{f.path}:{f.line} {f.message}" for f in findings]


def test_fig3_reads_its_values_from_a_record(tmp_path):
    import experiments.figures as mod

    run = tmp_path / "2026-09-03-neorx-7disease-aaaaaa"
    (run / "inputs").mkdir(parents=True)
    (run / "rows.jsonl").write_text(
        "\n".join(
            json.dumps({"disease": d, "F1": f1, "corr_F1": c, "P": 0.4, "R": 0.7})
            for d, f1, c in [("HIV", 0.545, 0.167), ("malaria", 0.333, 0.074)]
        )
        + "\n"
    )
    (run / "record.json").write_text(
        json.dumps({"run_id": run.name, "status": "complete", "citable": True,
                    "n_rows": 2})
    )
    series = mod.fig3_series(run.name, runs_dir=tmp_path)
    assert series["diseases"] == ["HIV", "malaria"]
    assert series["neorx_f1"] == [0.545, 0.333]
    assert series["corr_f1"] == [0.167, 0.074]


def test_a_non_citable_run_is_refused(tmp_path):
    import experiments.figures as mod

    run = tmp_path / "2026-09-03-neorx-7disease-bbbbbb"
    (run / "inputs").mkdir(parents=True)
    (run / "rows.jsonl").write_text('{"disease":"HIV","F1":0.5,"corr_F1":0.1}\n')
    (run / "record.json").write_text(
        json.dumps({"run_id": run.name, "status": "complete", "citable": False,
                    "n_rows": 1})
    )
    with pytest.raises(ValueError, match="citable"):
        mod.fig3_series(run.name, runs_dir=tmp_path)
