"""The gates, wired to the real repository."""

from pathlib import Path

from neorx.experiments.gates import (
    check_cited_runs,
    check_committed_records_are_citable,
    check_record_integrity,
    check_records_wellformed,
    find_hardcoded_metrics,
)

REPO = Path(__file__).resolve().parent.parent


def test_no_experiment_module_carries_a_metric_literal():
    offenders = []
    for path in sorted((REPO / "experiments").glob("*.py")):
        offenders += find_hardcoded_metrics(path)
    assert offenders == [], [f"{o.path}:{o.line} {o.message}" for o in offenders]


def test_every_stored_run_is_wellformed():
    findings = check_records_wellformed(REPO / "runs")
    assert findings == [], [f.message for f in findings]


def test_every_cited_run_exists_and_is_citable():
    findings = check_cited_runs(REPO / "docs" / "run-manifest.toml", REPO / "runs")
    assert findings == [], [f.message for f in findings]


def test_no_uncitable_record_is_committed():
    findings = check_committed_records_are_citable(REPO / "runs")
    assert findings == [], [f.message for f in findings]


def test_every_stored_run_has_an_intact_rows_digest():
    findings = check_record_integrity(REPO / "runs")
    assert findings == [], [f.message for f in findings]
