"""The gates that make the guarantee enforceable rather than aspirational."""

import hashlib
import json

from neorx.experiments import gates as gates_mod
from neorx.experiments.gates import (
    check_cited_runs,
    check_committed_records_are_citable,
    check_record_integrity,
    check_records_wellformed,
    find_hardcoded_metrics,
)


def test_a_metric_literal_in_figure_code_is_found(tmp_path):
    src = tmp_path / "figures.py"
    src.write_text("def fig3():\n    neorx_f1 = [0.545, 0.333, 0.556]\n    return neorx_f1\n")
    findings = find_hardcoded_metrics(src)
    assert findings, "0.545 is a metric-shaped literal and must be flagged"
    assert findings[0].line == 2
    assert "0.545" in findings[0].message


def test_coarse_constants_are_not_flagged(tmp_path):
    """Axis limits and thresholds are not metrics."""
    src = tmp_path / "figures.py"
    src.write_text("def fig():\n    ax.set_ylim(0, 1.0)\n    alpha = 0.5\n")
    assert find_hardcoded_metrics(src) == []


def test_allowlisted_values_are_permitted(tmp_path):
    src = tmp_path / "figures.py"
    src.write_text("GOLDEN = 1.618\n")
    assert find_hardcoded_metrics(src) == []  # 3 dp, but allowlisted below
    assert find_hardcoded_metrics(src, allowlist=set()) != []


def test_a_record_missing_its_summary_is_malformed(tmp_path):
    (tmp_path / "2026-09-03-x-aaaaaa").mkdir()
    findings = check_records_wellformed(tmp_path)
    assert findings and "record.json" in findings[0].message


def test_row_count_must_match_the_summary(tmp_path):
    run = tmp_path / "2026-09-03-x-aaaaaa"
    run.mkdir()
    (run / "record.json").write_text(
        json.dumps({"run_id": "x", "status": "complete", "citable": True, "n_rows": 5})
    )
    (run / "rows.jsonl").write_text('{"a":1}\n')
    findings = check_records_wellformed(tmp_path)
    assert findings and "5" in findings[0].message and "1" in findings[0].message


def test_an_unparseable_literal_is_reported_not_swallowed(tmp_path, monkeypatch):
    """A literal whose source segment cannot be parsed as a Decimal must

    surface as a Finding, not be silently skipped -- this gate's entire
    job is catching metric literals, so a literal it cannot classify has
    to be visible rather than exempted.
    """
    src = tmp_path / "figures.py"
    src.write_text("def fig():\n    x = 0.545\n")

    def broken_get_source_segment(source, node):
        # Simulate a real-world case where the recovered source text for
        # the literal is not valid Decimal syntax (e.g. tooling that hands
        # back a mangled or unexpected span).
        return "not-a-number"

    monkeypatch.setattr(gates_mod.ast, "get_source_segment", broken_get_source_segment)

    findings = find_hardcoded_metrics(src)

    assert len(findings) == 1
    assert findings[0].line == 2
    assert "not-a-number" in findings[0].message
    assert "could not classify" in findings[0].message


def test_a_cited_run_that_does_not_exist_is_found(tmp_path):
    manifest = tmp_path / "run-manifest.toml"
    manifest.write_text('[tables]\n"paper1.table2" = "2026-09-03-neorx-7disease-deadbe"\n')
    findings = check_cited_runs(manifest, tmp_path / "runs")
    assert findings and "deadbe" in findings[0].message


def test_a_cited_run_that_is_not_citable_is_found(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "2026-09-03-x-aaaaaa"
    run.mkdir(parents=True)
    (run / "record.json").write_text(
        json.dumps(
            {"run_id": "2026-09-03-x-aaaaaa", "status": "complete", "citable": False, "n_rows": 0}
        )
    )
    (run / "rows.jsonl").write_text("")
    manifest = tmp_path / "run-manifest.toml"
    manifest.write_text('[tables]\n"paper1.table2" = "2026-09-03-x-aaaaaa"\n')
    findings = check_cited_runs(manifest, runs)
    assert findings and "citable" in findings[0].message.lower()


def test_a_metric_spelled_out_in_a_string_literal_is_found(tmp_path):
    """The defect that let confidences = {"POL": "C=0.990", ...} through:

    a measured value transcribed as a string evades a float-only literal
    check just as completely as one assembled by computation.
    """
    src = tmp_path / "figures.py"
    src.write_text('def fig2():\n    confidences = {"POL": "C=0.990"}\n    return confidences\n')
    findings = find_hardcoded_metrics(src)
    assert findings, "'C=0.990' spells out a metric-shaped number and must be flagged"
    assert findings[0].line == 2
    assert "0.990" in findings[0].message


def test_a_committed_non_citable_record_is_found(tmp_path):
    """check_committed_records_are_citable exercised against a real finding,

    not the vacuous "runs/ holds only .gitkeep" case.
    """
    runs = tmp_path / "runs"
    run = runs / "2026-09-03-x-aaaaaa"
    run.mkdir(parents=True)
    (run / "record.json").write_text(
        json.dumps(
            {"run_id": "2026-09-03-x-aaaaaa", "status": "failed", "citable": False, "n_rows": 0}
        )
    )
    (run / "rows.jsonl").write_text("")

    findings = check_committed_records_are_citable(runs, tracked={"2026-09-03-x-aaaaaa"})

    assert findings, "a citable=false record committed to git must be flagged"
    assert "citable=false" in findings[0].message


def test_a_committed_citable_record_is_not_flagged(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "2026-09-03-x-aaaaaa"
    run.mkdir(parents=True)
    (run / "record.json").write_text(
        json.dumps(
            {"run_id": "2026-09-03-x-aaaaaa", "status": "complete", "citable": True, "n_rows": 0}
        )
    )
    (run / "rows.jsonl").write_text("")

    findings = check_committed_records_are_citable(runs, tracked={"2026-09-03-x-aaaaaa"})

    assert findings == []


def test_tampering_with_rows_after_finalise_is_detected(tmp_path):
    """Editing rows.jsonl (e.g. an F1 from 0.474 to 0.999) with the row

    count left intact must be caught -- this is the exact scenario that
    produced no finding from any other gate.
    """
    run = tmp_path / "2026-09-03-x-aaaaaa"
    run.mkdir()
    rows = '{"F1": 0.474}\n'
    (run / "rows.jsonl").write_text(rows)
    (run / "record.json").write_text(
        json.dumps(
            {
                "run_id": "2026-09-03-x-aaaaaa",
                "status": "complete",
                "citable": True,
                "n_rows": 1,
                "rows_sha256": hashlib.sha256(rows.encode()).hexdigest(),
            }
        )
    )

    assert check_record_integrity(tmp_path) == []

    # Tamper: change the metric, keep the row count identical.
    (run / "rows.jsonl").write_text('{"F1": 0.999}\n')

    findings = check_record_integrity(tmp_path)
    assert findings, "an edited rows.jsonl must be detected even with n_rows unchanged"
    assert "digest" in findings[0].message


def test_a_record_with_no_stored_digest_is_flagged(tmp_path):
    run = tmp_path / "2026-09-03-x-aaaaaa"
    run.mkdir()
    (run / "rows.jsonl").write_text('{"F1": 0.474}\n')
    (run / "record.json").write_text(
        json.dumps(
            {"run_id": "2026-09-03-x-aaaaaa", "status": "complete", "citable": True, "n_rows": 1}
        )
    )
    findings = check_record_integrity(tmp_path)
    assert findings and "rows_sha256" in findings[0].message
