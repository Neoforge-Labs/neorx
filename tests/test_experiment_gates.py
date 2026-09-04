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


def test_fabricated_statistics_gate_flags_a_reintroduced_fake_bootstrap(tmp_path):
    from neorx.experiments.gates import check_no_fabricated_statistics

    (tmp_path / "sneaky.py").write_text(
        "def _bootstrap_confidence_interval(effect):\n"
        "    return (0.1, 0.9)\n"
    )
    findings = check_no_fabricated_statistics(tmp_path)
    assert len(findings) == 1
    assert "_bootstrap_confidence_interval" in findings[0].message


def test_fabricated_statistics_gate_flags_a_reintroduced_p_value(tmp_path):
    from neorx.experiments.gates import check_no_fabricated_statistics

    (tmp_path / "sneaky.py").write_text("p_value = 0.5 * (1.0 - effect)\n")
    findings = check_no_fabricated_statistics(tmp_path)
    assert any("p_value" in f.message for f in findings)


def test_fabricated_statistics_gate_flags_the_removed_networkx_call(tmp_path):
    from neorx.experiments.gates import check_no_fabricated_statistics

    (tmp_path / "sneaky.py").write_text("if nx.d_separated(G, x, y, z):\n    pass\n")
    findings = check_no_fabricated_statistics(tmp_path)
    assert any("d_separated" in f.message for f in findings)


def test_fabricated_statistics_gate_passes_on_the_real_causal_package():
    from pathlib import Path

    from neorx.experiments.gates import check_no_fabricated_statistics
    import neorx.core.causal as causal

    findings = check_no_fabricated_statistics(Path(causal.__file__).parent)
    assert findings == [], [f.message for f in findings]


def test_fabricated_statistics_gate_does_not_reach_the_genuine_ks_p_value():
    # distribution.py reports a real Kolmogorov-Smirnov p-value. The gate
    # is scoped to the causal package precisely so it survives.
    from pathlib import Path

    from neorx.experiments.gates import check_no_fabricated_statistics
    import neorx.genmol.evaluation.distribution as dist

    scoped = Path(dist.__file__).parent
    findings = check_no_fabricated_statistics(scoped)
    assert any("p_value" in f.message for f in findings), (
        "sanity: the gate does flag p_value when pointed at this directory, "
        "which is why production wiring must never point it here"
    )
