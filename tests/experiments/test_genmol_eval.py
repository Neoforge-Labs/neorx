"""genmol-eval is offline and deterministic enough to assert on shape."""

import pytest

from neorx.experiments.registry import get_experiment, run_experiment


def test_the_experiment_is_registered():
    import experiments  # noqa: F401

    assert get_experiment("genmol-eval").name == "genmol-eval"


@pytest.mark.slow
def test_run_records_the_metrics_the_paper_reports(tmp_path):
    import experiments  # noqa: F401

    rec = run_experiment("genmol-eval", runs_dir=tmp_path)
    assert rec.status == "complete"
    rows = rec.rows()
    assert len(rows) == 1, "genmol-eval reports one summary row"

    row = rows[0]
    for key in (
        "n_params", "vocab_size", "n_generated", "validity",
        "uniqueness", "novelty", "diversity", "per_molecule_ms",
        "mw_mean", "mw_std", "logp_mean", "qed_mean",
    ):
        assert key in row, f"missing {key}"

    assert 0.0 <= row["validity"] <= 1.0
    assert row["vocab_size"] == 34
    assert row["n_params"] == 4_140_322
