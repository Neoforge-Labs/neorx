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


def test_generate_is_called_without_prefiltering(tmp_path, monkeypatch):
    """generate() defaults to validate=True, deduplicate=True, which would

    filter internally and make validity/uniqueness structurally always
    1.0 -- a better-looking number than the truth, not a measurement.
    genmol-eval must request raw sampler output instead.
    """
    import experiments  # noqa: F401
    import neorx.genmol as genmol_mod
    import neorx.genmol.data.download as download_mod

    calls: dict = {}

    class _FakeParam:
        def numel(self) -> int:
            return 10

    class _FakeModel:
        def parameters(self):
            return [_FakeParam(), _FakeParam()]

    class _FakeTokenizer:
        vocab_size = 5

    def fake_load_pretrained():
        return _FakeModel(), _FakeTokenizer()

    def fake_generate(model, tokenizer, **kwargs):
        calls.update(kwargs)
        # duplicate + invalid entries mixed in on purpose: proves the
        # experiment does not rely on generate() to have filtered them.
        return ["CCO", "CCO", "c1ccccc1", "not-a-smiles"]

    monkeypatch.setattr(genmol_mod, "load_pretrained", fake_load_pretrained)
    monkeypatch.setattr(genmol_mod, "generate", fake_generate)
    monkeypatch.setattr(download_mod, "load_smiles", lambda: ["CCO"])

    rec = run_experiment("genmol-eval", runs_dir=tmp_path)

    assert calls.get("validate") is False, calls
    assert calls.get("deduplicate") is False, calls
    assert rec.status == "complete"
