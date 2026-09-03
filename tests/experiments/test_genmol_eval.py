"""genmol-eval is offline and deterministic enough to assert on shape."""

import pytest
from rdkit import Chem

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
        "n_params",
        "vocab_size",
        "n_generated",
        "validity",
        "uniqueness",
        "novelty",
        "diversity",
        "per_molecule_ms",
        "mw_mean",
        "mw_std",
        "logp_mean",
        "qed_mean",
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


def test_diversity_is_computed_over_distinct_molecules_not_raw_duplicates():
    """Duplicates in ``valid`` must not deflate the reported diversity.

    _diversity scores every pair as 1 - Tanimoto similarity, and a molecule
    paired with itself always scores similarity 1.0. Feeding the raw
    (non-deduplicated) ``valid`` list to _diversity therefore mixes in a
    pile of self-pairs and drags the reported number down toward zero even
    though the *distinct* molecules present are genuinely diverse. This
    test constructs a ``valid`` list with an obvious duplicate (benzene
    appears 5 times alongside one different molecule, ethanol) and asserts
    the reported diversity equals the distinct-only computation -- a
    tolerant bound like ``0 <= diversity <= 1`` would pass on the broken
    version and proves nothing.
    """
    from experiments.genmol_eval import _distinct_mols, _diversity

    benzene = "c1ccccc1"
    ethanol = "CCO"
    # 5 copies of benzene + 1 ethanol: heavily duplicated raw output.
    valid = [(benzene, Chem.MolFromSmiles(benzene))] * 5 + [(ethanol, Chem.MolFromSmiles(ethanol))]

    distinct_expected = _diversity([Chem.MolFromSmiles(benzene), Chem.MolFromSmiles(ethanol)])
    reported = _diversity(_distinct_mols(valid))

    assert reported == distinct_expected
    # Sanity: feeding the raw duplicated list directly (the bug) gives a
    # different, deflated answer -- proving this test would catch it.
    raw_mols = [m for _, m in valid]
    assert _diversity(raw_mols) != reported
