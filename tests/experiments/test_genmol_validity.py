"""genmol-validity keeps the four definitions of "valid" apart.

The experiment exists because one number called "validity" was quoted
without saying which of these it meant. Tests that only asserted
``0 <= rate <= 1`` would pass on an implementation that conflated them
again, so each one is pinned to a hand-checked input instead.
"""

import pytest
from rdkit import Chem

from neorx.experiments.registry import get_experiment, run_experiment


def test_the_experiment_is_registered():
    import experiments  # noqa: F401

    assert get_experiment("genmol-validity").name == "genmol-validity"


def test_an_empty_decode_is_neither_parseable_nor_sanitisable():
    """Chem.MolFromSmiles("") returns a molecule, not None.

    So the obvious ``m is not None`` test counts an empty decode as a
    valid molecule. The sampler does emit empty strings at high enough
    temperature, and each one counted that way inflates validity.
    """
    from experiments.genmol_validity import _classify

    assert Chem.MolFromSmiles("") is not None, (
        "sanity: RDKit really does return a molecule for the empty string, "
        "which is why the atom count is part of the question"
    )

    counts = _classify(["", "   ", "CCO"])
    assert counts.n_empty == 2
    assert counts.n_parseable == 1
    assert counts.n_sanitisable == 1


def test_a_parseable_but_unsanitisable_smiles_is_counted_as_parseable_only():
    """Pentavalent carbon reads as a graph and fails sanitisation.

    This is the distinction the single word "validity" hides: a string can
    be perfectly readable SMILES and still describe an impossible
    molecule. Collapsing the two would report this molecule as simply
    invalid and lose the fact that the sampler's syntax is better than its
    chemistry.
    """
    from experiments.genmol_validity import _classify

    impossible = "C(C)(C)(C)(C)C"
    assert Chem.MolFromSmiles(impossible) is None, (
        "sanity: RDKit rejects this outright when sanitising, so it would "
        "be indistinguishable from unreadable syntax under one rate"
    )

    counts = _classify([impossible])
    assert counts.n_parseable == 1
    assert counts.n_sanitisable == 0
    assert counts.distinct == set()


def test_two_spellings_of_one_molecule_count_as_one_distinct_molecule():
    """ "CCO" and "OCC" are both ethanol.

    Counting raw strings instead of canonical ones would report two
    distinct molecules, moving both ``unique`` and ``novel`` in the
    flattering direction for no chemical reason.
    """
    from experiments.genmol_validity import _classify

    counts = _classify(["CCO", "OCC"])
    assert counts.n_sanitisable == 2, "both strings are valid molecules"
    assert counts.distinct == {"CCO"}, "but they are one molecule"


def test_summarise_reports_the_spread_across_seeds_not_a_point():
    """A rate from a single draw has no error bar.

    That is precisely what the 97% claim was, so the summary must carry a
    standard deviation computed across seeds and must not silently
    collapse two temperatures into one row.
    """
    from experiments.genmol_validity import summarise

    rows = [
        {
            "temperature": 1.0,
            "seed": 0,
            "parseable": 1.0,
            "sanitisable": 0.9,
            "unique": 1.0,
            "novel": 1.0,
        },
        {
            "temperature": 1.0,
            "seed": 1,
            "parseable": 1.0,
            "sanitisable": 0.7,
            "unique": 1.0,
            "novel": 1.0,
        },
        {
            "temperature": 1.2,
            "seed": 0,
            "parseable": 1.0,
            "sanitisable": 0.5,
            "unique": 1.0,
            "novel": 1.0,
        },
    ]
    summary = summarise(rows)

    assert sorted(summary) == [1.0, 1.2], "temperatures stay separate"
    mean, std = summary[1.0]["sanitisable"]
    assert mean == pytest.approx(0.8)
    assert std == pytest.approx(0.1), "the spread between the two seeds, not zero"


def test_sanitisable_agrees_with_plain_rdkit_validity():
    """``sanitisable`` must be the same rate everyone else calls validity.

    Parsing and sanitising are split into two steps here, which is a
    reimplementation of what ``Chem.MolFromSmiles(s)`` does in one. If the
    split drifted from it, ``sanitisable`` would no longer be comparable
    to the 97% claim it was written to check, and the comparison would be
    between two different measurements.
    """
    from experiments.genmol_validity import _classify

    probe = [
        "CCO",  # valid
        "c1ccccc1",  # valid, aromatic
        "C(C)(C)(C)(C)C",  # parseable, bad valence
        "c1ccc1",  # parseable, cannot kekulise
        "not-a-smiles",  # unreadable
        "",  # empty
    ]
    plain = sum(
        1 for s in probe if (m := Chem.MolFromSmiles(s)) is not None and m.GetNumAtoms() > 0
    )
    assert _classify(probe).n_sanitisable == plain


@pytest.mark.slow
def test_run_measures_every_definition_at_every_temperature(tmp_path):
    import experiments  # noqa: F401
    from experiments.genmol_validity import N_PER_SEED, SEEDS, TEMPERATURES
    from neorx.genmol.data.download import _DEFAULT_CSV

    if not _DEFAULT_CSV.exists():
        pytest.skip(
            f"ChEMBL corpus not present at {_DEFAULT_CSV} -- run `genmol download` "
            f"or call download_chembl() first. `novel` is not computable without "
            f"it, and a missing prerequisite is not a failing behaviour."
        )

    rec = run_experiment("genmol-validity", runs_dir=tmp_path)
    assert rec.status == "complete"

    rows = rec.rows()
    assert len(rows) == len(TEMPERATURES) * len(SEEDS), "one row per temperature per seed"
    assert {(r["temperature"], r["seed"]) for r in rows} == {
        (t, s) for t in TEMPERATURES for s in SEEDS
    }

    for row in rows:
        assert row["n_generated"] == N_PER_SEED
        assert row["vocab_size"] == 34
        assert row["n_params"] == 4_140_322
        assert row["corpus_sha256"], "the corpus novelty is measured against must be cited"
        # Parsing is a weaker requirement than sanitising, so its rate can
        # never be the lower of the two. An implementation that swapped
        # them, or asked RDKit the same question twice, breaks this.
        assert row["n_parseable"] >= row["n_sanitisable"]
        assert row["n_distinct"] <= row["n_sanitisable"]
        assert row["n_novel"] <= row["n_distinct"]
        for rate in ("parseable", "sanitisable", "unique", "novel"):
            assert 0.0 <= row[rate] <= 1.0
