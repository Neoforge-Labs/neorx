"""Every rate in README.md's GenMol table must come from the cited run.

``docs/run-manifest.toml`` plus ``check_cited_runs`` already prove that a
cited run exists and is citable. They do not prove that the numbers
printed beside the citation came out of it -- and that is the exact gap
the "97% validity" claim lived in for three releases: a figure in prose,
no record, and nothing that would notice.

So this recomputes the table from ``rows.jsonl`` and compares. It fails if
a rate is edited by hand, if the record is re-run and the README is not
updated, or if a row is added at a temperature the README does not show.
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
MANIFEST = REPO / "docs" / "run-manifest.toml"

HEADING = "### GenMol generation quality"
RATES = ("parseable", "sanitisable", "unique", "novel")

#: "0.974 ± 0.002" -- mean and across-seed standard deviation.
_CELL = re.compile(r"^(\d\.\d+)\s*±\s*(\d\.\d+)$")


def _cited_run_id() -> str:
    readme_claims = tomllib.loads(MANIFEST.read_text()).get("readme", {})
    run_id = readme_claims.get("genmol_generation_quality")
    assert run_id, (
        f"{MANIFEST} has no readme.genmol_generation_quality entry, so the "
        f"table in README.md cites no run and nothing can check it"
    )
    return str(run_id)


def _readme_table() -> dict[float, dict[str, tuple[str, str]]]:
    """The GenMol table as written, values left as strings.

    Kept as strings deliberately: the comparison is against the README's
    own stated precision, so parsing them to floats here and reformatting
    would hide a mismatch in the digits actually printed.
    """
    body = README.read_text()
    start = body.index(HEADING)
    table: dict[float, dict[str, tuple[str, str]]] = {}

    for line in body[start:].splitlines():
        if not line.startswith("|"):
            if table:
                break  # table ended
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # The temperature cell carries a note on some rows ("1.0
        # (`generate()` default)"), so take the leading number only.
        head = re.match(r"^(\d+(?:\.\d+)?)\b", cells[0])
        if head is None:
            continue  # header row, or the |---| separator
        parsed: dict[str, tuple[str, str]] = {}
        for rate, cell in zip(RATES, cells[1:]):
            match = _CELL.match(cell)
            assert match, f"{rate} cell {cell!r} is not 'mean ± sd'"
            parsed[rate] = (match.group(1), match.group(2))
        table[float(head.group(1))] = parsed

    assert table, f"found no data rows under {HEADING!r} in {README}"
    return table


def test_every_rate_in_the_readme_table_is_recomputed_from_the_cited_run():
    import experiments  # noqa: F401  -- registers the experiment
    from experiments.genmol_validity import summarise
    from neorx.experiments.record import RunRecord

    run_id = _cited_run_id()
    rows = RunRecord.load(run_id).rows()
    measured = summarise(rows)
    written = _readme_table()

    assert sorted(written) == sorted(measured), (
        f"README.md shows temperatures {sorted(written)} but run {run_id} "
        f"recorded {sorted(measured)} -- a temperature measured and not "
        f"shown, or shown and not measured"
    )

    for temperature, cells in written.items():
        for rate, (mean_text, sd_text) in cells.items():
            mean, sd = measured[temperature][rate]
            # Compared at the precision the README prints, so a rate
            # quoted to three decimals is held to three decimals.
            assert mean_text == f"{mean:.{len(mean_text) - 2}f}", (
                f"README.md says {rate} at T={temperature} is {mean_text}; "
                f"run {run_id} gives {mean}"
            )
            assert sd_text == f"{sd:.{len(sd_text) - 2}f}", (
                f"README.md says {rate} at T={temperature} has sd {sd_text}; "
                f"run {run_id} gives {sd}"
            )


def test_the_parameter_count_in_the_readme_is_the_shipped_model_s():
    """The README states the count twice; both must be the real one.

    "~4.1M" was right, but it sat in the same parenthesis as the
    unsourced validity figure, so it was never checked against anything
    either.
    """
    import experiments  # noqa: F401
    from neorx.experiments.record import RunRecord

    rows = RunRecord.load(_cited_run_id()).rows()
    n_params = {row["n_params"] for row in rows}
    assert len(n_params) == 1, f"one model, one count, got {n_params}"

    written = n_params.pop()
    assert f"{written:,}" in README.read_text(), (
        f"README.md does not state the shipped model's parameter count "
        f"({written:,}) as the run record measured it"
    )


def test_the_readme_quotes_no_validity_rate_outside_that_table():
    """A rate in prose is a rate nothing checks.

    The claim this replaces was exactly that: a percentage in a
    parenthesis in the pipeline description. Any future "NN% validity"
    belongs in the table, where the test above recomputes it.
    """
    stray = re.findall(r"\d+(?:\.\d+)?\s*%\s*validity", README.read_text(), re.IGNORECASE)
    assert stray == [], (
        f"README.md states {stray} as prose. Put the rate in the GenMol "
        f"table, with its definition and temperature, so it is checked "
        f"against the run record."
    )


@pytest.mark.parametrize("claim", ["sample_size", "seeds", "corpus_size"])
def test_the_readme_states_the_conditions_the_rates_were_measured_under(claim):
    """A rate without its sample size and seeds is not reproducible.

    That is half of why the 97% could not be checked: nothing recorded
    how many molecules it was over, or from which draw.
    """
    import experiments  # noqa: F401
    from neorx.experiments.record import RunRecord

    rows = RunRecord.load(_cited_run_id()).rows()
    body = README.read_text()

    if claim == "sample_size":
        per_seed = {row["n_requested"] for row in rows}
        assert len(per_seed) == 1
        assert f"{per_seed.pop():,}" in body, "the per-cell sample size is not stated"
    elif claim == "seeds":
        seeds = sorted({row["seed"] for row in rows})
        assert f"seeds {seeds[0]}–{seeds[-1]}" in body, (
            f"the seeds measured ({seeds}) are not stated"
        )
    else:
        corpus = {row["n_corpus"] for row in rows}
        assert len(corpus) == 1
        assert f"{corpus.pop():,}" in body, (
            "novelty is quoted without saying how large the corpus it is measured against was"
        )
