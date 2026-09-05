"""How many diseases the corpus actually contains.

The spec gives criteria rather than a number, because the count is a
property of each release. This measures it -- and does so through the run
recorder, so the answer is an artifact with provenance rather than a
figure someone typed into a document.
"""

import polars as pl

from experiments.corpus_census import census_row
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.schema import ASSOCIATION_COLUMNS


def _store(tmp_path, rows):
    d = tmp_path / "opentargets" / "18.06"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=ASSOCIATION_COLUMNS).write_parquet(d / "associations.parquet")
    return SnapshotStore(tmp_path)


def _row(symbol, disease, datatype="genetic_association", score=0.5):
    return {
        "target_id": f"ENSG_{symbol}",
        "target_symbol": symbol,
        "disease_id": disease,
        "datatype": datatype,
        "score": score,
    }


def test_counts_diseases_with_and_without_genetic_evidence(tmp_path):
    store = _store(
        tmp_path,
        [
            _row("A", "EFO_1"),
            _row("B", "EFO_1"),
            _row("C", "EFO_2"),
            _row("D", "EFO_3", datatype="literature"),
        ],
    )
    row = census_row(store, "18.06")
    assert row["n_diseases_total"] == 3
    assert row["n_diseases_with_genetic_evidence"] == 2


def test_reports_the_median_frame_size(tmp_path):
    store = _store(
        tmp_path,
        [
            _row("A", "EFO_1"),
            _row("B", "EFO_1"),
            _row("C", "EFO_1"),
            _row("D", "EFO_2"),
        ],
    )
    row = census_row(store, "18.06")
    assert row["median_frame_size"] == 2.0


def test_counts_distinct_target_disease_pairs_not_rows(tmp_path):
    store = _store(
        tmp_path,
        [
            _row("A", "EFO_1", datatype="genetic_association"),
            _row("A", "EFO_1", datatype="somatic_mutation"),
        ],
    )
    assert census_row(store, "18.06")["n_genetic_target_disease_pairs"] == 1


def test_an_empty_release_reports_zeros_not_an_error(tmp_path):
    store = _store(tmp_path, [])
    row = census_row(store, "18.06")
    assert row["n_diseases_with_genetic_evidence"] == 0
    assert row["median_frame_size"] == 0.0


def test_pair_count_spans_genetic_subset_only(tmp_path):
    # The pair count is scoped to the genetic subset: a fixture with both
    # genetic and literature-only associations for different diseases must
    # not count the literature-only pairs. This guards against inadvertent
    # "fixes" that broaden the scope to the whole extract, since the genetic
    # criterion is the only admissible evidence downstream (Phase II data
    # from task 5 is not yet available, so this count is an upper bound).
    store = _store(
        tmp_path,
        [
            # Genetic pairs: 2 for EFO_1, 1 for EFO_2
            _row("A", "EFO_1", datatype="genetic_association"),
            _row("B", "EFO_1", datatype="genetic_association"),
            _row("C", "EFO_2", datatype="genetic_association"),
            # Literature pairs (not counted): 2 for EFO_3
            _row("D", "EFO_3", datatype="literature"),
            _row("E", "EFO_3", datatype="literature"),
        ],
    )
    row = census_row(store, "18.06")
    # Only 2 diseases have genetic evidence
    assert row["n_diseases_with_genetic_evidence"] == 2
    # Only 3 genetic pairs are counted; the 2 literature-only pairs for EFO_3
    # are not included
    assert row["n_genetic_target_disease_pairs"] == 3
