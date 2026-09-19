"""The candidate frame -- the population a past date would have evaluated.

Features computed from 2018 data are useless if the population was chosen
with hindsight. A gene that entered the pool because a 2026 source
connected it to something is a gene 2018 would never have looked at, and
that is leakage in the sampling frame rather than in the predictor. So the
frame comes from the pinned release and nothing else.
"""

import polars as pl

from neorx.snapshots.frame import (
    candidate_frame,
    corpus_diseases,
    diseases_with_genetic_evidence,
)
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.schema import ASSOCIATION_COLUMNS


def _store(tmp_path, rows):
    d = tmp_path / "opentargets" / "18.06"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=ASSOCIATION_COLUMNS).write_parquet(d / "associations.parquet")
    return SnapshotStore(tmp_path, on_read=None)


def _row(symbol, disease="EFO_1", datatype="genetic_association", score=0.5):
    return {
        "target_id": f"ENSG_{symbol}",
        "target_symbol": symbol,
        "disease_id": disease,
        "datatype": datatype,
        "score": score,
    }


def test_frame_is_drawn_from_genetic_evidence_only(tmp_path):
    store = _store(
        tmp_path,
        [
            _row("PIK3CA", datatype="genetic_association"),
            _row("TP53", datatype="somatic_mutation"),
            _row("TNF", datatype="literature"),
            _row("IL6", datatype="rna_expression"),
        ],
    )
    assert candidate_frame(store, "18.06", "EFO_1") == frozenset({"PIK3CA", "TP53"})


def test_frame_is_scoped_to_one_disease(tmp_path):
    store = _store(
        tmp_path,
        [
            _row("PIK3CA", disease="EFO_1"),
            _row("BRCA1", disease="EFO_2"),
        ],
    )
    assert candidate_frame(store, "18.06", "EFO_1") == frozenset({"PIK3CA"})


def test_zero_scored_genetic_evidence_does_not_enter_the_frame(tmp_path):
    store = _store(tmp_path, [_row("PIK3CA", score=0.0)])
    assert candidate_frame(store, "18.06", "EFO_1") == frozenset()


def test_a_disease_absent_from_the_release_has_an_empty_frame(tmp_path):
    store = _store(tmp_path, [_row("PIK3CA")])
    assert candidate_frame(store, "18.06", "EFO_ABSENT") == frozenset()


def test_diseases_with_genetic_evidence_excludes_literature_only_diseases(tmp_path):
    store = _store(
        tmp_path,
        [
            _row("PIK3CA", disease="EFO_1", datatype="genetic_association"),
            _row("TNF", disease="EFO_2", datatype="literature"),
        ],
    )
    assert diseases_with_genetic_evidence(store, "18.06") == frozenset({"EFO_1"})


def test_corpus_requires_both_genetic_evidence_and_a_phase_two_target(tmp_path):
    store = _store(
        tmp_path,
        [
            _row("PIK3CA", disease="EFO_1"),  # genetic, and has a phase-2 target
            _row("BRCA1", disease="EFO_2"),  # genetic, but no phase-2 target
            _row("TNF", disease="EFO_3", datatype="literature"),  # phase-2, no genetics
        ],
    )
    corpus = corpus_diseases(
        store,
        "18.06",
        phase2_targets={"EFO_1": {"PIK3CA"}, "EFO_3": {"TNF"}},
    )
    assert corpus == frozenset({"EFO_1"})


def test_corpus_is_empty_when_no_disease_meets_both_criteria(tmp_path):
    store = _store(tmp_path, [_row("BRCA1", disease="EFO_2")])
    assert corpus_diseases(store, "18.06", phase2_targets={}) == frozenset()
