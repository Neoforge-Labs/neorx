"""The current-era reader.

25.06's association table is already tidy -- datatypeId, diseaseId,
targetId, score -- but carries no gene symbol at all. Symbols live in a
separate target table as id -> approvedSymbol, so this reader's real work
is the join, and its real risk is a join that silently produces nulls.
"""

import polars as pl
import pytest

from neorx.snapshots.opentargets import read_modern
from neorx.snapshots.schema import ASSOCIATION_COLUMNS


def _assoc(rows=None):
    return pl.DataFrame(rows if rows is not None else [
        {"datatypeId": "genetic_association", "datasourceId": "gwas_catalog",
         "diseaseId": "EFO_0000616", "targetId": "ENSG00000121879",
         "score": 0.61, "evidenceCount": 3},
        {"datatypeId": "literature", "datasourceId": "europepmc",
         "diseaseId": "EFO_0000616", "targetId": "ENSG00000121879",
         "score": 0.22, "evidenceCount": 9},
    ])


def _targets(rows=None):
    return pl.DataFrame(rows if rows is not None else [
        {"id": "ENSG00000121879", "approvedSymbol": "PIK3CA", "biotype": "protein_coding"},
    ])


def test_emits_the_canonical_schema():
    df = read_modern(_assoc(), _targets())
    assert dict(df.schema) == ASSOCIATION_COLUMNS


def test_symbol_is_joined_from_the_target_table():
    df = read_modern(_assoc(), _targets())
    assert set(df["target_symbol"].to_list()) == {"PIK3CA"}


def test_datatype_and_score_survive_the_join():
    df = read_modern(_assoc(), _targets())
    row = df.filter(pl.col("datatype") == "genetic_association").row(0, named=True)
    assert row["score"] == 0.61
    assert row["disease_id"] == "EFO_0000616"
    assert row["target_id"] == "ENSG00000121879"


def test_zero_scores_are_dropped_matching_the_1806_reader():
    a = _assoc([{"datatypeId": "genetic_association", "datasourceId": "x",
                 "diseaseId": "EFO_1", "targetId": "ENSG00000121879",
                 "score": 0.0, "evidenceCount": 0}])
    assert read_modern(a, _targets()).height == 0


def test_a_target_missing_from_the_target_table_raises():
    # A null symbol would drop silently out of every candidate frame and
    # look like an absence of evidence rather than a broken release.
    a = _assoc([{"datatypeId": "genetic_association", "datasourceId": "x",
                 "diseaseId": "EFO_1", "targetId": "ENSG_ABSENT",
                 "score": 0.5, "evidenceCount": 1}])
    with pytest.raises(ValueError, match="ENSG_ABSENT"):
        read_modern(a, _targets())


def test_targets_with_a_blank_symbol_are_treated_as_missing():
    a = _assoc([{"datatypeId": "genetic_association", "datasourceId": "x",
                 "diseaseId": "EFO_1", "targetId": "ENSG_BLANK",
                 "score": 0.5, "evidenceCount": 1}])
    t = _targets([{"id": "ENSG_BLANK", "approvedSymbol": "", "biotype": "protein_coding"}])
    with pytest.raises(ValueError, match="ENSG_BLANK"):
        read_modern(a, t)


def test_an_empty_association_table_yields_an_empty_frame():
    empty = pl.DataFrame(schema={
        "datatypeId": pl.Utf8, "datasourceId": pl.Utf8, "diseaseId": pl.Utf8,
        "targetId": pl.Utf8, "score": pl.Float64, "evidenceCount": pl.Int64,
    })
    df = read_modern(empty, _targets())
    assert df.height == 0
    assert dict(df.schema) == ASSOCIATION_COLUMNS
