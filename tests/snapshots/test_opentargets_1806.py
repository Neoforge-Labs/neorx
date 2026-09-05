"""The 2018 era reader.

18.06 ships one gzipped JSON object per line, nesting the datatype scores
under association_score.datatypes and carrying the gene symbol inline at
target.gene_info.symbol. Later eras do neither, which is why each era gets
its own reader and they all emit the same five columns.
"""

import json

import polars as pl

from neorx.snapshots.opentargets import read_1806
from neorx.snapshots.schema import ASSOCIATION_COLUMNS


def _record(symbol="PIK3CA", ensembl="ENSG00000121879",
            efo="EFO_0000616", datatypes=None):
    return json.dumps({
        "target": {"id": ensembl, "gene_info": {"symbol": symbol}},
        "disease": {"id": efo},
        "association_score": {
            "overall": 0.7,
            "datatypes": datatypes if datatypes is not None else {
                "genetic_association": 0.61,
                "literature": 0.22,
                "somatic_mutation": 0.0,
            },
        },
        "is_direct": True,
    })


def test_emits_the_canonical_schema():
    df = read_1806([_record()])
    assert dict(df.schema) == ASSOCIATION_COLUMNS


def test_one_row_per_nonzero_datatype():
    df = read_1806([_record()])
    # genetic_association and literature are non-zero; somatic_mutation is 0.0
    assert sorted(df["datatype"].to_list()) == ["genetic_association", "literature"]


def test_zero_scored_datatypes_are_dropped():
    df = read_1806([_record(datatypes={"genetic_association": 0.0})])
    assert df.height == 0


def test_identifiers_and_symbol_are_carried_through():
    df = read_1806([_record()])
    row = df.filter(pl.col("datatype") == "genetic_association").row(0, named=True)
    assert row["target_id"] == "ENSG00000121879"
    assert row["target_symbol"] == "PIK3CA"
    assert row["disease_id"] == "EFO_0000616"
    assert row["score"] == 0.61


def test_multiple_records_accumulate():
    df = read_1806([_record(), _record(symbol="TP53", ensembl="ENSG00000141510")])
    assert set(df["target_symbol"].to_list()) == {"PIK3CA", "TP53"}


def test_an_empty_stream_yields_an_empty_frame_with_the_schema():
    df = read_1806([])
    assert df.height == 0
    assert dict(df.schema) == ASSOCIATION_COLUMNS


def test_blank_lines_are_skipped():
    df = read_1806(["", "   ", _record()])
    assert df.height == 2


def test_a_record_missing_a_gene_symbol_is_dropped_not_guessed():
    # The frame is built from symbols. A record we cannot name is a record
    # we cannot put in a frame, and inventing one would corrupt it.
    bad = json.dumps({
        "target": {"id": "ENSG00000000001", "gene_info": {}},
        "disease": {"id": "EFO_0000616"},
        "association_score": {"datatypes": {"genetic_association": 0.9}},
    })
    assert read_1806([bad]).height == 0


def test_malformed_json_raises_rather_than_being_skipped():
    import pytest
    # A truncated download should fail loudly, not silently yield fewer rows.
    with pytest.raises(json.JSONDecodeError):
        read_1806(['{"target": {'])
