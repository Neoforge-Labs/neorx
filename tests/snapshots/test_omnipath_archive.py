"""The OmniPath archive reader.

Sub-project 3 deliberately split OmniPath's HTTP fetch from its parsing:
_interactions_to_edges(rows, known_genes) is a pure function of already-
parsed rows. That is what lets a snapshot feed the same parser as the live
API with no second implementation of edge construction, so this reader's
only job is to produce rows in that exact shape.
"""

import polars as pl

from neorx.snapshots.omnipath import read_archive_tsv, to_interaction_rows
from neorx.snapshots.schema import INTERACTION_COLUMNS

TSV = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\n"
    "EGFR\tSHC1\t1\t1\t1\t0\tSIGNOR;TRRUST\tSIGNOR:16331690\n"
    "TP53\tMDM2\t1\t1\t0\t1\tSIGNOR\tSIGNOR:14983059\n"
)


def test_emits_the_canonical_interaction_schema():
    df = read_archive_tsv(TSV)
    assert dict(df.schema) == INTERACTION_COLUMNS


def test_integer_flag_columns_become_booleans():
    df = read_archive_tsv(TSV)
    row = df.filter(pl.col("source_symbol") == "EGFR").row(0, named=True)
    assert row["is_directed"] is True
    assert row["is_stimulation"] is True
    assert row["is_inhibition"] is False


def test_rows_round_trip_into_the_shape_the_live_parser_consumes():
    rows = to_interaction_rows(read_archive_tsv(TSV))
    assert rows[0]["source_genesymbol"] == "EGFR"
    assert rows[0]["target_genesymbol"] == "SHC1"
    assert rows[0]["is_directed"] is True
    assert rows[0]["consensus_direction"] is True
    assert rows[0]["sources"] == ["SIGNOR", "TRRUST"]


def test_the_live_parser_accepts_those_rows_unchanged():
    # The actual integration point: sub-project 3's parser, unmodified.
    from neorx.core.sources.omnipath import _interactions_to_edges

    rows = to_interaction_rows(read_archive_tsv(TSV))
    edges = _interactions_to_edges(rows, {"EGFR", "SHC1", "TP53", "MDM2"})
    assert len(edges) == 2
    by_pair = {(e.source_id, e.target_id): e for e in edges}
    assert by_pair[("gene:EGFR", "gene:SHC1")].sign == 1
    assert by_pair[("gene:TP53", "gene:MDM2")].sign == -1


def test_an_empty_tsv_yields_an_empty_frame_with_the_schema():
    header = TSV.split("\n")[0] + "\n"
    df = read_archive_tsv(header)
    assert df.height == 0
    assert dict(df.schema) == INTERACTION_COLUMNS
