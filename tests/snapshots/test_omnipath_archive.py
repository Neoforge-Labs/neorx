"""The OmniPath archive reader.

Sub-project 3 deliberately split OmniPath's HTTP fetch from its parsing:
_interactions_to_edges(rows, known_genes) is a pure function of already-
parsed rows. That is what lets a snapshot feed the same parser as the live
API with no second implementation of edge construction, so this reader's
only job is to produce rows in that exact shape.

Two real-data defects drive most of the tests below:

- The ARCHIVE encodes booleans as '1'/'0'; the LIVE API encodes them as
  'True'/'False'. Both must parse to real booleans, not null.
- The 2018 archive has no `consensus_direction` column at all. Per T5,
  its absence is filled with `is_directed`, and the reader must tell the
  caller it did that rather than defaulting the flag away.
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

# The real 2018 archive column set (omnipath_webservice_interactions__
# 20180614-20181114.tsv.xz has 22 columns; consensus_direction is not
# among them). Only the leading, functionally relevant columns are
# included here, but they are the era's real shape.
TSV_2018_NO_CONSENSUS = (
    "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\t"
    "is_stimulation\tis_inhibition\tsources\treferences\tdip_url\tomnipath\n"
    "P04626\tP01133\tERBB2\tEGF\t1\t1\t0\tSIGNOR\tSIGNOR:12345678\t\t1\n"
)

# The LIVE API's boolean encoding: 'True'/'False' text rather than '1'/'0'.
TSV_LIVE_ENCODING = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\n"
    "EGFR\tSHC1\tTrue\tTrue\tTrue\tFalse\tSIGNOR;TRRUST\tSIGNOR:16331690\n"
    "TP53\tMDM2\tTrue\tTrue\tFalse\tTrue\tSIGNOR\tSIGNOR:14983059\n"
)

# An unrecognised / empty flag value must become False, not null.
TSV_UNRECOGNISED_FLAG = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\n"
    "EGFR\tSHC1\tmaybe\t1\t\t0\tSIGNOR\tSIGNOR:16331690\n"
)


def test_emits_the_canonical_interaction_schema():
    df, _ = read_archive_tsv(TSV)
    assert dict(df.schema) == INTERACTION_COLUMNS


def test_integer_flag_columns_become_booleans():
    df, _ = read_archive_tsv(TSV)
    row = df.filter(pl.col("source_symbol") == "EGFR").row(0, named=True)
    assert row["is_directed"] is True
    assert row["is_stimulation"] is True
    assert row["is_inhibition"] is False


def test_rows_round_trip_into_the_shape_the_live_parser_consumes():
    df, _ = read_archive_tsv(TSV)
    rows = to_interaction_rows(df)
    assert rows[0]["source_genesymbol"] == "EGFR"
    assert rows[0]["target_genesymbol"] == "SHC1"
    assert rows[0]["is_directed"] is True
    assert rows[0]["consensus_direction"] is True
    assert rows[0]["sources"] == ["SIGNOR", "TRRUST"]


def test_the_live_parser_accepts_those_rows_unchanged():
    # The actual integration point: sub-project 3's parser, unmodified.
    from neorx.core.sources.omnipath import _interactions_to_edges

    df, _ = read_archive_tsv(TSV)
    rows = to_interaction_rows(df)
    edges = _interactions_to_edges(rows, {"EGFR", "SHC1", "TP53", "MDM2"})
    assert len(edges) == 2
    by_pair = {(e.source_id, e.target_id): e for e in edges}
    assert by_pair[("gene:EGFR", "gene:SHC1")].sign == 1
    assert by_pair[("gene:TP53", "gene:MDM2")].sign == -1


def test_an_empty_tsv_yields_an_empty_frame_with_the_schema():
    header = TSV.split("\n")[0] + "\n"
    df, synthesised = read_archive_tsv(header)
    assert df.height == 0
    assert dict(df.schema) == INTERACTION_COLUMNS
    assert synthesised is False


# --- C1: boolean encoding differs between the archive and the live API ---


def test_live_api_true_false_strings_parse_to_real_booleans():
    df, _ = read_archive_tsv(TSV_LIVE_ENCODING)
    row = df.filter(pl.col("source_symbol") == "EGFR").row(0, named=True)
    assert row["is_directed"] is True
    assert row["is_stimulation"] is True
    assert row["is_inhibition"] is False
    assert row["consensus_direction"] is True
    # Never null: null is falsy in a way that hides the defect.
    assert df["is_directed"].null_count() == 0


def test_archive_1_0_strings_still_parse_to_real_booleans():
    # Regression guard: fixing the 'True'/'False' encoding must not break
    # the archive's existing '1'/'0' encoding.
    df, _ = read_archive_tsv(TSV)
    assert df["is_directed"].null_count() == 0
    row = df.filter(pl.col("source_symbol") == "TP53").row(0, named=True)
    assert row["is_directed"] is True
    assert row["is_stimulation"] is False
    assert row["is_inhibition"] is True


def test_unrecognised_or_empty_flag_value_becomes_false_not_null():
    df, _ = read_archive_tsv(TSV_UNRECOGNISED_FLAG)
    row = df.row(0, named=True)
    assert row["is_directed"] is False  # 'maybe' is not a recognised truthy token
    assert row["is_stimulation"] is False  # empty string
    assert df["is_directed"].null_count() == 0
    assert df["is_stimulation"].null_count() == 0


def test_live_encoding_feeds_the_live_parser_and_produces_edges():
    # End-to-end check of C1: before the fix, every is_directed became
    # null (falsy), so zero edges came out of _interactions_to_edges.
    from neorx.core.sources.omnipath import _interactions_to_edges

    df, _ = read_archive_tsv(TSV_LIVE_ENCODING)
    rows = to_interaction_rows(df)
    edges = _interactions_to_edges(rows, {"EGFR", "SHC1", "TP53", "MDM2"})
    assert len(edges) == 2


# --- C2: the 2018 archive has no consensus_direction column ---


def test_missing_consensus_direction_column_parses_without_raising():
    df, synthesised = read_archive_tsv(TSV_2018_NO_CONSENSUS)
    assert dict(df.schema) == INTERACTION_COLUMNS
    assert synthesised is True


def test_missing_consensus_direction_is_set_equal_to_is_directed():
    df, _ = read_archive_tsv(TSV_2018_NO_CONSENSUS)
    row = df.row(0, named=True)
    assert row["is_directed"] is True
    assert row["consensus_direction"] == row["is_directed"]


def test_present_consensus_direction_is_not_marked_synthesised():
    _, synthesised = read_archive_tsv(TSV)
    assert synthesised is False


def test_synthesised_2018_rows_still_reach_the_live_parser():
    from neorx.core.sources.omnipath import _interactions_to_edges

    df, synthesised = read_archive_tsv(TSV_2018_NO_CONSENSUS)
    assert synthesised is True
    rows = to_interaction_rows(df)
    edges = _interactions_to_edges(rows, {"ERBB2", "EGF"})
    assert len(edges) == 1
    assert edges[0].source_id == "gene:ERBB2"
    assert edges[0].target_id == "gene:EGF"
