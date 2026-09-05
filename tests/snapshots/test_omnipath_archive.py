"""The OmniPath archive reader.

Sub-project 3 deliberately split OmniPath's HTTP fetch from its parsing:
_interactions_to_edges(rows, known_genes) is a pure function of already-
parsed rows. That is what lets a snapshot feed the same parser as the live
API with no second implementation of edge construction, so this reader's
only job is to produce rows in that exact shape.

Three real-data defects drive most of the tests below:

- The ARCHIVE encodes booleans as '1'/'0'; the LIVE API encodes them as
  'True'/'False'. Both must parse to real booleans, not null.
- The 2018 archive has no `consensus_direction` column at all. Per T5,
  its absence is filled with `is_directed`, and the reader must tell the
  caller it did that rather than defaulting the flag away.
- The archive is not human. 316,327 of the real 2018 dump's 644,845 rows
  are mouse (242,338) or rat (73,989), the canonical schema carries
  nothing that marks them, and the columns that would (`ncbi_tax_id_source`,
  `ncbi_tax_id_target`) are dropped by the reader. So the filter has to
  run here or not at all. Every fixture below therefore carries the tax
  columns, because a fixture without them is a fixture of a file the
  reader must refuse.
"""

import polars as pl

from neorx.snapshots.omnipath import read_archive_tsv, to_interaction_rows
from neorx.snapshots.schema import INTERACTION_COLUMNS

TSV = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\t"
    "ncbi_tax_id_source\tncbi_tax_id_target\n"
    "EGFR\tSHC1\t1\t1\t1\t0\tSIGNOR;TRRUST\tSIGNOR:16331690\t9606\t9606\n"
    "TP53\tMDM2\t1\t1\t0\t1\tSIGNOR\tSIGNOR:14983059\t9606\t9606\n"
)

# The organism problem, in the four shapes it actually takes in the real
# dump. Mouse `Trp53` also demonstrates why the case rule cannot land
# without this filter: it matches human TP53 case-insensitively.
TSV_MIXED_ORGANISM = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\t"
    "ncbi_tax_id_source\tncbi_tax_id_target\n"
    "EGFR\tSHC1\t1\t1\t1\t0\tSIGNOR\tSIGNOR:16331690\t9606\t9606\n"
    # Mouse-mouse. This exact interaction (C3 -> C5) is one of the five
    # directed non-human rows that pass an exact-case human symbol filter
    # in the real 2018 dump.
    "C3\tC5\t1\t1\t1\t0\tSIGNOR\tSIGNOR:11111111\t10090\t10090\n"
    # Rat-rat, and one of the other four: F7 -> F10.
    "F7\tF10\t1\t1\t1\t0\tSIGNOR\tSIGNOR:22222222\t10116\t10116\n"
    # Mouse Title-case, which only a case-insensitive match would let in.
    "Trp53\tMdm2\t1\t1\t0\t1\tSIGNOR\tSIGNOR:33333333\t10090\t10090\n"
    # Cross-species: a mouse protein acting on a human one is not a human
    # regulatory arrow, so one human endpoint is not enough.
    "Egfr\tSHC1\t1\t1\t1\t0\tSIGNOR\tSIGNOR:44444444\t10090\t9606\n"
)

# The same columns as TSV, minus the two the organism filter reads.
TSV_NO_ORGANISM_COLUMNS = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\n"
    "EGFR\tSHC1\t1\t1\t1\t0\tSIGNOR;TRRUST\tSIGNOR:16331690\n"
)

# The real 2018 archive column set (omnipath_webservice_interactions__
# 20180614-20181114.tsv.xz has 22 columns; consensus_direction is not
# among them). Only the leading, functionally relevant columns are
# included here, but they are the era's real shape.
TSV_2018_NO_CONSENSUS = (
    "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\t"
    "is_stimulation\tis_inhibition\tsources\treferences\tdip_url\tomnipath\t"
    "ncbi_tax_id_source\tncbi_tax_id_target\n"
    "P04626\tP01133\tERBB2\tEGF\t1\t1\t0\tSIGNOR\tSIGNOR:12345678\t\t1\t"
    "9606\t9606\n"
)

# The LIVE API's boolean encoding: 'True'/'False' text rather than '1'/'0'.
TSV_LIVE_ENCODING = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\t"
    "ncbi_tax_id_source\tncbi_tax_id_target\n"
    "EGFR\tSHC1\tTrue\tTrue\tTrue\tFalse\tSIGNOR;TRRUST\tSIGNOR:16331690\t"
    "9606\t9606\n"
    "TP53\tMDM2\tTrue\tTrue\tFalse\tTrue\tSIGNOR\tSIGNOR:14983059\t9606\t9606\n"
)

# An unrecognised / empty flag value must become False, not null.
TSV_UNRECOGNISED_FLAG = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\t"
    "ncbi_tax_id_source\tncbi_tax_id_target\n"
    "EGFR\tSHC1\tmaybe\t1\t\t0\tSIGNOR\tSIGNOR:16331690\t9606\t9606\n"
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
    df, notes = read_archive_tsv(header)
    assert df.height == 0
    assert dict(df.schema) == INTERACTION_COLUMNS
    assert notes.consensus_direction_synthesised is False


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
    df, notes = read_archive_tsv(TSV_2018_NO_CONSENSUS)
    assert dict(df.schema) == INTERACTION_COLUMNS
    assert notes.consensus_direction_synthesised is True


def test_missing_consensus_direction_is_set_equal_to_is_directed():
    df, _ = read_archive_tsv(TSV_2018_NO_CONSENSUS)
    row = df.row(0, named=True)
    assert row["is_directed"] is True
    assert row["consensus_direction"] == row["is_directed"]


def test_present_consensus_direction_is_not_marked_synthesised():
    _, notes = read_archive_tsv(TSV)
    assert notes.consensus_direction_synthesised is False


def test_synthesised_2018_rows_still_reach_the_live_parser():
    from neorx.core.sources.omnipath import _interactions_to_edges

    df, notes = read_archive_tsv(TSV_2018_NO_CONSENSUS)
    assert notes.consensus_direction_synthesised is True
    rows = to_interaction_rows(df)
    edges = _interactions_to_edges(rows, {"ERBB2", "EGF"})
    assert len(edges) == 1
    assert edges[0].source_id == "gene:ERBB2"
    assert edges[0].target_id == "gene:EGF"


# --- C3: half the archive is mouse and rat ---


def test_non_human_rows_do_not_reach_the_extract():
    df, _ = read_archive_tsv(TSV_MIXED_ORGANISM)
    assert df["source_symbol"].to_list() == ["EGFR"]
    assert df["target_symbol"].to_list() == ["SHC1"]


def test_a_cross_species_row_is_not_human_enough():
    # One human endpoint is not a human regulatory arrow.
    df, _ = read_archive_tsv(TSV_MIXED_ORGANISM)
    assert ("Egfr", "SHC1") not in list(
        zip(df["source_symbol"].to_list(), df["target_symbol"].to_list())
    )


def test_the_dropped_non_human_rows_are_counted():
    # Filtering silently is the failure mode: an extract that quietly
    # halved is indistinguishable from one that read half a file.
    _, notes = read_archive_tsv(TSV_MIXED_ORGANISM)
    assert notes.non_human_rows_dropped == 4


def test_a_human_only_archive_reports_nothing_dropped():
    _, notes = read_archive_tsv(TSV)
    assert notes.non_human_rows_dropped == 0


def test_a_mouse_symbol_cannot_be_reached_by_a_case_insensitive_match():
    """Why the organism filter has to land before the case rule.

    Mouse `Trp53` differs from human `TP53` only in case, and 15,603 of
    the real dump's 34,211 symbols carry lowercase. Matching symbols
    case-insensitively over an unfiltered extract imports the mouse
    interactome into human causal graphs.
    """
    from neorx.core.sources.snapshot_sources import omnipath_from_snapshot

    df, _ = read_archive_tsv(TSV_MIXED_ORGANISM)

    class _Store:
        def interactions(self, release):
            return df

        def associations(self, release):  # pragma: no cover - unused here
            raise AssertionError("associations must not be read here")

    _nodes, edges = omnipath_from_snapshot(
        _Store(), "2018", ["TP53", "MDM2", "EGFR", "SHC1", "C3", "C5"]
    )
    assert sorted((e.source_id, e.target_id) for e in edges) == [
        ("gene:EGFR", "gene:SHC1"),
    ]


def test_an_archive_without_organism_columns_is_refused():
    """The extract's claim is that it is human; an unverifiable claim fails.

    Passing the rows through unfiltered would produce an extract that is
    indistinguishable from a filtered one and is not one.
    """
    import pytest

    with pytest.raises(ValueError) as excinfo:
        read_archive_tsv(TSV_NO_ORGANISM_COLUMNS)
    message = str(excinfo.value)
    assert "ncbi_tax_id_source" in message
    assert "ncbi_tax_id_target" in message
