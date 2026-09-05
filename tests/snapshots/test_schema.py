"""The one schema every era reader must produce.

Three OpenTargets format eras feed this pipeline. The whole point of a
derived extract is that downstream code cannot tell them apart, so the
column set is defined once, here, and asserted rather than assumed.
"""

import polars as pl

from neorx.snapshots.schema import (
    ASSOCIATION_COLUMNS,
    EXTRACTOR_VERSION,
    GENETIC_DATATYPES,
    INTERACTION_COLUMNS,
    empty_associations,
    empty_interactions,
)


def test_association_columns_are_exactly_the_five_the_causal_subgraph_needs():
    assert list(ASSOCIATION_COLUMNS) == [
        "target_id", "target_symbol", "disease_id", "datatype", "score",
    ]


def test_association_key_columns_are_strings_and_score_is_a_float():
    assert ASSOCIATION_COLUMNS["target_id"] == pl.Utf8
    assert ASSOCIATION_COLUMNS["target_symbol"] == pl.Utf8
    assert ASSOCIATION_COLUMNS["disease_id"] == pl.Utf8
    assert ASSOCIATION_COLUMNS["datatype"] == pl.Utf8
    assert ASSOCIATION_COLUMNS["score"] == pl.Float64


def test_empty_association_frame_matches_the_declared_schema():
    df = empty_associations()
    assert df.height == 0
    assert dict(df.schema) == ASSOCIATION_COLUMNS


def test_empty_interaction_frame_matches_the_declared_schema():
    df = empty_interactions()
    assert df.height == 0
    assert dict(df.schema) == INTERACTION_COLUMNS


def test_genetic_datatypes_are_exactly_the_two_that_confer_admissibility():
    # These are the OpenTargets datatype names that
    # neorx.core.causal.graph_semantics treats as causal-admissible for a
    # gene-disease edge. A third name added here silently widens what the
    # backdoor criterion is allowed to reason over.
    assert GENETIC_DATATYPES == frozenset({"genetic_association", "somatic_mutation"})


def test_extractor_version_is_an_integer_that_can_be_recorded():
    # A derived artifact needs its derivation pinned: two extracts of the
    # same release made by different extractor code are different inputs.
    assert isinstance(EXTRACTOR_VERSION, int)
    assert EXTRACTOR_VERSION >= 1
