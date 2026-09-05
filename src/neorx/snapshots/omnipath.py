"""
Reading an archived OmniPath interactions dump.

Sub-project 3 split OmniPath's fetch from its parsing on purpose:
``neorx.core.sources.omnipath._interactions_to_edges(rows, known_genes)``
is a pure function of already-parsed rows. A snapshot therefore feeds the
same parser the live API does, and there is exactly one implementation of
edge construction in the codebase rather than two that can drift apart.

Two archive-format quirks matter enough to be worth a comment:

Boolean encoding differs by source era. The ARCHIVE writes ``is_directed``,
``is_stimulation`` and ``is_inhibition`` as ``'1'``/``'0'``; the LIVE API
writes the same flags as ``'True'``/``'False'``. Both encodings are
normalised to real booleans in one place (``_to_bool``) rather than per
column, and an unrecognised or empty value reads as False rather than
null -- null is falsy in a way that silently drops rows instead of stating
plainly that the source did not make a directedness claim.

``consensus_direction`` does not exist in the 2018 archive column set at
all. Sub-project 3's admissibility gate requires ``is_directed AND
consensus_direction``; if a missing column simply meant False, every 2018
edge would be rejected and the causal subgraph would have no gene-to-gene
arrows at the earliest time point -- a wrong number that reads exactly
like a real finding. The honest reading is that "directed" is the
strongest directedness signal that era published, so where the column is
absent we set ``consensus_direction`` equal to ``is_directed``. That is a
weaker criterion than later eras provide, so ``read_archive_tsv`` reports
back whether it synthesised the column: a caller has to look at that flag
rather than being able to ignore the substitution silently.
"""

from __future__ import annotations

import io

import polars as pl

from neorx.snapshots.schema import INTERACTION_COLUMNS, empty_interactions

__all__ = ["read_archive_tsv", "to_interaction_rows"]

# Both encodings observed across OmniPath sources: the archive's '1'/'0'
# and the live API's 'True'/'False'. Anything else -- empty string,
# missing value, unrecognised text -- is conservatively False, not null.
_TRUTHY = {"1", "true"}


def _to_bool(flag: str) -> pl.Expr:
    """Normalise a flag column's text encoding to a real boolean.

    Case-insensitive; unrecognised or empty values become False rather
    than null, so a bad or absent value states itself as "not claimed"
    instead of silently disappearing from downstream boolean logic.
    """
    return pl.col(flag).fill_null("").str.to_lowercase().is_in(_TRUTHY).alias(flag)


def read_archive_tsv(text: str) -> tuple[pl.DataFrame, bool]:
    """Parse an archived interactions TSV into the canonical schema.

    Returns ``(frame, consensus_direction_synthesised)``. The flag is
    True when the source TSV had no ``consensus_direction`` column and
    the reader set it equal to ``is_directed`` instead -- see the module
    docstring for why that substitution is made and why it must be
    visible to the caller rather than defaulted away.
    """
    raw = pl.read_csv(io.StringIO(text), separator="\t", infer_schema_length=0)
    if raw.height == 0:
        return empty_interactions(), "consensus_direction" not in raw.columns

    synthesised = "consensus_direction" not in raw.columns
    consensus_expr = (
        _to_bool("is_directed").alias("consensus_direction")
        if synthesised
        else _to_bool("consensus_direction")
    )

    df = raw.select(
        pl.col("source_genesymbol").alias("source_symbol"),
        pl.col("target_genesymbol").alias("target_symbol"),
        _to_bool("is_directed"),
        consensus_expr,
        _to_bool("is_stimulation"),
        _to_bool("is_inhibition"),
        pl.col("sources").fill_null("").alias("primary_sources"),
        pl.col("references").fill_null("").alias("references"),
    ).cast(INTERACTION_COLUMNS)

    return df, synthesised


def to_interaction_rows(df: pl.DataFrame) -> list[dict]:
    """Convert canonical rows into the dicts the live parser consumes.

    Returns the exact key names
    ``neorx.core.sources.omnipath._interactions_to_edges`` reads, so the
    snapshot path reuses it without modification.
    """
    return [
        {
            "source_genesymbol": row["source_symbol"],
            "target_genesymbol": row["target_symbol"],
            "is_directed": row["is_directed"],
            "consensus_direction": row["consensus_direction"],
            "is_stimulation": row["is_stimulation"],
            "is_inhibition": row["is_inhibition"],
            "sources": [s for s in (row["primary_sources"] or "").split(";") if s],
            "references": row["references"] or "",
        }
        for row in df.iter_rows(named=True)
    ]
