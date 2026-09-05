"""
Reading an archived OmniPath interactions dump.

Sub-project 3 split OmniPath's fetch from its parsing on purpose:
``neorx.core.sources.omnipath._interactions_to_edges(rows, known_genes)``
is a pure function of already-parsed rows. A snapshot therefore feeds the
same parser the live API does, and there is exactly one implementation of
edge construction in the codebase rather than two that can drift apart.

The archive's TSV schema is uniform across the range this project uses, so
one reader covers every time point.
"""

from __future__ import annotations

import io

import polars as pl

from neorx.snapshots.schema import INTERACTION_COLUMNS, empty_interactions

__all__ = ["read_archive_tsv", "to_interaction_rows"]

_FLAGS = ("is_directed", "consensus_direction", "is_stimulation", "is_inhibition")


def read_archive_tsv(text: str) -> pl.DataFrame:
    """Parse an archived interactions TSV into the canonical schema.

    The archive encodes booleans as 0/1 integers; they become real
    booleans here so that downstream code never has to remember which
    convention a given source used.
    """
    raw = pl.read_csv(io.StringIO(text), separator="\t", infer_schema_length=0)
    if raw.height == 0:
        return empty_interactions()

    return raw.select(
        pl.col("source_genesymbol").alias("source_symbol"),
        pl.col("target_genesymbol").alias("target_symbol"),
        *[
            (pl.col(flag).cast(pl.Int8, strict=False) == 1).alias(flag)
            for flag in _FLAGS
        ],
        pl.col("sources").fill_null("").alias("primary_sources"),
        pl.col("references").fill_null("").alias("references"),
    ).cast(INTERACTION_COLUMNS)


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
