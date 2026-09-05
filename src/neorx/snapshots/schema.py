"""
The canonical shape of a derived extract.

Three OpenTargets format eras feed this pipeline: a flat gzipped JSON
dump in 2018, an ETL-era Parquet tree in 2021, and the current output
layout. Each gets its own reader, and each reader's job is to produce
exactly the columns below. Downstream code -- the frame, the resolver,
the graph builder -- never learns which era a row came from.

That is the whole value of a derived extract rather than a mirror: format
drift is absorbed once, at the edge, instead of leaking into every
consumer.
"""

from __future__ import annotations

import polars as pl

# Bump when a reader's output changes meaning for the same input. Recorded
# in the manifest beside the release id, because two extracts of the same
# release made by different extractor code are different inputs and must
# be distinguishable.
EXTRACTOR_VERSION = 1

ASSOCIATION_COLUMNS: dict[str, pl.DataType] = {
    # Ensembl gene id. The only identifier present in every era, so it is
    # the join key; 25.06 has no symbol in its association table at all.
    "target_id": pl.Utf8,
    # HGNC symbol. What graph_builder, STRING and OmniPath actually work
    # in, so it is carried rather than re-derived at every call site.
    "target_symbol": pl.Utf8,
    # EFO disease id.
    "disease_id": pl.Utf8,
    # OpenTargets datatype: genetic_association, somatic_mutation,
    # literature, known_drug, and so on. One row per datatype per pair.
    "datatype": pl.Utf8,
    "score": pl.Float64,
}

INTERACTION_COLUMNS: dict[str, pl.DataType] = {
    "source_symbol": pl.Utf8,
    "target_symbol": pl.Utf8,
    "is_directed": pl.Boolean,
    "consensus_direction": pl.Boolean,
    "is_stimulation": pl.Boolean,
    "is_inhibition": pl.Boolean,
    # Semicolon-joined; parsed by the reader into a list at use time.
    "primary_sources": pl.Utf8,
    "references": pl.Utf8,
}

# The OpenTargets datatypes that neorx.core.causal.graph_semantics treats
# as conferring causal admissibility on a gene-disease edge, on the
# Mendelian randomisation warrant. Adding a name here silently widens what
# the backdoor criterion may reason over.
GENETIC_DATATYPES: frozenset[str] = frozenset({
    "genetic_association",
    "somatic_mutation",
})

__all__ = [
    "ASSOCIATION_COLUMNS",
    "EXTRACTOR_VERSION",
    "GENETIC_DATATYPES",
    "INTERACTION_COLUMNS",
    "empty_associations",
    "empty_interactions",
]


def empty_associations() -> pl.DataFrame:
    """An empty association frame with the canonical schema."""
    return pl.DataFrame(schema=ASSOCIATION_COLUMNS)


def empty_interactions() -> pl.DataFrame:
    """An empty interaction frame with the canonical schema."""
    return pl.DataFrame(schema=INTERACTION_COLUMNS)
