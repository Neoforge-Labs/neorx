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

from dataclasses import dataclass

import polars as pl

# Bump when a reader's output changes meaning for the same input. Recorded
# in the manifest beside the release id, because two extracts of the same
# release made by different extractor code are different inputs and must
# be distinguishable.
#
# 2: the OmniPath archive reader filters to human-human interactions. 49%
#    of the 2018 dump is mouse and rat (242,338 and 73,989 rows of
#    644,845), and nothing downstream marks them, so an extract built
#    before this and one built after are different inputs for the same
#    release.
EXTRACTOR_VERSION = 2

# NCBI taxonomy id for Homo sapiens, as the archives spell it -- a string,
# because the extract's tax columns are read without schema inference so
# that "9606" is never silently widened to a float.
HUMAN_TAXON_ID = "9606"

# The archive columns naming each endpoint's organism. They do not survive
# into INTERACTION_COLUMNS: they are what the reader filters on, and once
# it has, every remaining row is human by construction.
ORGANISM_COLUMNS: tuple[str, str] = ("ncbi_tax_id_source", "ncbi_tax_id_target")


@dataclass(frozen=True)
class ExtractNotes:
    """What a reader had to do to the source that the rows no longer show.

    Both facts are recorded rather than defaulted away, because both change
    what the extract means and neither is visible in the extract itself.
    ``consensus_direction_synthesised`` says the 2018 archive had no such
    column and ``is_directed`` was used in its place -- a weaker criterion
    than later eras publish. ``non_human_rows_dropped`` says how many rows
    the organism filter removed; a count of zero on a real OmniPath archive
    means the filter did not fire, not that the dump was human-only.
    """

    consensus_direction_synthesised: bool = False
    non_human_rows_dropped: int = 0

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
    "HUMAN_TAXON_ID",
    "INTERACTION_COLUMNS",
    "ORGANISM_COLUMNS",
    "ExtractNotes",
    "empty_associations",
    "empty_interactions",
]


def empty_associations() -> pl.DataFrame:
    """An empty association frame with the canonical schema."""
    return pl.DataFrame(schema=ASSOCIATION_COLUMNS)


def empty_interactions() -> pl.DataFrame:
    """An empty interaction frame with the canonical schema."""
    return pl.DataFrame(schema=INTERACTION_COLUMNS)
