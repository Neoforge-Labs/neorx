"""
Reading an archived OmniPath interactions dump.

Sub-project 3 split OmniPath's fetch from its parsing on purpose:
``neorx.core.sources.omnipath._interactions_to_edges(rows, known_genes)``
is a pure function of already-parsed rows. A snapshot therefore feeds the
same parser the live API does, and there is exactly one implementation of
edge construction in the codebase rather than two that can drift apart.

Three archive-format facts matter enough to be worth a comment:

Boolean encoding differs by source era. The ARCHIVE writes ``is_directed``,
``is_stimulation`` and ``is_inhibition`` as ``'1'``/``'0'``; the LIVE API
writes the same flags as ``'True'``/``'False'``. Both encodings are
normalised to real booleans in one place (``_to_bool``) rather than per
column, and an unrecognised or empty value reads as False rather than
null -- null is falsy in a way that silently drops rows instead of stating
plainly that the source did not make a directedness claim.

Organism is the other one, and it is the reason the reader filters at
all. The archive is not human-only: 316,327 of the 2018 dump's 644,845
rows are mouse (242,338) or rat (73,989), and ``INTERACTION_COLUMNS``
carries nothing that marks them. Nothing downstream can tell them apart
afterwards, so the filter has to run here, on ``ncbi_tax_id_source`` and
``ncbi_tax_id_target``, before those columns are dropped. Both endpoints
must be human: a mouse protein acting on a human one is not a human
regulatory arrow.

This matters more than a 49% row count suggests, because gene symbols are
matched between the extract and a graph's gene list. Exact-case matching
hid most of the mouse rows by accident -- mouse symbols are Title-case,
``A1bg``, ``A2m`` -- and incompletely: 15 non-human rows pass an
exact-case human symbol filter today and 5 of them are directed, putting
mouse complement (C3->C5) and coagulation-cascade (F5->F2, F3->F7,
F7->F10) arrows into human graphs. The moment symbol matching becomes
case-insensitive, which is what a mixed-case human symbol like
``C9orf72`` requires, 15,603 of the dump's 34,211 symbols become
matchable and that 15 becomes thousands. Organism filtering is what makes
the case rule safe, so it lands first.

An archive without the two tax columns is refused rather than passed
through unfiltered. An extract's whole claim is that it is human; a
reader that cannot check that claim must say so, not assume it.

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

from neorx.snapshots.schema import (
    HUMAN_TAXON_ID,
    INTERACTION_COLUMNS,
    ORGANISM_COLUMNS,
    ExtractNotes,
    empty_interactions,
)

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


def read_archive_tsv(text: str) -> tuple[pl.DataFrame, ExtractNotes]:
    """Parse an archived interactions TSV into the canonical schema.

    Returns ``(frame, notes)``. Both fields of ``notes`` record something
    the rows themselves no longer show -- that ``consensus_direction`` was
    synthesised from ``is_directed``, and how many non-human rows the
    organism filter removed. See the module docstring for why each
    substitution is made and why neither may be defaulted away.

    Raises ``ValueError`` when the source has no organism columns. Every
    consumer of this extract treats its rows as human; a reader that
    cannot verify that must refuse rather than assume it.
    """
    raw = pl.read_csv(io.StringIO(text), separator="\t", infer_schema_length=0)
    missing = [c for c in ORGANISM_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"OmniPath archive has no organism column(s) {missing}; it has "
            f"{sorted(raw.columns)}. Both {list(ORGANISM_COLUMNS)} are "
            f"required: 49% of the 2018 dump is mouse and rat, nothing "
            f"downstream distinguishes them, and the extract's schema "
            f"cannot carry the distinction."
        )
    if raw.height == 0:
        return empty_interactions(), ExtractNotes(
            consensus_direction_synthesised="consensus_direction" not in raw.columns,
        )

    human = raw.filter(
        (pl.col(ORGANISM_COLUMNS[0]).str.strip_chars() == HUMAN_TAXON_ID)
        & (pl.col(ORGANISM_COLUMNS[1]).str.strip_chars() == HUMAN_TAXON_ID)
    )
    dropped = raw.height - human.height

    synthesised = "consensus_direction" not in raw.columns
    consensus_expr = (
        _to_bool("is_directed").alias("consensus_direction")
        if synthesised
        else _to_bool("consensus_direction")
    )

    df = human.select(
        pl.col("source_genesymbol").alias("source_symbol"),
        pl.col("target_genesymbol").alias("target_symbol"),
        _to_bool("is_directed"),
        consensus_expr,
        _to_bool("is_stimulation"),
        _to_bool("is_inhibition"),
        pl.col("sources").fill_null("").alias("primary_sources"),
        pl.col("references").fill_null("").alias("references"),
    ).cast(INTERACTION_COLUMNS)

    return df, ExtractNotes(
        consensus_direction_synthesised=synthesised,
        non_human_rows_dropped=dropped,
    )


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
