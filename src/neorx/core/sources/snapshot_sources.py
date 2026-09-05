"""
Building the causal subgraph from a pinned snapshot instead of a live API.

These are the readers ``neorx.snapshots.resolver.SourceResolver.resolve``
hands back on a dated run. They return exactly what their live
counterparts return -- ``tuple[list[GraphNode], list[GraphEdge]]`` -- so
the graph builder assembles a dated graph through the same code path it
assembles a live one, and a dated graph is structurally comparable to a
live one rather than being a second, subtly different object.

They live under ``neorx.core.sources`` rather than under
``neorx.snapshots`` because they must construct ``GraphNode`` and
``GraphEdge`` and reuse the live modules' parsing helpers. That makes
``neorx.snapshots.resolver`` depend on ``neorx.core``, while
``neorx.core.__init__`` imports ``graph_builder``, which wants the
resolver's name. Both imports at runtime resolve today, but only because
``neorx/__init__`` happens to import ``neorx.core`` first -- a mutual
dependency held open by import order. So the edge is broken on the side
that does not need it: the resolver imports this module for real, and
``graph_builder`` keeps ``SourceResolver`` to a type-checking annotation.
The runtime dependency runs one way only.

What a dated node score is, and what it is not
----------------------------------------------
The live Open Targets client puts OpenTargets' *overall* association
score on a gene node: a harmonic sum over datasources that the platform
computes and publishes. That number is not in the derived extract, which
carries one row per datatype per target-disease pair, and it is not
recoverable from those rows -- reconstructing it would mean inventing a
quantity and presenting it as a measurement.

So a dated node carries a different quantity, stated plainly: the
**maximum genetic-evidence score in the pinned release**, taken over
``GENETIC_DATATYPES``. Every part of that is a value that literally
appears in the extract; nothing is summed and nothing is synthesised. The
full per-datatype breakdown rides along in
``metadata["datatype_scores"]`` exactly as it does on the live path, so
nothing the extract knows is discarded.

A dated score and a live score are therefore **different quantities and
must never be placed in the same column of a table**, ranked against each
other, or pooled. Comparing releases to each other is sound, because
every release is read the same way; comparing a release to a live build
is not.

Two further differences follow from what the extract does and does not
record, and are stated here rather than patched over:

* Only targets with genetic evidence for the disease become nodes. A
  target supported by literature alone has no genetic score to carry, and
  giving it one would be an invention. This matches
  ``neorx.snapshots.frame.candidate_frame``, which draws the dated run's
  candidate population by the same rule.
* ``n_associated_diseases`` is counted from the extract rather than read
  from the API. The live client asks OpenTargets how many diseases a
  target is associated with, for the specificity penalty in
  ``neorx.core.causal.scoring``; that answer is today's, which in a dated
  build is the same anachronism as the overall score. Counting distinct
  diseases per target within the release gives the number as it stood at
  that date. It is a measurement of the pinned data, not a stand-in for
  the API's value, and the two will not agree.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Protocol

import polars as pl

from neorx.core.graph.dated_frame import match_key
from neorx.core.graph.models import EdgeType, GraphEdge, GraphNode, NodeType
from neorx.core.sources.omnipath import _interactions_to_edges
from neorx.core.sources.open_targets import _evidence_class_for
from neorx.snapshots.omnipath import to_interaction_rows
from neorx.snapshots.schema import GENETIC_DATATYPES

__all__ = [
    "omnipath_from_snapshot",
    "open_targets_from_snapshot",
    "snapshot_reader",
]

SourcePair = tuple[list[GraphNode], list[GraphEdge]]


class _Extracts(Protocol):
    """The part of ``SnapshotStore`` these readers use."""

    def associations(self, release: str) -> pl.DataFrame: ...

    def interactions(self, release: str) -> pl.DataFrame: ...


def open_targets_from_snapshot(
    store: _Extracts,
    release: str,
    disease_name: str,
    *,
    disease_id: str,
    max_results: int = 25,
) -> SourcePair:
    """Gene-disease nodes and edges for one disease in a pinned release.

    ``disease_id`` is the EFO id the extract is keyed on; ``disease_name``
    only supplies the disease node's id, using the same convention as the
    live client and the graph builder, so a dated graph wires up to the
    same disease node a live one does.

    The node score is the release's maximum genetic-evidence score for
    that target, **not** OpenTargets' overall association score, which the
    extract does not carry. See the module docstring: the two are
    different quantities and must not be tabulated together.
    """
    associations = store.associations(release)
    rows = associations.filter(pl.col("disease_id") == disease_id)
    if rows.height == 0:
        return [], []

    # How many diseases each target is associated with in this release.
    # Counted over every datatype, because the live client's count is over
    # every association rather than the genetic ones alone, and this is the
    # quantity standing in that slot. Counted release-wide, not within the
    # disease being built -- specificity is precisely a statement about the
    # rest of the release.
    n_diseases = dict(
        associations
        .select(
            pl.col("target_symbol").str.strip_chars().alias("symbol"),
            pl.col("disease_id"),
        )
        .filter(pl.col("symbol").is_not_null() & (pl.col("symbol") != ""))
        .group_by("symbol")
        .agg(pl.col("disease_id").n_unique().alias("n"))
        .iter_rows()
    )

    # Full per-datatype breakdown, exactly as the live path records it.
    # Two Ensembl ids occasionally share an approved symbol; the graph
    # builder merges such nodes by symbol keeping the higher score, so
    # the same rule is applied here rather than emitting two nodes that
    # would collide on `gene:<symbol>` anyway.
    breakdown: dict[str, dict[str, float]] = {}
    ensembl_ids: dict[str, str] = {}
    for row in rows.iter_rows(named=True):
        symbol = (row["target_symbol"] or "").strip()
        if not symbol:
            continue
        datatype = row["datatype"]
        score = float(row["score"])
        scores = breakdown.setdefault(symbol, {})
        scores[datatype] = max(scores.get(datatype, 0.0), score)
        ensembl_ids.setdefault(symbol, row["target_id"] or "")

    genetic_score: dict[str, float] = {}
    for symbol, scores in breakdown.items():
        supported = [
            score
            for datatype, score in scores.items()
            if datatype in GENETIC_DATATYPES and score > 0.0
        ]
        if supported:
            genetic_score[symbol] = min(max(supported), 1.0)

    ranked = sorted(genetic_score, key=lambda s: (-genetic_score[s], s))[:max_results]

    disease_node_id = f"disease:{disease_name.lower().replace(' ', '_')}"
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for symbol in ranked:
        score = genetic_score[symbol]
        node_id = f"gene:{symbol}"
        dt_scores = breakdown[symbol]

        nodes.append(GraphNode(
            node_id=node_id,
            name=symbol,
            node_type=NodeType.GENE,
            source="Open Targets",
            score=score,
            metadata={
                "ensembl_id": ensembl_ids[symbol],
                "datatype_scores": dt_scores,
                "has_known_drug": dt_scores.get("known_drug", 0.0) > 0.0,
                # Indexed, not .get(symbol, 0): every ranked symbol came
                # from a row of this same frame, so a miss is a real
                # inconsistency and a fabricated 0 would hide it.
                "n_associated_diseases": n_diseases[symbol],
                "snapshot_release": release,
                "score_is": "max genetic-evidence score in this release",
            },
        ))

        edges.append(GraphEdge(
            source_id=node_id,
            target_id=disease_node_id,
            edge_type=EdgeType.ASSOCIATED_WITH,
            weight=score,
            source_db="Open Targets",
            evidence=(
                f"Open Targets {release} genetic evidence: {score:.3f}"
            ),
            evidence_class=_evidence_class_for(dt_scores),
            primary_sources=["Open Targets"],
        ))

    return nodes, edges


def omnipath_from_snapshot(
    store: _Extracts,
    release: str,
    gene_symbols: list[str],
) -> SourcePair:
    """Directed regulatory edges among ``gene_symbols`` in a pinned release.

    Returns no new nodes, matching ``query_omnipath``: OmniPath enriches
    the structure among genes the other sources already found.

    Edge construction is not reimplemented here. The rows are handed to
    ``neorx.core.sources.omnipath._interactions_to_edges`` -- the same
    function the live client feeds -- so there is one implementation of
    what an OmniPath edge means. ``to_interaction_rows`` performs the
    key mapping between the extract's column names and the ones that
    function reads; neither side is renamed to meet the other.

    Symbols are matched through ``dated_frame.match_key`` rather than
    exactly. The extract spells a symbol the way its release does and the
    caller spells it the way *its* release does; an exact match therefore
    dropped every mixed-case HGNC symbol -- ``C9orf72`` and 871 other
    human symbols in the 2018 dump -- from the regulatory layer, without
    a log line. That is safe only because ``read_archive_tsv`` filters
    the extract to human at extraction time: 15,603 of the dump's 34,211
    symbols carry lowercase and most of them are mouse.

    The rows are then re-spelled to the caller's spelling before they
    reach the edge builder, which names each endpoint ``gene:<symbol>``.
    The archive decides which interactions exist; the graph's own nodes
    decide what they are called, or the edge would name a node that is
    not there.
    """
    by_key = {
        match_key(g): g.strip()
        for g in gene_symbols
        if g and g.strip()
    }
    if not by_key:
        return [], []

    keys = list(by_key)
    matching = (
        store.interactions(release)
        .with_columns(
            pl.col("source_symbol").str.strip_chars().str.to_lowercase()
            .alias("_source_key"),
            pl.col("target_symbol").str.strip_chars().str.to_lowercase()
            .alias("_target_key"),
        )
        .filter(pl.col("_source_key").is_in(keys) & pl.col("_target_key").is_in(keys))
        .with_columns(
            pl.col("_source_key").replace_strict(by_key).alias("source_symbol"),
            pl.col("_target_key").replace_strict(by_key).alias("target_symbol"),
        )
        .drop("_source_key", "_target_key")
    )
    return [], _interactions_to_edges(
        to_interaction_rows(matching), set(by_key.values())
    )


_READERS: dict[str, Callable[..., Callable[..., SourcePair]]] = {
    "opentargets": lambda store, release: partial(
        open_targets_from_snapshot, store, release
    ),
    "omnipath": lambda store, release: partial(
        omnipath_from_snapshot, store, release
    ),
}


def snapshot_reader(
    source: str,
    store: _Extracts,
    release: str,
) -> Callable[..., SourcePair]:
    """The reader for ``source`` at ``release``, bound to its extract.

    Raises ``KeyError`` for a source with no snapshot reader. That is not
    a case a caller should handle: the resolver only asks for sources in
    ``PINNED_SOURCES``, so reaching it means the two lists have drifted
    apart, and a silent live fetch is the one outcome a dated run must
    never produce.
    """
    try:
        build = _READERS[source]
    except KeyError:
        raise KeyError(
            f"No snapshot reader for source {source!r}. Sources pinned on a "
            f"dated run must have one; known readers: "
            f"{sorted(_READERS)}."
        ) from None
    return build(store, release)
