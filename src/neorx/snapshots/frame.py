"""
The candidate frame, and which diseases the corpus contains.

A dated run must evaluate the population that date would have evaluated.
If a gene enters the candidate pool because a later source connected it to
something, the analysis is scoring a gene the date would never have looked
at -- hindsight in the sampling frame rather than in the predictor, and
just as fatal to the result.

So the frame is drawn from the pinned release alone, and only from the
evidence types that confer causal admissibility. A gene supported by
literature co-mention is not in the frame, because such an edge is not
admissible as a causal arrow and the gene has nothing for identification
to reason about.

The disease corpus is chosen by the same principle one level up. Selecting
diseases because they look well studied today, then analysing them as of
2018, is the same leak. Both criteria are applied to the pinned release.
"""

from __future__ import annotations

import polars as pl

from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.schema import GENETIC_DATATYPES

__all__ = [
    "candidate_frame",
    "corpus_diseases",
    "diseases_with_genetic_evidence",
]


def _genetic(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("datatype").is_in(list(GENETIC_DATATYPES)) & (pl.col("score") > 0.0))


def candidate_frame(
    store: SnapshotStore,
    release: str,
    disease_id: str,
) -> frozenset[str]:
    """Gene symbols the release associated with a disease on genetic evidence.

    This is the population a dated run may evaluate. Nothing outside it may
    become a candidate, whatever a live source later contributes.
    """
    df = _genetic(store.associations(release)).filter(pl.col("disease_id") == disease_id)
    return frozenset(df["target_symbol"].to_list())


def diseases_with_genetic_evidence(
    store: SnapshotStore,
    release: str,
) -> frozenset[str]:
    """Diseases carrying at least one genetically-supported association.

    A disease without one has an empty causal subgraph, so its
    identifiability is trivially zero. Including such diseases would add
    noise to the corpus rather than signal.
    """
    return frozenset(_genetic(store.associations(release))["disease_id"].to_list())


def corpus_diseases(
    store: SnapshotStore,
    release: str,
    phase2_targets: dict[str, set[str]],
) -> frozenset[str]:
    """Diseases meeting both corpus criteria for this release.

    A disease enters when it has genetic evidence in the release *and* at
    least one target that reached Phase II or beyond, because without the
    latter there is no clinical outcome to predict against.

    ``phase2_targets`` maps disease id to the target symbols that reached
    Phase II. Sub-project 5 produces it; passing it in keeps this function
    a pure predicate over the release rather than a second place that
    knows about trials.
    """
    genetic = diseases_with_genetic_evidence(store, release)
    with_trials = {d for d, targets in phase2_targets.items() if targets}
    return frozenset(genetic & with_trials)
