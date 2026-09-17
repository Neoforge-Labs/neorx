"""How many diseases the corpus actually contains, per release.

The spec (``neorx.snapshots.frame``) gives corpus *criteria* rather than a
number, because the count is a property of each pinned release, not a
constant. This experiment measures it through sub-project 2's run
recorder, so the answer is a recorded artifact with provenance rather than
a figure someone typed into a manuscript -- the failure mode a previous
sub-project found in all four published papers, where no reported number
could be traced to a run that produced it.

IMPORTANT -- this measures only the first corpus criterion: genetic
evidence in the pinned release, via ``diseases_with_genetic_evidence`` and
``candidate_frame`` (task 8). The second criterion -- at least one target
that reached Phase II or beyond -- needs sub-project 5's trial data, which
does not exist yet. The counts recorded here are therefore an UPPER BOUND
on the eventual corpus, not the corpus itself: applying the Phase II
filter can only shrink these numbers, never grow them. Whoever reads these
rows later must not mistake this population for the final corpus.
"""

from __future__ import annotations

import statistics

from neorx.experiments.record import SNAPSHOTS_DIR, RunRecord
from neorx.experiments.registry import experiment
from neorx.snapshots.frame import candidate_frame, diseases_with_genetic_evidence
from neorx.snapshots.reader import SnapshotStore

# The three OpenTargets format eras this pipeline extracts from -- see the
# module docstring of neorx.snapshots.opentargets. One row is recorded per
# release, so a partial run (a release with no extract built yet) keeps
# whatever releases it already measured rather than losing all of them.
RELEASES = ("18.06", "21.11", "25.06")

# The repository's snapshot store, anchored to the repo root rather than
# the cwd. Opened through the run record, so the extracts read here are
# the ones the record cites.
STORE_ROOT = SNAPSHOTS_DIR


def census_row(store: SnapshotStore, release: str) -> dict:
    """The corpus census for one release. Pure and unit-testable.

    ``n_diseases_total`` is distinct ``disease_id`` values anywhere in the
    release's extract, genetic evidence or not -- the population a disease
    could be drawn from. ``n_diseases_with_genetic_evidence`` narrows that
    to task 8's genetic filter, reused rather than re-implemented so this
    can never drift from the filter a guard test pins.

    ``median_frame_size`` and ``n_target_disease_pairs`` both fall out of
    the same per-disease ``candidate_frame`` call: ``candidate_frame``
    already dedupes symbols within a disease, so ``len(candidate_frame(...))``
    for one disease *is* that disease's count of distinct
    ``(target_symbol, disease_id)`` pairs -- pairs from different diseases
    can never collide, since the disease id differs. Summing across
    diseases with genetic evidence therefore gives the release's total
    pair count directly, and the median of the same list is the median
    frame size, with no second traversal of the associations frame.

    ``0.0`` is reported for ``median_frame_size`` when no disease in the
    release carries genetic evidence -- an empty release is a finding
    ("this release evidences nothing"), not a division-by-zero error.

    Note that ``n_genetic_target_disease_pairs`` is scoped to the genetic
    subset only. The pair count for the whole extract, without the genetic
    filter, is not produced here, as only the genetic subset is used
    downstream (task 5's trial data is not yet available to apply the
    Phase II filter, so these counts are an upper bound on the eventual
    corpus).
    """
    associations = store.associations(release)
    n_diseases_total = int(associations["disease_id"].n_unique())

    genetic_diseases = diseases_with_genetic_evidence(store, release)
    frame_sizes = [len(candidate_frame(store, release, disease)) for disease in genetic_diseases]

    return {
        "release": release,
        "n_diseases_total": n_diseases_total,
        "n_diseases_with_genetic_evidence": len(genetic_diseases),
        "median_frame_size": float(statistics.median(frame_sizes)) if frame_sizes else 0.0,
        "n_genetic_target_disease_pairs": sum(frame_sizes),
    }


@experiment(
    name="corpus-census",
    reads_snapshots=True,
    help="Per-release disease counts: an upper bound on the corpus, pending Phase II data.",
)
def corpus_census(record: RunRecord) -> None:
    """Record the corpus census for all tracked releases.

    This experiment measures the first corpus criterion: genetic evidence in
    the pinned release. The second criterion -- at least one target that
    reached Phase II or beyond -- requires sub-project 5's trial data, which
    does not exist yet. The counts recorded here are therefore an UPPER BOUND
    on the eventual corpus: applying the Phase II filter can only shrink
    these numbers, never grow them.
    """
    store = record.snapshot_store(STORE_ROOT)
    for release in RELEASES:
        record.append_row(census_row(store, release))
