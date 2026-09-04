"""
Which evaluated targets get reported.

``identifier.evaluate_all_targets`` evaluates every candidate in the
disease graph; this module decides which of them the pipeline reports.
The two are separate because they answer different questions, and
conflating them corrupts any statistic about the evaluation: ranking is
by ``causal_confidence``, which awards a bonus for being identifiable,
so the reported slice is a sample selected partly by a property one
might want to measure over the candidates.
"""

from __future__ import annotations

from neorx.core.graph.models import NeoRxResult


def rank_causal_targets(
    results: list[NeoRxResult],
    top_n: int = 10,
    min_causal_confidence: float = 0.3,
) -> list[NeoRxResult]:
    """Filter and rank evaluated targets down to the reported top-N.

    ``results`` is what ``evaluate_all_targets`` returned. Kept separate
    from that evaluation so the full list stays reachable: the survivors
    are not a representative sample of what was evaluated.
    """
    # Split human and pathogen results to prevent pathogen targets
    # from completely crowding out human targets.  Each pool gets
    # at least half the slots (with leftover going to whichever
    # pool has more high-confidence results).
    human_results = [r for r in results if r.target_type != "PATHOGEN_DIRECT"]
    pathogen_results = [r for r in results if r.target_type == "PATHOGEN_DIRECT"]

    half = top_n // 2
    # Each pool gets at least half, remainder filled from the other
    top_human = [r for r in human_results if r.causal_confidence >= min_causal_confidence][:half]
    top_pathogen = [r for r in pathogen_results if r.causal_confidence >= min_causal_confidence][:half]

    # Fill remaining slots from whichever pool has leftovers
    remaining = top_n - len(top_human) - len(top_pathogen)
    if remaining > 0:
        used_ids = {r.protein_id for r in top_human} | {r.protein_id for r in top_pathogen}
        overflow = [
            r for r in results
            if r.protein_id not in used_ids
            and r.causal_confidence >= min_causal_confidence
        ][:remaining]
        combined = top_human + top_pathogen + overflow
    else:
        combined = top_human + top_pathogen

    # Re-sort the combined list by confidence
    combined.sort(key=lambda r: r.causal_confidence, reverse=True)

    if not combined:
        # If nothing passes threshold, return top_n anyway
        combined = results[:top_n]

    return combined[:top_n]
