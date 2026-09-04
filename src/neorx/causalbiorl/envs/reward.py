"""
Turning per-objective scores into a single reward.

Wraps ``neorx.causalbiorl.causal.reward_learner.AdaptiveRewardLearner``:
an adaptively-weighted combination of the objective scores
``screening.py`` produces. The learner is stateful -- it tracks running
per-objective statistics and trains a critic -- so it must persist
across calls; the caller owns its construction and lifetime, this only
scores against whatever instance it is handed.
"""

from __future__ import annotations

from typing import Any


def compute_reward(
    state: Any,
    obj_scores: dict[str, float],
    *,
    learner: Any,
) -> float:
    """Reward for these objective scores, recording them into ``learner``.

    Falls back to an equal-weight sum of the objective scores if
    ``learner`` is missing or raises -- a missing learner (construction
    failed) behaves exactly like one whose ``compute_reward`` call fails,
    since both simply skip straight to the fallback.
    """
    try:
        reward = learner.compute_reward(state, obj_scores)
        learner.update(state, obj_scores)
        return reward
    except Exception:
        # Fallback: equal-weight sum
        return sum(obj_scores.values()) / max(len(obj_scores), 1)


def score_reward(
    state: Any,
    obj_scores: dict[str, float],
    *,
    learner: Any,
) -> float:
    """The reward for these objective scores, recording nothing.

    Mirrors ``compute_reward`` exactly apart from the two mutations it
    performs -- the learner's weight/objective history and the critic
    update. A planner scoring a batch must not write hundreds of phantom
    history entries or train the critic on candidates it never takes.
    """
    try:
        return learner.score(state, obj_scores)
    except Exception:
        # Fallback: equal-weight sum. Mirrors compute_reward's own
        # fallback so evaluate_actions and step agree on every path.
        return sum(obj_scores.values()) / max(len(obj_scores), 1)
