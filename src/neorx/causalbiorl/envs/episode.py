"""
Assembling the episode's observation, info dict, and status line.

Pure functions over the pieces ``DrugDiscoveryEnv`` tracks -- the graph
embedding, current molecule features, and per-target state -- with
nothing here reading or mutating the environment itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def build_observation(
    *,
    graph_embedding: NDArray[np.floating],
    mol_features: NDArray[np.floating],
    targets: list[Any],
    step_count: int,
    n_targets: int,
    max_steps: int,
    episode_best_score: float,
    use_surrogate: bool,
    obs_dim: int,
    graph_embedding_dim: int,
    mol_feature_dim: int,
    target_feature_dim: int,
    max_targets: int,
) -> NDArray[np.floating]:
    """Build the full observation vector."""
    obs = np.zeros(obs_dim, dtype=np.float32)

    # Graph embedding
    obs[:graph_embedding_dim] = graph_embedding[:graph_embedding_dim]

    # Current molecule features
    offset = graph_embedding_dim
    obs[offset: offset + mol_feature_dim] = mol_features

    # Per-target summaries
    offset += mol_feature_dim
    for i, target in enumerate(targets):
        start = offset + i * target_feature_dim
        end = start + target_feature_dim
        obs[start:end] = target.summary_features()

    # Meta features
    meta_offset = offset + max_targets * target_feature_dim
    obs[meta_offset] = step_count / max(max_steps, 1)
    obs[meta_offset + 1] = n_targets / max_targets
    obs[meta_offset + 2] = max(episode_best_score, 0.0)
    obs[meta_offset + 3] = float(use_surrogate)

    return obs


def build_info(
    *,
    targets: list[Any],
    step_count: int,
    n_targets: int,
    max_steps: int,
    episode_best_score: float,
) -> dict[str, Any]:
    """Return the episode info dict."""
    target_summary = {}
    for t in targets:
        target_summary[t.gene_name] = {
            "n_attempts": t.n_attempts,
            "best_score": float(t.best_score) if t.best_score > -np.inf else 0.0,
            "best_smiles": t.best_smiles,
        }

    return {
        "step": step_count,
        "n_targets": n_targets,
        "budget_remaining": max_steps - step_count,
        "episode_best_score": (
            float(episode_best_score) if episode_best_score > -np.inf else 0.0
        ),
        "targets": target_summary,
    }


def render_status(disease: str, max_steps: int, info: dict[str, Any]) -> None:
    """Print current episode status."""
    print(f"\n─── DrugDiscovery Step {info['step']}/{max_steps} ───")
    print(f"  Disease: {disease}")
    print(f"  Best score: {info['episode_best_score']:.3f}")
    print(f"  Budget remaining: {info['budget_remaining']}")
    for gene, data in info["targets"].items():
        print(
            f"  {gene}: {data['n_attempts']} attempts, "
            f"best={data['best_score']:.3f}"
        )
