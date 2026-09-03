"""CausalBioRL agent comparison across the three control environments.

Replaces _bench_paper2.py and _bench_fast_baselines.py. The former wrote
its results only after all twelve env x agent cells finished, timed out on
the fourth, and lost 881 seconds of measurement that was then retyped by
hand into the latter as a literal dict. Here every cell is written the
moment it completes.

Offline: the environments are simulated locally, so no cassette is recorded.
"""

from __future__ import annotations

import time

import numpy as np

from neorx.causalbiorl.benchmark import run_single
from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

ENVS = ["GeneticToggle-v0", "MetabolicPathway-v0", "CellGrowth-v0"]
AGENTS = ["causal", "ppo", "sac", "random"]
N_SEEDS = 2
DIFFICULTY = "medium"

#: Episode budget per agent. Recorded on every row: the original benchmark
#: compared a 30-episode causal mean against 50-episode baseline means
#: without disclosing it, which is not a like-for-like comparison.
EPISODES = {"causal": 30, "ppo": 50, "sac": 50, "random": 50}


@experiment(
    name="causalbiorl-bench",
    help="RL agent comparison across control envs.",
    volatile_fields=("wall_clock_s",),
)
def causalbiorl_bench(record: RunRecord) -> None:
    for env_id in ENVS:
        for agent in AGENTS:
            n_episodes = EPISODES[agent]
            started = time.perf_counter()
            per_seed: list[list[float]] = []

            for seed in range(N_SEEDS):
                result = run_single(
                    env_id,
                    agent,
                    seed=seed,
                    n_episodes=n_episodes,
                    difficulty=DIFFICULTY,
                    verbose=False,
                )
                per_seed.append(list(result.episode_rewards))

            matrix = np.array(per_seed)
            record.append_row(
                {
                    "env": env_id,
                    "agent": agent,
                    "n_episodes": n_episodes,
                    "n_seeds": N_SEEDS,
                    "difficulty": DIFFICULTY,
                    "mean_episode_reward": float(np.mean(matrix)),
                    "std_across_seeds": float(np.std(matrix.mean(axis=1))),
                    "wall_clock_s": round(time.perf_counter() - started, 1),
                }
            )
