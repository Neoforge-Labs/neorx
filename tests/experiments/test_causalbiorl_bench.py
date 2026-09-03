"""One row per env x agent cell, written as each completes."""

import pytest

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import get_experiment


def test_the_experiment_is_registered():
    import experiments  # noqa: F401

    assert get_experiment("causalbiorl-bench").name == "causalbiorl-bench"


def test_each_cell_is_written_before_the_next_begins(tmp_path, monkeypatch):
    """The failure this migration exists to fix: a timeout must not lose
    the cells that already finished."""
    import experiments.causalbiorl_bench as mod

    seen: list[int] = []

    def fake_run_single(env_id, agent_type, *, seed, n_episodes, difficulty, verbose):
        if len(seen) == 2:
            raise TimeoutError("simulated timeout on the third cell")
        seen.append(1)

        class _R:
            episode_rewards = [-100.0, -200.0]

        return _R()

    monkeypatch.setattr(mod, "run_single", fake_run_single)
    monkeypatch.setattr(mod, "ENVS", ["GeneticToggle-v0"])
    monkeypatch.setattr(mod, "AGENTS", ["random", "ppo", "sac"])
    monkeypatch.setattr(mod, "N_SEEDS", 1)

    rec = RunRecord.create("causalbiorl-bench", runs_dir=tmp_path)
    with pytest.raises(TimeoutError):
        mod.causalbiorl_bench(rec)
    rec.finalise("failed")

    rows = rec.rows()
    assert len(rows) == 2, "cells completed before the timeout must be on disk"
    assert {r["agent"] for r in rows} == {"random", "ppo"}
    assert rec.citable is False


def test_row_carries_the_episode_count_it_was_measured_over(tmp_path, monkeypatch):
    """The original benchmark compared a 30-episode mean against 50-episode
    means without recording either. The count is part of the result."""
    import experiments.causalbiorl_bench as mod

    def fake_run_single(env_id, agent_type, *, seed, n_episodes, difficulty, verbose):
        class _R:
            episode_rewards = [-1.0] * n_episodes

        return _R()

    monkeypatch.setattr(mod, "run_single", fake_run_single)
    monkeypatch.setattr(mod, "ENVS", ["GeneticToggle-v0"])
    monkeypatch.setattr(mod, "AGENTS", ["random"])
    monkeypatch.setattr(mod, "N_SEEDS", 1)

    rec = RunRecord.create("causalbiorl-bench", runs_dir=tmp_path)
    mod.causalbiorl_bench(rec)
    row = rec.rows()[0]
    assert row["n_episodes"] == mod.EPISODES["random"]
    assert row["n_seeds"] == 1
