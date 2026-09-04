"""
The hierarchical planner must actually plan.

`pipeline.py` builds the agent, calls `init_hierarchical_planner()`, and
only then calls `train()`. `_reward_fn` used to be assigned inside
`train()`, and `HierarchicalPlanner` captures the value rather than a
reference -- so in the end-to-end pipeline the planner's reward function
was always None, its CEM never ran, and every molecule the pipeline
reported came from `rng.standard_normal(latent_dim) * 0.3`.
"""

import networkx as nx
import numpy as np
import pytest

from neorx.causalbiorl.envs.drug_discovery import DrugDiscoveryEnv


def _env() -> DrugDiscoveryEnv:
    G = nx.DiGraph()
    G.add_node("gene:A", score=0.9)
    G.add_node("disease:x", score=1.0)
    G.add_edge("gene:A", "disease:x", weight=0.9)
    return DrugDiscoveryEnv(
        disease="x",
        prebuilt_graph=G,
        prebuilt_targets=[{
            "gene_name": "A", "protein_id": "P1",
            "protein_name": "A", "causal_confidence": 0.8,
        }],
        max_steps=5,
    )


def test_reward_fn_is_set_before_train_is_called():
    from neorx.causalbiorl.agents.causal_agent import CausalAgent
    from neorx.causalbiorl.models import AgentConfig

    agent = CausalAgent(_env(), AgentConfig(agent_type="causal", n_episodes=1))
    assert agent._reward_fn is not None


def test_planner_built_in_the_pipelines_order_can_plan():
    from neorx.causalbiorl.agents.causal_agent import CausalAgent
    from neorx.causalbiorl.models import AgentConfig

    agent = CausalAgent(_env(), AgentConfig(agent_type="causal", n_episodes=1))
    agent.init_hierarchical_planner(n_targets=1, latent_dim=agent.env.latent_dim)
    assert agent._hierarchical_planner is not None
    assert agent._hierarchical_planner.reward_fn is not None


def test_train_propagates_a_reward_override_to_an_existing_planner():
    from neorx.causalbiorl.agents.causal_agent import CausalAgent
    from neorx.causalbiorl.models import AgentConfig

    agent = CausalAgent(_env(), AgentConfig(agent_type="causal", n_episodes=1))
    agent.init_hierarchical_planner(n_targets=1, latent_dim=agent.env.latent_dim)

    sentinel = lambda state, action: 0.5  # noqa: E731
    agent.train(n_episodes=1, reward_fn=sentinel, verbose=False)

    assert agent._hierarchical_planner.reward_fn is sentinel


def test_decode_latent_batch_returns_one_smiles_per_row():
    from neorx.causalbiorl.envs.generation import decode_latent_batch

    env = _env()
    env.reset(seed=0)
    env._init_genmol()
    Z = np.zeros((4, env.latent_dim), dtype=np.float32)

    smiles = decode_latent_batch(env._genmol_model, env._genmol_tokenizer, Z)

    assert len(smiles) == 4
    assert all(isinstance(s, str) for s in smiles)


def test_decode_latent_batch_agrees_with_single_decode():
    from neorx.causalbiorl.envs.generation import decode_latent_batch

    env = _env()
    env.reset(seed=0)
    z = np.zeros(env.latent_dim, dtype=np.float32)

    single = env._decode_latent(z)
    batched = decode_latent_batch(
        env._genmol_model, env._genmol_tokenizer, z.reshape(1, -1),
    )

    assert batched[0] == single


def test_evaluate_actions_returns_one_reward_per_action():
    env = _env()
    state, _ = env.reset(seed=0)
    actions = np.zeros((3, env.action_space.shape[0]), dtype=np.float32)
    actions[:, 1] = -1.0

    rewards = env.evaluate_actions(state, actions)

    assert rewards.shape == (3,)
    assert np.all(np.isfinite(rewards))


def test_evaluate_actions_does_not_mutate_environment_state():
    env = _env()
    state, _ = env.reset(seed=0)

    before_step = env._step_count
    before_best = [t.best_score for t in env._targets]
    before_zbase = [t.z_base.copy() for t in env._targets]
    before_attempts = [t.n_attempts for t in env._targets]

    actions = np.zeros((5, env.action_space.shape[0]), dtype=np.float32)
    actions[:, 1] = -1.0
    env.evaluate_actions(state, actions)

    assert env._step_count == before_step
    assert [t.best_score for t in env._targets] == before_best
    assert [t.n_attempts for t in env._targets] == before_attempts
    for after, before in zip(env._targets, before_zbase):
        assert np.array_equal(after.z_base, before)


def test_evaluate_actions_agrees_with_step_for_the_same_action():
    env = _env()
    state, _ = env.reset(seed=0)
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    action[1] = -1.0

    predicted = env.evaluate_actions(state, action.reshape(1, -1))[0]
    _, actual, _, _, _ = env.step(action)

    assert predicted == pytest.approx(actual, abs=1e-6)


def test_reward_learner_score_is_pure():
    from neorx.causalbiorl.causal.reward_learner import AdaptiveRewardLearner

    learner = AdaptiveRewardLearner(state_dim=4)
    state = np.zeros(4, dtype=np.float32)
    scores = {"binding": 0.5, "qed": 0.5}

    before = len(learner.get_weight_history()["binding"])
    learner.score(state, scores)
    after = len(learner.get_weight_history()["binding"])

    assert before == after


def test_reward_learner_compute_reward_still_records():
    from neorx.causalbiorl.causal.reward_learner import AdaptiveRewardLearner

    learner = AdaptiveRewardLearner(state_dim=4)
    state = np.zeros(4, dtype=np.float32)

    before = len(learner.get_weight_history()["binding"])
    learner.compute_reward(state, {"binding": 0.5})
    after = len(learner.get_weight_history()["binding"])

    assert after == before + 1
