"""
The reinforcement-learning candidate-generation stage.

Builds a DrugDiscoveryEnv over the identified targets, trains the causal
agent against it, and collects the best molecule found per target. Split
out of the pipeline module because it is the only stage that owns an
environment and an agent, and because the ordering bug that left the
planner unwired lived in this wiring.
"""

from __future__ import annotations

import logging
from typing import Any

from neorx.core.graph.models import ScoredCandidate
from neorx.core.scoring.scorer import score_candidate

logger = logging.getLogger(__name__)


def generate_candidates_with_rl(
    disease: str,
    nx_graph: Any,
    causal_only: list,
    *,
    n_episodes: int,
    max_steps_per_episode: int,
    latent_dim: int,
    seed: int,
) -> list[ScoredCandidate]:
    """Train the causal agent and return its best scored candidates."""
    from neorx.causalbiorl.envs.drug_discovery import DrugDiscoveryEnv
    from neorx.causalbiorl.agents.causal_agent import CausalAgent

    target_dicts = [
        {
            "gene_name": t.gene_name,
            "protein_id": t.protein_id,
            "protein_name": t.protein_name,
            "pdb_ids": t.pdb_ids if t.pdb_ids else [],
            "causal_confidence": t.causal_confidence,
        }
        for t in causal_only
    ]

    env = DrugDiscoveryEnv(
        disease=disease,
        prebuilt_graph=nx_graph,
        prebuilt_targets=target_dicts,
        max_steps=max_steps_per_episode,
        latent_dim=latent_dim,
    )

    # Build agent config
    from neorx.causalbiorl.models import AgentConfig
    agent_cfg = AgentConfig(
        agent_type="causal",
        n_episodes=n_episodes,
        seed=seed,
    )

    agent = CausalAgent(env, agent_cfg)
    agent.init_hierarchical_planner(
        n_targets=len(causal_only),
        latent_dim=latent_dim,
    )

    # Train (the training loop collects molecules internally)
    agent.train()

    # Extract best molecules from env's internal tracking. Each target
    # state records the raw measurements _screen_molecule took for its
    # best molecule -- binding in kcal/mol, SA on the 1-10 scale, QED in
    # [0, 1], exactly the units score_candidate documents. Pass those
    # through rather than the placeholders that used to stand in for
    # them, and rather than best_objectives, whose values are already
    # normalised to [0, 1]: feeding those back through score_candidate's
    # own normalisation would re-normalise an already-normalised number.
    # A measurement that was never taken is None, which score_candidate
    # reads as "unmeasured" -- never a number asserting a result.
    all_candidates: list[ScoredCandidate] = []
    for ts in env._targets:
        if ts.best_smiles is not None and ts.best_score > 0.0:
            candidate = score_candidate(
                smiles=ts.best_smiles,
                target_protein_id=causal_only[ts.target_idx].protein_id
                if ts.target_idx < len(causal_only) else "",
                target_protein_name=causal_only[ts.target_idx].protein_name
                if ts.target_idx < len(causal_only) else "",
                causal_confidence=causal_only[ts.target_idx].causal_confidence
                if ts.target_idx < len(causal_only) else 0.5,
                binding_affinity=ts.best_measurements.get("binding"),
                qed_score=ts.best_measurements.get("qed"),
                sa_score=ts.best_measurements.get("sa"),
            )
            all_candidates.append(candidate)

    logger.info("RL agent found %d candidate molecules.", len(all_candidates))
    return all_candidates
