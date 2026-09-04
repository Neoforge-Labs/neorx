"""
DrugDiscovery-v0 — Gymnasium environment for RL-driven drug discovery.

This is the **integration point** where CausalBioRL meets the real
NeoRx pipeline.  Instead of toy ODE environments, the RL agent
interacts with real drug-discovery modules:

    - **NeoRx** — disease graph, causal targets, SCM
    - **GenMol** — VAE-based molecule generation (latent space navigation)
    - **MolScreen** — drug-likeness scoring (QED, SA, filters)
    - **DockBot** — binding affinity (via surrogate for speed)
    - **MirrorFold** — structural stability assessment

Architecture
------------
State:
    Graph embedding (128D from R-GCN) + current best molecule features (32D)
    + per-target summary (n_targets × 8D) → flattened to fixed size.

Action (hierarchical):
    Level 1: Target selection (discrete: which target to pursue)
    Level 2: Molecule generation (continuous 128D delta-z in GenMol latent)

Reward:
    Adaptively-weighted multi-objective (AdaptiveRewardLearner):
    binding, QED, SA, novelty, causal confidence, stability.

Episode:
    One disease campaign.  Agent has a budget of N generate-screen
    cycles.  It decides which targets to invest in and how to explore
    the chemical space for each.

Termination:
    Budget exhausted OR agent outputs "stop" action (early confidence).

Design Decisions
-----------------
    - Disease graph is built once at ``reset()`` and cached.
    - DockBot uses a surrogate model during training; real docking only
      at episode end for recalibration (see ``screening.py``).
    - GenMol generates from ``z_base + delta_z``: the agent learns to
      navigate the latent space, not just sample randomly.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from neorx.causalbiorl.envs import episode
from neorx.causalbiorl.envs import reward as reward_scoring
from neorx.causalbiorl.envs import screening, target_setup
from neorx.causalbiorl.envs.generation import decode_latent, decode_latent_batch
from neorx.causalbiorl.envs.target_setup import GRAPH_EMBEDDING_DIM, LATENT_DIM, _TargetState
from neorx.genmol import load_pretrained


# ────────────────────────────────────────────────────────────────────────── #
#  Constants                                                                 #
# ────────────────────────────────────────────────────────────────────────── #

MOL_FEATURE_DIM = 32      # Molecular property features
TARGET_FEATURE_DIM = 8    # Per-target summary features
MAX_TARGETS = 10          # Maximum number of targets supported

# Total observation dim:
# graph_emb (128) + mol_features (32) + target_summaries (10*8) + meta (4)
OBS_DIM = GRAPH_EMBEDDING_DIM + MOL_FEATURE_DIM + (MAX_TARGETS * TARGET_FEATURE_DIM) + 4

# ────────────────────────────────────────────────────────────────────────── #
#  DrugDiscoveryEnv                                                          #
# ────────────────────────────────────────────────────────────────────────── #


class DrugDiscoveryEnv(gym.Env):
    """Gymnasium environment for RL-driven drug discovery campaigns.

    Parameters
    ----------
    disease : str
        Disease name (e.g. "Malaria", "HIV").
    max_steps : int
        Budget: maximum generate-screen cycles per episode.
    top_n_targets : int
        Number of causal targets to pursue (from NeoRx pipeline).
    latent_dim : int
        GenMol VAE latent space dimension.
    use_surrogate : bool
        Use surrogate docking model (fast) vs real DockBot (slow).
    recalibrate_interval : int
        Every N steps, run real docking to recalibrate surrogate.
    difficulty : str
        ``"easy"`` / ``"medium"`` / ``"hard"`` — affects noise & budget.
    prebuilt_graph : nx.DiGraph | None
        Pre-built disease graph.  If provided, skips graph building.
    prebuilt_targets : list[dict] | None
        Pre-identified targets.  If provided, skips identification.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    _DIFFICULTY = {
        "easy":   {"budget_mult": 2.0, "noise": 0.0},
        "medium": {"budget_mult": 1.0, "noise": 0.02},
        "hard":   {"budget_mult": 0.5, "noise": 0.05},
    }

    def __init__(
        self,
        disease: str = "Malaria",
        max_steps: int = 50,
        top_n_targets: int = 5,
        latent_dim: int = LATENT_DIM,
        use_surrogate: bool = True,
        recalibrate_interval: int = 10,
        difficulty: str = "medium",
        prebuilt_graph: nx.DiGraph | None = None,
        prebuilt_targets: list[dict[str, Any]] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        assert difficulty in self._DIFFICULTY

        self.disease = disease
        self.latent_dim = latent_dim
        self.use_surrogate = use_surrogate
        self.recalibrate_interval = recalibrate_interval
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.top_n_targets = min(top_n_targets, MAX_TARGETS)

        diff = self._DIFFICULTY[difficulty]
        self.max_steps = int(max_steps * diff["budget_mult"])
        self._noise_std: float = diff["noise"]

        self._prebuilt_graph = prebuilt_graph
        self._prebuilt_targets = prebuilt_targets

        # ── Observation Space ──────────────────────────────────
        # Fixed-size vector: graph_emb + mol_features + target_summaries + meta
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # ── Action Space (Hierarchical) ───────────────────────
        # [0] target_selection, [1] stop_signal, [2:2+latent_dim] delta_z
        action_dim = 2 + latent_dim
        self.action_space = spaces.Box(
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
        )

        # ── Internal State ─────────────────────────────────────
        self._graph: nx.DiGraph | None = None
        self._graph_embedding = np.zeros(GRAPH_EMBEDDING_DIM, dtype=np.float32)
        self._targets: list[_TargetState] = []
        self._n_targets: int = 0
        self._current_mol_features = np.zeros(MOL_FEATURE_DIM, dtype=np.float32)
        self._step_count: int = 0
        self._episode_best_score: float = -np.inf
        self._episode_rewards: list[float] = []

        # Lazy-loaded components
        self._graph_encoder: Any = None
        self._reward_learner: Any = None
        self._surrogate: Any = None
        self._genmol_model: Any = None
        self._genmol_tokenizer: Any = None

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """Reset for a new campaign: build the graph, identify targets,
        encode with R-GCN, and initialise per-target state."""
        super().reset(seed=seed)
        self._step_count = 0
        self._episode_best_score = -np.inf
        self._episode_rewards = []

        # ── Build disease graph ────────────────────────────────
        self._graph = target_setup.get_disease_graph(
            self.disease,
            prebuilt_graph=self._prebuilt_graph,
            top_n_targets=self.top_n_targets,
        )

        # ── Identify targets ──────────────────────────────────
        target_dicts = target_setup.get_targets(
            self.disease,
            top_n_targets=self.top_n_targets,
            prebuilt_targets=self._prebuilt_targets,
        )
        self._n_targets = min(len(target_dicts), MAX_TARGETS)

        # ── Encode graph ──────────────────────────────────────
        self._graph_embedding, self._node_embeddings, self._node_order = (
            target_setup.encode_disease_graph(
                self._graph,
                get_encoder=self._get_graph_encoder,
                embedding_dim=GRAPH_EMBEDDING_DIM,
            )
        )

        # ── Initialise target states ──────────────────────────
        self._targets = []
        for i, tdict in enumerate(target_dicts[: self._n_targets]):
            ts = _TargetState(tdict, i)
            # Assign node embedding if available
            ts.z_base = self.np_random.standard_normal(self.latent_dim).astype(np.float32) * 0.5
            self._targets.append(ts)

        # ── Assign node embeddings to targets ─────────────────
        target_setup.assign_target_embeddings(
            self._targets,
            node_order=self._node_order,
            node_embeddings=self._node_embeddings,
        )

        self._current_mol_features = np.zeros(MOL_FEATURE_DIM, dtype=np.float32)

        obs = self._build_observation()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Execute one generate-screen cycle.

        Action interpretation:
            action[0] — target selector (continuous → discretised)
            action[1] — stop signal (> 0.9 = early termination)
            action[2:] — delta-z for GenMol latent space navigation
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # ── Parse hierarchical action ─────────────────────────
        target_idx = self._select_target(float(action[0]))
        stop_signal = float(action[1]) > 0.9
        delta_z = action[2: 2 + self.latent_dim].astype(np.float32)

        target = self._targets[target_idx]
        target.n_attempts += 1

        # ── Generate molecule ─────────────────────────────────
        z = target.z_base + delta_z * 0.3  # scale delta for stability
        smiles = self._decode_latent(z)

        # ── Screen molecule ───────────────────────────────────
        obj_scores, measurements = self._screen_molecule(smiles, target)

        # ── Compute reward ────────────────────────────────────
        state_vec = self._build_observation()
        reward = self._compute_reward(state_vec, obj_scores)

        # ── Update target state ───────────────────────────────
        composite = sum(obj_scores.values()) / max(len(obj_scores), 1)
        if composite > target.best_score:
            target.best_score = composite
            target.best_smiles = smiles
            target.best_objectives = dict(obj_scores)
            target.best_measurements = dict(measurements)

        if composite > self._episode_best_score:
            self._episode_best_score = composite

        # ── Update latent base (drift toward good regions) ────
        if composite > 0.5:
            target.z_base = 0.8 * target.z_base + 0.2 * z

        # ── Update mol features for next observation ──────────
        self._current_mol_features = screening.encode_molecule(
            smiles, obj_scores, feature_dim=MOL_FEATURE_DIM,
        )

        self._step_count += 1
        self._episode_rewards.append(reward)

        # ── Periodic surrogate recalibration ──────────────────
        if (
            self.use_surrogate
            and self._step_count % self.recalibrate_interval == 0
            and self._step_count > 0
        ):
            self._recalibrate_surrogate()

        # ── Termination ───────────────────────────────────────
        terminated = stop_signal and self._step_count >= 5  # min 5 steps
        truncated = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = self._get_info()
        info["smiles"] = smiles
        info["target"] = target.gene_name
        info["objectives"] = obj_scores

        return obs, reward, terminated, truncated, info

    def evaluate_actions(
        self,
        state: NDArray[np.floating],
        actions: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Score candidate actions against real chemistry, mutating nothing.

        A planner needs to know what an action is actually worth before
        committing to it. ``step`` cannot answer that: it advances the
        step counter, updates per-target bests, drifts ``z_base``, and
        records into the reward learner's history. This decodes and
        screens the same molecules ``step`` would and returns their
        rewards, leaving the environment exactly as it found it.

        ``_screen_molecule`` draws from ``self.np_random`` to add
        per-difficulty noise to the objective scores. That draw advances
        the environment's shared random stream -- a real mutation, even
        though it touches no counter or cache. Left alone, evaluating a
        candidate here would desynchronise the noise a later ``step()``
        draws for the same action from what it would have drawn
        otherwise. The RNG state is saved before screening and restored
        after, so this method's random consumption is invisible to the
        rest of the environment.

        ``step`` also bumps the chosen target's ``n_attempts`` *before*
        building the observation it scores the reward against, so that
        counter is baked into the state the reward learner sees. To
        agree with ``step`` this replicates the same bump on a copy of
        ``state`` while scoring each candidate, and never writes it back
        to the target.

        Parameters
        ----------
        state
            The observation the rewards are computed against.
        actions
            Array of shape ``(n, action_dim)``.

        Returns
        -------
        ndarray of shape ``(n,)``
        """
        if actions.ndim != 2:
            raise ValueError(
                f"evaluate_actions expects a 2-D action array, got shape {actions.shape}"
            )

        if self._genmol_model is None:
            self._init_genmol()

        clipped = np.clip(actions, self.action_space.low, self.action_space.high)

        target_indices = [self._select_target(float(a[0])) for a in clipped]
        latents = np.stack([
            self._targets[idx].z_base
            + clipped[i, 2: 2 + self.latent_dim].astype(np.float32) * 0.3
            for i, idx in enumerate(target_indices)
        ])

        smiles = decode_latent_batch(
            self._genmol_model, self._genmol_tokenizer, latents,
        )

        target_block_offset = GRAPH_EMBEDDING_DIM + MOL_FEATURE_DIM

        rng_state = self.np_random.bit_generator.state
        rewards = np.empty(len(clipped), dtype=np.float64)
        for i, (mol, idx) in enumerate(zip(smiles, target_indices)):
            obj_scores, _ = self._screen_molecule(mol, self._targets[idx])

            target = self._targets[idx]
            state_for_reward = state.copy()
            block_start = target_block_offset + idx * TARGET_FEATURE_DIM
            target.n_attempts += 1
            state_for_reward[block_start: block_start + TARGET_FEATURE_DIM] = (
                target.summary_features()
            )
            target.n_attempts -= 1

            rewards[i] = self._score_reward(state_for_reward, obj_scores)
        self.np_random.bit_generator.state = rng_state

        return rewards

    # ------------------------------------------------------------------ #
    #  Causal Interface (shared with toy envs)                             #
    # ------------------------------------------------------------------ #

    def get_causal_graph(self) -> nx.DiGraph:
        """Return the disease causal graph (or empty graph if not built)."""
        if self._graph is not None:
            return self._graph
        return nx.DiGraph()

    def _get_graph_encoder(self) -> Any:
        """Lazily construct and cache the R-GCN encoder (persists across resets).

        Raises straight through -- ``target_setup.encode_disease_graph``'s
        try/except is what decides whether construction failure falls back.
        """
        if self._graph_encoder is None:
            from neorx.causalbiorl.causal.graph_encoder import DiseaseGraphEncoder
            self._graph_encoder = DiseaseGraphEncoder(embedding_dim=GRAPH_EMBEDDING_DIM)
            self._graph_encoder.eval()
        return self._graph_encoder

    # ------------------------------------------------------------------ #
    #  Internal: Molecule Generation & Screening                           #
    # ------------------------------------------------------------------ #

    def _decode_latent(self, z: NDArray[np.floating]) -> str:
        """Decode a latent vector to SMILES via the trained VAE (greedy).

        See ``neorx.causalbiorl.envs.generation.decode_latent`` for why
        greedy decoding is the right default here.
        """
        if self._genmol_model is None:
            self._init_genmol()

        return decode_latent(self._genmol_model, self._genmol_tokenizer, z)

    def _init_genmol(self) -> None:
        """Load the packaged GenMol model.

        Errors propagate. A randomly initialised VAE emits plausible SMILES,
        so a silent failure here is undetectable downstream.

        The environment's latent action space *is* the model's latent
        space -- there is no padding/truncation shim reconciling the two.
        If they disagree, that is a real misconfiguration and must fail
        loudly here rather than surface later as a shape-mismatch error
        deep inside ``decode()``.
        """
        model, tokenizer = load_pretrained()

        if self.latent_dim != model.latent_dim:
            raise ValueError(
                f"latent_dim={self.latent_dim} does not match the shipped "
                f"GenMol model's latent dimension "
                f"({model.latent_dim}). The environment's "
                f"latent action space must match the decoder it drives -- "
                f"construct the environment with "
                f"latent_dim={model.latent_dim}, or omit the "
                f"argument to use the default."
            )

        # Only commit to instance state once validation has passed, so a
        # failed check leaves self._genmol_model as None and a retry (e.g.
        # the next env.step()) re-validates instead of skipping the guard.
        self._genmol_model, self._genmol_tokenizer = model, tokenizer

    def _screen_molecule(
        self,
        smiles: str,
        target: _TargetState,
    ) -> tuple[dict[str, float], dict[str, float | None]]:
        """Screen a molecule; return its normalised scores and raw measurements.

        Thin orchestration over ``screening.py``'s pure scoring functions:
        owns the surrogate's lazy construction and the per-difficulty noise
        draw (which consumes ``self.np_random`` -- a real mutation).

        The first element is the per-objective scores in [0, 1] that the
        reward and the observation consume. The second is the same
        molecule's raw measurements in their own units -- binding in
        kcal/mol, SA on the 1-10 scale, QED in [0, 1] -- with None for
        anything that could not be measured. The measurements carry no
        noise and no neutral-prior substitution, so a reporter can publish
        them as the measurements they are.
        """
        if self.use_surrogate and self._surrogate is None:
            try:
                from neorx.causalbiorl.causal.surrogate_docker import SurrogateDockingModel
                self._surrogate = SurrogateDockingModel()
            except Exception:
                pass  # screening.measure_binding treats a missing surrogate as a failed lookup

        from neorx.causalbiorl.causal.reward_learner import (
            normalise_binding,
            normalise_sa,
        )

        measurements: dict[str, float | None] = {
            "binding": screening.measure_binding(
                smiles, target,
                use_surrogate=self.use_surrogate,
                surrogate=self._surrogate,
            ),
            "qed": screening.measure_qed(smiles),
            "sa": screening.measure_synthetic_accessibility(smiles),
        }

        qed = measurements["qed"]
        scores: dict[str, float] = {
            "binding": normalise_binding(measurements["binding"]),
            "qed": 0.5 if qed is None else qed,
            "sa": normalise_sa(measurements["sa"]),
            "novelty": screening.score_novelty(smiles),
            "causal": target.causal_confidence,
            "stability": screening.score_stability(smiles, target),
        }

        # Add noise for difficulty
        if self._noise_std > 0:
            for key in scores:
                scores[key] = float(np.clip(
                    scores[key] + self.np_random.normal(0, self._noise_std),
                    0.0, 1.0,
                ))

        return scores, measurements

    def _recalibrate_surrogate(self) -> None:
        """Run real docking on recent molecules to recalibrate the surrogate."""
        screening.recalibrate_surrogate(self._targets, self._surrogate)

    # ------------------------------------------------------------------ #
    #  Internal: Reward                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_reward_learner(self) -> None:
        """Lazily construct the reward learner, once, and cache it.

        A None learner (construction failed) is treated by ``reward.py``
        exactly like one whose call raised -- same fallback either way.
        """
        if self._reward_learner is None:
            try:
                from neorx.causalbiorl.causal.reward_learner import AdaptiveRewardLearner
                self._reward_learner = AdaptiveRewardLearner(state_dim=OBS_DIM)
            except Exception:
                pass

    def _compute_reward(
        self,
        state: NDArray[np.floating],
        obj_scores: dict[str, float],
    ) -> float:
        """Compute adaptively-weighted reward, recording it into the learner."""
        self._ensure_reward_learner()
        return reward_scoring.compute_reward(state, obj_scores, learner=self._reward_learner)

    def _score_reward(
        self,
        state: NDArray[np.floating],
        obj_scores: dict[str, float],
    ) -> float:
        """The reward for these objective scores, recording nothing."""
        self._ensure_reward_learner()
        return reward_scoring.score_reward(state, obj_scores, learner=self._reward_learner)

    # ------------------------------------------------------------------ #
    #  Internal: Observation, Target Selection & Reporting                 #
    # ------------------------------------------------------------------ #

    def _build_observation(self) -> NDArray[np.floating]:
        """Build the full observation vector."""
        return episode.build_observation(
            graph_embedding=self._graph_embedding,
            mol_features=self._current_mol_features,
            targets=self._targets,
            step_count=self._step_count,
            n_targets=self._n_targets,
            max_steps=self.max_steps,
            episode_best_score=self._episode_best_score,
            use_surrogate=self.use_surrogate,
            obs_dim=OBS_DIM,
            graph_embedding_dim=GRAPH_EMBEDDING_DIM,
            mol_feature_dim=MOL_FEATURE_DIM,
            target_feature_dim=TARGET_FEATURE_DIM,
            max_targets=MAX_TARGETS,
        )

    def _select_target(self, action_value: float) -> int:
        """Convert continuous action to discrete target index."""
        if self._n_targets == 0:
            return 0

        # Map [-1, 1] → [0, n_targets-1]
        normalised = (action_value + 1.0) / 2.0  # → [0, 1]
        idx = int(normalised * self._n_targets)
        return max(0, min(idx, self._n_targets - 1))

    def _get_info(self) -> dict[str, Any]:
        """Return episode info dict."""
        return episode.build_info(
            targets=self._targets,
            step_count=self._step_count,
            n_targets=self._n_targets,
            max_steps=self.max_steps,
            episode_best_score=self._episode_best_score,
        )

    def render(self) -> None:
        """Print current episode status."""
        if self.render_mode == "human":
            episode.render_status(self.disease, self.max_steps, self._get_info())

    def close(self) -> None:
        """Clean up resources."""
        self._graph = None
        self._graph_encoder = None
        self._surrogate = None
        self._genmol_model = None
        self._genmol_tokenizer = None
