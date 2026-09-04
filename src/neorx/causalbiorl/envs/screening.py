"""
Scoring a molecule against a target.

Binding affinity via the docking surrogate, drug-likeness via QED,
synthetic accessibility, and the feature encoding the observation uses.
These are pure functions of a SMILES string and a target: nothing here
advances the environment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neorx.causalbiorl.envs.target_setup import _TargetState

# The order the reward learner assigns per-objective weights in. Kept in
# sync with ``neorx.causalbiorl.causal.reward_learner.OBJECTIVE_NAMES``.
_OBJECTIVE_ORDER = ("binding", "qed", "sa", "novelty", "causal", "stability")


def measure_binding(
    smiles: str,
    target: _TargetState,
    *,
    use_surrogate: bool,
    surrogate: Any,
) -> float | None:
    """Binding affinity in kcal/mol (negative = better), or None.

    None means *not measured* -- no surrogate, no fingerprint, no PDB to
    dock against, or a failed lookup. Callers that need a [0, 1] score
    hand that None to ``normalise_binding``, which supplies the neutral
    prior; callers that report the affinity itself must report the
    absence, never a number standing in for one.

    ``surrogate`` must already be constructed when ``use_surrogate`` is
    True -- lazily creating and caching it is the environment's job, since
    the model persists (and accumulates recalibration observations) across
    calls. This only scores against whatever surrogate it is handed.
    """
    if use_surrogate:
        return _surrogate_binding(smiles, target, surrogate)
    return _real_binding(smiles, target)


def _surrogate_binding(
    smiles: str, target: _TargetState, surrogate: Any,
) -> float | None:
    """Fast binding estimate via surrogate model, in kcal/mol.

    ``surrogate`` is None when the caller's lazy construction failed --
    unmeasured, the same as a fingerprint that could not be built, and
    without attempting a lookup that could not be scored anyway.
    """
    if surrogate is None:
        return None

    try:
        from neorx.causalbiorl.causal.surrogate_docker import smiles_to_fingerprint

        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            return None

        return float(surrogate.predict(fp, target.node_embedding))

    except Exception:
        return None


def _real_binding(smiles: str, target: _TargetState) -> float | None:
    """Real docking via DockBot (slow but accurate), in kcal/mol."""
    try:
        from neorx.core.pipeline import _run_docking

        if not target.pdb_ids:
            return None

        affinity = _run_docking(smiles, target.pdb_ids[0])
        return None if affinity is None else float(affinity)

    except Exception:
        return None


def measure_qed(smiles: str) -> float | None:
    """QED in [0, 1] (higher = more drug-like), or None if not computable."""
    try:
        from neorx.molscreen.accessibility import qed_score
        q = qed_score(smiles)
        return None if q is None else float(q)
    except Exception:
        try:
            from rdkit import Chem
            from rdkit.Chem import QED
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return float(QED.qed(mol))
        except Exception:
            pass
        return None


def measure_synthetic_accessibility(smiles: str) -> float | None:
    """Raw SA score on the 1 (easy) to 10 (hard) scale, or None.

    None means *not measured*; ``normalise_sa`` turns that into the
    neutral prior, and reporters must report the absence rather than a
    stand-in number.
    """
    try:
        from neorx.molscreen.accessibility import sa_score
        sa = sa_score(smiles)
        return None if sa is None else float(sa)
    except Exception:
        return None


def screen_molecule(
    smiles: str,
    target: _TargetState,
    *,
    use_surrogate: bool,
    surrogate: Any,
) -> tuple[dict[str, float], dict[str, float | None]]:
    """Screen a molecule; return its normalised scores and raw measurements.

    The first element is the per-objective scores in [0, 1] the reward and
    the observation consume. The second is the same molecule's raw
    measurements in their own units -- binding in kcal/mol, SA on the
    1-10 scale, QED in [0, 1] -- with None for anything not measured.

    Both are returned because the normalisations are lossy: they
    substitute a neutral prior for a missing measurement and clamp
    out-of-range values, so a reporter that inverted them would publish
    numbers no measurement supports.
    """
    from neorx.causalbiorl.causal.reward_learner import (
        normalise_binding,
        normalise_sa,
    )

    measurements: dict[str, float | None] = {
        "binding": measure_binding(
            smiles, target, use_surrogate=use_surrogate, surrogate=surrogate,
        ),
        "qed": measure_qed(smiles),
        "sa": measure_synthetic_accessibility(smiles),
    }

    qed = measurements["qed"]
    scores: dict[str, float] = {
        "binding": normalise_binding(measurements["binding"]),
        "qed": 0.5 if qed is None else qed,
        "sa": normalise_sa(measurements["sa"]),
        "novelty": score_novelty(smiles),
        "causal": target.causal_confidence,
        "stability": score_stability(smiles, target),
    }

    return scores, measurements


def score_novelty(smiles: str) -> float:
    """Get structural novelty vs known drugs."""
    try:
        from neorx.molscreen.similarity import find_similar_drugs
        similar = find_similar_drugs(smiles, top_k=1, threshold=0.3)
        if similar:
            max_sim = max(s[1] for s in similar) if similar else 0.0
            return 1.0 - max_sim
        return 0.9
    except Exception:
        return 0.5


def score_stability(smiles: str, target: _TargetState) -> float:
    """Get structural stability via MirrorFold therapeutic assessment."""
    try:
        from neorx.mirrorfold import therapeutic_assessment

        # MirrorFold operates on protein sequences, not SMILES.
        # For drug discovery, we assess the target protein's
        # stability as a drug target (is it structurally tractable?)
        if target.pdb_ids:
            from neorx.mirrorfold import fetch_pdb_structure, pdb_to_sequence
            pdb_text = fetch_pdb_structure(target.pdb_ids[0])
            if pdb_text:
                seq = pdb_to_sequence(pdb_text)
                if seq and len(seq) > 10:
                    result = therapeutic_assessment(seq[:500])  # cap length
                    return float(result.get("therapeutic_score", 0.5))
    except Exception:
        pass
    return 0.5  # neutral prior


def recalibrate_surrogate(targets: list, surrogate: Any) -> None:
    """Run real docking on recent best molecules to recalibrate ``surrogate``.

    Mutates ``surrogate`` in place -- new observations, and a fit pass
    once enough have accumulated. A no-op if ``surrogate`` is None (the
    environment never lazily constructed one, e.g. because it never ran
    with ``use_surrogate=True``).
    """
    if surrogate is None:
        return

    try:
        from neorx.causalbiorl.causal.surrogate_docker import smiles_to_fingerprint

        recal_count = 0
        for target in targets:
            if target.best_smiles and target.pdb_ids:
                real_aff = measure_binding(
                    target.best_smiles, target,
                    use_surrogate=False, surrogate=None,
                )
                if real_aff is None:
                    continue  # docking failed -- nothing measured to fit to
                fp = smiles_to_fingerprint(target.best_smiles)
                if fp is not None:
                    surrogate.add_observation(
                        fp, target.node_embedding, real_aff,
                    )
                    recal_count += 1

        if recal_count > 0 and surrogate.buffer_size >= 10:
            surrogate.fit(epochs=50)
            logger.debug(
                "Surrogate recalibrated with %d new observations.", recal_count,
            )

    except Exception as e:
        logger.debug("Surrogate recalibration failed: %s", e)


def encode_molecule(
    smiles: str,
    obj_scores: dict[str, float],
    *,
    feature_dim: int,
) -> NDArray[np.floating]:
    """Encode a molecule into a ``feature_dim``-D feature vector."""
    features = np.zeros(feature_dim, dtype=np.float32)

    # Objective scores (6 values)
    for i, name in enumerate(_OBJECTIVE_ORDER):
        if i < feature_dim:
            features[i] = obj_scores.get(name, 0.0)

    # Molecular fingerprint summary (top 26 bits of Morgan fp)
    try:
        from neorx.causalbiorl.causal.surrogate_docker import smiles_to_fingerprint
        fp = smiles_to_fingerprint(smiles, n_bits=256)
        if fp is not None:
            # Compress 256-bit fp to 26 values via chunked sums
            chunk_size = 256 // 26
            for i in range(26):
                start = i * chunk_size
                end = min(start + chunk_size, 256)
                features[6 + i] = fp[start:end].sum() / chunk_size
    except Exception:
        pass

    return features
