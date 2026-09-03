"""GenMol generation quality, as reported in the GenMol manuscript section 4.3.

Offline: loads the shipped weights and generates locally. No HTTP, so no
cassette is recorded.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

N_GENERATE = 1000
TEMPERATURE = 0.8


@experiment(name="genmol-eval", help="GenMol generation quality metrics.")
def genmol_eval(record: RunRecord) -> None:
    from neorx.genmol import generate, load_pretrained

    model, tokenizer = load_pretrained()
    n_params = sum(p.numel() for p in model.parameters())

    started = time.perf_counter()
    # validate=False, deduplicate=False: generate() otherwise filters
    # internally, which would make validity and uniqueness structurally
    # always 1.0 -- reporting a better number than the truth instead of a
    # measurement. These metrics must be computed over raw sampler output,
    # matching what the paper actually reported (~0.97 validity).
    smiles = generate(
        model,
        tokenizer,
        n=N_GENERATE,
        temperature=TEMPERATURE,
        validate=False,
        deduplicate=False,
    )
    elapsed = time.perf_counter() - started

    mols = [(s, Chem.MolFromSmiles(s)) for s in smiles]
    valid = [(s, m) for s, m in mols if m is not None]
    unique = {s for s, _ in valid}

    mw = [Descriptors.MolWt(m) for _, m in valid]  # type: ignore[attr-defined]
    logp = [Crippen.MolLogP(m) for _, m in valid]  # type: ignore[attr-defined]
    qed = [QED.qed(m) for _, m in valid]

    record.append_row(
        {
            "n_params": n_params,
            "vocab_size": tokenizer.vocab_size,
            "temperature": TEMPERATURE,
            "n_requested": N_GENERATE,
            "n_generated": len(smiles),
            "n_valid": len(valid),
            "validity": len(valid) / len(smiles) if smiles else 0.0,
            "uniqueness": len(unique) / len(valid) if valid else 0.0,
            "novelty": _novelty(unique),
            "diversity": _diversity(_distinct_mols(valid)),
            "per_molecule_ms": 1000 * elapsed / len(smiles) if smiles else 0.0,
            "mw_mean": float(np.mean(mw)) if mw else 0.0,
            "mw_std": float(np.std(mw)) if mw else 0.0,
            "logp_mean": float(np.mean(logp)) if logp else 0.0,
            "logp_std": float(np.std(logp)) if logp else 0.0,
            "qed_mean": float(np.mean(qed)) if qed else 0.0,
            "qed_std": float(np.std(qed)) if qed else 0.0,
        }
    )


def _novelty(generated: set[str]) -> float:
    """Fraction of generated molecules absent from the training corpus."""
    from neorx.genmol.data.download import load_smiles

    training = set(load_smiles())
    if not generated:
        return 0.0
    return len(generated - training) / len(generated)


def _distinct_mols(valid: Sequence[tuple[str, object]]) -> list:
    """One RDKit Mol per distinct SMILES in ``valid``, first occurrence kept.

    ``valid`` is raw sampler output and may repeat the same molecule many
    times (see the ``deduplicate=False`` note above). Feeding duplicates to
    _diversity would score every duplicate pair as similarity 1.0 and
    deflate the reported diversity -- the metric would be measuring how
    repetitive the sample is, not how diverse the distinct molecules are.
    """
    seen: set[str] = set()
    out = []
    for s, m in valid:
        if s not in seen:
            seen.add(s)
            out.append(m)
    return out


def _diversity(mols: list) -> float:
    """1 - mean pairwise Tanimoto over Morgan fingerprints, full pairwise.

    The original script sampled a 50-neighbour sliding window over the first
    200 molecules; this computes the real statistic over a capped sample so
    the number means what its name says. Callers must pass distinct
    molecules -- see ``_distinct_mols`` -- since duplicate pairs score
    similarity 1.0 and would deflate the result.
    """
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    sample = mols[:500]
    if len(sample) < 2:
        return 0.0
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)  # type: ignore[attr-defined]
        for m in sample
    ]
    sims = [
        s
        for i, fp in enumerate(fps[:-1])
        for s in DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])
    ]
    return 1.0 - (sum(sims) / len(sims)) if sims else 0.0
