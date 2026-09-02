"""The environment generates via the trained VAE, not a hardcoded list."""

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from neorx.causalbiorl.envs.drug_discovery import DrugDiscoveryEnv

from .scaffold_fixture import DELETED_FALLBACK_SCAFFOLDS


def _env():
    g = nx.DiGraph()
    for gene, score in [("gene_EGFR", 0.9), ("gene_TP53", 0.8), ("gene_ALK", 0.7)]:
        g.add_node(gene, type="gene", score=score, tissue_relevant=True)
    g.add_node("disease_X", type="disease", score=1.0, tissue_relevant=True)
    for gene in ("gene_EGFR", "gene_TP53", "gene_ALK"):
        g.add_edge(gene, "disease_X", edge_type="associated_with", weight=0.8)
    targets = [
        {"gene": n, "causal_confidence": c, "classification": "CAUSAL"}
        for n, c in [("EGFR", 0.85), ("TP53", 0.7), ("ALK", 0.65)]
    ]
    return DrugDiscoveryEnv(
        disease="TestDisease",
        prebuilt_graph=g,
        prebuilt_targets=targets,
        max_steps=20,
    )


def _episode_molecules(env, n_steps=20, seed=0):
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_steps):
        action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
        _, _, term, trunc, info = env.step(action)
        smiles = info.get("smiles") or info.get("molecule")
        if smiles:
            out.append(smiles)
        if term or trunc:
            break
    return out


def test_molecules_do_not_come_from_the_deleted_scaffold_list():
    """Provenance, not diversity.

    The model has documented partial posterior collapse, so outputs may
    cluster. Asserting diversity here would fail for a reason unrelated to
    whether the wiring works.
    """
    mols = _episode_molecules(_env())
    assert mols, "episode produced no molecules"
    leaked = set(mols) & DELETED_FALLBACK_SCAFFOLDS
    assert not leaked, f"environment emitted deleted fallback scaffolds: {leaked}"


def test_molecules_are_chemically_valid():
    mols = _episode_molecules(_env())
    assert all(Chem.MolFromSmiles(s) is not None for s in mols)


def test_fallback_generate_no_longer_exists():
    assert not hasattr(DrugDiscoveryEnv, "_fallback_generate"), (
        "a reachable fallback is the bug; an unreachable one is dead code"
    )


def test_missing_assets_surface_rather_than_degrade(monkeypatch):
    from neorx.genmol import GenMolAssetError
    import neorx.causalbiorl.envs.drug_discovery as dd

    def boom(*a, **k):
        raise GenMolAssetError("no trained checkpoint")

    monkeypatch.setattr(dd, "load_pretrained", boom)
    env = _env()
    with pytest.raises(GenMolAssetError):
        _episode_molecules(env, n_steps=1)
