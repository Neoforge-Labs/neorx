"""
Disease graph construction and causal target identification.

Turns a disease name into a causal graph, the causal targets pursued
within it, and per-target R-GCN node embeddings -- everything
``DrugDiscoveryEnv.reset()`` needs before an episode can start. Synthetic
fallbacks keep this available without the full NeoRx graph-building
pipeline (e.g. in tests), and are used whenever the real pipeline raises.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

GRAPH_EMBEDDING_DIM = 128
LATENT_DIM = 128  # GenMol VAE latent dimension


class _TargetState:
    """Tracks the agent's progress on a single target."""

    def __init__(
        self,
        target_info: dict[str, Any],
        target_idx: int,
    ) -> None:
        self.target_info = target_info
        self.target_idx = target_idx
        self.gene_name: str = target_info.get("gene_name", f"target_{target_idx}")
        self.causal_confidence: float = target_info.get("causal_confidence", 0.5)
        self.pdb_ids: list[str] = target_info.get("pdb_ids", [])
        self.node_embedding: NDArray[np.floating] = np.zeros(
            GRAPH_EMBEDDING_DIM, dtype=np.float32,
        )

        # Best molecule found for this target
        self.best_smiles: str = ""
        self.best_score: float = -np.inf
        # Normalised per-objective scores in [0, 1], as the reward and the
        # observation consume them.
        self.best_objectives: dict[str, float] = {}
        # The same molecule's raw measurements, in their own units --
        # binding in kcal/mol, SA on the 1-10 scale, QED in [0, 1]. A None
        # entry means the quantity was not measured. Kept alongside
        # ``best_objectives`` because the normalisations are lossy: they
        # substitute a neutral prior for a missing measurement and clamp
        # out-of-range values, so a reporter that inverted them would
        # publish numbers no measurement supports.
        self.best_measurements: dict[str, float | None] = {}

        # Running z-vector for latent space navigation
        self.z_base: NDArray[np.floating] = np.random.randn(LATENT_DIM).astype(np.float32) * 0.5

        # Number of attempts on this target
        self.n_attempts: int = 0

    def summary_features(self) -> NDArray[np.floating]:
        """Return an 8-D summary vector for observation space."""
        return np.array([
            self.causal_confidence,
            min(self.n_attempts / 20.0, 1.0),  # normalised attempt count
            max(self.best_score, 0.0),           # best composite score
            self.best_objectives.get("binding", 0.0),
            self.best_objectives.get("qed", 0.0),
            self.best_objectives.get("sa", 0.0),
            self.best_objectives.get("novelty", 0.0),
            self.best_objectives.get("stability", 0.0),
        ], dtype=np.float32)


def get_disease_graph(
    disease: str,
    *,
    prebuilt_graph: nx.DiGraph | None,
    top_n_targets: int,
) -> nx.DiGraph:
    """Build or retrieve the disease knowledge graph."""
    if prebuilt_graph is not None:
        return prebuilt_graph

    try:
        from neorx.core.graph.graph_builder import build_disease_graph
        from neorx.causalbiorl.causal.graph_encoder import disease_graph_to_networkx

        dg = build_disease_graph(disease)
        return disease_graph_to_networkx(dg)
    except Exception as e:
        logger.warning("Could not build disease graph: %s — using synthetic", e)
        return synthetic_graph(disease, top_n_targets)


def synthetic_graph(disease: str, top_n_targets: int) -> nx.DiGraph:
    """Create a small synthetic graph for testing/fallback."""
    G = nx.DiGraph()
    targets = [f"gene_{i}" for i in range(top_n_targets)]
    disease_node = f"disease_{disease}"
    G.add_node(disease_node, node_type="disease", score=1.0)

    for i, gene in enumerate(targets):
        G.add_node(gene, node_type="gene", score=0.3 + 0.1 * i)
        G.add_edge(gene, disease_node, edge_type="causes", weight=0.5 + 0.05 * i)
        # Add some inter-gene edges
        if i > 0:
            G.add_edge(targets[i - 1], gene, edge_type="regulates", weight=0.3)

    return G


def get_targets(
    disease: str,
    *,
    top_n_targets: int,
    prebuilt_targets: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Identify causal targets or use pre-built list."""
    if prebuilt_targets is not None:
        return prebuilt_targets

    try:
        from neorx.core.causal.identifier import identify_causal_targets
        from neorx.core.graph.graph_builder import build_disease_graph

        dg = build_disease_graph(disease)
        results = identify_causal_targets(dg, top_n=top_n_targets)
        return [
            {
                "gene_name": r.gene_name,
                "protein_id": r.protein_id,
                "protein_name": r.protein_name,
                "causal_confidence": r.causal_confidence,
                "pdb_ids": r.pdb_ids,
                "is_causal": r.is_causal_target,
                "target_type": r.target_type,
                "node_id": r.protein_id,
            }
            for r in results
            if r.is_causal_target
        ]
    except Exception as e:
        logger.warning("Could not identify targets: %s — using synthetic", e)
        return synthetic_targets(top_n_targets)


def synthetic_targets(top_n_targets: int) -> list[dict[str, Any]]:
    """Create synthetic target list for testing."""
    return [
        {
            "gene_name": f"Gene{i}",
            "protein_id": f"P{i:05d}",
            "protein_name": f"Protein_{i}",
            "causal_confidence": 0.4 + 0.1 * i,
            "pdb_ids": [],
            "is_causal": True,
            "target_type": "CAUSAL",
            "node_id": f"gene_{i}",
        }
        for i in range(top_n_targets)
    ]


def encode_disease_graph(
    graph: nx.DiGraph | None,
    *,
    get_encoder: Any,
    embedding_dim: int = GRAPH_EMBEDDING_DIM,
) -> tuple[NDArray[np.floating], NDArray[np.floating], list]:
    """R-GCN embed the graph, falling back to degree-based features.

    ``get_encoder`` is a zero-argument callable that lazily constructs
    and caches the encoder -- it holds learned weights and needs to
    persist across calls, so the caller owns that lifecycle and this
    only calls it. Construction and encoding share one try/except, so a
    construction failure falls back exactly like an encoding failure,
    without a partially-applied state in between.

    Returns ``(graph_embedding, node_embeddings, node_order)``. When the
    graph is empty, ``node_embeddings`` and ``node_order`` are empty --
    callers that assign per-target embeddings from these must treat that
    as "nothing to assign", the same as the encoder never having run.
    """
    if graph is None or len(graph) == 0:
        return (
            np.zeros(embedding_dim, dtype=np.float32),
            np.zeros((0, embedding_dim), dtype=np.float32),
            [],
        )

    try:
        encoder = get_encoder()
        import torch
        with torch.no_grad():
            graph_emb, node_embs, node_order = encoder.encode_disease_graph(graph)
            return graph_emb.cpu().numpy(), node_embs.cpu().numpy(), node_order
    except Exception as e:
        logger.debug("R-GCN encoding failed: %s — using degree-based features", e)

    return (
        fallback_graph_features(graph, embedding_dim=embedding_dim),
        np.zeros((len(graph), embedding_dim), dtype=np.float32),
        list(graph.nodes()),
    )


def fallback_graph_features(
    graph: nx.DiGraph | None,
    *,
    embedding_dim: int = GRAPH_EMBEDDING_DIM,
) -> NDArray[np.floating]:
    """Simple graph statistics as fallback embedding."""
    if graph is None or len(graph) == 0:
        return np.zeros(embedding_dim, dtype=np.float32)

    features = np.zeros(embedding_dim, dtype=np.float32)
    features[0] = len(graph.nodes()) / 100.0
    features[1] = len(graph.edges()) / 500.0
    features[2] = nx.density(graph)

    try:
        pr = nx.pagerank(graph, max_iter=50)
        top_pr = sorted(pr.values(), reverse=True)[:10]
        for i, v in enumerate(top_pr):
            features[3 + i] = v
    except Exception:
        pass

    return features


def assign_target_embeddings(
    targets: list[_TargetState],
    *,
    node_order: list,
    node_embeddings: NDArray[np.floating],
) -> None:
    """Assign R-GCN node embeddings to targets, in place."""
    if not node_order:
        return

    node_to_idx = {n: i for i, n in enumerate(node_order)}
    for target in targets:
        node_id = target.target_info.get("node_id", "")
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            target.node_embedding = node_embeddings[idx].copy()
