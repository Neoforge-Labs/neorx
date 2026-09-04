"""
Which edges may carry a causal claim.

A disease graph assembled from public sources mixes two kinds of edge.
Some assert a mechanism with a direction: OmniPath's signed regulatory
interactions, where a curator has said that A acts on B. Others assert
only that two things occur together: a literature co-mention, membership
of a shared pathway, an undirected protein interaction whose arrow exists
because one protein landed in the first column of an API response.

Do-calculus over the second kind assumes the very thing a causal method
is supposed to establish. So this module partitions the edge types and
identification runs only on the causal-admissible subgraph. Associational
edges keep contributing to evidence scoring; they stop being arrows.

Gene -> disease is admitted only on genetic evidence. A germline variant
is randomised at conception, so a genetic association carries a
natural-experiment warrant that a co-mention does not -- the Mendelian
randomisation argument (Davey Smith & Ebrahim, 2003, Int J Epidemiol
32:1-22). That warrant is what earns the arrow.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

# Directed, mechanistic, with an asserted direction.
CAUSAL_EDGE_TYPES: frozenset[str] = frozenset({
    "activates",
    "inhibits",
    "phosphorylates",
    "regulates",
    "upregulates",
    "downregulates",
    "causes",
})

# Gene -> disease edge types, admissible only under genetic evidence.
_ASSOCIATION_EDGE_TYPES: frozenset[str] = frozenset({"associated_with"})

GENETIC_EVIDENCE_CLASSES: frozenset[str] = frozenset({
    "genetic_association",
    "somatic_mutation",
})

__all__ = [
    "CAUSAL_EDGE_TYPES",
    "GENETIC_EVIDENCE_CLASSES",
    "AcyclicCore",
    "acyclic_core",
    "causal_subgraph",
    "cyclic_components",
    "is_causal_admissible",
]


def is_causal_admissible(attrs: dict) -> bool:
    """Whether an edge with these attributes may carry a causal claim.

    Parameters
    ----------
    attrs
        Edge attribute mapping, as stored on the NetworkX graph. Reads
        ``edge_type`` and ``evidence_class``.
    """
    edge_type = attrs.get("edge_type", "")
    if edge_type in CAUSAL_EDGE_TYPES:
        return True
    if edge_type in _ASSOCIATION_EDGE_TYPES:
        return attrs.get("evidence_class", "") in GENETIC_EVIDENCE_CLASSES
    return False


def causal_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """The subgraph of causal-admissible edges, attributes preserved.

    Nodes left with no admissible edge are dropped: a node that
    participates in no causal relation cannot be a treatment, an
    outcome, or a confounder.
    """
    admissible = [
        (u, v) for u, v, attrs in G.edges(data=True)
        if is_causal_admissible(attrs)
    ]
    return G.edge_subgraph(admissible).copy()


def cyclic_components(G: nx.DiGraph) -> list[frozenset[str]]:
    """Strongly connected components of more than one node."""
    return [
        frozenset(component)
        for component in nx.strongly_connected_components(G)
        if len(component) > 1
    ]


@dataclass(frozen=True)
class AcyclicCore:
    """A DAG carved out of a possibly-cyclic graph, and what was removed.

    ``excluded`` names every node that sat in a feedback loop. Those
    nodes are not silently repaired: d-separation is undefined on them,
    and identification reports ``cyclic_component`` rather than guessing
    an acyclic orientation.
    """

    dag: nx.DiGraph
    excluded: frozenset[str]


def acyclic_core(G: nx.DiGraph) -> AcyclicCore:
    """Remove every node in a non-trivial strongly connected component.

    Biological regulatory networks are full of feedback, and
    ``nx.is_d_separator`` requires a DAG of the whole graph -- not merely
    of the neighbourhood under test. So cyclic nodes come out entirely,
    and the caller is told which.
    """
    cyclic_nodes: set[str] = set()
    for component in cyclic_components(G):
        cyclic_nodes |= set(component)

    dag = G.copy()
    dag.remove_nodes_from(cyclic_nodes)
    return AcyclicCore(dag=dag, excluded=frozenset(cyclic_nodes))
