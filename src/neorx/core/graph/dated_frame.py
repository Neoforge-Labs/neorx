"""What an unpinned source may contribute to a dated build.

A dated build pins the two sources that feed the causal subgraph --
OpenTargets and OmniPath -- and leaves the associational sources
(Monarch, ChEMBL, STRING, KEGG, Reactome, UniProt, PDB) live. That is
deliberate: their *edges* are not causal-admissible, so identification
filters them out before it runs.

Their **nodes** are not filtered by anything, and a live gene node does
three things a filtered edge cannot.

1. It wins the score merge. ``_merge_nodes`` keeps the highest score, and
   Monarch gives every causal association a flat 0.85 -- above most
   genetic scores in a release. ChEMBL is worse: its score is 60% of
   today's ``max_phase``, so the clinical outcome a dated build exists to
   predict enters the graph as one of its own predictors.
2. It draws causal-admissible arrows out of the *pinned* OmniPath
   extract. OmniPath is not disease-scoped, so whichever symbols reach
   the reader decide which regulatory edges a dated graph contains, and
   ``activates``/``inhibits`` are unconditionally causal-admissible. A
   single live symbol can close a cycle and turn an identifiable target
   into ``cyclic_component``.
3. It displaces frame genes from the ``max_genes`` cap, which is ranked
   by score -- the score a live source just inflated.

So, on a dated build: **the gene node population is the frame.** Unpinned
sources may enrich nodes already in the frame -- pathways, structures,
protein metadata, associational edges among frame genes -- but may not
introduce gene, protein or pathogen-gene nodes, and may not overwrite a
pinned node's score or the metadata the snapshot reader wrote.

The frame is not recomputed here. On a dated build the pinned Open
Targets reader's own node set *is* the frame:
``neorx.snapshots.frame.candidate_frame`` and
``neorx.core.sources.snapshot_sources.open_targets_from_snapshot`` select
by the same rule (genetic datatypes, score > 0), so the symbols coming
out of step 1 of the build are the population.

Pathogen genes are in scope, and that is a deliberate widening of the
list above. A pathogen target is a ChEMBL node whose score is 60% of
today's ``max_phase`` and nothing else; no source in the dated stack pins
it, so no release can vouch for it. Admitting it would put today's
clinical phase into a 2018 graph as a candidate's score -- the leak this
module exists to close, in its purest form. A dated build therefore has
no pathogen targets, and says so in the log rather than quietly shipping
a shorter list.
"""

from __future__ import annotations

from neorx.core.graph.models import GraphEdge, GraphNode, NodeType

__all__ = [
    "GENE_LIKE_TYPES",
    "SNAPSHOT_RELEASE_KEY",
    "frame_symbols",
    "is_pinned",
    "restrict_to_frame",
]

# Node types an unpinned source may not introduce on a dated build. These
# are the types that become identification candidates; a pathway or a
# structure is enrichment hanging off a candidate, not a candidate.
GENE_LIKE_TYPES: tuple[NodeType, ...] = (
    NodeType.GENE,
    NodeType.PROTEIN,
    NodeType.PATHOGEN_GENE,
)

# What marks a node as read from a pinned release. Written by
# ``open_targets_from_snapshot``; read here rather than threaded through
# the builder as a flag, so the fact travels with the node it is about.
SNAPSHOT_RELEASE_KEY = "snapshot_release"


def is_pinned(node: GraphNode) -> bool:
    """Whether this node was read from a pinned snapshot release."""
    return bool(node.metadata.get(SNAPSHOT_RELEASE_KEY))


def frame_symbols(nodes: list[GraphNode]) -> frozenset[str]:
    """The gene population a dated build may evaluate.

    ``nodes`` is the pinned Open Targets reader's output. Symbols are
    upper-cased because that is the form ``_extract_gene_symbols`` and
    every gene-list-taking source work in.
    """
    return frozenset(
        node.name.upper()
        for node in nodes
        if node.node_type in GENE_LIKE_TYPES and node.name
    )


def restrict_to_frame(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    frame: frozenset[str],
) -> tuple[list[GraphNode], list[GraphEdge], list[str]]:
    """Keep only what an unpinned source may contribute to a dated build.

    Returns the surviving nodes, the surviving edges, and the names of the
    dropped nodes so the caller can report them. An edge touching a
    dropped node goes with it: an edge to a node that does not exist is
    worse than no edge, and the identification code would read it as a
    real arrow to a nameless target.
    """
    kept_nodes: list[GraphNode] = []
    dropped_ids: set[str] = set()
    dropped_names: list[str] = []

    for node in nodes:
        if node.node_type in GENE_LIKE_TYPES and node.name.upper() not in frame:
            dropped_ids.add(node.node_id)
            dropped_names.append(node.name)
            continue
        kept_nodes.append(node)

    if not dropped_ids:
        return nodes, edges, []

    kept_edges = [
        edge for edge in edges
        if edge.source_id not in dropped_ids and edge.target_id not in dropped_ids
    ]
    return kept_nodes, kept_edges, dropped_names
