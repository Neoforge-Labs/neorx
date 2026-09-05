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
introduce gene or protein nodes, and may not overwrite a pinned node's
score or the metadata the snapshot reader wrote.

The frame is not recomputed here. On a dated build the pinned Open
Targets reader's own node set *is* the frame:
``neorx.snapshots.frame.candidate_frame`` and
``neorx.core.sources.snapshot_sources.open_targets_from_snapshot`` select
by the same rule (genetic datatypes, score > 0), so the symbols coming
out of step 1 of the build are the population.

Identity, and why there is one function for it
----------------------------------------------
A node's identity is its ``node_id``. ``_merge_nodes`` merges on it, the
graph is keyed on it, and every edge names two of them. So the frame is a
population of *identities*, and admitting a node means deciding whether
its identity is one of them.

Enforcing that on the node's bare ``name`` instead is what produced four
separate defects, because the two normalise differently:

* ``C9orf72`` is a real HGNC symbol carrying lowercase, and the snapshot
  reader keeps the release's spelling while ``chembl.py`` upper-cases
  every symbol it resolves. A name check on ``name.upper()`` passes; the
  id check ``gene:C9orf72`` vs ``gene:C9ORF72`` does not, so the gene
  ends up as two nodes and the second carries today's clinical phase.
* A pathogen target's id is namespaced (``pathogen:<organism>:<SYMBOL>``)
  but its name is the bare symbol, so a name check admits it whenever the
  human gene of the same name is in the frame -- and never counts it as
  dropped.
* The gene list handed to the pinned OmniPath reader was upper-cased
  while the extract is filtered with an exact match, so every mixed-case
  symbol lost its whole regulatory layer, silently. The 2018 dump holds
  321 ``C#orf#`` symbols and 872 mixed-case human symbols.

``match_key`` is the one normalisation all of that uses:
``restrict_to_frame`` below, ``omnipath_from_snapshot``'s extract filter,
``_extract_gene_symbols``' deduplication, and ``_get_candidate_nodes``'
frame comparison in the identifier. It is ``.lower()`` rather than
``.casefold()`` deliberately, because the extract side of the comparison
is polars' ``str.to_lowercase`` and the two sides must be the same
operation, not merely similar ones.

Case-insensitive matching is only safe because the OmniPath extract is
filtered to human at extraction time (see
``neorx.snapshots.omnipath.read_archive_tsv``): 49% of the 2018 archive
is mouse and rat, most of it Title-case symbols that an exact-case match
excluded by accident. The organism filter is the precondition for this
rule, not an independent improvement.

Pathogen genes are excluded outright, and by node type. A pathogen target
is a ChEMBL node whose score is 60% of today's ``max_phase`` and nothing
else; no source in the dated stack pins it, so no release can vouch for
it. Admitting it would put today's clinical phase into a 2018 graph as a
candidate's score -- the leak this module exists to close, in its purest
form. The exclusion is by type because that is the only property a
pathogen node has that a colliding human symbol cannot forge, and it is
counted and logged like every other drop rather than quietly shipping a
shorter list.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from neorx.core.graph.models import GraphEdge, GraphNode, NodeType

__all__ = [
    "EXCLUDED_TYPES",
    "GENE_LIKE_TYPES",
    "SNAPSHOT_RELEASE_KEY",
    "Restriction",
    "enrich",
    "frame_identity",
    "is_pinned",
    "match_key",
    "restrict_to_frame",
]

# Node types an unpinned source may not introduce on a dated build. These
# are the types that become identification candidates -- the same three
# ``neorx.core.causal.identifier._get_candidate_nodes`` admits, minus the
# one excluded outright below; a pathway or a structure is enrichment
# hanging off a candidate, not a candidate.
GENE_LIKE_TYPES: tuple[NodeType, ...] = (
    NodeType.GENE,
    NodeType.PROTEIN,
)

# Node types no dated build may contain at all, whatever the frame says.
# Nothing in the dated stack pins a pathogen target, so nothing can date
# its score.
EXCLUDED_TYPES: tuple[NodeType, ...] = (NodeType.PATHOGEN_GENE,)

# What marks a node as read from a pinned release. Written by
# ``open_targets_from_snapshot``; read here rather than threaded through
# the builder as a flag, so the fact travels with the node it is about.
SNAPSHOT_RELEASE_KEY = "snapshot_release"


def match_key(value: str) -> str:
    """The one form in which identities and symbols are compared.

    Applied to a ``node_id`` it yields an identity key; applied to a gene
    symbol it yields a symbol key. Every comparison between a frame and a
    node in a dated build goes through this function, so no two of them
    can normalise differently -- which is precisely how a live node
    survived a frame check and then failed the merge that would have
    absorbed it.

    ``.lower()``, not ``.casefold()``: the extract side of the OmniPath
    comparison is polars' ``str.to_lowercase``, and one rule means one
    operation.
    """
    return value.strip().lower()


def is_pinned(node: GraphNode) -> bool:
    """Whether this node was read from a pinned snapshot release."""
    return bool(node.metadata.get(SNAPSHOT_RELEASE_KEY))


def enrich(node: GraphNode, key: str, value: Any) -> None:
    """Write ``key`` on ``node`` without overwriting what a release said.

    The same protection ``_merge_nodes`` applies, for the enrichment
    passes that write metadata directly rather than through a merge. A
    live source may add what the release is silent about; it may not
    restate what the release said.
    """
    if is_pinned(node):
        node.metadata.setdefault(key, value)
    else:
        node.metadata[key] = value


def frame_identity(nodes: list[GraphNode]) -> dict[str, GraphNode]:
    """The gene identities a dated build may evaluate.

    ``nodes`` is the pinned Open Targets reader's output. Maps each
    identity key to the release's own node, so an unpinned source
    reporting the same gene under a different spelling is reconciled *to
    the release's* identity -- both its ``node_id`` and its ``name``,
    since the gene list every other source is given is built from names --
    rather than the release being reconciled to today's.
    """
    return {
        match_key(node.node_id): node
        for node in nodes
        if node.node_type in GENE_LIKE_TYPES and node.node_id
    }


class Restriction(NamedTuple):
    """What survived an unpinned source's contribution, and what did not."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    off_frame: list[str]
    excluded_by_type: list[str]


def restrict_to_frame(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    frame: dict[str, GraphNode],
) -> Restriction:
    """Keep only what an unpinned source may contribute to a dated build.

    Three things happen, and the third is the one a bare-name check could
    not do. A node of an excluded type goes, whatever it is called. A
    gene-like node whose identity is not in the frame goes. A gene-like
    node whose identity *is* in the frame but spelled differently is
    rewritten to the release's ``node_id``, along with every edge naming
    it -- otherwise the frame check passes and the merge still produces
    two nodes for one gene, which is the defect rather than its fix.

    An edge touching a dropped node goes with it: an edge to a node that
    does not exist is worse than no edge, and the identification code
    would read it as a real arrow to a nameless target.

    Nodes are copied rather than mutated; the caller's list is a source's
    own output and re-spelling it in place would make the source's return
    value depend on who consumed it.
    """
    kept_nodes: list[GraphNode] = []
    dropped_ids: set[str] = set()
    off_frame: list[str] = []
    excluded_by_type: list[str] = []
    remapped: dict[str, str] = {}

    for node in nodes:
        if node.node_type in EXCLUDED_TYPES:
            dropped_ids.add(node.node_id)
            excluded_by_type.append(node.node_id)
            continue

        if node.node_type in GENE_LIKE_TYPES:
            pinned = frame.get(match_key(node.node_id))
            if pinned is None:
                dropped_ids.add(node.node_id)
                off_frame.append(node.name or node.node_id)
                continue
            if (pinned.node_id, pinned.name) != (node.node_id, node.name):
                remapped[node.node_id] = pinned.node_id
                node = node.model_copy(
                    update={"node_id": pinned.node_id, "name": pinned.name}
                )

        kept_nodes.append(node)

    if not dropped_ids and not remapped:
        return Restriction(nodes, edges, [], [])

    kept_edges: list[GraphEdge] = []
    for edge in edges:
        if edge.source_id in dropped_ids or edge.target_id in dropped_ids:
            continue
        source_id = remapped.get(edge.source_id, edge.source_id)
        target_id = remapped.get(edge.target_id, edge.target_id)
        if (source_id, target_id) != (edge.source_id, edge.target_id):
            edge = edge.model_copy(
                update={"source_id": source_id, "target_id": target_id}
            )
        kept_edges.append(edge)

    return Restriction(kept_nodes, kept_edges, off_frame, excluded_by_type)
