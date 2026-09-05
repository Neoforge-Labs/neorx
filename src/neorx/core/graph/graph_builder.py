"""
Disease Causal Graph Builder
=============================

The graph builder is the first major step in the NeoRx
pipeline.  It queries all 8 biomedical data sources and assembles
a unified **causal knowledge graph** for a specific disease.

Architecture
------------
1. Query gene–disease associations (Monarch Initiative, Open Targets)
1b. Query validated drug targets (ChEMBL — human + pathogen)
2. Query pathway memberships (KEGG, Reactome)
3. Query protein–protein interactions (STRING)
4. Enrich proteins with UniProt metadata (function, PDB IDs)
5. Query PDB structures for docking targets
6. Merge duplicate nodes by gene symbol
7. Build a NetworkX DiGraph ready for DoWhy causal inference

Node merging
------------
Monarch might call it "CCR5" while Open Targets uses
"ENSG00000160791".  We normalise to gene symbols and merge,
keeping the highest confidence score and union of all metadata.

Edge semantics
--------------
- ``ASSOCIATED_WITH`` → gene ↔ disease (from Monarch, OT)
- ``PARTICIPATES_IN`` → gene → pathway (from KEGG, Reactome)
- ``INTERACTS_WITH`` → protein ↔ protein (from STRING)
- ``CAUSES`` → the disease node → disease outcome

The causal identifier (next stage) then tests whether each
gene's path to the disease outcome is *causal* or merely
*correlational* using do-calculus.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

import networkx as nx

from neorx.core.graph.dated_frame import (
    frame_symbols,
    is_pinned,
    restrict_to_frame,
)
from neorx.core.graph.models import (
    DiseaseGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)
from neorx.core.sources import (
    query_monarch,
    query_open_targets,
    query_kegg_pathways,
    query_reactome_pathways,
    query_string_interactions,
    query_omnipath,
    query_uniprot,
    query_pdb_structures,
    query_chembl,
)
from neorx.core.cache import get_cache, _cache_key, GRAPH_TTL

if TYPE_CHECKING:
    # `neorx.snapshots.resolver` imports `neorx.core.sources.snapshot_sources`,
    # so that a dated run resolves to real snapshot readers rather than to a
    # placeholder, and `neorx.core.__init__` imports this module. A runtime
    # import here would make that dependency mutual. It does not currently
    # fail -- `neorx/__init__` imports `neorx.core` first, so the resolver is
    # always reached with `neorx.core.graph.models` already loaded -- but it
    # would be a cycle held open by import order alone, and nothing here needs
    # the name at runtime: `from __future__ import annotations` leaves the
    # annotation a string.
    from neorx.snapshots.resolver import SourceResolver

logger = logging.getLogger(__name__)


def build_disease_graph(
    disease: str,
    *,
    max_genes: int = 20,
    string_min_score: int = 400,
    use_cache: bool = True,
    allow_mocks: bool = False,
    as_of: str | None = None,
    resolver: SourceResolver | None = None,
    disease_id: str | None = None,
) -> DiseaseGraph:
    """Build a comprehensive causal knowledge graph for a disease.

    Parameters
    ----------
    disease : str
        Disease name (e.g. "HIV", "Type 2 Diabetes").
    max_genes : int
        Cap on number of genes to include.
    string_min_score : int
        STRING combined score threshold (0–1000).
    use_cache : bool
        Check/store results in the cache layer.
    allow_mocks : bool
        If *True*, data source clients may fall back to curated
        mock data when a live API call fails.  If *False*
        (default), failed API calls produce empty results. Refused
        outright together with ``as_of``: curated mock data is a
        measurement of no release, and a dated build would carry it as a
        node score for the date.
    as_of : str | None
        A release date for a dated build. ``None`` (default) means
        undated, unchanged behaviour -- every source is queried live.
        When set, requires ``resolver`` and ``disease_id``.
    resolver : SourceResolver | None
        Resolves ``opentargets`` and ``omnipath`` -- the sources that
        feed the causal subgraph -- to their pinned snapshot for
        ``as_of``. Resolved before any fetching begins, so a missing
        snapshot is refused immediately rather than after minutes of
        work against the other sources. It will not fall back to live
        data, and on a dated build the live Open Targets and OmniPath
        clients are not called at all.
    disease_id : str | None
        EFO disease id. Required when ``as_of`` is set: the extract is
        keyed on EFO id and records no disease names, so a dated build
        cannot look one up from the snapshot, and looking it up from the
        API would put a live call inside a dated build. Undated, it is
        optional and resolved from Open Targets as before.

    Returns
    -------
    DiseaseGraph
        Assembled graph with merged nodes and unified edges.

    Dated builds and live builds are not interchangeable
    ----------------------------------------------------
    On a **live** build a gene node's ``score`` is OpenTargets' overall
    association score: a harmonic sum over datasources, published by the
    platform. On a **dated** build it is the maximum genetic-evidence
    score in the pinned release, because the derived extract carries one
    row per datatype and does not carry the overall score -- and
    reconstructing it would mean putting a computed quantity where a
    measurement belongs. Both are real numbers from Open Targets, but
    they are *different quantities*: they must not be placed in the same
    column of a table, ranked against each other, or pooled. The full
    per-datatype breakdown is on both, in
    ``metadata["datatype_scores"]``. See
    ``neorx.core.sources.snapshot_sources``.

    The gene node population of a dated build is the frame
    ----------------------------------------------------
    The unpinned sources stay live on a dated build, because their edges
    are not causal-admissible and identification filters them out. Their
    *nodes* are another matter, and on a dated build they are restricted
    to the pinned release's own gene population -- they may enrich a node
    the release put in the graph, never introduce one, never overwrite a
    pinned score or a pinned node's metadata, and never take a place in
    the ``max_genes`` cap. See ``neorx.core.graph.dated_frame`` for what
    each of those channels does when it is left open.
    """
    ot_reader = None
    omnipath_reader = None
    if as_of is not None:
        if allow_mocks:
            raise ValueError(
                f"build_disease_graph(as_of={as_of!r}, allow_mocks=True) is "
                f"refused: mock data is a measurement of no release, and on "
                f"a dated build it would enter the graph as a node score for "
                f"{disease!r} as of {as_of}. Pass allow_mocks=False, or drop "
                f"as_of."
            )
        if resolver is None:
            raise ValueError(
                "build_disease_graph(as_of=...) requires a resolver: "
                "a dated run cannot be built without one to pin "
                "opentargets and omnipath to that date's snapshot."
            )
        if not disease_id:
            raise ValueError(
                f"build_disease_graph(as_of={as_of!r}) requires disease_id: "
                f"the pinned extract is keyed on EFO id and records no "
                f"disease names, so {disease!r} cannot be resolved from the "
                f"snapshot, and resolving it from the Open Targets API "
                f"would be a live call inside a dated build. Pass "
                f"disease_id='EFO_...' (or the MONDO/Orphanet id the "
                f"release uses)."
            )
        # Resolve the pinned sources before any fetching begins, so a
        # missing snapshot raises UnpinnedSourceError here rather than
        # after the other six sources have already been queried. These
        # are the readers the build actually uses: on a dated build
        # query_open_targets and query_omnipath are never called.
        ot_reader = resolver.resolve("opentargets", as_of)
        omnipath_reader = resolver.resolve("omnipath", as_of)

    # ── Check cache first ─────────────────────────────────────
    if use_cache:
        cache = get_cache()
        cache_key = _cache_key(
            "graph", disease=disease.lower(),
            max_genes=max_genes, string_min_score=string_min_score,
            as_of=as_of, disease_id=disease_id,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            try:
                graph = DiseaseGraph.model_validate(cached)
                logger.info("Graph loaded from cache (%d nodes, %d edges).",
                            len(graph.nodes), len(graph.edges))
                return graph
            except Exception:
                pass  # Invalid cache entry — rebuild

    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    sources_queried: list[str] = []

    # Filled once the pinned reader has run: on a dated build its node set
    # is the frame, and every unpinned source is restricted to it. ``None``
    # on an undated build, where no source is restricted.
    frame: frozenset[str] | None = None
    dropped_by_source: dict[str, list[str]] = {}

    def admit(
        source_name: str,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Take an unpinned source's contribution into the build."""
        if frame is not None:
            nodes, edges, dropped = restrict_to_frame(nodes, edges, frame)
            if dropped:
                dropped_by_source.setdefault(source_name, []).extend(dropped)
        all_nodes.extend(nodes)
        all_edges.extend(edges)
        return nodes, edges

    # ── Step 1: Gene–Disease Associations (parallel) ──────────

    if ot_reader is None:
        ot_call = partial(
            query_open_targets, disease,
            max_results=max_genes, allow_mocks=allow_mocks,
        )
        logger.info("Querying Monarch + Open Targets for '%s' (parallel)…", disease)
    else:
        ot_call = partial(
            ot_reader, disease,
            disease_id=disease_id, max_results=max_genes,
        )
        logger.info(
            "Querying Monarch live + Open Targets pinned at %s for '%s'…",
            as_of, disease,
        )
    with ThreadPoolExecutor(max_workers=2) as pool:
        mn_future = pool.submit(
            query_monarch, disease,
            max_results=max_genes,
            allow_mocks=allow_mocks,
        )
        ot_future = pool.submit(ot_call)
        mn_nodes, mn_edges = mn_future.result()
        ot_nodes, ot_edges = ot_future.result()

    # Open Targets goes in first on a dated build, because it *is* the
    # frame: nothing else may add a gene the pinned release did not.
    all_nodes.extend(ot_nodes)
    all_edges.extend(ot_edges)
    sources_queried.append("OpenTargets")
    logger.info("  Open Targets: %d nodes, %d edges.", len(ot_nodes), len(ot_edges))

    if ot_reader is not None:
        frame = frame_symbols(ot_nodes)
        logger.info(
            "Dated build frame: %d genes from the pinned release. Unpinned "
            "sources may enrich them and may not add to them.", len(frame),
        )

    mn_nodes, mn_edges = admit("Monarch", mn_nodes, mn_edges)
    sources_queried.append("Monarch")
    logger.info("  Monarch: %d nodes, %d edges.", len(mn_nodes), len(mn_edges))

    # ── Step 1b: ChEMBL Drug Targets (local SQLite) ────────────

    logger.info("Querying ChEMBL for '%s'…", disease)
    chembl_nodes, chembl_edges = query_chembl(
        disease, max_results=max_genes * 3,
        allow_mocks=allow_mocks,
    )
    chembl_nodes, chembl_edges = admit("ChEMBL", chembl_nodes, chembl_edges)
    if chembl_nodes:
        sources_queried.append("ChEMBL")
    n_pathogen = sum(
        1 for n in chembl_nodes
        if n.metadata.get("is_pathogen_target")
    )
    logger.info(
        "  ChEMBL: %d nodes (%d pathogen), %d edges.",
        len(chembl_nodes), n_pathogen, len(chembl_edges),
    )

    # Resolve disease ontology ID from Open Targets. A dated build was
    # given one -- it had to be, the snapshot is keyed on it -- and must
    # not make this live call.
    if disease_id is None:
        try:
            from neorx.core.sources.open_targets import resolve_disease_id
            disease_id = resolve_disease_id(disease)
        except Exception:
            pass

    # ── Step 2: Pathway Memberships ─────────────────────────────

    gene_symbols = _extract_gene_symbols(all_nodes, max_genes=max_genes)
    logger.info("Found %d unique gene symbols (capped at %d).",
                len(gene_symbols), max_genes)

    logger.info("Querying KEGG pathways…")
    kegg_nodes, kegg_edges = admit(
        "KEGG", *query_kegg_pathways(gene_symbols, allow_mocks=allow_mocks),
    )
    sources_queried.append("KEGG")
    logger.info("  KEGG: %d nodes, %d edges.", len(kegg_nodes), len(kegg_edges))

    logger.info("Querying Reactome pathways…")
    react_nodes, react_edges = admit(
        "Reactome",
        *query_reactome_pathways(gene_symbols, allow_mocks=allow_mocks),
    )
    sources_queried.append("Reactome")
    logger.info("  Reactome: %d nodes, %d edges.", len(react_nodes), len(react_edges))

    # ── Step 3: Protein–Protein Interactions ────────────────────

    logger.info("Querying STRING interactions…")
    string_nodes, string_edges = admit(
        "STRING",
        *query_string_interactions(
            gene_symbols, min_score=string_min_score,
            allow_mocks=allow_mocks,
        ),
    )
    sources_queried.append("STRING")
    logger.info("  STRING: %d nodes, %d edges.", len(string_nodes), len(string_edges))

    # ── Step 3b: Directed Regulatory Interactions ───────────────

    if omnipath_reader is None:
        logger.info("Querying OmniPath regulatory interactions…")
        omni_nodes, omni_edges = query_omnipath(
            gene_symbols, allow_mocks=allow_mocks,
        )
    else:
        logger.info("Reading OmniPath regulatory interactions pinned at %s…", as_of)
        omni_nodes, omni_edges = omnipath_reader(gene_symbols)
    if omnipath_reader is None:
        omni_nodes, omni_edges = admit("OmniPath", omni_nodes, omni_edges)
    else:
        # Pinned: it is not an unpinned source and is not restricted. It
        # only ever saw frame symbols, because ``gene_symbols`` is drawn
        # from a node population that is already the frame.
        all_nodes.extend(omni_nodes)
        all_edges.extend(omni_edges)
    sources_queried.append("OmniPath")
    logger.info("  OmniPath: %d directed edges.", len(omni_edges))

    if frame is not None and dropped_by_source:
        n_dropped = sum(len(names) for names in dropped_by_source.values())
        logger.info(
            "Dated build of '%s' as of %s: dropped %d off-frame gene node(s) "
            "from unpinned sources (%s). The pinned release's %d genes are "
            "the population; an unpinned source may not add to it.",
            disease, as_of, n_dropped,
            ", ".join(
                f"{source}: {len(names)} ({', '.join(sorted(set(names)))})"
                for source, names in sorted(dropped_by_source.items())
            ),
            len(frame),
        )

    # ── Step 4: UniProt Enrichment ──────────────────────────────

    logger.info("Enriching with UniProt metadata…")
    uniprot_data = query_uniprot(gene_symbols, allow_mocks=allow_mocks)
    _enrich_nodes_with_uniprot(all_nodes, uniprot_data)
    sources_queried.append("UniProt")
    logger.info("  UniProt: enriched %d/%d proteins.", len(uniprot_data), len(gene_symbols))

    # ── Step 5: PDB Structures ──────────────────────────────────

    uniprot_map = {
        gene: info["uniprot_id"]
        for gene, info in uniprot_data.items()
        if info.get("uniprot_id")
    }
    if uniprot_map:
        logger.info("Querying PDB structures for %d proteins…", len(uniprot_map))
        pdb_data = query_pdb_structures(uniprot_map, allow_mocks=allow_mocks)
        _enrich_nodes_with_pdb(all_nodes, pdb_data)
        sources_queried.append("PDB")
        logger.info("  PDB: structures for %d proteins.", len(pdb_data))

    # ── Step 6: Merge & Build ───────────────────────────────────

    merged_nodes, merged_edges = _merge_nodes(all_nodes, all_edges)

    # Add disease outcome node
    disease_node = GraphNode(
        node_id=f"disease:{disease.lower().replace(' ', '_')}",
        name=disease,
        node_type=NodeType.DISEASE,
        source="NeoRx",
        score=1.0,
    )
    merged_nodes.append(disease_node)

    # Connect genes directly to disease via ASSOCIATED_WITH if
    # they don't already have that edge
    existing_disease_edges = {
        (e.source_id, e.target_id) for e in merged_edges
        if e.edge_type == EdgeType.ASSOCIATED_WITH
    }
    for node in merged_nodes:
        if node.node_type in (NodeType.GENE, NodeType.PROTEIN, NodeType.PATHOGEN_GENE):
            pair = (node.node_id, disease_node.node_id)
            if pair not in existing_disease_edges:
                merged_edges.append(GraphEdge(
                    source_id=node.node_id,
                    target_id=disease_node.node_id,
                    edge_type=EdgeType.ASSOCIATED_WITH,
                    weight=node.score,
                    source_db="NeoRx",
                ))

    graph = DiseaseGraph(
        disease_name=disease,
        disease_id=disease_id,
        nodes=merged_nodes,
        edges=merged_edges,
        sources_queried=sources_queried,
        as_of=as_of,
    )

    logger.info(
        "Graph built: %d nodes (%d genes, %d proteins, %d pathways), %d edges.",
        len(graph.nodes), graph.n_genes, graph.n_proteins,
        graph.n_pathways, len(graph.edges),
    )

    # ── Cache & persist ───────────────────────────────────────
    if use_cache:
        try:
            cache.set(cache_key, graph.model_dump(mode="json"), ttl=GRAPH_TTL)
        except Exception:
            pass

    try:
        from neorx.core.graph.persistence import save_graph_to_db
        # The upsert key is (disease_name, parameters). Without the date
        # and the ontology id in there, a dated and a live build of the
        # same disease are the same row and overwrite each other.
        save_graph_to_db(graph, params={
            "max_genes": max_genes,
            "string_min_score": string_min_score,
            "as_of": as_of,
            "disease_id": disease_id,
        })
    except Exception:
        pass

    return graph


def disease_graph_to_networkx(graph: DiseaseGraph) -> nx.DiGraph:
    """Convert a DiseaseGraph to a NetworkX directed graph.

    This is needed by DoWhy for causal inference.  Node attributes
    include ``node_type``, ``score``, ``uniprot_id``, etc.  Edge
    attributes include ``edge_type``, ``weight``, ``source_db``.
    """
    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(
            node.node_id,
            name=node.name,
            node_type=node.node_type.value,
            score=node.score,
            uniprot_id=node.uniprot_id or "",
            pdb_ids=node.pdb_ids,
            description=node.description or "",
            source=node.source or "",
            metadata=dict(node.metadata),
        )

    for (source_id, target_id), attrs in _resolve_edge_collisions(graph.edges).items():
        G.add_edge(source_id, target_id, **attrs)

    return G


def _resolve_edge_collisions(
    edges: list[GraphEdge],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Pick one edge per node pair, by an explicit rule.

    A ``DiGraph`` holds at most one edge between two nodes, but the
    assembled disease graph routinely contains several: STRING and
    OmniPath both report protein relationships, so the same gene pair
    arrives as an undirected ``interacts_with`` and again as a directed,
    signed regulatory edge.

    Left to ``add_edge``, the last writer would win and the outcome
    would depend on the order the graph builder happens to append
    sources in. That is not a tie-break, it is an accident: an
    associational edge overwriting a causal one deletes an arrow from
    the causal subgraph, which lowers the identifiability rate and
    presents as a finding rather than as a bug.

    So the rule is explicit and order-independent:

    1. A causal-admissible edge always beats an associational one
       (see :mod:`neorx.core.causal.graph_semantics`). Admissibility is
       a claim about what the edge means; weight is only a confidence
       in it, and no amount of confidence in an association promotes it
       to a mechanism.
    2. Between two edges of equal standing, the heavier wins.
    3. ``primary_sources`` are unioned across every colliding edge
       whatever the outcome, because corroboration counts distinct
       primary evidence and a collision must not discard the loser's
       provenance.
    """
    from neorx.core.causal.graph_semantics import is_causal_admissible

    resolved: dict[tuple[str, str], dict[str, Any]] = {}

    for edge in edges:
        key = (edge.source_id, edge.target_id)
        attrs: dict[str, Any] = {
            "edge_type": edge.edge_type.value,
            "weight": edge.weight,
            "source_db": edge.source_db,
            "evidence": edge.evidence or "",
            "evidence_class": edge.evidence_class,
            "sign": edge.sign,
            "primary_sources": list(edge.primary_sources),
        }

        incumbent = resolved.get(key)
        if incumbent is None:
            resolved[key] = attrs
            continue

        merged_sources = list(
            dict.fromkeys(incumbent["primary_sources"] + attrs["primary_sources"])
        )

        challenger_rank = (is_causal_admissible(attrs), attrs["weight"])
        incumbent_rank = (is_causal_admissible(incumbent), incumbent["weight"])

        winner = attrs if challenger_rank > incumbent_rank else incumbent
        winner["primary_sources"] = merged_sources
        resolved[key] = winner

    return resolved


# ── Internal Helpers ────────────────────────────────────────────────

def _extract_gene_symbols(
    nodes: list[GraphNode],
    max_genes: int = 0,
) -> list[str]:
    """Extract unique gene symbols from node names.

    Parameters
    ----------
    nodes : list[GraphNode]
        All nodes collected so far.
    max_genes : int
        If > 0, keep only the top-scoring genes (by their node
        score) to avoid sending hundreds of genes to per-gene APIs.
    """
    # Collect unique genes, remembering the best score for each
    best_score: dict[str, float] = {}
    for node in nodes:
        if node.node_type in (NodeType.GENE, NodeType.PROTEIN):
            sym = node.name.upper()
            if sym not in best_score or node.score > best_score[sym]:
                best_score[sym] = node.score

    # Sort by score descending, then cap
    ranked = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
    if max_genes > 0:
        ranked = ranked[:max_genes]
    return [sym for sym, _ in ranked]


def _enrich_nodes_with_uniprot(
    nodes: list[GraphNode],
    uniprot_data: dict[str, dict[str, Any]],
) -> None:
    """In-place enrichment of nodes with UniProt metadata."""
    for node in nodes:
        gene = node.name.upper()
        info = uniprot_data.get(gene)
        if not info:
            continue

        if not node.uniprot_id and info.get("uniprot_id"):
            node.uniprot_id = info["uniprot_id"]
        if not node.pdb_ids and info.get("pdb_ids"):
            node.pdb_ids = info["pdb_ids"]
        if not node.description and info.get("function"):
            node.description = info["function"]

        # Store druggability in metadata
        node.metadata["is_druggable"] = info.get("is_druggable", False)
        node.metadata["subcellular_location"] = info.get("subcellular_location", "")
        node.metadata["go_terms"] = info.get("go_terms", [])


def _enrich_nodes_with_pdb(
    nodes: list[GraphNode],
    pdb_data: dict[str, list[dict[str, Any]]],
) -> None:
    """In-place enrichment of nodes with PDB structure IDs."""
    for node in nodes:
        gene = node.name.upper()
        structs = pdb_data.get(gene)
        if not structs:
            continue
        # Prefer structures with ligands (defines binding pocket)
        sorted_structs = sorted(structs, key=lambda s: (s.get("has_ligand", False), -(s.get("resolution") or 99)))
        pdb_ids = [s["pdb_id"] for s in sorted_structs]
        if not node.pdb_ids:
            node.pdb_ids = pdb_ids
        else:
            # Union
            existing = set(node.pdb_ids)
            for pid in pdb_ids:
                if pid not in existing:
                    node.pdb_ids.append(pid)


def _merge_nodes(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Merge duplicate nodes by gene/pathway name.

    When the same gene appears from multiple sources (Monarch
    and Open Targets both report CCR5), we keep the entry with
    the highest score and merge metadata.

    A node read from a pinned release is the exception, and it is not an
    exception this function is told about: the fact rides on the node, in
    ``metadata["snapshot_release"]`` (see
    ``neorx.core.graph.dated_frame.is_pinned``). Such a node keeps its
    score and every metadata key the snapshot reader wrote, whatever a
    live source reports. A live source may still *add* keys the pinned
    node does not carry -- that is enrichment, and it is the whole reason
    the unpinned sources are still queried on a dated build -- but
    ``0.85`` from Monarch may not become the release's genetic score, and
    ``has_known_drug`` from today's ChEMBL may not become the release's
    answer about known drugs. On an undated build no node is pinned and
    the behaviour is exactly what it was.

    Sources are still unioned onto a pinned node's ``source`` string.
    That is a record of corroboration, not an overwrite: the score
    ``collect_source_scores`` then attributes to Monarch is the pinned
    node's own.
    """
    merged: dict[str, GraphNode] = {}

    for node in nodes:
        key = node.node_id
        if key in merged:
            existing = merged[key]
            existing_pinned = is_pinned(existing)
            incoming_pinned = is_pinned(node)

            if existing_pinned and not incoming_pinned:
                # Enrichment only: add what the release does not say,
                # change nothing it does.
                for meta_key, value in node.metadata.items():
                    existing.metadata.setdefault(meta_key, value)
            elif incoming_pinned and not existing_pinned:
                # The pinned node arrived second (Monarch is queried
                # first). The release's answer replaces the live one.
                existing.score = node.score
                existing.metadata.update(node.metadata)
            else:
                # Keep highest score
                if node.score > existing.score:
                    existing.score = node.score
                # Merge metadata
                existing.metadata.update(node.metadata)

            # Merge UniProt
            if node.uniprot_id and not existing.uniprot_id:
                existing.uniprot_id = node.uniprot_id
            # Merge PDB IDs
            existing_pdb = set(existing.pdb_ids)
            for pid in node.pdb_ids:
                if pid not in existing_pdb:
                    existing.pdb_ids.append(pid)
            # Record multiple sources
            if node.source and node.source not in existing.source:
                existing.source = f"{existing.source}, {node.source}"
        else:
            # Deep, so that merging into the copy cannot reach back into
            # the source node's own metadata dict -- which is exactly what
            # a shallow copy shares, and what makes "the pinned node keeps
            # its metadata" true by accident rather than by construction.
            merged[key] = node.model_copy(deep=True)

    # Deduplicate edges
    seen_edges: set[tuple[str, str, str]] = set()
    unique_edges: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.edge_type.value)
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)

    return list(merged.values()), unique_edges
