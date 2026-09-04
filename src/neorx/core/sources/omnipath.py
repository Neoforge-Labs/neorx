"""
OmniPath — signed, directed regulatory interactions.

OmniPath (https://omnipathdb.org/) aggregates causal molecular
interactions from SIGNOR, TRRUST, PhosphoSite, and around a hundred
other resources, resolving direction and sign by consensus across them.

This is the only source in NeoRx that supplies gene -> gene arrows a
causal claim may rest on. STRING supplies protein interactions, but they
are undirected: an arrow drawn from a STRING row records which protein
appeared in the first column, not which acts on which.

Aggregation does not cost provenance here. Each interaction carries the
list of primary databases that assert it and the PubMed IDs behind them,
both of which are recorded on the edge -- so multi-source corroboration
counts distinct primary evidence rather than counting OmniPath twice for
an interaction STRING already reported.

Endpoint: ``GET https://omnipathdb.org/interactions``. Note that the
``resources=`` filter returns an empty list for values that
``datasets=omnipath`` covers; use ``datasets``.

Filtering by gene: use ``partners=<comma-separated symbols>``, not
``genes=``. ``genes=`` returns HTTP 200 with a body that decodes as a
list, but each element is itself a list rather than an interaction
record -- it is not a usable filter and must not be used. ``partners=``
returns every interaction touching *any* of the named genes (a gene's
whole neighbourhood, not just edges within the set); the caller narrows
that down by keeping only interactions whose source and target are both
in the requested set -- see ``_interactions_to_edges``.
"""

from __future__ import annotations

import logging

import requests

from neorx.core.graph.models import EdgeType, GraphEdge, GraphNode

logger = logging.getLogger(__name__)

OMNIPATH_URL = "https://omnipathdb.org/interactions"
TIMEOUT = 45
HUMAN_TAXON = 9606

__all__ = ["query_omnipath"]


def _pmids_from_references(references: str) -> list[str]:
    """Extract PubMed IDs from OmniPath's ``DB:pmid;DB:pmid`` format."""
    pmids: set[str] = set()
    for token in (references or "").split(";"):
        _, _, pmid = token.rpartition(":")
        pmid = pmid.strip()
        if pmid.isdigit():
            pmids.add(pmid)
    return sorted(pmids)


def _interactions_to_edges(
    rows: list[dict],
    known_genes: set[str],
) -> list[GraphEdge]:
    """Convert OmniPath rows to graph edges.

    Only directed interactions with a consensus direction become edges.
    A sign is claimed only when stimulation and inhibition do not
    contradict each other.
    """
    edges: list[GraphEdge] = []

    for row in rows:
        source = (row.get("source_genesymbol") or "").strip()
        target = (row.get("target_genesymbol") or "").strip()
        if not source or not target or source == target:
            continue
        if source not in known_genes or target not in known_genes:
            continue
        if not row.get("is_directed") or not row.get("consensus_direction"):
            continue

        stimulates = bool(row.get("is_stimulation"))
        inhibits = bool(row.get("is_inhibition"))
        if stimulates and not inhibits:
            edge_type, sign = EdgeType.ACTIVATES, 1
        elif inhibits and not stimulates:
            edge_type, sign = EdgeType.INHIBITS, -1
        else:
            edge_type, sign = EdgeType.REGULATES, 0

        sources = [str(s) for s in row.get("sources") or []]

        edges.append(GraphEdge(
            source_id=f"gene:{source}",
            target_id=f"gene:{target}",
            edge_type=edge_type,
            weight=min(1.0, 0.5 + 0.05 * len(sources)),
            source_db="OmniPath",
            evidence=f"OmniPath consensus over {len(sources)} resources",
            evidence_class="regulatory",
            sign=sign,
            primary_sources=sources,
            pmids=_pmids_from_references(row.get("references", "")),
        ))

    return edges


def query_omnipath(
    gene_symbols: list[str],
    *,
    allow_mocks: bool = False,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Fetch directed regulatory interactions among ``gene_symbols``.

    Returns no new nodes -- OmniPath enriches the structure among genes
    the other sources already found. ``allow_mocks`` is accepted for
    parity with the other source modules; there is no curated mock set,
    so it has no effect and a failure returns empty results either way.
    """
    known = {g.strip() for g in gene_symbols if g and g.strip()}
    if not known:
        return [], []

    try:
        response = requests.get(
            OMNIPATH_URL,
            params={
                "genesymbols": "yes",
                "organisms": HUMAN_TAXON,
                "datasets": "omnipath",
                "fields": "sources,references",
                "format": "json",
                "partners": ",".join(sorted(known)),
            },
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        logger.warning("OmniPath request failed: %s.", e)
        return [], []

    if response.status_code != 200:
        logger.warning("OmniPath returned HTTP %d.", response.status_code)
        return [], []

    rows = response.json()
    if not isinstance(rows, list):
        logger.warning("OmniPath returned %s, expected a list.", type(rows).__name__)
        return [], []

    edges = _interactions_to_edges(rows, known)
    logger.info(
        "OmniPath: %d directed edges from %d interactions over %d genes.",
        len(edges), len(rows), len(known),
    )
    return [], edges
