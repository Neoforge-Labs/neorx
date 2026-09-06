"""A dated graph does not depend on what the live sources say.

Every defect found on this path was a route by which post-`as_of` data
reached a dated graph while the graph still looked correct: a cache key
without a date, a resolver whose result was discarded, unpinned nodes
overwriting pinned scores, a frame matched on a bare name, an extract half
of which was mouse, and today's druggability worth +0.015 on the headline
score.

Each was closed by its own test. Those tests check the mechanism that was
broken, which is what a fix needs -- but the property actually wanted is
stronger and simpler than any of them: run a dated build twice, once with
the unpinned sources returning everything they can and once returning
nothing, and the graph must be identical. Every one of the six defects
violates that single statement.

This is deliberately a whole-graph comparison rather than a check on named
fields. A test naming the fields it knows about cannot fail for a field
nobody has thought of yet, and that is precisely how the last three got
through.
"""

import polars as pl
import pytest

from neorx.core.cache import FileCache
from neorx.core.graph.models import EdgeType, GraphEdge, GraphNode, NodeType
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.resolver import SourceResolver
from neorx.snapshots.schema import ASSOCIATION_COLUMNS, INTERACTION_COLUMNS

RELEASE = "18.06"
DISEASE_ID = "EFO_1"
UNPINNED = (
    "query_monarch",
    "query_chembl",
    "query_kegg_pathways",
    "query_reactome_pathways",
    "query_string_interactions",
)


def _write_snapshot(root):
    associations = root / "opentargets" / RELEASE
    associations.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "target_id": ["E1", "E2", "E3"],
            # C9orf72 is mixed-case on purpose: the identity rule and the
            # OmniPath filter must agree about it.
            "target_symbol": ["CCR5", "C9orf72", "CXCR4"],
            "disease_id": [DISEASE_ID] * 3,
            "datatype": ["genetic_association"] * 3,
            "score": [0.75, 0.30, 0.40],
        },
        schema=ASSOCIATION_COLUMNS,
    ).write_parquet(associations / "associations.parquet")

    interactions = root / "omnipath" / RELEASE
    interactions.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "source_symbol": ["CCR5", "C9orf72"],
            "target_symbol": ["C9orf72", "CXCR4"],
            "is_directed": [True, True],
            "consensus_direction": [True, True],
            "is_stimulation": [True, False],
            "is_inhibition": [False, True],
            "primary_sources": ["SIGNOR", "SIGNOR"],
            "references": ["1", "2"],
        },
        schema=INTERACTION_COLUMNS,
    ).write_parquet(interactions / "interactions.parquet")


def _everything_a_live_source_could_say():
    """A deliberately hostile unpinned contribution.

    One node re-spelling a frame gene at a higher score with today's
    clinical answers on it; one node off the frame entirely; an edge
    between them. Each of these is a defect that shipped.
    """
    return (
        [
            GraphNode(
                node_id="gene:C9ORF72",
                name="C9ORF72",
                node_type=NodeType.GENE,
                source="ChEMBL",
                score=0.99,
                metadata={
                    "clinical_phase": 4,
                    "n_drugs": 7,
                    "drugs": ["a", "b"],
                    "has_known_drug": True,
                    "is_druggable": True,
                    "chembl_drug_evidence_score": 0.9,
                },
            ),
            GraphNode(
                node_id="gene:OFFFRAME",
                name="OFFFRAME",
                node_type=NodeType.GENE,
                source="Monarch",
                score=0.98,
            ),
            GraphNode(
                node_id="pathogen:plasmodium_falciparum:CCR5",
                name="CCR5",
                node_type=NodeType.PATHOGEN_GENE,
                source="ChEMBL",
                score=0.97,
                metadata={"clinical_phase": 4},
            ),
        ],
        [
            GraphEdge(
                source_id="gene:C9ORF72",
                target_id="gene:OFFFRAME",
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=0.9,
                source_db="ChEMBL",
                evidence="live",
            )
        ],
    )


def _build(tmp_path, monkeypatch, *, live_active, tag):
    import neorx.core.graph.graph_builder as gb

    root = tmp_path / tag
    _write_snapshot(root / "snapshots")

    def empty(*_a, **_kw):
        return [], []

    def empty_dict(*_a, **_kw):
        return {}

    contribution = _everything_a_live_source_could_say() if live_active else ([], [])
    for name in UNPINNED:
        monkeypatch.setattr(gb, name, lambda *_a, **_kw: contribution)
    monkeypatch.setattr(gb, "query_uniprot", empty_dict)
    monkeypatch.setattr(gb, "query_pdb_structures", empty_dict)
    monkeypatch.setattr(gb, "get_cache", lambda: FileCache(cache_dir=root / "cache"))

    resolver = SourceResolver(
        store=SnapshotStore(root / "snapshots"),
        release_for=lambda _source, _as_of: RELEASE,
        live={"opentargets": empty, "omnipath": empty},
    )
    return gb.build_disease_graph(
        "d", as_of="2018-06", disease_id=DISEASE_ID, resolver=resolver
    )


def _fingerprint(graph):
    return {
        "nodes": sorted(
            (n.node_id, n.name, n.node_type.value, round(n.score, 9)) for n in graph.nodes
        ),
        "edges": sorted(
            (
                e.source_id,
                e.target_id,
                e.edge_type.value,
                round(e.weight, 9),
                e.evidence_class or "",
            )
            for e in graph.edges
        ),
        # Values too, not just keys: a key present in both with a live
        # value in one of them is the druggability defect exactly.
        "metadata": sorted(
            (n.node_id, k, repr(v)) for n in graph.nodes for k, v in n.metadata.items()
        ),
    }


@pytest.mark.parametrize("part", ["nodes", "edges", "metadata"])
def test_a_dated_graph_is_identical_whatever_the_live_sources_say(
    tmp_path, monkeypatch, part
):
    silent = _fingerprint(_build(tmp_path, monkeypatch, live_active=False, tag="silent"))
    loud = _fingerprint(_build(tmp_path, monkeypatch, live_active=True, tag="loud"))
    assert loud[part] == silent[part]


def test_the_hostile_contribution_is_not_vacuous(tmp_path, monkeypatch):
    """The invariance above must not hold because nothing was offered.

    Without this, stubbing the live sources to return nothing would make
    every assertion above pass -- which is the exact shape of the blind
    spot that let three of these defects ship.
    """
    nodes, edges = _everything_a_live_source_could_say()
    assert len(nodes) == 3
    assert len(edges) == 1

    # And an UNDATED build does take them, so the fixture really is
    # something a live source can contribute.
    import neorx.core.graph.graph_builder as gb

    def empty_dict(*_a, **_kw):
        return {}

    for name in UNPINNED:
        monkeypatch.setattr(gb, name, lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(gb, "query_monarch", lambda *_a, **_kw: (nodes, edges))
    monkeypatch.setattr(gb, "query_open_targets", lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(gb, "query_omnipath", lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(gb, "query_uniprot", empty_dict)
    monkeypatch.setattr(gb, "query_pdb_structures", empty_dict)
    monkeypatch.setattr(
        gb, "get_cache", lambda: FileCache(cache_dir=tmp_path / "undated-cache")
    )

    graph = gb.build_disease_graph("d")
    live_names = {n.name for n in graph.nodes}
    assert "OFFFRAME" in live_names
    assert "C9ORF72" in live_names
