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

    Every element here survived a dated build at some point, and the two
    edges are the correction that matters: the first version of this
    fixture had one edge pointing at an off-frame node, so the edge was
    dropped along with its target and the `edges` comparison compared two
    empty lists. It passed for the same reason three of the leaks it was
    written to catch had passed -- the fixture offered nothing that could
    survive. `_the_hostile_contribution_is_not_vacuous` now checks each
    dimension can actually move, rather than only that an undated build
    accepts the fixture.
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
                    # Removing this key from OUTCOME_KEYS survived the
                    # whole suite until the fixture set it.
                    "tractability": [{"value": True, "modality": "SM"}],
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
            # Off-frame target: dropped with its endpoint.
            GraphEdge(
                source_id="gene:C9ORF72",
                target_id="gene:OFFFRAME",
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=0.9,
                source_db="ChEMBL",
                evidence="live",
            ),
            # BOTH endpoints in the frame, spelled as the release spells
            # them. This one survived frame restriction, and moved CCR5's
            # causal_confidence 0.5342 -> 0.5819 and robustness
            # 0.5404 -> 0.6142 through corroboration_factor -- without
            # ever reaching identification, which is why four passes over
            # the identification path did not see it.
            GraphEdge(
                source_id="gene:CCR5",
                target_id="gene:CXCR4",
                edge_type=EdgeType.INTERACTS_WITH,
                weight=0.95,
                source_db="STRING",
                evidence="live interaction",
                primary_sources=["STRING", "BioGRID"],
            ),
        ],
    )


def _what_uniprot_really_returns():
    """Faithful to ``uniprot.py``'s documented contract, not ``{}``.

    The invariance test stubbed UniProt and PDB empty, so it was blind
    exactly where the largest remaining leak was: pdb_ids +0.20 of
    assess_druggability, uniprot_id +0.10, a `function` description +0.15
    through its keyword match, and go_terms an evidence stream. Together
    +0.045 on causal_confidence -- three times the leak whose fix prompted
    the test.
    """
    return {
        "CCR5": {
            "uniprot_id": "P51681",
            "function": "C-C chemokine receptor type 5, a GPCR",
            "pdb_ids": ["5UIW", "4MBS"],
            "is_druggable": True,
            "subcellular_location": "Cell membrane",
            "go_terms": ["GO:0004950"],
        },
        "C9orf72": {
            "uniprot_id": "Q96LT7",
            "function": "Guanine nucleotide exchange factor",
            "pdb_ids": ["6LT0"],
            "is_druggable": True,
            "subcellular_location": "Cytoplasm",
            "go_terms": ["GO:0005085"],
        },
    }


def _what_pdb_really_returns():
    return {
        "CCR5": [{"pdb_id": "5UIW", "has_ligand": True, "resolution": 2.1}],
        "C9orf72": [{"pdb_id": "6LT0", "has_ligand": False, "resolution": 3.2}],
    }


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
    # Faithful returns, not {}. Stubbing these empty is what made the
    # first version of this test blind to the largest remaining leak.
    uniprot = _what_uniprot_really_returns() if live_active else {}
    pdb = _what_pdb_really_returns() if live_active else {}
    monkeypatch.setattr(gb, "query_uniprot", lambda *_a, **_kw: uniprot)
    monkeypatch.setattr(gb, "query_pdb_structures", lambda *_a, **_kw: pdb)
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
    """Everything about the graph that can reach a reported number.

    The omissions in the first version were not incidental: `source` is
    split by evidence.py into evidence streams and the consensus
    denominator, `pdb_ids`/`uniprot_id`/`description` are worth 0.45 of
    assess_druggability between them, and an edge's `primary_sources`
    feeds corroboration_factor. None of them is a score, and all of them
    move one.
    """
    return {
        "nodes": sorted(
            (n.node_id, n.name, n.node_type.value, round(n.score, 9))
            for n in graph.nodes
        ),
        "node_provenance": sorted(
            (n.node_id, n.source, n.uniprot_id or "", tuple(n.pdb_ids),
             n.description or "")
            for n in graph.nodes
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
        "edge_provenance": sorted(
            (e.source_id, e.target_id, e.source_db or "",
             tuple(e.primary_sources or ()), str(getattr(e, "sign", "")))
            for e in graph.edges
        ),
        # Values too, not just keys: a key present in both with a live
        # value in one of them is the druggability defect exactly.
        "metadata": sorted(
            (n.node_id, k, repr(v)) for n in graph.nodes for k, v in n.metadata.items()
        ),
        # A hand-listed set of fields cannot fail for a field nobody has
        # thought of yet -- this module's own docstring says so, and then
        # this happened anyway: `sources_queried` appended "PDB" only
        # `if uniprot_map:`, so a dated graph's source list depended on
        # what live UniProt answered. Every score was invariant and a
        # persisted, user-visible field was not.
        #
        # So the whole model is compared, minus only the two fields
        # DESIGNED to differ: build_timestamp, and frame_exclusions, which
        # is the record OF the difference.
        #
        # An earlier version of this also dropped `nodes` and `edges` on
        # the grounds that the dimensions above cover them. They did not:
        # GraphEdge.evidence and GraphEdge.pmids appear in no dimension,
        # and pmids is persisted and rendered in reports. That is the
        # hand-listed-fields failure this dimension exists to end,
        # committed inside the fix for it. Nothing is dropped for being
        # "already covered" again.
        "everything_else": {
            k: v
            for k, v in graph.model_dump(mode="json").items()
            if k not in ("build_timestamp", "frame_exclusions")
        },
    }


@pytest.mark.parametrize(
    "part",
    [
        "nodes",
        "node_provenance",
        "edges",
        "edge_provenance",
        "metadata",
        "everything_else",
    ],
)
def test_a_dated_graph_is_identical_whatever_the_live_sources_say(
    tmp_path, monkeypatch, part
):
    silent = _fingerprint(_build(tmp_path, monkeypatch, live_active=False, tag="silent"))
    loud = _fingerprint(_build(tmp_path, monkeypatch, live_active=True, tag="loud"))
    assert loud[part] == silent[part]


def test_the_hostile_contribution_is_not_vacuous(tmp_path, monkeypatch):
    """Every dimension above must be one the fixture can actually move.

    This is spec criterion 12, and it exists because the first version of
    this file did not satisfy it. That version asserted the fixture was
    non-empty and that an UNDATED build accepted it -- neither of which
    says anything about whether an element survives the DATED restriction,
    which is the only thing the invariance claim rests on. Its `edges`
    parametrisation was comparing two empty lists.

    So the check is made against the undated build, where nothing is
    withheld: if a dimension moves there, the fixture can exercise it, and
    the dated build holding it fixed is a real result rather than an
    absence of input.
    """
    import neorx.core.graph.graph_builder as gb

    nodes, edges = _everything_a_live_source_could_say()
    assert len(nodes) == 3
    assert len(edges) == 2

    def _undated(live_active, tag):
        contribution = (nodes, edges) if live_active else ([], [])
        uniprot = _what_uniprot_really_returns() if live_active else {}
        pdb = _what_pdb_really_returns() if live_active else {}
        for name in UNPINNED:
            monkeypatch.setattr(gb, name, lambda *_a, **_kw: ([], []))
        monkeypatch.setattr(gb, "query_monarch", lambda *_a, **_kw: contribution)
        monkeypatch.setattr(gb, "query_open_targets", lambda *_a, **_kw: _pinned_like())
        monkeypatch.setattr(gb, "query_omnipath", lambda *_a, **_kw: ([], []))
        monkeypatch.setattr(gb, "query_uniprot", lambda *_a, **_kw: uniprot)
        monkeypatch.setattr(gb, "query_pdb_structures", lambda *_a, **_kw: pdb)
        monkeypatch.setattr(
            gb, "get_cache", lambda: FileCache(cache_dir=tmp_path / tag)
        )
        return _fingerprint(gb.build_disease_graph("d"))

    silent = _undated(False, "u-silent")
    loud = _undated(True, "u-loud")

    moved = [part for part in silent if silent[part] != loud[part]]
    # Not "some dimension moved": every one of them, or the fixture is
    # vacuous for the ones that did not.
    assert set(moved) == set(silent), (
        f"fixture cannot exercise {sorted(set(silent) - set(moved))}; "
        f"the invariance assertions for those dimensions prove nothing"
    )


def _pinned_like():
    """An unpinned stand-in for the release's genes, for the undated probe.

    The undated path has no snapshot, so the frame genes have to arrive
    from somewhere for the live edges to have endpoints.
    """
    return (
        [
            GraphNode(
                node_id=f"gene:{sym}",
                name=sym,
                node_type=NodeType.GENE,
                source="Open Targets",
                score=score,
                metadata={"datatype_scores": {"genetic_association": score}},
            )
            for sym, score in (("CCR5", 0.75), ("C9orf72", 0.30), ("CXCR4", 0.40))
        ],
        [],
    )
