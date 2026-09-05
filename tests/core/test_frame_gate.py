"""No gene outside the frame may become a candidate.

_get_candidate_nodes admits every gene, protein and pathogen_gene node in
the assembled graph, so any source contributing a gene node contributes a
candidate -- Monarch, ChEMBL, STRING, KEGG and Reactome all do. On a dated
run that is leakage in the sampling frame: a gene the date would never
have evaluated, scored as though it would have been.

The gate filters rather than raises, because Monarch legitimately returns
genes OpenTargets did not and refusing would fail every dated run. What
makes the filter safe is that every exclusion is recorded with the source
that contributed it, so a run that suddenly drops four hundred genes where
it used to drop twelve is visible rather than absorbed.
"""

import networkx as nx

from neorx.core.causal.identifier import _get_candidate_nodes


def _graph():
    G = nx.DiGraph()
    G.add_node("disease:d", node_type="disease", name="d", source="NeoRx")
    G.add_node("gene:PIK3CA", node_type="gene", name="PIK3CA", source="Open Targets")
    G.add_node("gene:TP53", node_type="gene", name="TP53", source="Open Targets")
    # Contributed by a source that bypasses any gene-list restriction.
    G.add_node("gene:LATER", node_type="gene", name="LATER", source="Monarch")
    return G


def test_without_a_frame_every_gene_is_a_candidate():
    # The undated path must behave exactly as before.
    candidates, excluded = _get_candidate_nodes(_graph(), "disease:d")
    assert set(candidates) == {"gene:PIK3CA", "gene:TP53", "gene:LATER"}
    assert excluded == []


def test_a_gene_outside_the_frame_never_becomes_a_candidate():
    candidates, _ = _get_candidate_nodes(
        _graph(),
        "disease:d",
        frame=frozenset({"PIK3CA", "TP53"}),
    )
    assert "gene:LATER" not in candidates
    assert set(candidates) == {"gene:PIK3CA", "gene:TP53"}


def test_every_exclusion_is_recorded_with_its_source():
    _, excluded = _get_candidate_nodes(
        _graph(),
        "disease:d",
        frame=frozenset({"PIK3CA", "TP53"}),
    )
    assert excluded == [
        {"node_id": "gene:LATER", "symbol": "LATER", "source": "Monarch"},
    ]


def test_an_empty_frame_excludes_every_gene_and_records_them_all():
    candidates, excluded = _get_candidate_nodes(
        _graph(),
        "disease:d",
        frame=frozenset(),
    )
    assert candidates == []
    assert len(excluded) == 3


def test_the_disease_node_is_never_a_candidate_or_an_exclusion():
    _, excluded = _get_candidate_nodes(
        _graph(),
        "disease:d",
        frame=frozenset({"PIK3CA"}),
    )
    assert all(e["node_id"] != "disease:d" for e in excluded)


def test_non_gene_nodes_are_neither_candidates_nor_exclusions():
    G = _graph()
    G.add_node("pathway:1", node_type="pathway", name="p", source="KEGG")
    candidates, excluded = _get_candidate_nodes(
        G,
        "disease:d",
        frame=frozenset({"PIK3CA"}),
    )
    assert "pathway:1" not in candidates
    assert all(e["node_id"] != "pathway:1" for e in excluded)


def test_pathogen_genes_are_gated_like_any_other_candidate():
    G = _graph()
    G.add_node("gene:POL", node_type="pathogen_gene", name="POL", source="ChEMBL")
    candidates, excluded = _get_candidate_nodes(
        G,
        "disease:d",
        frame=frozenset({"PIK3CA"}),
    )
    assert "gene:POL" not in candidates
    assert any(e["symbol"] == "POL" for e in excluded)
