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
    # `reason` distinguishes a gene the release did not associate with
    # this disease from a pathogen node excluded by type -- which the
    # symbol comparison alone used to admit, since ChEMBL namespaces a
    # pathogen's node id but leaves its name the bare symbol.
    assert excluded == [
        {
            "node_id": "gene:LATER",
            "symbol": "LATER",
            "source": "Monarch",
            "reason": "off_frame",
        },
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


# ── Pathogen nodes, excluded by type rather than by symbol ──────────
#
# ChEMBL namespaces a pathogen's node id (`pathogen:<organism>:<SYMBOL>`)
# but leaves `name` as the bare symbol, and this gate compares names. So a
# pathogen whose symbol collides with a human frame gene was admitted --
# and P. falciparum DHFR then OUTRANKED human DHFR, on a score that is 60%
# today's clinical phase. The builder's gate catches these first, which
# made this one inert; the spec puts the invariant here, on the assembled
# graph, precisely so it does not depend on that.


def _graph_with_a_colliding_pathogen():
    G = _graph()
    G.add_node(
        "pathogen:plasmodium_falciparum:PIK3CA",
        node_type="pathogen_gene",
        name="PIK3CA",  # the bare symbol, colliding with the frame gene
        source="ChEMBL",
    )
    return G


def test_a_pathogen_node_is_excluded_even_when_its_symbol_is_in_the_frame():
    candidates, excluded = _get_candidate_nodes(
        _graph_with_a_colliding_pathogen(),
        "disease:d",
        frame=frozenset({"PIK3CA", "TP53"}),
    )
    assert "pathogen:plasmodium_falciparum:PIK3CA" not in candidates
    # The human gene of the same name is unaffected.
    assert "gene:PIK3CA" in candidates
    by_id = {e["node_id"]: e for e in excluded}
    assert by_id["pathogen:plasmodium_falciparum:PIK3CA"]["reason"] == (
        "excluded_by_type"
    )
    assert by_id["pathogen:plasmodium_falciparum:PIK3CA"]["source"] == "ChEMBL"


def test_an_undated_run_still_evaluates_pathogen_targets():
    """The exclusion is about dated provenance, not about pathogens.

    Undated behaviour is unchanged: nothing pins a pathogen target, so a
    dated build cannot score one honestly, but a live build can.
    """
    candidates, _ = _get_candidate_nodes(
        _graph_with_a_colliding_pathogen(), "disease:d"
    )
    assert "pathogen:plasmodium_falciparum:PIK3CA" in candidates
