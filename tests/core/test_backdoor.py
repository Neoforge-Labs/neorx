"""
Tests for Pearl's backdoor criterion.

The textbook confounding graph is X <- U -> D with X -> D: U is a common
cause of treatment and outcome, so the effect of X on D is identifiable
only after adjusting for U. Every test here is built from that shape or
a deliberate corruption of it.
"""

import networkx as nx
import pytest

from neorx.core.causal.backdoor import (
    Identification,
    IdentificationReason,
    confounding_sensitivity,
    find_adjustment_set,
    mutilated_graph,
    satisfies_backdoor,
)

REG = {"edge_type": "activates", "evidence_class": "regulatory"}
GEN = {"edge_type": "associated_with", "evidence_class": "genetic_association"}
LIT = {"edge_type": "associated_with", "evidence_class": "literature"}


def _confounded_graph() -> nx.DiGraph:
    """X <- U -> D, X -> D. U confounds; adjusting for U identifies."""
    G = nx.DiGraph()
    G.add_edge("U", "X", **REG)
    G.add_edge("U", "D", **GEN)
    G.add_edge("X", "D", **GEN)
    return G


def test_mutilated_graph_removes_edges_into_the_treatment_only():
    G = _confounded_graph()
    Gx = mutilated_graph(G, "X")
    assert ("U", "X") not in Gx.edges()
    assert ("X", "D") in Gx.edges()
    assert ("U", "D") in Gx.edges()
    # The original is untouched.
    assert ("U", "X") in G.edges()


def test_empty_set_does_not_satisfy_backdoor_when_a_confounder_is_open():
    G = _confounded_graph()
    assert not satisfies_backdoor(G, "X", "D", frozenset())


def test_confounder_satisfies_backdoor():
    G = _confounded_graph()
    assert satisfies_backdoor(G, "X", "D", frozenset({"U"}))


def test_adjustment_set_containing_a_descendant_of_the_treatment_is_rejected():
    G = _confounded_graph()
    G.add_edge("X", "M", **REG)   # M is a descendant of X
    G.add_edge("M", "D", **GEN)
    assert not satisfies_backdoor(G, "X", "D", frozenset({"U", "M"}))


def test_adjustment_set_containing_treatment_or_outcome_is_rejected():
    G = _confounded_graph()
    assert not satisfies_backdoor(G, "X", "D", frozenset({"X"}))
    assert not satisfies_backdoor(G, "X", "D", frozenset({"D"}))


def test_find_adjustment_set_identifies_by_adjustment():
    result = find_adjustment_set(_confounded_graph(), "X", "D")
    assert result == Identification(
        identifiable=True,
        adjustment_set=("U",),
        reason=IdentificationReason.IDENTIFIABLE_BY_ADJUSTMENT,
        search_truncated=False,
        n_near_miss_confounders=0,
    )


def test_find_adjustment_set_identifies_trivially_when_nothing_confounds():
    G = nx.DiGraph()
    G.add_edge("X", "D", **GEN)
    result = find_adjustment_set(G, "X", "D")
    assert result.identifiable
    assert result.adjustment_set == ()
    assert result.reason is IdentificationReason.IDENTIFIABLE_TRIVIALLY


def test_todays_star_topology_has_no_causal_path():
    # Every current gene -> disease edge is a literature association, so
    # the causal subgraph contains no path at all. This is the honest
    # verdict on the graph the builder produces today.
    G = nx.DiGraph()
    for gene in ("g1", "g2", "g3"):
        G.add_edge(gene, "d", **LIT)
    result = find_adjustment_set(G, "g1", "d")
    assert not result.identifiable
    assert result.reason is IdentificationReason.NO_CAUSAL_PATH


def test_target_in_a_feedback_loop_reports_cyclic_component():
    G = _confounded_graph()
    G.add_edge("X", "U", **REG)   # U -> X -> U is a 2-cycle
    result = find_adjustment_set(G, "X", "D")
    assert not result.identifiable
    assert result.reason is IdentificationReason.CYCLIC_COMPONENT


def test_feedback_loop_elsewhere_does_not_break_identification():
    G = _confounded_graph()
    G.add_edge("p", "q", **REG)
    G.add_edge("q", "p", **REG)   # an unrelated cycle
    result = find_adjustment_set(G, "X", "D")
    assert result.identifiable
    assert result.adjustment_set == ("U",)


def test_absent_treatment_and_outcome_are_named_separately():
    G = _confounded_graph()
    assert find_adjustment_set(G, "ZZZ", "D").reason is (
        IdentificationReason.TREATMENT_ABSENT
    )
    assert find_adjustment_set(G, "X", "ZZZ").reason is (
        IdentificationReason.OUTCOME_ABSENT
    )


def test_self_loop_is_a_cyclic_component():
    # A gene regulating itself -- autoregulation, common in OmniPath -- is
    # the purest cycle there is. It is its own strongly connected component
    # of size one, so a naive len(component) > 1 filter misses it entirely
    # and hands d-separation a graph that is not a DAG.
    G = nx.DiGraph()
    G.add_edge("X", "X", **REG)
    G.add_edge("X", "D", **GEN)
    result = find_adjustment_set(G, "X", "D")
    assert not result.identifiable
    assert result.reason is IdentificationReason.CYCLIC_COMPONENT


def test_search_truncation_is_reported_not_hidden():
    # A wide fan of confounders, none of which alone or in threes blocks
    # the path, forces the bound to be reached.
    G = nx.DiGraph()
    G.add_edge("X", "D", **GEN)
    for i in range(30):
        G.add_edge(f"U{i}", "X", **REG)
        G.add_edge(f"U{i}", "D", **GEN)
    result = find_adjustment_set(G, "X", "D")
    assert not result.identifiable
    assert result.reason is IdentificationReason.NO_VALID_ADJUSTMENT_SET
    assert result.search_truncated


def test_confounding_sensitivity_counts_near_miss_confounders():
    # W regulates X and is associated with D, but only by literature, so
    # it is not an admissible arrow -- it is exactly the kind of missing
    # knowledge a trivial verdict is exposed to.
    G = nx.DiGraph()
    G.add_edge("X", "D", **GEN)
    G.add_edge("W", "X", **REG)
    G.add_edge("W", "D", **LIT)

    result = find_adjustment_set(G, "X", "D")
    assert result.reason is IdentificationReason.IDENTIFIABLE_TRIVIALLY
    assert result.n_near_miss_confounders == 1


def test_confounding_sensitivity_is_zero_with_no_near_misses():
    G = nx.DiGraph()
    G.add_edge("X", "D", **GEN)
    result = find_adjustment_set(G, "X", "D")
    assert result.n_near_miss_confounders == 0


def test_identification_is_immutable():
    result = find_adjustment_set(_confounded_graph(), "X", "D")
    with pytest.raises(Exception):
        result.identifiable = False
