"""
Pearl's backdoor criterion, and an honest verdict when it cannot be met.

Z satisfies the backdoor criterion relative to (X, Y) when no node in Z
is a descendant of X, and Z blocks every path between X and Y that
starts with an arrow into X (Pearl, *Causality*, 2nd ed., Def. 3.3.1).
Both halves matter: the descendant condition rules out conditioning on
a mediator or a collider downstream of the treatment, and blocking only
the *backdoor* paths -- never the causal path itself -- is what a
plain d-separation test does not do for free. To test the second half
with a generic d-separation routine, the edges leaving the treatment
are removed first: that deletes the causal path (and anything X causes)
while leaving every backdoor path, the ones entering X through an
arrow, exactly as they were.

This module reports *why* identification failed, not merely that it did.
The reasons are a closed enum so that a run's failures aggregate into a
distribution rather than a pile of prose.

A note on trivial identification. When no confounder appears in the
graph, the empty set satisfies the criterion and the effect is
identifiable -- but a knowledge graph is open-world, and a missing edge
is missing knowledge rather than evidence of no confounding. So a
trivial verdict is reported under its own reason and carries
``n_near_miss_confounders``: how many nodes regulate the treatment and
have *some* association with the outcome that fell short of admissibility.
A target with many near misses is fragile. This is the same move as an
E-value in observational epidemiology -- not "there is no confounding"
but "here is how much it would take to break this".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import networkx as nx

from neorx.core.causal.graph_semantics import acyclic_core, causal_subgraph

# A larger bound buys little and costs combinatorially: the number of
# candidate subsets grows as C(pool, size). Reaching the bound is
# reported via Identification.search_truncated, never hidden.
MAX_ADJUSTMENT_SET_SIZE = 3
MAX_CANDIDATE_POOL = 20

__all__ = [
    "MAX_ADJUSTMENT_SET_SIZE",
    "MAX_CANDIDATE_POOL",
    "Identification",
    "IdentificationReason",
    "confounding_sensitivity",
    "find_adjustment_set",
    "mutilated_graph",
    "satisfies_backdoor",
]


class IdentificationReason(str, Enum):
    """Why identification succeeded or failed. A closed set."""

    IDENTIFIABLE_BY_ADJUSTMENT = "identifiable_by_adjustment"
    IDENTIFIABLE_TRIVIALLY = "identifiable_trivially"
    NO_CAUSAL_PATH = "no_causal_path"
    CYCLIC_COMPONENT = "cyclic_component"
    NO_VALID_ADJUSTMENT_SET = "no_valid_adjustment_set"
    TREATMENT_ABSENT = "treatment_absent"
    OUTCOME_ABSENT = "outcome_absent"


@dataclass(frozen=True)
class Identification:
    """The verdict on one (treatment, outcome) pair."""

    identifiable: bool
    adjustment_set: tuple[str, ...]
    reason: IdentificationReason
    search_truncated: bool = False
    n_near_miss_confounders: int = 0


def mutilated_graph(G: nx.DiGraph, treatment: str) -> nx.DiGraph:
    """G with every edge into ``treatment`` removed. G is not modified."""
    mutilated = G.copy()
    mutilated.remove_edges_from(list(G.in_edges(treatment)))
    return mutilated


def satisfies_backdoor(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    adjustment: frozenset[str],
) -> bool:
    """Whether ``adjustment`` satisfies the backdoor criterion.

    ``G`` must be a DAG containing both nodes; callers reach this through
    ``find_adjustment_set``, which guarantees that.
    """
    if treatment in adjustment or outcome in adjustment:
        return False

    if adjustment & nx.descendants(G, treatment):
        return False

    # The backdoor criterion asks whether Z blocks every path between
    # X and Y that starts with an arrow *into* X -- it says nothing
    # about the causal path itself. Testing that with a generic
    # d-separation routine means removing X's *out*-edges first, so the
    # direct causal edge (and any directed path through it) cannot
    # register as "still connected" and mask an open backdoor path.
    # This is a different mutilation from ``mutilated_graph``, which
    # removes edges *into* the treatment for the interventional graph
    # G_X-bar used elsewhere (e.g. the do-operator's truncated
    # factorization); the two must not be confused.
    backdoor_graph = G.copy()
    backdoor_graph.remove_edges_from(list(G.out_edges(treatment)))

    return bool(
        nx.is_d_separator(
            backdoor_graph,
            {treatment},
            {outcome},
            set(adjustment),
        )
    )


def confounding_sensitivity(
    dag: nx.DiGraph,
    full_graph: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> int:
    """Count near-miss confounders of ``treatment`` and ``outcome``.

    A near miss is a node that reaches the treatment in the causal DAG
    and carries some edge to the outcome in the full graph that was not
    admissible -- a literature association, a shared pathway. Each one is
    a confounder the graph hints at but does not license adjusting for.
    """
    if treatment not in dag:
        return 0

    ancestors = nx.ancestors(dag, treatment)
    admissible_parents = set(dag.predecessors(outcome)) if outcome in dag else set()

    return sum(
        1
        for node in ancestors
        if node not in admissible_parents
        and full_graph.has_edge(node, outcome)
    )


def find_adjustment_set(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> Identification:
    """Search for a minimal valid backdoor adjustment set.

    ``G`` is the *full* disease graph. Restriction to the causal subgraph
    and to its acyclic core happens here, so no caller can forget it.
    """
    causal = causal_subgraph(G)
    core = acyclic_core(causal)
    dag = core.dag

    if treatment in core.excluded or outcome in core.excluded:
        return Identification(
            identifiable=False,
            adjustment_set=(),
            reason=IdentificationReason.CYCLIC_COMPONENT,
        )

    # TREATMENT_ABSENT / OUTCOME_ABSENT mean the node does not exist in
    # the graph at all -- a typo, a stale ID. That is a different fact
    # from a node that exists but has no causal-admissible edge: such a
    # node is dropped from ``dag`` by causal_subgraph, and reporting it
    # as "absent" would blame the query when the honest complaint is
    # that there is no causal path in what evidence exists. So presence
    # is checked against the full input graph, not against ``dag``.
    if treatment not in G:
        return Identification(
            identifiable=False,
            adjustment_set=(),
            reason=IdentificationReason.TREATMENT_ABSENT,
        )

    if outcome not in G:
        return Identification(
            identifiable=False,
            adjustment_set=(),
            reason=IdentificationReason.OUTCOME_ABSENT,
        )

    if (
        treatment not in dag
        or outcome not in dag
        or not nx.has_path(dag, treatment, outcome)
    ):
        return Identification(
            identifiable=False,
            adjustment_set=(),
            reason=IdentificationReason.NO_CAUSAL_PATH,
        )

    if satisfies_backdoor(dag, treatment, outcome, frozenset()):
        return Identification(
            identifiable=True,
            adjustment_set=(),
            reason=IdentificationReason.IDENTIFIABLE_TRIVIALLY,
            n_near_miss_confounders=confounding_sensitivity(
                dag, G, treatment, outcome,
            ),
        )

    descendants = nx.descendants(dag, treatment)
    pool = sorted(
        (
            node for node in dag.nodes()
            if node not in descendants
            and node not in (treatment, outcome)
        ),
        key=lambda n: (-dag.nodes[n].get("score", 0.0), n),
    )
    truncated = len(pool) > MAX_CANDIDATE_POOL
    pool = pool[:MAX_CANDIDATE_POOL]

    for size in range(1, MAX_ADJUSTMENT_SET_SIZE + 1):
        for candidate in combinations(pool, size):
            if satisfies_backdoor(dag, treatment, outcome, frozenset(candidate)):
                return Identification(
                    identifiable=True,
                    adjustment_set=tuple(sorted(candidate)),
                    reason=IdentificationReason.IDENTIFIABLE_BY_ADJUSTMENT,
                    search_truncated=truncated,
                )

    return Identification(
        identifiable=False,
        adjustment_set=(),
        reason=IdentificationReason.NO_VALID_ADJUSTMENT_SET,
        search_truncated=truncated or len(pool) > MAX_ADJUSTMENT_SET_SIZE,
    )
