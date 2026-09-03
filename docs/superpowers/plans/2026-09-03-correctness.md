# Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make identifiability a reported outcome of the causal engine rather than an assumed precondition, and make the CEM planner optimise molecules that exist.

**Architecture:** Edge types partition into causal-admissible and associational; identification runs only on the causal subgraph, via a correct backdoor implementation that reports a typed verdict with a reason. Fabricated statistics are deleted. The RL planner's elite set is confirmed against real chemistry through a new batched environment method.

**Tech Stack:** Python 3.12+, networkx, pydantic v2, numpy, torch, RDKit, Gymnasium, pytest, uv (never pip), polars (never pandas).

**Spec:** `docs/superpowers/specs/2026-09-03-correctness-design.md`

## Global Constraints

- **No shims, no patch jobs.** Proper implementation only. A zero-padding, a
  silent truncation, a `try/except` that swallows a failure and continues, or a
  boolean flag added to make one caller behave differently are all rejected at
  review. If two things disagree, fail loudly and name both.
- **No `except Exception` that continues.** In identification code an exception
  is a bug. Let it propagate.
- `MAX_ADJUSTMENT_SET_SIZE = 3`; candidate pool capped at `MAX_CANDIDATE_POOL = 20`
  nodes, ordered by descending node `score`. A truncated search sets
  `search_truncated=True` and is never presented as exhaustive.
- `IdentificationReason` is a closed enum of exactly seven values:
  `identifiable_by_adjustment`, `identifiable_trivially`, `no_causal_path`,
  `cyclic_component`, `no_valid_adjustment_set`, `treatment_absent`,
  `outcome_absent`.
- The networkx floor stays `networkx>=3.3` unless `scripts/check_floors.sh`
  fails; raise it only on that evidence, with a verification comment in the
  style of the existing `pyproject.toml` entries.
- No module in `src/neorx/core/causal/` or `src/neorx/causalbiorl/envs/`
  exceeds 600 lines.
- Statistics gate is scoped to `src/neorx/core/causal/` only. The genuine KS
  p-value at `src/neorx/genmol/evaluation/distribution.py:114` must not be
  flagged.
- `robustness_score` and `_sensitivity_analysis` are correct and stay.
- Run tests with `.venv/bin/python -m pytest`. Install with `uv pip install`.
- Commit messages carry no AI attribution and no `Co-Authored-By` trailer.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/neorx/core/causal/graph_semantics.py` | **New.** Which edges are causal-admissible; causal subgraph extraction; acyclic core |
| `src/neorx/core/causal/backdoor.py` | **New.** `Identification`, mutilated graph, backdoor test, adjustment-set search, confounding sensitivity |
| `src/neorx/core/causal/scoring.py` | **New.** Druggability, causal confidence, target classification — moved out of `identifier.py` |
| `src/neorx/core/causal/evidence.py` | **New.** Source scores, pathway/PPI counts, corroboration factor — moved out of `identifier.py` |
| `src/neorx/core/causal/identifier.py` | Orchestration only, after the moves |
| `src/neorx/core/sources/omnipath.py` | **New.** Directed signed regulatory interactions |
| `src/neorx/core/graph/models.py` | `GraphEdge` gains `evidence_class`, `sign`, `primary_sources`; `NeoRxResult` loses fabricated fields, gains identification |
| `src/neorx/causalbiorl/envs/generation.py` | **New.** Batched latent decode |
| `src/neorx/causalbiorl/envs/screening.py` | **New.** Per-objective molecule screening — moved out of `drug_discovery.py` |
| `src/neorx/core/pipeline/` | **New package.** `__init__.py` re-exports; `rl_stage.py` holds the RL orchestration |
| `src/neorx/experiments/gates.py` | Gains the fabricated-statistics gate |

---

## Task 1: Edge provenance and evidence class

**Files:**
- Modify: `src/neorx/core/graph/models.py:113-126` (`GraphEdge`)
- Modify: `src/neorx/core/graph/graph_builder.py:311-320` (`disease_graph_to_networkx`)
- Test: `tests/core/test_graph_semantics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `GraphEdge.evidence_class: str`, `GraphEdge.sign: int`,
  `GraphEdge.primary_sources: list[str]`. These three attributes appear on the
  NetworkX graph under the same names.

Policy lives in Task 2; this task only carries the data. `evidence_class` is
one of `""`, `"genetic_association"`, `"somatic_mutation"`, `"literature"`,
`"regulatory"`. `sign` is `+1` stimulation, `-1` inhibition, `0` unknown.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_graph_semantics.py
from neorx.core.graph.models import (
    DiseaseGraph, GraphEdge, GraphNode, EdgeType, NodeType,
)
from neorx.core.graph.graph_builder import disease_graph_to_networkx


def _graph_with_edge(**edge_kwargs) -> DiseaseGraph:
    return DiseaseGraph(
        disease_name="test",
        nodes=[
            GraphNode(node_id="gene:A", name="A", node_type=NodeType.GENE,
                      source="X", score=0.9),
            GraphNode(node_id="disease:test", name="test",
                      node_type=NodeType.DISEASE, source="X", score=1.0),
        ],
        edges=[GraphEdge(source_id="gene:A", target_id="disease:test",
                         **edge_kwargs)],
    )


def test_graph_edge_defaults_are_unclassified_unsigned_and_unsourced():
    edge = GraphEdge(
        source_id="gene:A", target_id="disease:test",
        edge_type=EdgeType.ASSOCIATED_WITH,
    )
    assert edge.evidence_class == ""
    assert edge.sign == 0
    assert edge.primary_sources == []


def test_evidence_class_sign_and_primary_sources_reach_the_networkx_graph():
    graph = _graph_with_edge(
        edge_type=EdgeType.ASSOCIATED_WITH,
        weight=0.7,
        source_db="Open Targets",
        evidence_class="genetic_association",
        sign=-1,
        primary_sources=["SIGNOR", "TRRUST"],
    )
    G = disease_graph_to_networkx(graph)
    attrs = G.edges["gene:A", "disease:test"]
    assert attrs["evidence_class"] == "genetic_association"
    assert attrs["sign"] == -1
    assert attrs["primary_sources"] == ["SIGNOR", "TRRUST"]


def test_sign_rejects_values_outside_minus_one_to_one():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GraphEdge(
            source_id="gene:A", target_id="disease:test",
            edge_type=EdgeType.ASSOCIATED_WITH, sign=2,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_graph_semantics.py -v`
Expected: FAIL — `ValidationError` for the unexpected `evidence_class` keyword, or `KeyError: 'evidence_class'`.

- [ ] **Step 3: Add the fields to `GraphEdge`**

In `src/neorx/core/graph/models.py`, inside `class GraphEdge`, after the
`pmids` field:

```python
    evidence_class: str = Field(
        "",
        description=(
            "What kind of evidence backs this edge: genetic_association, "
            "somatic_mutation, literature, regulatory, or empty if unclassified. "
            "Causal admissibility is derived from this plus edge_type -- see "
            "neorx.core.causal.graph_semantics."
        ),
    )
    sign: int = Field(
        0,
        description="+1 stimulation, -1 inhibition, 0 unknown or unsigned",
        ge=-1, le=1,
    )
    primary_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Primary databases asserting this interaction. An aggregator such "
            "as OmniPath records the databases it drew from here, so multi-source "
            "corroboration counts distinct primary evidence rather than distinct "
            "aggregators."
        ),
    )
```

- [ ] **Step 4: Propagate them onto the NetworkX graph**

In `src/neorx/core/graph/graph_builder.py`, in `disease_graph_to_networkx`,
replace the `G.add_edge(...)` call with:

```python
        G.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            source_db=edge.source_db,
            evidence=edge.evidence or "",
            evidence_class=edge.evidence_class,
            sign=edge.sign,
            primary_sources=list(edge.primary_sources),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/test_graph_semantics.py tests/core/test_graph_builder.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/neorx/core/graph/models.py src/neorx/core/graph/graph_builder.py tests/core/test_graph_semantics.py
git commit -m "feat: carry evidence class, sign, and primary sources on graph edges"
```

---

## Task 2: Graph semantics — the causal subgraph

**Files:**
- Create: `src/neorx/core/causal/graph_semantics.py`
- Test: `tests/core/test_graph_semantics.py` (append)

**Interfaces:**
- Consumes: edge attributes `edge_type`, `evidence_class` from Task 1.
- Produces:
  - `CAUSAL_EDGE_TYPES: frozenset[str]`
  - `GENETIC_EVIDENCE_CLASSES: frozenset[str]`
  - `is_causal_admissible(attrs: dict) -> bool`
  - `causal_subgraph(G: nx.DiGraph) -> nx.DiGraph`
  - `cyclic_components(G: nx.DiGraph) -> list[frozenset[str]]`
  - `AcyclicCore` (frozen dataclass: `dag: nx.DiGraph`, `excluded: frozenset[str]`)
  - `acyclic_core(G: nx.DiGraph) -> AcyclicCore`

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_graph_semantics.py`:

```python
import networkx as nx
from neorx.core.causal.graph_semantics import (
    acyclic_core,
    causal_subgraph,
    cyclic_components,
    is_causal_admissible,
)


def test_regulatory_gene_gene_edge_is_admissible():
    assert is_causal_admissible(
        {"edge_type": "activates", "evidence_class": "regulatory"}
    )


def test_gene_disease_edge_is_admissible_only_with_genetic_evidence():
    genetic = {"edge_type": "associated_with",
               "evidence_class": "genetic_association"}
    somatic = {"edge_type": "associated_with",
               "evidence_class": "somatic_mutation"}
    literature = {"edge_type": "associated_with",
                  "evidence_class": "literature"}
    assert is_causal_admissible(genetic)
    assert is_causal_admissible(somatic)
    assert not is_causal_admissible(literature)


def test_unclassified_association_is_not_admissible():
    assert not is_causal_admissible(
        {"edge_type": "associated_with", "evidence_class": ""}
    )


def test_string_interaction_is_never_admissible():
    # STRING data is undirected; its arrow direction is an artefact of
    # which protein landed in the first response column.
    assert not is_causal_admissible(
        {"edge_type": "interacts_with", "evidence_class": "regulatory"}
    )


def test_pathway_membership_is_never_admissible():
    assert not is_causal_admissible(
        {"edge_type": "participates_in", "evidence_class": ""}
    )


def test_causal_subgraph_keeps_admissible_edges_and_drops_the_rest():
    G = nx.DiGraph()
    G.add_edge("gene:U", "gene:X", edge_type="activates",
               evidence_class="regulatory")
    G.add_edge("gene:X", "disease:d", edge_type="associated_with",
               evidence_class="genetic_association")
    G.add_edge("gene:P", "disease:d", edge_type="associated_with",
               evidence_class="literature")
    G.add_edge("gene:X", "pathway:1", edge_type="participates_in",
               evidence_class="")

    sub = causal_subgraph(G)

    assert set(sub.edges()) == {("gene:U", "gene:X"), ("gene:X", "disease:d")}
    assert "pathway:1" not in sub
    assert "gene:P" not in sub


def test_causal_subgraph_preserves_edge_attributes():
    G = nx.DiGraph()
    G.add_edge("gene:U", "gene:X", edge_type="inhibits",
               evidence_class="regulatory", sign=-1,
               primary_sources=["SIGNOR"], weight=0.8)
    sub = causal_subgraph(G)
    assert sub.edges["gene:U", "gene:X"]["sign"] == -1
    assert sub.edges["gene:U", "gene:X"]["primary_sources"] == ["SIGNOR"]


def test_cyclic_components_reports_only_nontrivial_sccs():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "a")
    G.add_edge("b", "c")
    assert cyclic_components(G) == [frozenset({"a", "b"})]


def test_acyclic_core_removes_cyclic_nodes_and_names_them():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "a")
    G.add_edge("b", "c")
    G.add_edge("c", "d")

    core = acyclic_core(G)

    assert core.excluded == frozenset({"a", "b"})
    assert nx.is_directed_acyclic_graph(core.dag)
    assert set(core.dag.nodes()) == {"c", "d"}


def test_acyclic_core_of_a_dag_excludes_nothing():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    core = acyclic_core(G)
    assert core.excluded == frozenset()
    assert set(core.dag.nodes()) == {"a", "b"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/core/test_graph_semantics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.core.causal.graph_semantics'`

- [ ] **Step 3: Write the module**

Create `src/neorx/core/causal/graph_semantics.py`:

```python
"""
Which edges may carry a causal claim.

A disease graph assembled from public sources mixes two kinds of edge.
Some assert a mechanism with a direction: OmniPath's signed regulatory
interactions, where a curator has said that A acts on B. Others assert
only that two things occur together: a literature co-mention, membership
of a shared pathway, an undirected protein interaction whose arrow exists
because one protein landed in the first column of an API response.

Do-calculus over the second kind assumes the very thing a causal method
is supposed to establish. So this module partitions the edge types and
identification runs only on the causal-admissible subgraph. Associational
edges keep contributing to evidence scoring; they stop being arrows.

Gene -> disease is admitted only on genetic evidence. A germline variant
is randomised at conception, so a genetic association carries a
natural-experiment warrant that a co-mention does not -- the Mendelian
randomisation argument (Davey Smith & Ebrahim, 2003, Int J Epidemiol
32:1-22). That warrant is what earns the arrow.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

# Directed, mechanistic, with an asserted direction.
CAUSAL_EDGE_TYPES: frozenset[str] = frozenset({
    "activates",
    "inhibits",
    "phosphorylates",
    "regulates",
    "upregulates",
    "downregulates",
    "causes",
})

# Gene -> disease edge types, admissible only under genetic evidence.
_ASSOCIATION_EDGE_TYPES: frozenset[str] = frozenset({"associated_with"})

GENETIC_EVIDENCE_CLASSES: frozenset[str] = frozenset({
    "genetic_association",
    "somatic_mutation",
})

__all__ = [
    "CAUSAL_EDGE_TYPES",
    "GENETIC_EVIDENCE_CLASSES",
    "AcyclicCore",
    "acyclic_core",
    "causal_subgraph",
    "cyclic_components",
    "is_causal_admissible",
]


def is_causal_admissible(attrs: dict) -> bool:
    """Whether an edge with these attributes may carry a causal claim.

    Parameters
    ----------
    attrs
        Edge attribute mapping, as stored on the NetworkX graph. Reads
        ``edge_type`` and ``evidence_class``.
    """
    edge_type = attrs.get("edge_type", "")
    if edge_type in CAUSAL_EDGE_TYPES:
        return True
    if edge_type in _ASSOCIATION_EDGE_TYPES:
        return attrs.get("evidence_class", "") in GENETIC_EVIDENCE_CLASSES
    return False


def causal_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """The subgraph of causal-admissible edges, attributes preserved.

    Nodes left with no admissible edge are dropped: a node that
    participates in no causal relation cannot be a treatment, an
    outcome, or a confounder.
    """
    admissible = [
        (u, v) for u, v, attrs in G.edges(data=True)
        if is_causal_admissible(attrs)
    ]
    return G.edge_subgraph(admissible).copy()


def cyclic_components(G: nx.DiGraph) -> list[frozenset[str]]:
    """Strongly connected components of more than one node."""
    return [
        frozenset(component)
        for component in nx.strongly_connected_components(G)
        if len(component) > 1
    ]


@dataclass(frozen=True)
class AcyclicCore:
    """A DAG carved out of a possibly-cyclic graph, and what was removed.

    ``excluded`` names every node that sat in a feedback loop. Those
    nodes are not silently repaired: d-separation is undefined on them,
    and identification reports ``cyclic_component`` rather than guessing
    an acyclic orientation.
    """

    dag: nx.DiGraph
    excluded: frozenset[str]


def acyclic_core(G: nx.DiGraph) -> AcyclicCore:
    """Remove every node in a non-trivial strongly connected component.

    Biological regulatory networks are full of feedback, and
    ``nx.is_d_separator`` requires a DAG of the whole graph -- not merely
    of the neighbourhood under test. So cyclic nodes come out entirely,
    and the caller is told which.
    """
    cyclic_nodes: set[str] = set()
    for component in cyclic_components(G):
        cyclic_nodes |= set(component)

    dag = G.copy()
    dag.remove_nodes_from(cyclic_nodes)
    return AcyclicCore(dag=dag, excluded=frozenset(cyclic_nodes))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/test_graph_semantics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/core/causal/graph_semantics.py tests/core/test_graph_semantics.py
git commit -m "feat: partition graph edges into causal-admissible and associational"
```

---

## Task 3: The backdoor criterion

**Files:**
- Create: `src/neorx/core/causal/backdoor.py`
- Test: `tests/core/test_backdoor.py`

**Interfaces:**
- Consumes: `acyclic_core`, `causal_subgraph` from Task 2.
- Produces:
  - `IdentificationReason` (str Enum, seven members)
  - `Identification` (frozen dataclass: `identifiable: bool`,
    `adjustment_set: tuple[str, ...]`, `reason: IdentificationReason`,
    `search_truncated: bool`, `n_near_miss_confounders: int`)
  - `mutilated_graph(G, treatment) -> nx.DiGraph`
  - `satisfies_backdoor(G, treatment, outcome, adjustment) -> bool`
  - `find_adjustment_set(G, treatment, outcome, full_graph=None) -> Identification`
  - `confounding_sensitivity(dag, full_graph, treatment, outcome) -> int`
  - `MAX_ADJUSTMENT_SET_SIZE = 3`, `MAX_CANDIDATE_POOL = 20`

`find_adjustment_set` takes the **full** graph and does the causal-subgraph and
acyclic-core extraction itself, so callers cannot forget.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_backdoor.py
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


def test_unblockable_backdoor_path_reports_no_valid_adjustment_set():
    # X <- U -> D where U itself is a descendant of X: adjusting for U is
    # forbidden, and nothing else blocks the path.
    G = nx.DiGraph()
    G.add_edge("X", "U", **REG)
    G.add_edge("U", "X2", **REG)
    G.add_edge("X2", "X", **REG)
    G.add_edge("U", "D", **GEN)
    G.add_edge("X", "D", **GEN)
    result = find_adjustment_set(G, "X", "D")
    assert not result.identifiable
    assert result.reason in {
        IdentificationReason.NO_VALID_ADJUSTMENT_SET,
        IdentificationReason.CYCLIC_COMPONENT,
    }


def test_search_truncation_is_reported_not_hidden():
    # A wide fan of confounders, none of which alone or in threes blocks
    # the path, forces the bound to be reached.
    G = nx.DiGraph()
    G.add_edge("X", "D", **GEN)
    for i in range(30):
        G.add_edge(f"U{i}", "X", **REG)
        G.add_edge(f"U{i}", "D", **GEN)
    result = find_adjustment_set(G, "X", "D")
    if not result.identifiable:
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/core/test_backdoor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.core.causal.backdoor'`

- [ ] **Step 3: Write the module**

Create `src/neorx/core/causal/backdoor.py`:

```python
"""
Pearl's backdoor criterion, and an honest verdict when it cannot be met.

Z satisfies the backdoor criterion relative to (X, Y) when no node in Z
is a descendant of X, and Z d-separates X from Y in the graph with all
edges into X removed (Pearl, *Causality*, 2nd ed., Def. 3.3.1). Both
halves matter: the descendant condition rules out conditioning on a
mediator or a collider downstream of the treatment, and the mutilated
graph is what distinguishes a backdoor path from the causal path itself.

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

    return nx.is_d_separator(
        mutilated_graph(G, treatment),
        {treatment},
        {outcome},
        set(adjustment),
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

    if treatment not in dag:
        return Identification(
            identifiable=False,
            adjustment_set=(),
            reason=IdentificationReason.TREATMENT_ABSENT,
        )

    if outcome not in dag:
        return Identification(
            identifiable=False,
            adjustment_set=(),
            reason=IdentificationReason.OUTCOME_ABSENT,
        )

    if not nx.has_path(dag, treatment, outcome):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/test_backdoor.py -v`
Expected: PASS

- [ ] **Step 5: Verify the networkx floor supplies `is_d_separator`**

Run: `bash scripts/check_floors.sh`
Expected: PASS with the declared `networkx>=3.3`.

If it fails with an `ImportError` or `AttributeError` on `is_d_separator`,
raise the floor in `pyproject.toml` to the smallest version that passes,
re-running this script at each candidate, and add a comment above the entry in
the style of the neighbouring pins, naming `is_d_separator` as the reason and
today's date as the verification date. Do not raise the floor without a failing
run to point at.

- [ ] **Step 6: Commit**

```bash
git add src/neorx/core/causal/backdoor.py tests/core/test_backdoor.py
git commit -m "feat: implement the backdoor criterion with a typed verdict"
```

---

## Task 4: Classify Open Targets gene–disease evidence

**Files:**
- Modify: `src/neorx/core/sources/open_targets.py:178-186` (the `GraphEdge` construction)
- Test: `tests/core/test_data_sources.py` (append)

**Interfaces:**
- Consumes: `GraphEdge.evidence_class` from Task 1.
- Produces: Open Targets `ASSOCIATED_WITH` edges carry
  `evidence_class="genetic_association"`, `"somatic_mutation"`, or
  `"literature"`, chosen from the `datatypeScores` the query already fetches.

No new API call. `dt_scores` is already built at `open_targets.py:151` and
stashed in node metadata; this task reads it for the edge as well.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_data_sources.py`:

```python
from neorx.core.sources.open_targets import _evidence_class_for


def test_genetic_association_wins_when_it_is_the_strongest_datatype():
    assert _evidence_class_for(
        {"genetic_association": 0.6, "literature": 0.2}
    ) == "genetic_association"


def test_somatic_mutation_is_admissible_evidence():
    assert _evidence_class_for(
        {"somatic_mutation": 0.5, "literature": 0.9}
    ) == "somatic_mutation"


def test_genetic_association_outranks_somatic_mutation_when_both_present():
    assert _evidence_class_for(
        {"genetic_association": 0.1, "somatic_mutation": 0.9}
    ) == "genetic_association"


def test_literature_only_association_is_classified_as_literature():
    assert _evidence_class_for(
        {"literature": 0.8, "rna_expression": 0.4}
    ) == "literature"


def test_absent_datatype_scores_yield_no_classification():
    assert _evidence_class_for({}) == ""


def test_zero_scored_genetic_evidence_does_not_count():
    # A datatype present with a zero score is not evidence.
    assert _evidence_class_for(
        {"genetic_association": 0.0, "literature": 0.7}
    ) == "literature"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_data_sources.py -k evidence_class -v`
Expected: FAIL — `ImportError: cannot import name '_evidence_class_for'`

- [ ] **Step 3: Add the classifier and use it**

In `src/neorx/core/sources/open_targets.py`, after the module constants
(`TIMEOUT = 30`):

```python
def _evidence_class_for(datatype_scores: dict[str, float]) -> str:
    """Pick the evidence class for a gene-disease edge.

    Open Targets breaks an association into datatypes. Two of them --
    genetic_association and somatic_mutation -- carry a
    natural-experiment warrant that lets the edge be treated as causal
    downstream (see neorx.core.causal.graph_semantics). Everything else
    is association. Genetic association is preferred over somatic
    mutation when both are present, being the stronger warrant: germline
    variants are randomised at conception, somatic ones are not.
    """
    for genetic in ("genetic_association", "somatic_mutation"):
        if datatype_scores.get(genetic, 0.0) > 0.0:
            return genetic
    if any(score > 0.0 for score in datatype_scores.values()):
        return "literature"
    return ""
```

Then in `query_open_targets`, replace the `edges.append(GraphEdge(...))` call
that follows the node append with:

```python
            edges.append(GraphEdge(
                source_id=node_id,
                target_id=disease_node_id,
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=min(score, 1.0),
                source_db="Open Targets",
                evidence=f"OT score: {score:.3f}",
                evidence_class=_evidence_class_for(dt_scores),
                primary_sources=["Open Targets"],
            ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/test_data_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/core/sources/open_targets.py tests/core/test_data_sources.py
git commit -m "feat: classify Open Targets gene-disease edges by evidence datatype"
```

---

## Task 5: OmniPath as a regulatory source

**Files:**
- Create: `src/neorx/core/sources/omnipath.py`
- Modify: `src/neorx/core/sources/__init__.py`
- Modify: `src/neorx/core/graph/graph_builder.py` (after the STRING block ending at line 202)
- Test: `tests/core/test_omnipath.py`

**Interfaces:**
- Consumes: `GraphEdge` fields from Task 1.
- Produces: `query_omnipath(gene_symbols: list[str], *, allow_mocks: bool = False)
  -> tuple[list[GraphNode], list[GraphEdge]]`, exported from
  `neorx.core.sources`. Emits `EdgeType.ACTIVATES` / `EdgeType.INHIBITS` /
  `EdgeType.REGULATES` gene→gene edges with `evidence_class="regulatory"`,
  `sign` set, and `primary_sources` from the response's `sources` list.

The endpoint is `https://omnipathdb.org/interactions`. Verified working
parameters: `genesymbols=yes`, `organisms=9606`, `datasets=omnipath`,
`fields=sources,references`, `format=json`. Note that `resources=SIGNOR`
returns an empty list — do not use it.

Only interactions where `is_directed` and `consensus_direction` are both true
become edges; an undirected or direction-disputed interaction is not an arrow.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_omnipath.py
"""
Tests for the OmniPath regulatory source.

OmniPath aggregates signed, directed causal interactions from SIGNOR,
TRRUST, and others. Only interactions the aggregator marks as directed
with a consensus direction may become arrows -- an undirected or
direction-disputed interaction carries no causal claim.
"""

from neorx.core.graph.models import EdgeType
from neorx.core.sources.omnipath import _interactions_to_edges


def _row(**overrides) -> dict:
    row = {
        "source_genesymbol": "A",
        "target_genesymbol": "B",
        "is_directed": True,
        "is_stimulation": True,
        "is_inhibition": False,
        "consensus_direction": True,
        "sources": ["SIGNOR", "TRRUST"],
        "references": "SIGNOR:16331690;TRRUST:14983059",
    }
    row.update(overrides)
    return row


def test_stimulation_becomes_a_positively_signed_activates_edge():
    edges = _interactions_to_edges([_row()], {"A", "B"})
    assert len(edges) == 1
    edge = edges[0]
    assert edge.source_id == "gene:A"
    assert edge.target_id == "gene:B"
    assert edge.edge_type is EdgeType.ACTIVATES
    assert edge.sign == 1
    assert edge.evidence_class == "regulatory"


def test_inhibition_becomes_a_negatively_signed_inhibits_edge():
    edges = _interactions_to_edges(
        [_row(is_stimulation=False, is_inhibition=True)], {"A", "B"},
    )
    assert edges[0].edge_type is EdgeType.INHIBITS
    assert edges[0].sign == -1


def test_directed_but_unsigned_interaction_becomes_an_unsigned_regulates_edge():
    edges = _interactions_to_edges(
        [_row(is_stimulation=False, is_inhibition=False)], {"A", "B"},
    )
    assert edges[0].edge_type is EdgeType.REGULATES
    assert edges[0].sign == 0


def test_undirected_interaction_is_not_an_arrow():
    assert _interactions_to_edges([_row(is_directed=False)], {"A", "B"}) == []


def test_direction_without_consensus_is_not_an_arrow():
    assert _interactions_to_edges(
        [_row(consensus_direction=False)], {"A", "B"},
    ) == []


def test_contradictory_sign_is_recorded_as_unsigned_regulation():
    # Curators disagree on the sign; the direction still holds, so the
    # arrow stands but makes no claim about which way it pushes.
    edges = _interactions_to_edges(
        [_row(is_stimulation=True, is_inhibition=True)], {"A", "B"},
    )
    assert edges[0].edge_type is EdgeType.REGULATES
    assert edges[0].sign == 0


def test_primary_sources_are_preserved_for_corroboration_counting():
    edges = _interactions_to_edges([_row()], {"A", "B"})
    assert edges[0].primary_sources == ["SIGNOR", "TRRUST"]
    assert edges[0].source_db == "OmniPath"


def test_pubmed_ids_are_extracted_from_the_references_field():
    edges = _interactions_to_edges([_row()], {"A", "B"})
    assert sorted(edges[0].pmids) == ["14983059", "16331690"]


def test_interactions_outside_the_requested_gene_set_are_dropped():
    # OmniPath returns a gene's whole neighbourhood; only edges between
    # genes already in the disease graph are relevant.
    assert _interactions_to_edges([_row(target_genesymbol="ZZZ")], {"A", "B"}) == []


def test_self_loops_are_dropped():
    assert _interactions_to_edges([_row(target_genesymbol="A")], {"A"}) == []


def test_missing_gene_symbols_are_dropped_rather_than_guessed():
    assert _interactions_to_edges([_row(source_genesymbol="")], {"A", "B"}) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/core/test_omnipath.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.core.sources.omnipath'`

- [ ] **Step 3: Write the module**

Create `src/neorx/core/sources/omnipath.py`:

```python
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
                "genes": ",".join(sorted(known)),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/test_omnipath.py -v`
Expected: PASS

- [ ] **Step 5: Export it from the sources package**

In `src/neorx/core/sources/__init__.py`, add the import after the
`string_db` import:

```python
from .omnipath import query_omnipath
```

and add `"query_omnipath"` to `__all__`.

- [ ] **Step 6: Wire it into the graph builder**

In `src/neorx/core/graph/graph_builder.py`, add `query_omnipath` to the
`from neorx.core.sources import (...)` block. Then, immediately after the
STRING block (the line `logger.info("  STRING: %d nodes, %d edges.", ...)`),
insert:

```python
    # ── Step 3b: Directed Regulatory Interactions ───────────────

    logger.info("Querying OmniPath regulatory interactions…")
    omni_nodes, omni_edges = query_omnipath(
        gene_symbols, allow_mocks=allow_mocks,
    )
    all_nodes.extend(omni_nodes)
    all_edges.extend(omni_edges)
    sources_queried.append("OmniPath")
    logger.info("  OmniPath: %d directed edges.", len(omni_edges))
```

- [ ] **Step 7: Run the graph builder tests**

Run: `.venv/bin/python -m pytest tests/core/test_graph_builder.py tests/test_public_api.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/neorx/core/sources/omnipath.py src/neorx/core/sources/__init__.py src/neorx/core/graph/graph_builder.py tests/core/test_omnipath.py
git commit -m "feat: add OmniPath as a directed regulatory graph source"
```

---

## Task 6: Wire identification into the identifier

**Files:**
- Modify: `src/neorx/core/causal/identifier.py:270-276` (adjustment set and identifiability)
- Modify: `src/neorx/core/causal/identifier.py:623-672` (delete `_compute_adjustment_set`)
- Modify: `src/neorx/core/graph/models.py` (`NeoRxResult`: add identification fields)
- Modify: `tests/core/test_identifier.py` (drop the `_compute_adjustment_set` import)
- Test: `tests/core/test_identifier.py` (append)

**Interfaces:**
- Consumes: `find_adjustment_set`, `Identification`, `IdentificationReason` from Task 3.
- Produces: `NeoRxResult.identifiable: bool`,
  `NeoRxResult.identification_reason: str`,
  `NeoRxResult.adjustment_set: list[str]` (existing field, now meaningful),
  `NeoRxResult.n_near_miss_confounders: int`,
  `NeoRxResult.adjustment_search_truncated: bool`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_identifier.py`:

```python
class TestIdentificationIsReported:
    """The verdict, not an assumption.

    Identifiability used to be `len(causal_pathway) > 0` -- the existence
    of any path -- while the adjustment set was computed and discarded.
    These tests pin the replacement.
    """

    def test_result_carries_the_identification_reason(self, hiv_graph):
        from neorx.core.causal.backdoor import IdentificationReason

        targets = identify_causal_targets(hiv_graph, top_n=5)
        assert targets
        valid = {r.value for r in IdentificationReason}
        for t in targets:
            assert t.identification_reason in valid

    def test_non_identifiable_targets_have_an_empty_adjustment_set(self, hiv_graph):
        targets = identify_causal_targets(hiv_graph, top_n=10)
        for t in targets:
            if not t.identifiable:
                assert t.adjustment_set == []

    def test_identifiability_no_longer_tracks_mere_path_existence(self, hiv_graph):
        # A literature-only graph has paths in the full graph but none in
        # the causal subgraph, so nothing may be identifiable.
        targets = identify_causal_targets(hiv_graph, top_n=10)
        for t in targets:
            if t.identification_reason == "no_causal_path":
                assert not t.identifiable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_identifier.py -k IdentificationIsReported -v`
Expected: FAIL — `AttributeError: 'NeoRxResult' object has no attribute 'identification_reason'`

- [ ] **Step 3: Add the fields to `NeoRxResult`**

In `src/neorx/core/graph/models.py`, in `class NeoRxResult`, replace the
`adjustment_set` field and add the neighbours, so the causal block reads:

```python
    causal_confidence: float = Field(0.0, description="Combined confidence 0-1", ge=0.0, le=1.0)
    identifiable: bool = Field(
        False,
        description="Whether the backdoor criterion is satisfied for this target",
    )
    identification_reason: str = Field(
        "",
        description=(
            "Why identification succeeded or failed -- an "
            "IdentificationReason value. See neorx.core.causal.backdoor."
        ),
    )
    adjustment_set: list[str] = Field(
        default_factory=list,
        description="Variables that must be adjusted for (backdoor); empty when not identifiable or trivially identifiable",
    )
    n_near_miss_confounders: int = Field(
        0,
        description=(
            "For a trivially identifiable target: how many nodes regulate it "
            "and have some non-admissible association with the disease. A "
            "sensitivity measure on knowledge-graph incompleteness."
        ),
    )
    adjustment_search_truncated: bool = Field(
        False,
        description="Whether the adjustment-set search hit its size or pool bound",
    )
```

- [ ] **Step 4: Replace the adjustment-set computation**

In `src/neorx/core/causal/identifier.py`, add to the imports:

```python
from neorx.core.causal.backdoor import find_adjustment_set
```

Replace lines 270-276 (`# Compute adjustment set (simplified backdoor criterion)`
through `is_identifiable = len(causal_pathway) > 0`) with:

```python
    # Identification: does the causal subgraph license a claim here?
    identification = find_adjustment_set(G, target_id, disease_id)
    adjustment_set = list(identification.adjustment_set)
    is_identifiable = identification.identifiable
```

Then in the same function's `NeoRxResult(...)` construction (around line 347),
add alongside `adjustment_set=adjustment_set`:

```python
        identifiable=identification.identifiable,
        identification_reason=identification.reason.value,
        n_near_miss_confounders=identification.n_near_miss_confounders,
        adjustment_search_truncated=identification.search_truncated,
```

- [ ] **Step 5: Delete `_compute_adjustment_set`**

Delete the whole function at `src/neorx/core/causal/identifier.py:623-672`,
and remove `_compute_adjustment_set` from the import list at
`tests/core/test_identifier.py:20`.

- [ ] **Step 6: Give the pathogen path an identification too**

The pathogen branch constructs a `NeoRxResult` at around line 518 with
`adjustment_set=[]`. Pathogen targets are not in the human causal subgraph, so
add beside it:

```python
        identifiable=False,
        identification_reason="treatment_absent",
```

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest tests/core/ -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/neorx/core/causal/identifier.py src/neorx/core/graph/models.py tests/core/test_identifier.py
git commit -m "feat: report a backdoor identification verdict per target"
```

---

## Task 7: Delete the fabricated statistics

**Files:**
- Modify: `src/neorx/core/causal/identifier.py` (`_estimate_causal_effect`, `_bootstrap_confidence_interval`, both call sites, both `NeoRxResult` constructions)
- Modify: `src/neorx/core/causal/counterfactual.py:66,130,140,148,158,282-312`
- Modify: `src/neorx/core/graph/models.py` (`NeoRxResult.causal_effect`, `NeoRxResult.confidence_interval`)
- Modify: `src/neorx/core/api.py:169`
- Modify: `src/neorx/core/__main__.py:238-242`
- Modify: `src/neorx/core/report.py:156,163,245-246,360`
- Modify: `src/neorx/core/templates/report.html:283,310`
- Modify: `tests/core/test_new_modules.py:229-254`
- Modify: `tests/core/test_identifier.py:155,168,181`

**Interfaces:**
- Consumes: `identification_reason` from Task 6.
- Produces: `NeoRxResult` and `CounterfactualResult` without `causal_effect`,
  `confidence_interval`. `_estimate_causal_effect` becomes
  `_estimate_evidence_score(G, treatment, outcome, adjustment_set) -> float`,
  returning a single score and no p-value.

The report tables do not lose a column — the CI column becomes an
identifiability column, which is the number this sub-project exists to
surface.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_new_modules.py`:

```python
def test_neorx_result_has_no_fabricated_statistics():
    """No effect size, no p-value, no confidence interval.

    All three were computed from heuristic scores with no data behind
    them: the CI was percentiles of hand-chosen Gaussian noise, and the
    p-value was a rescaling of the effect. robustness_score stays --
    leave-one-source-out is a real measurement.
    """
    from neorx.core.graph.models import NeoRxResult

    fields = set(NeoRxResult.model_fields)
    assert "causal_effect" not in fields
    assert "confidence_interval" not in fields
    assert "p_value" not in fields
    assert "robustness_score" in fields


def test_counterfactual_result_has_no_confidence_interval():
    from neorx.core.causal.counterfactual import CounterfactualResult

    assert "confidence_interval" not in set(
        CounterfactualResult.__dataclass_fields__
        if hasattr(CounterfactualResult, "__dataclass_fields__")
        else CounterfactualResult.model_fields
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_new_modules.py -k fabricated -v`
Expected: FAIL — `assert 'causal_effect' not in fields`

- [ ] **Step 3: Rename the estimator and drop the p-value**

In `src/neorx/core/causal/identifier.py`, rename `_estimate_causal_effect` to
`_estimate_evidence_score`, change its return annotation to `float`, delete the
`p_value` computation and return only `effect`, renaming the local to
`evidence`. Replace the docstring's `Returns` block with:

```
    Returns
    -------
    float
        A weighted multi-source evidence score. This is not a causal
        effect size: no interventional or patient-level data enters it.
        It ranks candidates; it does not estimate a magnitude.
```

Update the call site at line 278 to:

```python
    evidence_score = _estimate_evidence_score(
        G, target_id, disease_id, adjustment_set,
    )
```

and update the `_sensitivity_analysis` call on the next line to pass
`evidence_score`. Replace every later use of `causal_effect` in the function
with `evidence_score`, and drop `causal_effect=` from both `NeoRxResult`
constructions.

- [ ] **Step 4: Delete both fake bootstraps**

Delete `_bootstrap_confidence_interval` entirely from
`src/neorx/core/causal/identifier.py`, along with the `ci_lo, ci_hi = ...` call
sites and the `confidence_interval=(ci_lo, ci_hi)` arguments in both
`NeoRxResult` constructions.

Delete `CounterfactualValidator._bootstrap_ci` from
`src/neorx/core/causal/counterfactual.py`, the `ci_lo, ci_hi = self._bootstrap_ci(...)`
call at line 130, the `confidence_interval` field at line 66, and the
`confidence_interval=(round(ci_lo, 4), round(ci_hi, 4))` argument at line 158.
Replace the two reasoning strings so they no longer interpolate an interval:

```python
        if would_help:
            reasoning = (
                f"Counterfactual analysis: inhibiting {gene} reduces "
                f"disease score by {cf_effect:.3f} "
                f"(factual={factual:.3f} → counterfactual="
                f"{counterfactual:.3f}).  "
                f"Intervention would likely reduce disease severity."
            )
        else:
            reasoning = (
                f"Counterfactual analysis: inhibiting {gene} has "
                f"minimal effect (ΔY={cf_effect:.3f}).  "
                f"This target may be correlational rather than causal."
            )
```

If `self._n_bootstrap` and `self._rng` become unused after the deletion, remove
them from `__init__` and from the class docstring.

- [ ] **Step 5: Remove the fields from the models**

In `src/neorx/core/graph/models.py`, delete the `causal_effect` field and the
whole `# Uncertainty quantification` block containing `confidence_interval`.
In the class docstring, replace the numbered list items 2 and 3 with:

```
    2. **Evidence aggregation** — How much independent support does
       this target have?  Reported as ``evidence_score``, not as a
       causal effect size: no interventional data enters it.

    3. **Sensitivity analysis** — Is the finding robust to removing
       any single source?  Leave-one-source-out stability.
```

- [ ] **Step 6: Update the four consumers**

`src/neorx/core/api.py:169` — replace the `confidence_interval` entry with:

```python
                "identifiable": t.identifiable,
                "identification_reason": t.identification_reason,
```

`src/neorx/core/__main__.py:238-242` — replace the `ci` local and its use:

```python
    for t in targets:
        icon = "✓" if t.is_causal_target else "✗"
        typer.echo(
            f"  {icon} {t.gene_name:12s} "
            f"conf={t.causal_confidence:.3f} "
            f"id={t.identification_reason:26s} "
```

`src/neorx/core/report.py` — at line 156 remove the `"causal_effect"` entry; at
line 163 replace the `"confidence_interval"` entry with:

```python
                    "identifiable": t.identifiable,
                    "identification_reason": t.identification_reason,
```

At lines 245-246 replace the `ci`/`ci_str` locals with:

```python
        id_str = t.get("identification_reason", "—").replace("_", " ")
```

and use `{id_str}` where `{ci_str}` was used in the row template. At line 360
change `<th>95% CI</th>` to `<th>Identifiability</th>`.

`src/neorx/core/templates/report.html` — at line 283 change
`<th>95% CI</th>` to `<th>Identifiability</th>`; at line 310 change the cell to:

```html
          <td style="font-size:0.85em">{{ t.identification_reason|replace("_", " ") }}</td>
```

- [ ] **Step 7: Update the tests that construct the removed fields**

In `tests/core/test_identifier.py`, remove the `causal_effect=` argument from
the three `NeoRxResult` constructions at lines 155, 168, and 181. In
`tests/core/test_new_modules.py`, delete the test at lines 229-254 that asserts
on `confidence_interval` — the new test from Step 1 replaces it.

- [ ] **Step 8: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add -A src/neorx tests
git commit -m "refactor: delete fabricated effect sizes, p-values, and confidence intervals"
```

---

## Task 8: The fabricated-statistics gate

**Files:**
- Modify: `src/neorx/experiments/gates.py`
- Test: `tests/test_experiment_gates.py` (append)

**Interfaces:**
- Consumes: `Finding` from `gates.py`.
- Produces: `check_no_fabricated_statistics(causal_dir: Path) -> list[Finding]`.

Scoped to `src/neorx/core/causal/` by argument, because
`src/neorx/genmol/evaluation/distribution.py:114` reports a genuine
Kolmogorov–Smirnov p-value that must not be flagged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_experiment_gates.py`:

```python
def test_fabricated_statistics_gate_flags_a_reintroduced_fake_bootstrap(tmp_path):
    from neorx.experiments.gates import check_no_fabricated_statistics

    (tmp_path / "sneaky.py").write_text(
        "def _bootstrap_confidence_interval(effect):\n"
        "    return (0.1, 0.9)\n"
    )
    findings = check_no_fabricated_statistics(tmp_path)
    assert len(findings) == 1
    assert "_bootstrap_confidence_interval" in findings[0].message


def test_fabricated_statistics_gate_flags_a_reintroduced_p_value(tmp_path):
    from neorx.experiments.gates import check_no_fabricated_statistics

    (tmp_path / "sneaky.py").write_text("p_value = 0.5 * (1.0 - effect)\n")
    findings = check_no_fabricated_statistics(tmp_path)
    assert any("p_value" in f.message for f in findings)


def test_fabricated_statistics_gate_flags_the_removed_networkx_call(tmp_path):
    from neorx.experiments.gates import check_no_fabricated_statistics

    (tmp_path / "sneaky.py").write_text("if nx.d_separated(G, x, y, z):\n    pass\n")
    findings = check_no_fabricated_statistics(tmp_path)
    assert any("d_separated" in f.message for f in findings)


def test_fabricated_statistics_gate_passes_on_the_real_causal_package():
    from pathlib import Path

    from neorx.experiments.gates import check_no_fabricated_statistics
    import neorx.core.causal as causal

    findings = check_no_fabricated_statistics(Path(causal.__file__).parent)
    assert findings == [], [f.message for f in findings]


def test_fabricated_statistics_gate_does_not_reach_the_genuine_ks_p_value():
    # distribution.py reports a real Kolmogorov-Smirnov p-value. The gate
    # is scoped to the causal package precisely so it survives.
    from pathlib import Path

    from neorx.experiments.gates import check_no_fabricated_statistics
    import neorx.genmol.evaluation.distribution as dist

    scoped = Path(dist.__file__).parent
    findings = check_no_fabricated_statistics(scoped)
    assert any("p_value" in f.message for f in findings), (
        "sanity: the gate does flag p_value when pointed at this directory, "
        "which is why production wiring must never point it here"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_experiment_gates.py -k fabricated -v`
Expected: FAIL — `ImportError: cannot import name 'check_no_fabricated_statistics'`

- [ ] **Step 3: Implement the gate**

Append to `src/neorx/experiments/gates.py`:

```python
# Names whose presence in the causal package means a fabricated statistic
# has come back. Each was removed for a specific reason:
#   _bootstrap_confidence_interval / _bootstrap_ci -- percentiles of
#       hand-chosen Gaussian noise added to deterministic scores, so the
#       interval width was a readout of two constants
#   p_value -- a rescaling of a heuristic score, with no null distribution
#   d_separated -- removed from NetworkX; every call raised, and the
#       except branch continued as though it had passed
_FABRICATED_STATISTIC_NAMES = (
    "_bootstrap_confidence_interval",
    "_bootstrap_ci",
    "p_value",
    "d_separated",
)


def check_no_fabricated_statistics(causal_dir: Path) -> list[Finding]:
    """Assert no fabricated statistic has been reintroduced.

    Scoped to a directory by argument, never to ``src/`` as a whole:
    ``neorx/genmol/evaluation/distribution.py`` reports a genuine
    Kolmogorov-Smirnov p-value, and a repository-wide search would flag
    it. Point this at ``neorx/core/causal/`` only.
    """
    findings: list[Finding] = []

    for path in sorted(causal_dir.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1,
        ):
            for name in _FABRICATED_STATISTIC_NAMES:
                if name in line:
                    findings.append(Finding(
                        path=str(path),
                        line=lineno,
                        message=(
                            f"{name} is a fabricated statistic removed in "
                            f"sub-project 3; it must not return. See "
                            f"docs/superpowers/specs/2026-09-03-correctness-design.md"
                        ),
                    ))

    return findings
```

`Finding` is a frozen dataclass with `path: str`, `line: int`, `message: str`
at `gates.py:64-68`, which is why `path` is stringified above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_experiment_gates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/experiments/gates.py tests/test_experiment_gates.py
git commit -m "feat: gate against reintroducing fabricated statistics"
```

---

## Task 9: Split identifier.py

**Files:**
- Create: `src/neorx/core/causal/scoring.py`
- Create: `src/neorx/core/causal/evidence.py`
- Modify: `src/neorx/core/causal/identifier.py`
- Modify: `tests/core/test_identifier.py` (imports)
- Test: `tests/core/test_module_sizes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces, in `scoring.py`: `assess_druggability`, `compute_causal_confidence`,
  `classify_target`, `organism_disease_relevance`. In `evidence.py`:
  `collect_source_scores`, `count_pathway_connections`,
  `count_protein_interactions`, `count_evidence_streams`,
  `compute_path_strength`, `corroboration_factor`.

The moved functions lose their leading underscore, becoming the modules' public
API. `identifier.py` imports them by name. Behaviour does not change in this
task — it is a move, verified by the suite passing unchanged.

- [ ] **Step 1: Write the failing size test**

```python
# tests/core/test_module_sizes.py
"""
Module size ceilings.

A file that has grown past a few hundred lines is usually doing more
than one thing, and both humans and agents edit focused files more
reliably. identifier.py reached 1291 lines while holding graph
semantics, identification, scoring, and evidence counting at once.
"""

from pathlib import Path

import pytest

import neorx

MAX_LINES = 600

_PACKAGES = ("core/causal", "causalbiorl/envs")


def _modules() -> list[Path]:
    root = Path(neorx.__file__).parent
    return sorted(
        path
        for package in _PACKAGES
        for path in (root / package).rglob("*.py")
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_module_is_under_the_line_ceiling(path):
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= MAX_LINES, (
        f"{path.name} is {n_lines} lines, over the {MAX_LINES}-line ceiling. "
        f"Split it by responsibility rather than raising the ceiling."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_module_sizes.py -v`
Expected: FAIL — `identifier.py is <N> lines, over the 600-line ceiling`

- [ ] **Step 3: Create `evidence.py`**

Create `src/neorx/core/causal/evidence.py` with a module docstring:

```python
"""
Counting the evidence behind a candidate target.

How many pathways mention it, how many proteins it interacts with, how
many independent databases report it, and how strongly the graph's edge
weights connect it to the disease. None of this is causal inference --
it is the bookkeeping that ranks candidates once identification has said
which claims are admissible.
"""
```

Move these functions from `identifier.py` verbatim, dropping the leading
underscore from each name and updating internal references between them:
`_collect_source_scores`, `_count_pathway_connections`,
`_count_protein_interactions`, `_count_evidence_streams`,
`_compute_path_strength`.

- [ ] **Step 4: Add the de-duplicating corroboration factor**

`_estimate_evidence_score` currently counts distinct `source_db` values on
edges incident to the treatment. OmniPath re-reports interactions STRING
already contributed, so counting aggregators double-counts one piece of
evidence. Add to `evidence.py`:

```python
def corroboration_factor(G: nx.DiGraph, node: str) -> float:
    """Multi-source corroboration, counted over primary evidence.

    An aggregator is not a source. OmniPath re-reports interactions
    STRING also reports, so counting ``source_db`` values would let one
    curated interaction inflate the factor twice. Where an edge names its
    ``primary_sources``, those are counted instead of the aggregator.
    """
    primary: set[str] = set()

    for _, _, attrs in list(G.edges(node, data=True)) + list(
        G.in_edges(node, data=True)
    ):
        named = attrs.get("primary_sources") or []
        if named:
            primary.update(named)
        elif attrs.get("source_db"):
            primary.add(attrs["source_db"])

    return min(1.5, 1.0 + len(primary) * 0.1)
```

In `_estimate_evidence_score`, delete the `source_dbs` loop and the
`source_factor = min(1.5, 1.0 + len(source_dbs) * 0.1)` line, replacing them
with:

```python
    source_factor = corroboration_factor(G, treatment)
```

- [ ] **Step 5: Write the corroboration test**

Append to `tests/core/test_identifier.py`:

```python
class TestCorroborationCountsPrimaryEvidence:
    def test_an_aggregator_does_not_double_count_a_shared_interaction(self):
        import networkx as nx
        from neorx.core.causal.evidence import corroboration_factor

        G = nx.DiGraph()
        # STRING and OmniPath both report the same underlying interaction;
        # OmniPath names STRING among its primary sources.
        G.add_edge("gene:A", "gene:B", source_db="STRING",
                   primary_sources=["STRING"])
        G.add_edge("gene:A", "gene:C", source_db="OmniPath",
                   primary_sources=["STRING"])

        # One distinct primary source, not two aggregators.
        assert corroboration_factor(G, "gene:A") == 1.1

    def test_distinct_primary_sources_each_count(self):
        import networkx as nx
        from neorx.core.causal.evidence import corroboration_factor

        G = nx.DiGraph()
        G.add_edge("gene:A", "gene:B", source_db="OmniPath",
                   primary_sources=["SIGNOR", "TRRUST"])
        assert corroboration_factor(G, "gene:A") == pytest.approx(1.2)

    def test_an_edge_without_primary_sources_falls_back_to_its_database(self):
        import networkx as nx
        from neorx.core.causal.evidence import corroboration_factor

        G = nx.DiGraph()
        G.add_edge("gene:A", "disease:d", source_db="Monarch",
                   primary_sources=[])
        assert corroboration_factor(G, "gene:A") == pytest.approx(1.1)
```

- [ ] **Step 6: Create `scoring.py`**

Create `src/neorx/core/causal/scoring.py` with a module docstring:

```python
"""
Turning evidence into a ranked, labelled target.

Druggability, the combined causal confidence, and the final
causal-versus-correlational classification. These are weighted
heuristics over the evidence counts, and they are named as such: nothing
here estimates a causal effect.
"""
```

Move these functions from `identifier.py` verbatim, dropping the leading
underscore and updating references between them: `_assess_druggability`,
`_compute_causal_confidence`, `_classify_target`,
`_organism_disease_relevance`.

- [ ] **Step 7: Rewire `identifier.py`**

Add to `identifier.py`:

```python
from neorx.core.causal.evidence import (
    collect_source_scores,
    compute_path_strength,
    corroboration_factor,
    count_evidence_streams,
    count_pathway_connections,
    count_protein_interactions,
)
from neorx.core.causal.scoring import (
    assess_druggability,
    classify_target,
    compute_causal_confidence,
    organism_disease_relevance,
)
```

Update every call site in `identifier.py` to the un-underscored name. Update
`tests/core/test_identifier.py`'s import block, which currently imports
`_compute_causal_confidence` and `_classify_target` from `identifier` — point
those at `neorx.core.causal.scoring` under their new names.

- [ ] **Step 8: Run the full suite and the size test**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, with `test_module_sizes.py` now passing for `identifier.py`.

- [ ] **Step 9: Commit**

```bash
git add -A src/neorx/core/causal tests/core
git commit -m "refactor: split identifier into evidence and scoring modules"
```

---

## Task 10: Fix the planner wiring and add batched evaluation

**Files:**
- Modify: `src/neorx/causalbiorl/agents/causal_agent.py:224,260,333-350`
- Modify: `src/neorx/causalbiorl/causal/reward_learner.py:197-228`
- Create: `src/neorx/causalbiorl/envs/generation.py`
- Modify: `src/neorx/causalbiorl/envs/drug_discovery.py` (`_decode_latent`, plus the new method)
- Test: `tests/causalbiorl/test_planner_wiring.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `generation.decode_latent_batch(model, tokenizer, Z: NDArray) -> list[str]`
  - `DrugDiscoveryEnv.evaluate_actions(state, actions: NDArray) -> NDArray[np.float64]`
  - `AdaptiveRewardLearner.score(state, objective_scores) -> float` (pure)
  - `CausalAgent._reward_fn` non-`None` from `__init__` onward
  - `CausalAgent.train(reward_fn=...)` propagates the override to an existing
    hierarchical planner

`evaluate_actions` must not mutate environment state. `compute_reward` appends
to weight and objective history, so it is split: `score` is pure, and
`compute_reward` calls `score` then records.

- [ ] **Step 1: Write the failing tests**

```python
# tests/causalbiorl/test_planner_wiring.py
"""
The hierarchical planner must actually plan.

`pipeline.py` builds the agent, calls `init_hierarchical_planner()`, and
only then calls `train()`. `_reward_fn` used to be assigned inside
`train()`, and `HierarchicalPlanner` captures the value rather than a
reference -- so in the end-to-end pipeline the planner's reward function
was always None, its CEM never ran, and every molecule the pipeline
reported came from `rng.standard_normal(latent_dim) * 0.3`.
"""

import networkx as nx
import numpy as np
import pytest

from neorx.causalbiorl.envs.drug_discovery import DrugDiscoveryEnv


def _env() -> DrugDiscoveryEnv:
    G = nx.DiGraph()
    G.add_node("gene:A", score=0.9)
    G.add_node("disease:x", score=1.0)
    G.add_edge("gene:A", "disease:x", weight=0.9)
    return DrugDiscoveryEnv(
        disease="x",
        prebuilt_graph=G,
        prebuilt_targets=[{
            "gene_name": "A", "protein_id": "P1",
            "protein_name": "A", "causal_confidence": 0.8,
        }],
        max_steps=5,
    )


def test_reward_fn_is_set_before_train_is_called():
    from neorx.causalbiorl.agents.causal_agent import CausalAgent
    from neorx.causalbiorl.models import AgentConfig

    agent = CausalAgent(_env(), AgentConfig(agent_type="causal", n_episodes=1))
    assert agent._reward_fn is not None


def test_planner_built_in_the_pipelines_order_can_plan():
    from neorx.causalbiorl.agents.causal_agent import CausalAgent
    from neorx.causalbiorl.models import AgentConfig

    agent = CausalAgent(_env(), AgentConfig(agent_type="causal", n_episodes=1))
    agent.init_hierarchical_planner(n_targets=1, latent_dim=agent.env.latent_dim)
    assert agent._hierarchical_planner is not None
    assert agent._hierarchical_planner.reward_fn is not None


def test_train_propagates_a_reward_override_to_an_existing_planner():
    from neorx.causalbiorl.agents.causal_agent import CausalAgent
    from neorx.causalbiorl.models import AgentConfig

    agent = CausalAgent(_env(), AgentConfig(agent_type="causal", n_episodes=1))
    agent.init_hierarchical_planner(n_targets=1, latent_dim=agent.env.latent_dim)

    sentinel = lambda state, action: 0.5  # noqa: E731
    agent.train(n_episodes=1, reward_fn=sentinel, verbose=False)

    assert agent._hierarchical_planner.reward_fn is sentinel


def test_decode_latent_batch_returns_one_smiles_per_row():
    from neorx.causalbiorl.envs.generation import decode_latent_batch

    env = _env()
    env.reset(seed=0)
    env._init_genmol()
    Z = np.zeros((4, env.latent_dim), dtype=np.float32)

    smiles = decode_latent_batch(env._genmol_model, env._genmol_tokenizer, Z)

    assert len(smiles) == 4
    assert all(isinstance(s, str) for s in smiles)


def test_decode_latent_batch_agrees_with_single_decode():
    from neorx.causalbiorl.envs.generation import decode_latent_batch

    env = _env()
    env.reset(seed=0)
    z = np.zeros(env.latent_dim, dtype=np.float32)

    single = env._decode_latent(z)
    batched = decode_latent_batch(
        env._genmol_model, env._genmol_tokenizer, z.reshape(1, -1),
    )

    assert batched[0] == single


def test_evaluate_actions_returns_one_reward_per_action():
    env = _env()
    state, _ = env.reset(seed=0)
    actions = np.zeros((3, env.action_space.shape[0]), dtype=np.float32)
    actions[:, 1] = -1.0

    rewards = env.evaluate_actions(state, actions)

    assert rewards.shape == (3,)
    assert np.all(np.isfinite(rewards))


def test_evaluate_actions_does_not_mutate_environment_state():
    env = _env()
    state, _ = env.reset(seed=0)

    before_step = env._step_count
    before_best = [t.best_score for t in env._targets]
    before_zbase = [t.z_base.copy() for t in env._targets]
    before_attempts = [t.n_attempts for t in env._targets]

    actions = np.zeros((5, env.action_space.shape[0]), dtype=np.float32)
    actions[:, 1] = -1.0
    env.evaluate_actions(state, actions)

    assert env._step_count == before_step
    assert [t.best_score for t in env._targets] == before_best
    assert [t.n_attempts for t in env._targets] == before_attempts
    for after, before in zip(env._targets, before_zbase):
        assert np.array_equal(after.z_base, before)


def test_evaluate_actions_agrees_with_step_for_the_same_action():
    env = _env()
    state, _ = env.reset(seed=0)
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    action[1] = -1.0

    predicted = env.evaluate_actions(state, action.reshape(1, -1))[0]
    _, actual, _, _, _ = env.step(action)

    assert predicted == pytest.approx(actual, abs=1e-6)


def test_reward_learner_score_is_pure():
    from neorx.causalbiorl.causal.reward_learner import AdaptiveRewardLearner

    learner = AdaptiveRewardLearner(state_dim=4)
    state = np.zeros(4, dtype=np.float32)
    scores = {"binding": 0.5, "qed": 0.5}

    before = len(learner.get_weight_history()["binding"])
    learner.score(state, scores)
    after = len(learner.get_weight_history()["binding"])

    assert before == after


def test_reward_learner_compute_reward_still_records():
    from neorx.causalbiorl.causal.reward_learner import AdaptiveRewardLearner

    learner = AdaptiveRewardLearner(state_dim=4)
    state = np.zeros(4, dtype=np.float32)

    before = len(learner.get_weight_history()["binding"])
    learner.compute_reward(state, {"binding": 0.5})
    after = len(learner.get_weight_history()["binding"])

    assert after == before + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/causalbiorl/test_planner_wiring.py -v`
Expected: FAIL — `assert agent._reward_fn is not None` fails, and
`ModuleNotFoundError` for `neorx.causalbiorl.envs.generation`.

- [ ] **Step 3: Fix the agent's reward wiring**

In `src/neorx/causalbiorl/agents/causal_agent.py`, change line 224 from the
`None` initialisation to:

```python
        # Assigned here, not in train(): pipeline.py calls
        # init_hierarchical_planner() before train(), and
        # HierarchicalPlanner captures this value rather than a
        # reference. Leaving it None until train() meant the planner's
        # CEM never ran in the end-to-end pipeline.
        self._reward_fn: Callable[..., float] = self._reward_model.predict
```

At line 260, replace the unconditional assignment with an override that
propagates:

```python
        if reward_fn is not None:
            self._reward_fn = reward_fn
            if self._hierarchical_planner is not None:
                self._hierarchical_planner.reward_fn = reward_fn
            if self._planner is not None:
                self._planner.reward_fn = reward_fn
```

- [ ] **Step 4: Split the reward learner's scoring from its recording**

In `src/neorx/causalbiorl/causal/reward_learner.py`, replace the body of
`compute_reward` and add `score`:

```python
    def score(
        self,
        state: NDArray[np.floating],
        objective_scores: dict[str, float],
    ) -> float:
        """The adaptively-weighted scalar reward, recording nothing.

        ``compute_reward`` records each call into the weight and
        objective histories, which makes it unusable for scoring
        candidate actions -- a planner evaluating a batch would write
        hundreds of phantom entries per environment step. This is the
        pure half.
        """
        weights = self._compute_weights(state)
        scores = np.array([
            objective_scores.get(name, 0.0)
            for name in OBJECTIVE_NAMES
        ], dtype=np.float32)
        return float(np.dot(weights, scores))

    def compute_reward(
        self,
        state: NDArray[np.floating],
        objective_scores: dict[str, float],
    ) -> float:
        """Compute the reward and record it in the learner's history."""
        weights = self._compute_weights(state)
        scores = np.array([
            objective_scores.get(name, 0.0)
            for name in OBJECTIVE_NAMES
        ], dtype=np.float32)

        self._weight_history.append(weights.copy())
        self._objective_history.append(scores.copy())

        return float(np.dot(weights, scores))
```

Keep `compute_reward`'s existing docstring parameters block below the new
summary line.

- [ ] **Step 5: Create the batched decoder**

Create `src/neorx/causalbiorl/envs/generation.py`:

```python
"""
Batched latent-space decoding.

The VAE decoder is a single forward pass, so decoding one molecule at a
time wastes almost all of it. Measured on CPU: 13.9 ms per molecule
one at a time, 2.0 ms per molecule at batch 200 -- a sevenfold
difference. A planner that scores candidate molecules needs the batched
form or it cannot afford to decode at all.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

__all__ = ["decode_latent_batch"]


def decode_latent_batch(model, tokenizer, Z: NDArray[np.floating]) -> list[str]:
    """Decode a batch of latent vectors to SMILES.

    Greedy decoding, matching ``DrugDiscoveryEnv._decode_latent``: the
    shipped model has partial posterior collapse, and sampled decodes
    produce chemically invalid SMILES often enough to waste steps.

    Parameters
    ----------
    Z
        Array of shape ``(n, latent_dim)``.
    """
    if Z.ndim != 2:
        raise ValueError(
            f"decode_latent_batch expects a 2-D array of latents, got shape {Z.shape}"
        )
    if Z.shape[1] != model.latent_dim:
        raise ValueError(
            f"latent vectors have width {Z.shape[1]} but the decoder's latent "
            f"dimension is {model.latent_dim}"
        )

    z_tensor = torch.as_tensor(np.ascontiguousarray(Z), dtype=torch.float32)
    with torch.no_grad():
        token_ids = model.decode(z_tensor, greedy=True)

    return [tokenizer.decode(row.tolist()) for row in token_ids]
```

- [ ] **Step 6: Add `evaluate_actions` to the environment**

In `src/neorx/causalbiorl/envs/drug_discovery.py`, add to the imports:

```python
from neorx.causalbiorl.envs.generation import decode_latent_batch
```

and add this method next to `step`:

```python
    def evaluate_actions(
        self,
        state: NDArray[np.floating],
        actions: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Score candidate actions against real chemistry, mutating nothing.

        A planner needs to know what an action is actually worth before
        committing to it. ``step`` cannot answer that: it advances the
        step counter, updates per-target bests, drifts ``z_base``, and
        records into the reward learner's history. This decodes and
        screens the same molecules ``step`` would and returns their
        rewards, leaving the environment exactly as it found it.

        Parameters
        ----------
        state
            The observation the rewards are computed against.
        actions
            Array of shape ``(n, action_dim)``.

        Returns
        -------
        ndarray of shape ``(n,)``
        """
        if actions.ndim != 2:
            raise ValueError(
                f"evaluate_actions expects a 2-D action array, got shape {actions.shape}"
            )

        if self._genmol_model is None:
            self._init_genmol()

        clipped = np.clip(actions, self.action_space.low, self.action_space.high)

        target_indices = [self._select_target(float(a[0])) for a in clipped]
        latents = np.stack([
            self._targets[idx].z_base
            + clipped[i, 2: 2 + self.latent_dim].astype(np.float32) * 0.3
            for i, idx in enumerate(target_indices)
        ])

        smiles = decode_latent_batch(
            self._genmol_model, self._genmol_tokenizer, latents,
        )

        rewards = np.empty(len(clipped), dtype=np.float64)
        for i, (mol, idx) in enumerate(zip(smiles, target_indices)):
            obj_scores = self._screen_molecule(mol, self._targets[idx])
            rewards[i] = self._score_reward(state, obj_scores)

        return rewards
```

`_select_target` and `_screen_molecule` must not mutate. `_select_target`
currently only reads; confirm `_screen_molecule` does the same and, if it
increments a counter or caches onto the target, move that mutation into `step`.

- [ ] **Step 7: Add the non-recording reward path to the environment**

`step` calls `self._compute_reward(state_vec, obj_scores)`. Add its pure twin
beside it and have `evaluate_actions` use it:

```python
    def _score_reward(
        self,
        state: NDArray[np.floating],
        obj_scores: dict[str, float],
    ) -> float:
        """The reward for these objective scores, recording nothing.

        Mirrors ``_compute_reward`` exactly apart from the two mutations
        it performs -- the learner's weight/objective history and the
        critic update. A planner scoring a batch must not write hundreds
        of phantom history entries or train the critic on candidates it
        never takes.
        """
        try:
            if self._reward_learner is None:
                from neorx.causalbiorl.causal.reward_learner import AdaptiveRewardLearner
                self._reward_learner = AdaptiveRewardLearner(
                    state_dim=OBS_DIM,
                )
            return self._reward_learner.score(state, obj_scores)
        except Exception:
            # Fallback: equal-weight sum. Mirrors _compute_reward's own
            # fallback so evaluate_actions and step agree on every path.
            return sum(obj_scores.values()) / max(len(obj_scores), 1)
```

`_compute_reward` at `drug_discovery.py:709-731` computes the reward *before*
calling `self._reward_learner.update(...)`, so a preceding `evaluate_actions`
that skips the update leaves `step`'s reward unchanged and the agreement test
holds.

The `except Exception` swallow is copied deliberately rather than fixed: it is
pre-existing behaviour in `_compute_reward`, and the agreement test requires
both paths to behave identically. Removing it from both belongs to
sub-project 4, not here — note it in the task report so it is not lost.

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest tests/causalbiorl/ -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add -A src/neorx/causalbiorl tests/causalbiorl
git commit -m "fix: planner reward function was never wired; add batched action evaluation"
```

---

## Task 11: Confirm CEM elites against real chemistry

**Files:**
- Modify: `src/neorx/causalbiorl/causal/planner.py:290-333` (`_plan_molecule_cem`)
- Modify: `src/neorx/causalbiorl/causal/planner.py:198-256` (constructor and `plan`)
- Test: `tests/causalbiorl/test_planner_wiring.py` (append)

**Interfaces:**
- Consumes: `DrugDiscoveryEnv.evaluate_actions` from Task 10.
- Produces: `HierarchicalPlanner(..., evaluate_actions: Callable | None = None)`;
  `HierarchicalPlanner.last_exploitation_gap: float | None`.

The inner iterations keep using the learned reward model, which is what makes
the search affordable. The final elite set is decoded and screened for real,
and the returned delta-z is the best real-scored elite rather than the mean of
model-scored ones. The gap between the two is recorded: it measures model
exploitation directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/causalbiorl/test_planner_wiring.py`:

```python
class TestEliteConfirmation:
    """CEM searches with the learned model; reality picks the winner.

    Optimising a 64-unit MLP over 128 latent dimensions with a
    1000-sample CEM finds the model's artefacts. Decoding every
    candidate is unaffordable -- 1000 evaluations per step is ~15 s even
    with batched decode -- so the elites, and only the elites, are
    checked against real chemistry.
    """

    def test_planner_returns_the_best_real_scored_elite(self):
        from neorx.causalbiorl.causal.planner import HierarchicalPlanner

        # The learned model prefers large positive delta-z; reality
        # prefers the opposite. The planner must follow reality.
        def model_reward(state, action):
            return float(np.sum(action[2:]))

        def real_rewards(state, actions):
            return -np.sum(actions[:, 2:], axis=1)

        planner = HierarchicalPlanner(
            reward_fn=model_reward,
            evaluate_actions=real_rewards,
            n_targets=1,
            latent_dim=4,
            cem_samples=32,
            cem_iterations=2,
        )
        action = planner.plan(np.zeros(4), rng=np.random.default_rng(0))

        # Reality rewards small sums, so the chosen delta-z must not be
        # the model's preferred large-positive direction.
        assert float(np.sum(action[2:])) < 0.0

    def test_planner_records_the_exploitation_gap(self):
        from neorx.causalbiorl.causal.planner import HierarchicalPlanner

        planner = HierarchicalPlanner(
            reward_fn=lambda s, a: float(np.sum(a[2:])),
            evaluate_actions=lambda s, a: -np.sum(a[:, 2:], axis=1),
            n_targets=1,
            latent_dim=4,
            cem_samples=16,
            cem_iterations=1,
        )
        planner.plan(np.zeros(4), rng=np.random.default_rng(0))

        assert planner.last_exploitation_gap is not None
        assert planner.last_exploitation_gap > 0.0

    def test_planner_without_real_evaluation_still_runs_cem(self):
        # No evaluate_actions supplied: fall back to the model-scored
        # elite mean, as before. This must not silently become random.
        from neorx.causalbiorl.causal.planner import HierarchicalPlanner

        planner = HierarchicalPlanner(
            reward_fn=lambda s, a: -float(np.sum(np.abs(a[2:]))),
            evaluate_actions=None,
            n_targets=1,
            latent_dim=4,
            cem_samples=64,
            cem_iterations=3,
        )
        action = planner.plan(np.zeros(4), rng=np.random.default_rng(0))

        # The model rewards delta-z near zero, so CEM must converge there.
        assert float(np.sum(np.abs(action[2:]))) < 1.0
        assert planner.last_exploitation_gap is None

    def test_cem_evaluates_more_than_one_candidate(self):
        from neorx.causalbiorl.causal.planner import HierarchicalPlanner

        calls = []

        def counting_reward(state, action):
            calls.append(action.copy())
            return 0.0

        planner = HierarchicalPlanner(
            reward_fn=counting_reward,
            n_targets=1,
            latent_dim=4,
            cem_samples=8,
            cem_iterations=2,
        )
        planner.plan(np.zeros(4), rng=np.random.default_rng(0))

        assert len(calls) == 16
        assert not np.allclose(calls[0], calls[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/causalbiorl/test_planner_wiring.py -k EliteConfirmation -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'evaluate_actions'`

- [ ] **Step 3: Accept the real evaluator**

In `src/neorx/causalbiorl/causal/planner.py`, add to `HierarchicalPlanner.__init__`
a parameter after `reward_fn`:

```python
        evaluate_actions: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ] | None = None,
```

and in the body:

```python
        self.evaluate_actions = evaluate_actions
        self.last_exploitation_gap: float | None = None
```

Add to the class docstring, after the `reward_fn` entry:

```
    evaluate_actions : callable | None
        ``(state, actions) -> rewards`` scoring a batch of actions
        against real chemistry. The CEM inner loop uses ``reward_fn``
        because decoding every candidate is unaffordable; the final
        elite set is confirmed through this, and the best real-scored
        elite is what the planner emits. ``None`` falls back to the
        model-scored elite mean.
```

- [ ] **Step 4: Confirm the elites**

Replace the tail of `_plan_molecule_cem` — the `return mean` and the loop's
final elite computation — so the whole method reads:

```python
    def _plan_molecule_cem(
        self,
        state: NDArray[np.floating],
        target_idx: int,
        target_embeddings: NDArray[np.floating] | None,
        rng: np.random.Generator,
    ) -> NDArray[np.floating]:
        """CEM in the decoder's latent space, elites confirmed for real.

        The inner iterations score candidates with the learned reward
        model: at 200 samples over 5 iterations that is 1000 evaluations
        per environment step, and decoding all of them costs roughly 15
        seconds even batched. The elites are few enough to decode and
        screen properly, so they are, and the winner is chosen on that
        real score rather than on the model's opinion of it.
        """
        self.last_exploitation_gap = None

        if self.reward_fn is None:
            raise ValueError(
                "HierarchicalPlanner has no reward_fn, so its CEM cannot "
                "score candidates. Pass one at construction -- returning "
                "random latents here would silently reduce planning to noise."
            )

        mean = np.zeros(self.latent_dim, dtype=np.float32)
        std = np.full(self.latent_dim, 0.3, dtype=np.float32)
        n_elite = max(int(self.cem_samples * self.cem_elite_frac), 1)
        elite = np.tile(mean, (n_elite, 1))
        elite_model_scores = np.zeros(n_elite, dtype=np.float32)

        for _ in range(self.cem_iterations):
            samples = rng.normal(
                loc=mean, scale=std,
                size=(self.cem_samples, self.latent_dim),
            ).astype(np.float32)
            samples = np.clip(samples, -1.0, 1.0)

            rewards = np.array([
                self.reward_fn(state, self._compose_action(target_idx, dz))
                for dz in samples
            ], dtype=np.float32)

            elite_idx = np.argsort(rewards)[-n_elite:]
            elite = samples[elite_idx]
            elite_model_scores = rewards[elite_idx]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-6

        if self.evaluate_actions is None:
            return mean

        elite_actions = np.stack([
            self._compose_action(target_idx, dz) for dz in elite
        ])
        real_scores = np.asarray(
            self.evaluate_actions(state, elite_actions), dtype=np.float64,
        )

        best = int(np.argmax(real_scores))
        self.last_exploitation_gap = float(
            np.max(elite_model_scores) - real_scores[best]
        )
        return elite[best]

    def _compose_action(
        self,
        target_idx: int,
        delta_z: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Assemble ``[target_selector, stop_signal, delta_z...]``."""
        action = np.zeros(2 + self.latent_dim, dtype=np.float32)
        action[0] = (2.0 * target_idx / max(self.n_targets - 1, 1)) - 1.0
        action[1] = -1.0
        action[2:] = delta_z
        return action
```

Replace the same three lines inside `plan` that build the action with a call to
`self._compose_action(target_idx, delta_z)`.

Note that the `reward_fn is None` early return becomes a raised error. Task 10
guarantees the agent always supplies one, so a `None` here is a wiring bug of
exactly the kind this sub-project exists to remove — it must not degrade to
noise silently.

- [ ] **Step 5: Wire the environment's evaluator into the agent's planner**

In `src/neorx/causalbiorl/agents/causal_agent.py`, in
`init_hierarchical_planner`, pass the environment's method through:

```python
        self._hierarchical_planner = HierarchicalPlanner(
            scm=self._scm,
            reward_fn=self._reward_fn,
            evaluate_actions=getattr(self.env, "evaluate_actions", None),
            n_targets=n_targets,
            latent_dim=latent_dim,
        )
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/causalbiorl/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -A src/neorx/causalbiorl tests/causalbiorl
git commit -m "feat: confirm CEM elites against real chemistry and record the exploitation gap"
```

---

## Task 12: Split drug_discovery.py and the pipeline

**Files:**
- Create: `src/neorx/causalbiorl/envs/screening.py`
- Modify: `src/neorx/causalbiorl/envs/drug_discovery.py`
- Create: `src/neorx/core/pipeline/__init__.py`
- Create: `src/neorx/core/pipeline/rl_stage.py`
- Delete: `src/neorx/core/pipeline.py`
- Test: `tests/core/test_module_sizes.py` (already written in Task 9)

**Interfaces:**
- Consumes: nothing new.
- Produces: `screening.score_binding`, `screening.score_qed`,
  `screening.score_synthetic_accessibility`, `screening.encode_molecule` —
  whichever of `drug_discovery.py`'s `_get_*` / `_screen_*` / `_encode_*`
  helpers are pure functions of a molecule and a target.
  `neorx.core.pipeline` keeps every name it exports today.

- [ ] **Step 1: Run the size test to see the current failures**

Run: `.venv/bin/python -m pytest tests/core/test_module_sizes.py -v`
Expected: FAIL for `drug_discovery.py` (884 lines).

- [ ] **Step 2: Create `screening.py`**

Create `src/neorx/causalbiorl/envs/screening.py` with a module docstring:

```python
"""
Scoring a molecule against a target.

Binding affinity via the docking surrogate, drug-likeness via QED,
synthetic accessibility, and the feature encoding the observation uses.
These are pure functions of a SMILES string and a target: nothing here
advances the environment.
"""
```

Move the screening helpers out of `DrugDiscoveryEnv` as module-level functions,
taking explicitly what they previously read from `self`. Each helper that used
`self.use_surrogate`, `self._surrogate`, or similar gains that as a parameter.
The environment's `_screen_molecule` becomes a thin call into them.

- [ ] **Step 3: Run the environment tests**

Run: `.venv/bin/python -m pytest tests/causalbiorl/ -q`
Expected: PASS — this is a move, so behaviour is unchanged.

- [ ] **Step 4: Convert the pipeline to a package**

```bash
mkdir -p src/neorx/core/pipeline
git mv src/neorx/core/pipeline.py src/neorx/core/pipeline/__init__.py
```

- [ ] **Step 5: Extract the RL stage**

Create `src/neorx/core/pipeline/rl_stage.py` with a module docstring:

```python
"""
The reinforcement-learning candidate-generation stage.

Builds a DrugDiscoveryEnv over the identified targets, trains the causal
agent against it, and collects the best molecule found per target. Split
out of the pipeline module because it is the only stage that owns an
environment and an agent, and because the ordering bug that left the
planner unwired lived in this wiring.
"""
```

Move the RL block — the `try:` beginning with the
`from neorx.causalbiorl.envs.drug_discovery import DrugDiscoveryEnv` import
through the `logger.info("RL agent found %d candidate molecules.", ...)` line
and its `except ImportError` handler — into a function:

```python
def generate_candidates_with_rl(
    disease: str,
    nx_graph,
    causal_only: list,
    *,
    n_episodes: int,
    max_steps_per_episode: int,
    latent_dim: int,
    seed: int,
) -> list:
    """Train the causal agent and return its best scored candidates."""
```

Import and call it from `__init__.py` at the original site, passing the same
values the inlined code used.

- [ ] **Step 6: Fix the two hardcoded screening placeholders**

The RL stage builds each `score_candidate(...)` with `qed_score=0.5` and
`sa_score=5.0` — constants standing in for values the environment already
measured. The environment records per-objective scores on each target state, so
pass the real ones:

```python
                    candidate = score_candidate(
                        smiles=ts.best_smiles,
                        target_protein_id=...,
                        target_protein_name=...,
                        causal_confidence=...,
                        binding_affinity=ts.best_objectives.get("binding", 0.0) * -10.0,
                        qed_score=ts.best_objectives.get("qed", 0.0),
                        sa_score=ts.best_objectives.get("sa", 0.0),
                    )
```

Match the objective key names `_screen_molecule` actually produces; if it does
not produce an `sa` key, use the key it does and do not invent one.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, including `test_module_sizes.py`.

- [ ] **Step 8: Commit**

```bash
git add -A src/neorx tests
git commit -m "refactor: split environment screening and the pipeline RL stage into modules"
```

---

## Task 13: Record identifiability metrics and arm the manifest

**Files:**
- Modify: `experiments/neorx_7disease.py`
- Modify: `docs/run-manifest.toml`
- Test: `tests/experiments/test_neorx_7disease.py` (append — the file exists and
  already imports `pytest`)

**Interfaces:**
- Consumes: `NeoRxResult.identifiable`, `.identification_reason`,
  `.n_near_miss_confounders` from Task 6.
- Produces: per-disease run-record rows carrying
  `n_candidates`, `n_identifiable_by_adjustment`, `n_identifiable_trivially`,
  `nontrivial_identifiability_rate`, `cyclic_fraction`,
  `mean_near_miss_confounders`, and a `reason_counts` mapping.

- [ ] **Step 1: Write the failing test**

Append to `tests/experiments/test_neorx_7disease.py`, whose existing header
already provides `import pytest`:

```python
# ── Identifiability metrics ────────────────────────────────────────
#
# The non-trivial identifiability rate is what the rewritten manuscript
# leads with, and the failure breakdown is what makes a low rate legible
# rather than a bare zero. Both have to reach the run record.

from neorx.core.causal.backdoor import IdentificationReason


def test_identifiability_summary_counts_every_candidate():
    from experiments.neorx_7disease import summarise_identification

    class _R:
        def __init__(self, reason, identifiable, near_miss=0):
            self.identification_reason = reason
            self.identifiable = identifiable
            self.n_near_miss_confounders = near_miss

    results = [
        _R("identifiable_by_adjustment", True),
        _R("identifiable_trivially", True, near_miss=3),
        _R("no_causal_path", False),
        _R("cyclic_component", False),
    ]

    summary = summarise_identification(results)

    assert summary["n_candidates"] == 4
    assert sum(summary["reason_counts"].values()) == 4
    assert summary["n_identifiable_by_adjustment"] == 1
    assert summary["n_identifiable_trivially"] == 1


def test_nontrivial_rate_excludes_trivial_verdicts():
    from experiments.neorx_7disease import summarise_identification

    class _R:
        def __init__(self, reason, identifiable):
            self.identification_reason = reason
            self.identifiable = identifiable
            self.n_near_miss_confounders = 0

    results = [
        _R("identifiable_by_adjustment", True),
        _R("identifiable_trivially", True),
        _R("identifiable_trivially", True),
        _R("no_causal_path", False),
    ]

    summary = summarise_identification(results)

    assert summary["nontrivial_identifiability_rate"] == 0.25
    assert summary["cyclic_fraction"] == 0.0


def test_cyclic_fraction_is_reported():
    from experiments.neorx_7disease import summarise_identification

    class _R:
        def __init__(self, reason):
            self.identification_reason = reason
            self.identifiable = False
            self.n_near_miss_confounders = 0

    summary = summarise_identification(
        [_R("cyclic_component"), _R("cyclic_component"), _R("no_causal_path")]
    )
    assert summary["cyclic_fraction"] == pytest.approx(2 / 3)


def test_summary_of_no_candidates_is_zero_not_a_division_error():
    from experiments.neorx_7disease import summarise_identification

    summary = summarise_identification([])
    assert summary["n_candidates"] == 0
    assert summary["nontrivial_identifiability_rate"] == 0.0
    assert summary["cyclic_fraction"] == 0.0


def test_every_reason_appears_in_the_breakdown_even_at_zero():
    from experiments.neorx_7disease import summarise_identification

    summary = summarise_identification([])
    assert set(summary["reason_counts"]) == {
        r.value for r in IdentificationReason
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/experiments/test_neorx_7disease.py -v`
Expected: FAIL — `ImportError: cannot import name 'summarise_identification'`

- [ ] **Step 3: Implement the summary**

Add to `experiments/neorx_7disease.py`:

```python
def summarise_identification(results) -> dict:
    """Aggregate per-target identification verdicts for one disease.

    The non-trivial rate is the headline: a trivial verdict means the
    graph held no confounder to adjust for, which in an open-world
    knowledge graph is a statement about coverage rather than about
    biology. Both rates are reported, along with the mean near-miss
    count that says how exposed the trivial verdicts are.

    Every reason appears in ``reason_counts`` even at zero, so a run's
    breakdown has the same shape whatever it found.
    """
    from neorx.core.causal.backdoor import IdentificationReason

    counts = {reason.value: 0 for reason in IdentificationReason}
    for result in results:
        reason = result.identification_reason
        if reason in counts:
            counts[reason] += 1

    n = len(results)
    n_by_adjustment = counts[IdentificationReason.IDENTIFIABLE_BY_ADJUSTMENT.value]
    n_trivially = counts[IdentificationReason.IDENTIFIABLE_TRIVIALLY.value]
    n_cyclic = counts[IdentificationReason.CYCLIC_COMPONENT.value]

    trivial_near_misses = [
        r.n_near_miss_confounders
        for r in results
        if r.identification_reason
        == IdentificationReason.IDENTIFIABLE_TRIVIALLY.value
    ]

    return {
        "n_candidates": n,
        "n_identifiable_by_adjustment": n_by_adjustment,
        "n_identifiable_trivially": n_trivially,
        "nontrivial_identifiability_rate": (n_by_adjustment / n) if n else 0.0,
        "trivial_identifiability_rate": (n_trivially / n) if n else 0.0,
        "cyclic_fraction": (n_cyclic / n) if n else 0.0,
        "mean_near_miss_confounders": (
            sum(trivial_near_misses) / len(trivial_near_misses)
            if trivial_near_misses else 0.0
        ),
        "reason_counts": counts,
    }
```

Then merge its output into the per-disease row the experiment already appends,
so the metrics land in `rows.jsonl` alongside the existing fields.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/experiments/ -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and both gates**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/neorx_7disease.py tests/experiments/test_neorx_7disease.py
git commit -m "feat: record identifiability rates and failure breakdown per disease"
```

- [ ] **Step 7: Record a real run and arm the manifest**

This step touches eight live APIs and takes roughly 25 minutes. It is the last
step of the sub-project because it is the first run whose numbers the corrected
engine produces.

```bash
.venv/bin/python -m neorx exp run neorx_7disease
```

Read the run ID from the command's output, then add the manifest entries that
bind each rewritten manuscript table and figure to it in
`docs/run-manifest.toml`, following the format `check_cited_runs` expects.
Verify with:

```bash
.venv/bin/python -m pytest tests/test_experiment_gates.py -q
```

If the run fails or reports zero candidates for a disease, stop and report the
failure with its run ID rather than editing the manifest — an armed manifest
pointing at a broken run is worse than an empty one.

- [ ] **Step 8: Commit the manifest**

```bash
git add docs/run-manifest.toml runs/
git commit -m "chore: arm the run manifest against the first corrected-engine run"
```

---

## Self-Review

**Spec coverage.**

| Spec section | Task |
|---|---|
| The contribution / non-trivial identifiability rate | 13 |
| Graph semantics, causal-admissible partition | 1, 2 |
| Mendelian justification for gene → disease | 2 (docstring), 4 |
| OmniPath as a source, provenance preserved | 5 |
| Corroboration de-duplication | 9 (step 4) |
| Identification: mutilated graph, `is_d_separator`, typed verdict | 3 |
| Closed `reason` taxonomy | 3 |
| Search bound reported, not hidden | 3 |
| The open-world problem, trivial vs non-trivial, sensitivity | 3, 13 |
| Cyclic components refused and reported | 2, 3, 13 |
| Statistics deleted (effect, p-value, both CIs) | 7 |
| Gate scoped to the causal package | 8 |
| RL: wiring bug | 10 |
| RL: `evaluate_actions`, batched | 10 |
| RL: elite confirmation and exploitation gap | 11 |
| Structure: `identifier.py` split | 9 |
| Structure: `drug_discovery.py`, pipeline package | 12 |
| networkx floor verified not assumed | 3 (step 5) |
| Manifest armed at the end | 13 (step 7) |

**Placeholder scan.** Three steps direct the implementer to match an existing
name rather than quoting it: Task 8 step 3 (`Finding`'s constructor), Task 10
steps 6-7 (the reward-learner attribute and fallback in `_compute_reward`), and
Task 12 step 6 (the objective key for synthetic accessibility). These are
instructions to read one adjacent line, not deferred decisions, and each says
explicitly what not to do instead — do not change `Finding`, do not invent an
`sa` key. Task 12's moves name the responsibility rather than every line,
because the helper set is what `drug_discovery.py` contains at that point and
the size test is the objective check.

**Type consistency.** `Identification` is constructed in Task 3 and consumed in
Task 6 with the same five fields. `IdentificationReason` values are quoted as
strings in Tasks 6, 8, and 13 and as enum members in Task 3; `identification_reason`
is a `str` on `NeoRxResult`, matching. `evidence_class`, `sign`, and
`primary_sources` are added in Task 1, read in Task 2, written in Tasks 4 and 5,
and consumed in Task 9. `evaluate_actions` has signature
`(state, actions) -> NDArray` in Task 10 and is called with that signature in
Task 11. `decode_latent_batch(model, tokenizer, Z)` matches between Task 10's
definition and its two call sites.

**One deliberate behaviour change beyond the spec.** Task 11 turns
`_plan_molecule_cem`'s `reward_fn is None` early return into a raised
`ValueError`. The spec says to fix the wiring; returning noise on a `None`
reward function is the mechanism that hid the bug for as long as it did, so the
guard is inverted rather than merely bypassed. Flagged here because it can
break a caller that relied on the silent fallback.
