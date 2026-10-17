# Identification

Identification asks whether the causal effect of intervening on a target
can be computed from observational data at all. It is a question about
the *graph*, answered before any number is estimated — and it is the
question that separates NeoRx from ranking by association score.

The criterion is Pearl's backdoor criterion (Pearl, *Causality*, 2nd ed.,
Def. 3.3.1): a set `Z` is admissible relative to `(X, Y)` when no node in
`Z` is a descendant of `X`, and `Z` blocks every path between `X` and `Y`
that starts with an arrow **into** `X`.

Both halves matter, and the second is the one a naive implementation gets
wrong.

## The entry point

`find_adjustment_set` takes the **full** graph. Restriction to the
[causal-admissible subgraph](admissibility.md) and to its acyclic core
happens inside, so no caller can forget that step.

```python
>>> import networkx as nx
>>> from neorx.core.causal.backdoor import find_adjustment_set
>>> G = nx.DiGraph()
>>> G.add_edge("CCL5", "CCR5", edge_type="regulates",
...            evidence_class="regulatory")
>>> G.add_edge("CCL5", "HIV", edge_type="associated_with",
...            evidence_class="genetic_association")
>>> G.add_edge("CCR5", "HIV", edge_type="associated_with",
...            evidence_class="genetic_association")
>>> verdict = find_adjustment_set(G, "CCR5", "HIV")
>>> verdict.identifiable
True
>>> verdict.adjustment_set
('CCL5',)
>>> verdict.reason.value
'identifiable_by_adjustment'

```

CCL5 regulates CCR5 and is itself associated with the disease, opening
the backdoor path `CCR5 ← CCL5 → HIV`. Conditioning on CCL5 closes it.

## It tells you why it failed

A method reporting only pass or fail hands you prose to read. NeoRx's
reasons are a closed enum, so a run's failures aggregate into a
distribution:

```python
>>> from neorx.core.causal.backdoor import IdentificationReason
>>> [r.value for r in IdentificationReason]
['identifiable_by_adjustment', 'identifiable_trivially', 'no_causal_path',
 'cyclic_component', 'no_valid_adjustment_set', 'treatment_absent',
 'outcome_absent']

```

| Reason | What it means |
|---|---|
| `identifiable_by_adjustment` | A valid non-empty adjustment set was found, and it is named on the verdict |
| `identifiable_trivially` | The empty set satisfies the criterion — nothing in the graph confounds the pair. See below |
| `no_causal_path` | The node exists, but has no admissible route to the outcome. This is the verdict for a target supported only by co-mention |
| `cyclic_component` | The treatment or outcome sits in a feedback loop; d-separation is undefined there |
| `no_valid_adjustment_set` | A backdoor path exists and no admissible set within the search bounds blocks it |
| `treatment_absent` / `outcome_absent` | The node is not in the input graph at all — a typo or a stale identifier |

`no_causal_path` and `treatment_absent` are deliberately different.
Presence is checked against the **full** input graph, not the admissible
subgraph, because a node that exists but carries no admissible edge is
a finding about the evidence — reporting it as "absent" would blame the
query instead.

```python
>>> find_adjustment_set(G, "NOT_A_GENE", "HIV").reason.value
'treatment_absent'

```

## Trivial is not clean

When no confounder appears in the graph, the empty set satisfies the
criterion and the effect is identifiable. But a knowledge graph is
open-world: **a missing edge is missing knowledge, not evidence of no
confounding.**

So a trivial verdict gets its own reason, and carries a count of near
misses — nodes that regulate the treatment and have *some* association
with the outcome that fell short of admissibility:

```python
>>> H = nx.DiGraph()
>>> H.add_edge("CCL5", "CCR5", edge_type="regulates")
>>> H.add_edge("CCR5", "HIV", edge_type="associated_with",
...            evidence_class="genetic_association")
>>> H.add_edge("CCL5", "HIV", edge_type="associated_with",
...            evidence_class="literature")
>>> trivial = find_adjustment_set(H, "CCR5", "HIV")
>>> trivial.reason.value
'identifiable_trivially'
>>> trivial.n_near_miss_confounders
1

```

The graph says CCL5 confounds this pair. It just does not say so with
evidence strong enough to adjust on. A target with many near misses is
fragile, and this is the same move as an E-value in observational
epidemiology: not *"there is no confounding"* but *"here is how much it
would take to break this"*.

## Two different mutilations

The module exposes both, and conflating them is a real bug class.

`mutilated_graph` removes edges **into** the treatment. That is the
interventional graph `G_X̄` — what the do-operator's truncated
factorisation is defined on.

```python
>>> from neorx.core.causal.backdoor import mutilated_graph
>>> M = nx.DiGraph([("U", "X"), ("X", "Y"), ("U", "Y")])
>>> sorted(mutilated_graph(M, "X").edges)
[('U', 'Y'), ('X', 'Y')]
>>> sorted(M.edges)   # the input is not modified
[('U', 'X'), ('U', 'Y'), ('X', 'Y')]

```

`satisfies_backdoor` removes edges **out of** the treatment instead. This
is what makes a generic d-separation routine answer the right question:
deleting the out-edges removes the causal path (and anything `X` causes)
while leaving every backdoor path — the ones entering `X` through an
arrow — exactly as it was. Without it, the direct causal edge registers
as "still connected" and masks an open backdoor path.

```python
>>> from neorx.core.causal.backdoor import satisfies_backdoor
>>> D = nx.DiGraph([("U", "X"), ("U", "Y"), ("X", "Y")])
>>> satisfies_backdoor(D, "X", "Y", frozenset())        # U is unblocked
False
>>> satisfies_backdoor(D, "X", "Y", frozenset({"U"}))   # blocked
True
>>> satisfies_backdoor(D, "X", "Y", frozenset({"Y"}))   # outcome in Z
False

```

The third case is the descendant condition doing its job: conditioning on
a mediator or a collider downstream of the treatment is not adjustment,
it is a different estimand.

## Search bounds, reported rather than hidden

The search enumerates candidate subsets, so it is bounded:

```python
>>> from neorx.core.causal.backdoor import (MAX_ADJUSTMENT_SET_SIZE,
...                                         MAX_CANDIDATE_POOL)
>>> MAX_ADJUSTMENT_SET_SIZE
3
>>> MAX_CANDIDATE_POOL
20

```

The candidate pool is non-descendants of the treatment, ordered by node
score so the strongest evidence is considered first. A larger bound buys
little and costs combinatorially — the number of subsets grows as
`C(pool, size)`.

When either bound is reached, `Identification.search_truncated` is set.
A truncated search that found nothing means *"no valid set within these
bounds"*, which is weaker than *"no valid set"*, and the flag is how a
reader can tell the difference.

## Cycles are refused, not repaired

```python
>>> C = nx.DiGraph()
>>> C.add_edge("A", "B", edge_type="activates")
>>> C.add_edge("B", "A", edge_type="inhibits")
>>> C.add_edge("A", "D", edge_type="associated_with",
...            evidence_class="genetic_association")
>>> find_adjustment_set(C, "A", "D").reason.value
'cyclic_component'

```

`nx.is_d_separator` requires a DAG of the whole graph, not just of the
neighbourhood under test. Breaking the loop would mean choosing a
direction the data does not support, so the nodes come out and the
verdict says so.

## From verdict to target

A verdict is not a target. Identifiability feeds into a causal
confidence alongside evidence counts, druggability, a tissue gate and a
robustness check — see [From verdict to target](targets.md).

## The standalone package

This module has no biomedical content: it reads edge attributes and graph
structure. It has been extracted as
[CausalPhylo](https://causalphylo.neoforgelabs.tech/) for use over any
NetworkX graph. NeoRx still carries its own copy and does not yet depend
on the package.
