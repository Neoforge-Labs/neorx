# Which edges earn an arrow

A disease graph assembled from public sources mixes two kinds of edge,
and only one of them is an arrow.

Some edges assert **a mechanism with a direction**: OmniPath's signed
regulatory interactions, where a curator has established that A acts on
B. Others assert only that two things **occur together** — a literature
co-mention, membership of a shared pathway, an undirected protein
interaction whose arrow exists because one protein happened to land in
the first column of an API response.

Running do-calculus over the second kind assumes the very thing a causal
method is supposed to establish. So NeoRx partitions the edges before
identification runs, and identification runs only on the
causal-admissible subgraph.

Associational edges are not discarded. They keep contributing to evidence
scoring and robustness — see [From verdict to target](targets.md). They
stop being arrows.

## The rule

Two edge attributes decide it: `edge_type` and `evidence_class`.

```python
>>> from neorx.core.causal.graph_semantics import CAUSAL_EDGE_TYPES
>>> sorted(CAUSAL_EDGE_TYPES)
['activates', 'causes', 'downregulates', 'inhibits', 'phosphorylates',
 'regulates', 'upregulates']

```

Those seven are directed and mechanistic: the source asserts which way
the influence runs. An edge of one of these types is admissible on its
type alone.

```python
>>> from neorx.core.causal.graph_semantics import is_causal_admissible
>>> is_causal_admissible({"edge_type": "inhibits"})
True
>>> is_causal_admissible({"edge_type": "interacts_with"})
False
>>> is_causal_admissible({"edge_type": "participates_in"})
False

```

`interacts_with` is the instructive one. A STRING protein–protein
interaction is real biology, and it has no direction. The arrow in the
graph points whichever way the API happened to order the pair.

## Gene to disease needs genetics

`associated_with` — the edge every association database produces — is
admissible only under genetic evidence:

```python
>>> from neorx.core.causal.graph_semantics import GENETIC_EVIDENCE_CLASSES
>>> sorted(GENETIC_EVIDENCE_CLASSES)
['genetic_association', 'somatic_mutation']
>>> is_causal_admissible({"edge_type": "associated_with",
...                       "evidence_class": "genetic_association"})
True
>>> is_causal_admissible({"edge_type": "associated_with",
...                       "evidence_class": "literature"})
False
>>> is_causal_admissible({"edge_type": "associated_with",
...                       "evidence_class": "known_drug"})
False

```

The warrant is Mendelian randomisation. A germline variant is randomised
at conception, independently of the confounders that plague an
observational comparison, so a genetic association carries a
natural-experiment argument that a co-mention does not (Davey Smith &
Ebrahim, 2003, *Int J Epidemiol* 32:1–22). That argument is what earns
the arrow.

`known_drug` evidence is excluded for a different reason worth stating:
it is the strongest-looking evidence class in Open Targets and it is
circular here. A gene is linked to a disease by `known_drug` *because*
someone already developed a drug for it. Admitting that edge would let
the pipeline rediscover approved targets and call it a prediction.

## Taking the subgraph

`causal_subgraph` keeps the admissible edges with their attributes and
drops nodes left with none — a node participating in no causal relation
cannot be a treatment, an outcome, or a confounder.

```python
>>> import networkx as nx
>>> from neorx.core.causal.graph_semantics import causal_subgraph
>>> G = nx.DiGraph()
>>> G.add_edge("CCL5", "CCR5", edge_type="regulates",
...            evidence_class="regulatory")
>>> G.add_edge("CCR5", "HIV", edge_type="associated_with",
...            evidence_class="genetic_association")
>>> G.add_edge("TNF", "HIV", edge_type="associated_with",
...            evidence_class="literature")
>>> G.add_edge("TNF", "IL6", edge_type="interacts_with",
...            evidence_class="ppi")
>>> sub = causal_subgraph(G)
>>> sorted(sub.nodes)
['CCL5', 'CCR5', 'HIV']
>>> sorted(sub.edges)
[('CCL5', 'CCR5'), ('CCR5', 'HIV')]

```

TNF and IL6 are gone from the subgraph entirely. That is the mechanism
behind the `no_causal_path` verdict in
[the worked example](worked-example.md): once restricted, TNF has no
route to the outcome.

## Cycles

Biology has feedback loops, and a loop is not a DAG. Rather than
silently breaking cycles — which would mean choosing a direction the
data does not support — the admissible subgraph is reduced to its
acyclic core, and what was removed is reported.

```python
>>> from neorx.core.causal.graph_semantics import cyclic_components, acyclic_core
>>> H = nx.DiGraph()
>>> H.add_edge("A", "B", edge_type="activates")
>>> H.add_edge("B", "A", edge_type="inhibits")
>>> H.add_edge("B", "C", edge_type="activates")
>>> [sorted(component) for component in cyclic_components(H)]
[['A', 'B']]
>>> core = acyclic_core(H)
>>> sorted(core.dag.nodes)
['C']
>>> sorted(core.excluded)
['A', 'B']

```

A node inside a cycle is not a candidate target in that run, and the
verdict for it says `cyclic_component` rather than reporting a result
computed on a graph that was quietly rewritten. See
[Identification](identification.md).

## Widening this is a decision, not a tweak

`CAUSAL_EDGE_TYPES` and `GENETIC_EVIDENCE_CLASSES` are the whole
interface between "what the sources said" and "what the causal engine may
reason over". Adding a name to either silently widens what the backdoor
criterion is allowed to treat as a mechanism.

The same frozensets appear in the snapshot layer as `GENETIC_DATATYPES`,
for the same reason — the candidate frame for a dated run is drawn only
from evidence types that confer admissibility. See
[Versioned snapshots](snapshots.md).

## This part is also a standalone package

Admissibility and identification have no biomedical model of their own:
they read two edge attributes and graph structure. They have been
extracted into [CausalPhylo](https://causalphylo.neoforgelabs.tech/),
which does this over any NetworkX graph with one runtime dependency.
If identification over your own graph is what you want, depend on that
rather than on NeoRx.

NeoRx still carries its own copy of these modules; CausalPhylo is not
yet a declared dependency here.
