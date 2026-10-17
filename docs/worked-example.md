# A worked example

Everything on this page runs offline and is executed as a doctest, so the
numbers below are what the code produced rather than what it was hoped it
would produce.

The example builds a small disease graph by hand instead of querying the
live sources. That is not a convenience: it makes the example
reproducible, and it puts the edge attributes — the things the causal
engine actually reads — in plain sight.

## The graph

Four entities and four edges. The disease, a target with a genetic
association (CCR5), a target with only a literature co-mention (TNF), and
a regulator of CCR5 that is itself genetically associated with the
disease (CCL5).

```python
>>> from neorx import DiseaseGraph, GraphNode, GraphEdge, NodeType, EdgeType
>>> graph = DiseaseGraph(
...     disease_name="HIV",
...     as_of="2026-01-01",
...     nodes=[
...         GraphNode(node_id="MONDO:0005109", name="HIV",
...                   node_type=NodeType.DISEASE),
...         GraphNode(node_id="CCR5", name="CCR5", node_type=NodeType.GENE,
...                   source="OpenTargets", score=0.8),
...         GraphNode(node_id="TNF", name="TNF", node_type=NodeType.GENE,
...                   source="Monarch", score=0.9),
...         GraphNode(node_id="CCL5", name="CCL5", node_type=NodeType.GENE,
...                   source="OmniPath", score=0.5),
...     ],
...     edges=[
...         GraphEdge(source_id="CCR5", target_id="MONDO:0005109",
...                   edge_type=EdgeType.ASSOCIATED_WITH, weight=0.8,
...                   evidence_class="genetic_association"),
...         GraphEdge(source_id="TNF", target_id="MONDO:0005109",
...                   edge_type=EdgeType.ASSOCIATED_WITH, weight=0.9,
...                   evidence_class="literature"),
...         GraphEdge(source_id="CCL5", target_id="CCR5",
...                   edge_type=EdgeType.REGULATES, weight=0.7,
...                   evidence_class="regulatory"),
...         GraphEdge(source_id="CCL5", target_id="MONDO:0005109",
...                   edge_type=EdgeType.ASSOCIATED_WITH, weight=0.6,
...                   evidence_class="genetic_association"),
...     ],
...     sources_queried=["OpenTargets", "Monarch", "OmniPath"],
... )
>>> graph.n_genes
3

```

Note what TNF has: the *highest* node score in the graph, 0.9, and the
heaviest edge to the disease, 0.9. On an association ranking it wins.

!!! note "Why `as_of` is set here"
    `as_of` marks this as a dated build. A dated build does not apply the
    tissue-expression gate, because that gate queries the Human Protein
    Atlas live and nothing pins it — today's answer must not decide a
    classification on a graph built from a past release. Setting it here
    is what makes this example reach no network at all. See
    [From verdict to target](targets.md#the-tissue-gate).

## Evaluating every candidate

`evaluate_all_targets` evaluates the whole candidate population and
filters nothing. This is the function to use whenever you want a
statistic *about* the procedure.

```python
>>> from neorx.core.causal.identifier import evaluate_all_targets
>>> for result in evaluate_all_targets(graph):
...     print(f"{result.gene_name:5} {result.identifiable!s:5} "
...           f"{result.identification_reason:26} "
...           f"adj={result.adjustment_set} "
...           f"conf={result.causal_confidence:.3f} "
...           f"{result.classification.value}")
CCR5  True  identifiable_by_adjustment adj=['CCL5'] conf=0.478 inconclusive
CCL5  True  identifiable_trivially     adj=[] conf=0.408 inconclusive
TNF   False no_causal_path             adj=[] conf=0.324 correlational

```

Read that from the right. **TNF is classified correlational** despite
carrying the strongest association in the graph, and the reason is not a
threshold: it is `no_causal_path`. TNF's only edge to the disease is a
literature co-mention, which is not admissible as an arrow, so once
identification restricts to the admissible subgraph TNF has no path to
the outcome at all. There is nothing for the backdoor criterion to
reason about.

**CCR5 is identifiable by adjustment**, and the adjustment set is named:
`['CCL5']`. CCL5 regulates CCR5 and is associated with the disease, so it
opens a backdoor path `CCR5 ← CCL5 → HIV`. Conditioning on CCL5 closes
it. The verdict carries the set, not just the boolean.

**CCL5 is identifiable trivially** — the empty set satisfies the
criterion, because nothing in this graph confounds it. That is a weaker
result than it looks, and NeoRx reports it under its own reason for
exactly that purpose; see
[Identification](identification.md#trivial-is-not-clean).

Neither CCR5 nor CCL5 reaches `causal` here. With three genes and two
sources there is not enough evidence to clear the threshold, and
`inconclusive` is the honest label. Pushing a four-node toy graph to
`causal` would require weakening the classifier, which is the opposite
of the point.

## What the pipeline would do with that

`identify_causal_targets` is `evaluate_all_targets` followed by ranking
and truncation:

```python
>>> from neorx import identify_causal_targets
>>> top = identify_causal_targets(graph, top_n=2, min_causal_confidence=0.4)
>>> [t.gene_name for t in top]
['CCR5', 'CCL5']

```

Ranking is by `causal_confidence`, which awards a bonus for being
identifiable. So the returned list is a sample selected partly by a
property you might want to measure over the candidates — which is why
the two functions are separate, and why an identifiability rate computed
over `identify_causal_targets` output would be measuring itself.

## Scoring a molecule against that target

Given a target's causal confidence, a candidate molecule is scored on six
dimensions. Here is aspirin, measured rather than guessed, scored against
CCR5 at the confidence the run just produced:

```python
>>> from neorx.molscreen import (parse_smiles, qed_score, sa_score,
...                              run_all_filters, classify_drug_likeness)
>>> aspirin = parse_smiles("CC(=O)Oc1ccccc1C(O)=O")
>>> round(qed_score(aspirin), 3)
0.55
>>> round(sa_score(aspirin), 3)
1.58
>>> filters = run_all_filters(aspirin)
>>> [(f.name, f.passed) for f in filters]
[('Lipinski', True), ('Veber', True), ('Ghose', False), ('Egan', True), ('PAINS', True), ('Brenk', False)]
>>> classify_drug_likeness(filters)
'Lead-like'

```

```python
>>> from neorx import score_candidate
>>> candidate = score_candidate(
...     smiles="CC(=O)Oc1ccccc1C(O)=O",
...     target_protein_id="CCR5",
...     target_protein_name="CCR5",
...     causal_confidence=0.478,
...     binding_affinity=-7.5,
...     qed_score=0.55,
...     sa_score=1.58,
...     molecular_weight=180.04,
...     n_filters_passed=4,
... )
>>> candidate.composite_score
0.5757
>>> candidate.score_breakdown
{'causal_confidence': 0.1434, 'binding_affinity': 0.1562, 'qed': 0.0825,
 'sa': 0.0936, 'admet': 0.05, 'novelty': 0.05}

```

The breakdown is per-dimension and additive, so you can see where a
score came from. `admet` and `novelty` sit at their 0.5 defaults here
because nothing supplied them. The −7.5 kcal/mol is an illustrative
number, not a docking result: a real one comes from
[DockBot](dockbot.md).

## What this example left out

Two whole stages, both because they cannot be made offline and
deterministic:

- **Graph assembly.** `build_disease_graph("HIV")` queries eight sources
  over HTTP. See [The graph and its sources](graph.md).
- **Docking.** AutoDock Vina needs a prepared receptor and a binding
  site. See [DockBot](dockbot.md).

The run record system exists for precisely this gap: a real run writes
what it read and what it produced, so its numbers are traceable even
though they cannot be reproduced from a doctest. See
[Experiments and run records](experiments.md).
