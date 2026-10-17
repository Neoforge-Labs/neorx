# NeoRx

> Correlation is not causation. NeoRx is built around telling the difference.

NeoRx is a drug-target discovery platform that asks an interventional
question of a biomedical knowledge graph: **if we inhibited this protein,
would the disease outcome change?** Association databases answer a
different question — whether the gene and the disease occur together —
and the two answers routinely disagree.

It is one Python package with several engines: graph assembly from
biomedical sources, causal identification over that graph, molecule
generation, drug-likeness screening, docking, a reinforcement-learning
mode, and an experiment runner that refuses to report a number without a
run record.

## Why the distinction matters

Association databases report that **TNF** is strongly associated with
HIV, and they are right: TNF-α is elevated during infection. But that
elevation is *downstream* of immune activation. Inhibiting TNF does not
treat HIV — it removes an immune response the patient needs.

**CCR5** is the HIV-1 co-receptor. The CCR5-Δ32 loss-of-function variant
confers near-complete resistance, and maraviroc, which blocks CCR5, is an
approved antiretroviral. CCR5 is a target you can intervene on.

Both genes sit high in the same association tables. What separates them
is not the strength of the association but its *direction* — and nothing
in an association score carries direction.

NeoRx's answer is to refuse to treat the two kinds of claim alike:

```python
>>> from causalphylo import is_causal_admissible
>>> is_causal_admissible({"edge_type": "associated_with",
...                       "evidence_class": "genetic_association"})
True
>>> is_causal_admissible({"edge_type": "associated_with",
...                       "evidence_class": "literature"})
False

```

A germline variant is randomised at conception, so a genetic association
carries a natural-experiment warrant a literature co-mention does not.
That warrant is what earns an arrow. Everything the causal engine does
runs on the subgraph of edges that have one — see
[Which edges earn an arrow](admissibility.md).

## The shape of a run

1. **Build the graph.** Query the biomedical sources for a disease, merge
   duplicate entities to gene symbols, and record the provenance of every
   edge. See [The graph and its sources](graph.md).
2. **Identify targets.** Restrict to the causal-admissible subgraph, then
   apply Pearl's backdoor criterion per candidate, reporting *why*
   identification succeeded or failed. See
   [Identification](identification.md).
3. **Rank and classify.** Combine identifiability with evidence counts,
   druggability and a tissue-expression gate into a causal confidence and
   a `causal` / `correlational` / `inconclusive` label. See
   [From verdict to target](targets.md).
4. **Design molecules.** Sample the [GenMol](genmol.md) VAE for each
   surviving target, screen with [MolScreen](molscreen.md), dock with
   [DockBot](dockbot.md).
5. **Score.** A six-dimension composite in which causal confidence
   carries the largest weight. See [Ranking candidates](scoring.md).
6. **Report.** An HTML report with the graph, the target reasoning, and
   the ranked candidate table.

Steps 1–6 are `neorx run <disease>`, or `run_pipeline()` in Python. Each
is also reachable on its own.

## Thirty seconds

!!! warning "Requires network"
    `build_disease_graph` queries the live biomedical APIs, so this block
    is not executed as a doctest and no output is shown for it — what
    comes back depends on what those sources say today.

```python
from neorx import build_disease_graph, identify_causal_targets

graph = build_disease_graph("HIV")
for target in identify_causal_targets(graph, top_n=5):
    print(target.gene_name, target.classification.value)
```

For a version you can run offline and check line by line, see
[A worked example](worked-example.md).

## Where the causal engine lives

The graph-only part of this work — admissibility and the backdoor
criterion, with no biomedical model of its own — has been extracted into
a standalone package, [CausalPhylo](https://causalphylo.neoforgelabs.tech/).
If you want identification over your own graph rather than drug
discovery, that is the smaller thing to depend on: one runtime
dependency, NetworkX.

NeoRx depends on it rather than carrying a second copy, so
`is_causal_admissible` above is CausalPhylo's. The division is along the
dependency graph, not by subject: everything that needs only a graph
went, and everything written against NeoRx's own data model — evidence
counting, scoring, ranking, target classification — stayed.

!!! note "Mid-move, as of 2026-10-17"
    The extraction is open as a pull request and not yet on `main`. Until
    it merges, a checkout of `main` still contains
    `neorx/core/causal/graph_semantics.py` and `backdoor.py`, and
    `causalphylo` is declared only in the `docs` dependency group — these
    pages are written for the merged state, and import from
    `causalphylo`, because that is the package the code is moving to and
    the one you should write against. Delete this note once the move
    lands.

## Reproducibility is not a feature of this project

It is a constraint on it. A manuscript audit of this codebase found that
no reported number could be traced to an execution artifact: a benchmark
that persisted nothing, figures holding F₁ values as literals, a table
entry retyped by hand from lost terminal output. The
[experiment runner](experiments.md) exists so that an experiment has no
code path by which it reports a result without writing a record, and the
[snapshot layer](snapshots.md) exists so that a run over a past release
reads that release rather than today's.

## Author and licence

**Kelyn Paul Njeri** · [NeoForge Labs](mailto:kelyn@neoforgelabs.tech) ·
[ORCID 0009-0000-1068-4512](https://orcid.org/0009-0000-1068-4512)

Source available, non-commercial. See [Licence](licence.md).
