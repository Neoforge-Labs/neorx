# Correctness — Design

**Date:** 2026-09-03
**Status:** Approved for planning
**Sub-project:** 3 of 5

---

## Context

Sub-project 2 made every reported number traceable to a run record. It
deliberately did not ask whether the numbers were *right*. This sub-project
does.

The causal engine claims to apply Pearl's backdoor criterion. It does not, and
the reasons compound:

**The d-separation test never executes.** `identifier.py` calls
`nx.d_separated`, removed from NetworkX in favour of `nx.is_d_separator`. The
installed version is 3.6.1, so every call raises `AttributeError`. At
`_compute_adjustment_set` the `if`, the `else`, and the `except` all append the
candidate node, so the three branches are identical and the test is decorative.

**The test is the wrong one anyway.** The backdoor criterion requires
d-separation in the mutilated graph *G*<sub>X̄</sub>, with edges into the
treatment removed. The code tests d-separation in *G*.

**The graph cannot support the criterion regardless.** Every edge points away
from genes:

| Source | Edge | Direction |
|---|---|---|
| Monarch, OpenTargets, ChEMBL | `ASSOCIATED_WITH` | gene → disease |
| KEGG, Reactome | `PARTICIPATES_IN` | gene → pathway |
| STRING | `INTERACTS_WITH` | gene → gene |

A gene's only possible parent is another gene via STRING. STRING interactions
are undirected; an arrow exists only because one protein landed in the first
column of the API response. So `G.predecessors(treatment)` yields protein
interaction partners sorted by tuple position, and a PPI partner is a correlate
of a gene, not a common cause of a gene and a disease. The graph holds no
upstream variables at all — no cell type, no ancestry, no assay platform.
Verified on the star topology `graph_builder` produces: the adjustment set is
empty for every gene.

**Three reported statistics are manufactured.** `_estimate_causal_effect`
multiplies heuristic factors and derives `p_value = max(0.001, 0.5 * (1 -
|effect|))` — a p-value with no data and no null distribution.
`_bootstrap_confidence_interval` resamples nothing; it adds hand-chosen
Gaussian noise to deterministic scores, so every confidence interval in the
manuscripts is a readout of the constants `0.05` and `0.03`.

The RL half has a defect of the same character. `pipeline.py:812` calls
`agent.init_hierarchical_planner()` before `agent.train()`. `_reward_fn` is
`None` until `train()` assigns it, and `HierarchicalPlanner` captures the value
rather than a reference, so `_plan_molecule_cem` hits its `reward_fn is None`
guard and returns `rng.standard_normal(latent_dim) * 0.3`. The CEM inner loop
never executes in the end-to-end pipeline. Every molecule the pipeline has ever
reported came from Gaussian noise in latent space.

### Decomposition

Sub-project 3 of five. The fifth was added by this design; see *σ-separation*
under **Deferred**.

| # | Sub-project | Status |
|---|-------------|--------|
| 1 | Package skeleton | Complete — merged at `43d3ad2` |
| 2 | Experiment tracking | Complete — merged at `9117816` |
| **3** | **Correctness** | **This spec** |
| 4 | Test hardening | Behaviour tests; molscreen from zero coverage |
| 5 | σ-separation for cyclic SCMs | Deferred from this spec |

---

## Goals

1. Identifiability is a reported outcome, not an assumed precondition.
2. The backdoor criterion, where applied, is applied correctly.
3. No statistic is reported that the available data cannot support.
4. The CEM planner optimises molecules that exist.

## Non-goals

- **Reproducing the published F₁ = 0.474.** The papers are four months old and
  will be rewritten against the corrected engine. Preserving the old number is
  not a constraint on this work; it is the thing this work invalidates.
- **Causal inference from patient data.** No GEO, TCGA, or Biobank integration.
  The claim under test is what public knowledge graphs license, which is
  answerable without patient data and is the point.
- **σ-separation.** Deferred to sub-project 5.
- **Test hardening.** Sub-project 4 owns coverage. This sub-project writes the
  tests its own changes require and no more.

---

## Design

### The contribution

Target-identification pipelines score associations and call the top of the list
causal. NeoRx will instead answer, per target: *does the assembled public
evidence license a causal claim about this gene and this disease, and if so,
what must be adjusted for?*

The headline metric becomes the **non-trivial identifiability rate** — the
fraction of candidate targets for which the backdoor criterion is satisfied by
an adjustment set the graph actually supplies — reported per disease with
failure reasons broken out, alongside the trivial rate and the confounding
sensitivity distribution (see *The open-world problem*). It requires no patient
data, it is checkable by anyone with the same public sources, and a low value
is as publishable as a high one.

### Graph semantics

New module `src/neorx/core/causal/graph_semantics.py`. Edge types partition
into two classes, and identification runs only on the causal-admissible ones.

**Causal-admissible**

- gene → gene, from OmniPath, where `is_directed` and `consensus_direction`
  both hold; signed by `is_stimulation` / `is_inhibition`
- gene → disease, from OpenTargets, only where the evidence `datatypeId` is
  `genetic_association` or `somatic_mutation`

**Associational** — feeds scoring, never graph structure: `ASSOCIATED_WITH`
from literature or co-expression, `PARTICIPATES_IN`, `INTERACTS_WITH`, `BINDS`,
`TREATS`.

The gene → disease restriction carries the design. A germline variant is
randomised at conception, so a genetic association carries a natural-experiment
warrant that a literature co-mention does not — the Mendelian randomisation
argument (Davey Smith & Ebrahim, 2003). That warrant is what earns the arrow.

The restriction also gives the criterion real work: a gene *U* that regulates
*X* in OmniPath while carrying its own genetic association to *D* creates a
genuine backdoor path *X* ← *U* → *D*.

The module exposes:

- `CAUSAL_EDGE_TYPES: frozenset[EdgeType]`
- `causal_subgraph(G) -> nx.DiGraph` — the induced subgraph on
  causal-admissible edges, carrying edge sign and provenance
- `cyclic_components(G) -> list[set[str]]` — strongly connected components of
  size > 1

STRING's `INTERACTS_WITH` edges become associational. They keep contributing to
evidence scoring; they stop masquerading as arrows.

### OmniPath as a source

New `src/neorx/core/sources/omnipath.py`, following the existing source module
contract. The REST endpoint returns, per interaction: `is_directed`,
`is_stimulation`, `is_inhibition`, `consensus_direction`, a `sources` list of
contributing primary databases, and `references` carrying PMIDs
(`SIGNOR:16331690;KEA:14983059;…`).

Aggregation does not cost provenance: a SIGNOR-derived edge remains traceable
to SIGNOR, and the per-interaction `sources` list is recorded on the edge.

That list also has to be used, not merely stored. `_estimate_causal_effect`
computes a multi-source corroboration factor by counting distinct `source_db`
values on incident edges. OmniPath re-reports interactions that STRING already
contributed, so the count must be taken over the union of *primary* sources
after mapping OmniPath edges to their contributing databases — otherwise one
interaction inflates the factor twice.

### Identification

New module `src/neorx/core/causal/backdoor.py`:

- `mutilated_graph(G, X) -> nx.DiGraph` — G with edges *into* X removed
- `satisfies_backdoor(G, X, Y, Z) -> bool` — no z ∈ Z is a descendant of X, and
  Z d-separates X from Y in *G*<sub>X̄</sub>, via `nx.is_d_separator`
- `find_adjustment_set(G, X, Y) -> Identification` — searches candidate
  non-descendants of X by increasing set size and returns the minimal valid
  set, or a negative verdict naming the reason

`Identification` is a frozen dataclass:

```python
identifiable: bool
adjustment_set: tuple[str, ...]
reason: str
```

`reason` takes one of a closed set of values, so the failure breakdown is
aggregatable rather than free text:

| `reason` | Meaning |
|---|---|
| `identifiable_by_adjustment` | A non-empty valid adjustment set was found |
| `identifiable_trivially` | No backdoor paths exist, so the empty set suffices |
| `no_causal_path` | No directed path X → Y in the causal subgraph |
| `cyclic_component` | X or Y lies in a strongly connected component of size > 1 |
| `no_valid_adjustment_set` | Backdoor paths exist that no candidate set blocks |
| `treatment_absent` / `outcome_absent` | Node not present in the causal subgraph |

`networkx>=3.4` is pinned and `is_d_separator` imported by name. Every
`except Exception: append anyway` is deleted. An exception in identification is
a bug, not a fallback.

**Search bound.** `find_adjustment_set` searches subsets of increasing size up
to `MAX_ADJUSTMENT_SET_SIZE = 3` over a candidate pool capped at 20 nodes,
ordered by descending node score. When the bound is reached without a valid
set, the verdict is `no_valid_adjustment_set` and the run record notes the
search was truncated — a bounded search that reports its bound, never a silent
cap.

### The open-world problem

A knowledge graph is open-world. When no confounder of *X* and *D* appears in
it, the empty set trivially satisfies the backdoor criterion — and if that
counted as a success, the identifiability rate would read near 100% for the
same reason the current engine reports empty adjustment sets: absence of
knowledge, dressed as absence of confounding.

Two mechanisms keep the metric honest.

**The verdict distinguishes its two kinds.** `identifiable_trivially` and
`identifiable_by_adjustment` are separate reasons, reported separately. A
trivial verdict is a statement about the graph, not about biology, and the
manuscript must say so.

**A sensitivity analysis quantifies the exposure.** For each trivially
identifiable target, `confounding_sensitivity(G, X, Y)` reports how many
plausible confounders would have to be missing to overturn the verdict —
counted as the number of nodes that are ancestors of *X* in the causal
subgraph and carry any association to *D*, causal-admissible or not. A target
with many such near-misses is fragile; one with none is genuinely unconfounded
as far as the assembled evidence reaches.

This is the same move as an E-value in observational epidemiology: the claim is
not "there is no confounding" but "here is how much unmeasured confounding it
would take to break this." It is what makes a trivial verdict reportable rather
than embarrassing, and the headline metric is the **non-trivial**
identifiability rate, published alongside the trivial rate and the sensitivity
distribution.

### Cyclic components

OmniPath contains feedback loops; d-separation is defined on DAGs. Targets
inside a strongly connected component get `identifiable=False` with
`reason="cyclic_component"`.

Breaking cycles to raise the headline number would be a shim. Instead the
cyclic fraction is reported as its own finding: the proportion of candidate
targets sitting in feedback loops, where backdoor adjustment is undefined, is a
real observation about public biological knowledge graphs and the motivation
for sub-project 5.

### Statistics that get deleted

`effect`, `p_value`, and the confidence interval leave the public result. With
no patient data there is no null distribution and no sampling distribution;
all three are decoration, and a reviewer will say so.

They are replaced by:

- `identification: Identification` — the verdict and its adjustment set
- `evidence_score: float` — the weighted multi-source aggregate, named as what
  it is

This breaks `CausalTarget`'s public shape. That is the point of the
sub-project, and the package is pre-1.0.

`_bootstrap_confidence_interval` and the `p_value` computation are removed
outright rather than deprecated. A gate asserts neither name reappears.

### RL: the planner

Three changes.

**The wiring bug.** `_reward_fn` is assigned in `CausalAgent.__init__`, and
`train(reward_fn=…)` propagates the override to an existing hierarchical
planner instead of only rebinding the agent's attribute. A regression test
constructs agent and planner in the pipeline's exact order and asserts the
planner's reward function is not `None`.

**Batched evaluation.** `DrugDiscoveryEnv.evaluate_actions(state, actions) ->
NDArray` decodes and screens a batch of candidate actions and returns their
real rewards, mutating no environment state. Chemistry stays in the
environment; the planner stays generic. Batching matters — measured on this
machine, VAE decode costs 13.9 ms one at a time and 2.0 ms/molecule at batch
200, with screening at 13.4 ms/molecule.

**Elite confirmation.** `_plan_molecule_cem` runs its inner iterations against
the learned reward model as designed, then decodes and screens the final elite
set through `evaluate_actions` and returns the best *real*-scored elite rather
than the mean of model-scored ones. At 20 elites this costs roughly 0.3 s per
environment step.

Decoding every candidate was considered and rejected on cost: the default
budget of 200 samples × 5 iterations is 1000 evaluations per step, ~15 s/step
even with batched decode, or 43 hours for a 10,000-step run.

**A result falls out of this.** The gap between model-predicted and real scores
on the elite set measures model exploitation in latent-space molecular RL
directly. It is logged per step and becomes a figure in the rewritten
manuscript.

### Structure

| File | Now | After |
|---|---|---|
| `core/causal/identifier.py` | 1291 | orchestration, plus `backdoor.py`, `graph_semantics.py`, `scoring.py`, `evidence.py` |
| `causalbiorl/envs/drug_discovery.py` | 884 | env, plus `screening.py`, `generation.py` |
| `core/pipeline.py` | 920 | becomes a package: `pipeline/__init__.py` re-exporting the current public names, plus `pipeline/rl_stage.py` |

The splits follow the seams the work already creates. Converting `pipeline.py`
to a package keeps `from neorx.core.pipeline import …` working unchanged for
every existing caller. It is the lowest-value of the three splits and the first
to drop if the implementation plan runs long.

---

## Consequences

**F₁ will move, in a direction nobody can predict before running it.** It is
the number in all four manuscripts. This is expected and is not a regression.

**Manuscript revisions follow from this work**, beyond the two SP2 already
identified:

- p-values, confidence intervals, and effect sizes come out of every table
- OmniPath joins the data-sources table; STRING's role changes from graph
  structure to evidence scoring
- the trivial and non-trivial identifiability rates, the confounding
  sensitivity distribution, and the cyclic fraction are new headline results
- the causal claim is restricted to genetically-supported gene–disease pairs,
  which narrows scope and strengthens the warrant

**`docs/run-manifest.toml` is armed at the end of this sub-project**, not
before. Arming it earlier would bind manuscript tables to numbers from an
engine this work invalidates.

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Method framing | Report identifiability; don't assume it | The graph cannot support the current claim, and the honest verdict is itself the novel result |
| Causal arrows | Typed subgraph; genetic evidence only for gene → disease | Mendelian randomisation supplies the warrant a co-mention cannot |
| Regulatory source | OmniPath | Best coverage, and its per-interaction `sources` + PMIDs preserve provenance through aggregation |
| STRING | Demoted to associational | Undirected data; current arrows are tuple-column artefacts |
| Open-world graph | Split trivial from non-trivial identifiability; add a sensitivity count | Absence of a confounder in a KB is absence of knowledge; counting it as success would inflate the metric to ~100% |
| Cycles | Refuse, and report the fraction | Breaking cycles to lift the number would be a shim; the fraction is a finding |
| `p_value`, CI, `effect` | Deleted | No data supports them; the CI is a readout of two hand-chosen constants |
| CEM | Model searches, real chemistry confirms elites | Full decoding costs 43 h/run; elite confirmation costs 0.3 s/step |
| σ-separation | Sub-project 5 | Substantial lift; SP3 stays shippable |

---

## Risks

| Risk | Mitigation |
|---|---|
| OmniPath coverage is thin for some of the seven diseases, and the identifiability rate lands near zero | That is a result, reported with its failure breakdown. The `reason` taxonomy distinguishes "no causal path" from "no valid adjustment set" so the cause is legible rather than a bare zero |
| The cyclic fraction is high enough to dominate the headline | Report it as a headline in its own right; it is the motivation for SP5 |
| OmniPath's REST endpoint changes shape or its filter parameters drift | The SP2 capture gate already asserts each configured source records ≥ 1 interaction; a source contributing zero fails the run |
| Deleting `p_value` and the CI breaks downstream consumers | `CausalTarget` is a public type; the change is enumerated in the plan, and `test_public_api.py` is updated in the same task rather than after |
| Elite confirmation slows the benchmark past what a paper run tolerates | Cost is measured, not assumed: 20 elites × ~15 ms ≈ 0.3 s/step. The elite count is a parameter, and the plan records the measured wall-clock |
| The adjustment-set search bound hides valid larger sets | The bound is reported in the run record alongside the verdict, so a truncated search is never presented as an exhaustive one |

---

## Success criteria

Verified in CI on every change:

1. `satisfies_backdoor` rejects an adjustment set containing a descendant of
   the treatment, and accepts a known-valid set on a textbook confounding
   graph.
2. Identification on a graph with a genuine confounder *X* ← *U* → *D* returns
   `identifiable=True`, `adjustment_set=("U",)`, and
   `reason="identifiable_by_adjustment"`.
3. Identification on today's star topology returns `identifiable=False` with
   `reason="no_causal_path"` — the honest verdict, asserted as behaviour.
4. A graph with a causal path X → D and no backdoor path returns
   `reason="identifiable_trivially"` with an empty adjustment set, and
   `confounding_sensitivity` reports the count of near-miss confounders.
5. A target inside a strongly connected component returns
   `reason="cyclic_component"` and never raises.
6. A `HierarchicalPlanner` built in the pipeline's construction order has a
   non-`None` reward function, and its CEM performs more than one distinct
   evaluation.
7. `evaluate_actions` returns one reward per action, mutates no environment
   state, and agrees with `step()`'s reward for the same action.
8. `_bootstrap_confidence_interval`, `p_value`, and `nx.d_separated` appear
   nowhere in `src/`.
9. No module in `src/neorx/core/causal/` or `src/neorx/causalbiorl/envs/`
   exceeds 600 lines.

Verified out of band, because it touches live APIs and takes ~25 minutes:

10. A recorded `neorx_7disease` run reports trivial and non-trivial
    identifiability rates and a failure breakdown summing to the candidate
    count, and its replay reproduces its own rows.
