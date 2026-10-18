# From verdict to target

An identification verdict says whether an effect *could* be computed. It
does not say the gene is a drug target. Between the two sit an evidence
count, a biological classifier, a tissue gate, a sensitivity check and a
confidence score — each of which can veto.

This page is about that machinery, which lives in
`neorx.core.causal.evidence`, `neorx.core.causal.scoring`,
`neorx.core.causal.ranking` and `neorx.core.bio`.

## Causal confidence

`compute_causal_confidence` combines six terms. Identifiability is one of
them, weighted 0.15 — a target is not promoted on the backdoor criterion
alone.

| Term | Weight |
|---|---|
| Causal effect magnitude | 0.30 |
| Robustness (sensitivity analysis) | 0.25 |
| Identifiability (backdoor criterion) | 0.15 |
| Multi-source consensus | 0.10 |
| Disease specificity | 0.10 |
| Druggability | 0.10 |

```python
>>> from neorx.core.causal.scoring import compute_causal_confidence
>>> float(compute_causal_confidence(
...     effect=0.8,
...     robustness=0.9,
...     is_identifiable=True,
...     n_pathways=5,
...     n_interactions=10,
...     source_scores={"OpenTargets": 0.8, "Monarch": 0.7},
...     druggability=0.6,
...     n_active_sources=2,
...     n_associated_diseases=3,
... ))
0.7931

```

(The `float()` is not decoration: the function is annotated `-> float` but
returns a NumPy scalar, so the bare repr would read `np.float64(0.7931)`.)

**Disease specificity** deserves a note, because it replaced a network
centrality term and the reversal was deliberate. Hub genes — TP53, AKT1 —
are associated with thousands of diseases. Under a centrality weight they
score highest everywhere, which is the opposite of useful: they are
generic, not specific. Specificity is `1 / log2(n_diseases + 2)`, so a
gene linked to one disease scores 1.0 and a gene linked to a thousand
scores about 0.10. With no disease-count data the term is a neutral 0.5,
neither rewarding nor penalising.

## The biological classifier

Before confidence is consulted at all, a candidate is typed. This exists
because the HIV/TNF failure is not a scoring problem — TNF genuinely has
strong, multi-source, reproducible evidence of association. It is a
*category* problem.

```python
>>> from neorx.core.bio.classifier import (TargetClassifier, DiseaseType,
...                                        classify_disease)
>>> classify_disease("HIV").value
'infectious_viral'
>>> classify_disease("Malaria").value
'infectious_parasitic'
>>> classify_disease("Type 2 Diabetes").value
'metabolic'
>>> classify_disease("Alzheimer Disease").value
'neurodegenerative'

```

The classifier is disease-type aware, so the same gene can be typed
differently depending on what is being treated:

```python
>>> classifier = TargetClassifier()
>>> target_type, _ = classifier.classify("CCR5", DiseaseType.INFECTIOUS_VIRAL)
>>> target_type.value
'host_invasion'
>>> target_type, explanation = classifier.classify("TNF", DiseaseType.INFECTIOUS_VIRAL)
>>> target_type.value
'host_symptom'
>>> explanation.startswith("TNF is an inflammatory cytokine")
True

```

CCR5 is a host protein the pathogen uses to enter the cell — blocking the
interaction can prevent infection. TNF is a cytokine whose elevation is a
consequence of immune activation. The classifier says so in prose, and
that prose ends up in the report.

Five target types:

```python
>>> from neorx.core.bio.classifier import TargetType
>>> [t.value for t in TargetType]
['pathogen_direct', 'host_invasion', 'host_immune', 'host_symptom',
 'correlational']

```

`pathogen_direct` is the non-human case — *Plasmodium* DHFR-TS, HIV Pol —
reached through the ChEMBL pathogen target pipeline.

### The symptom-marker blocklist

Nine protein families are blocked outright as direct targets for
infectious disease, because they encode host neurophysiological and
systemic responses rather than pathogen biology:

```python
>>> from neorx.core.bio.classifier import SYMPTOM_MARKER_FAMILIES
>>> sorted(SYMPTOM_MARKER_FAMILIES)
['GABA_receptors', 'acute_phase_proteins', 'coagulation_factors',
 'dopamine_receptors', 'glutamate_receptors', 'neuronal_structural',
 'potassium_channels', 'serotonin_receptors', 'sodium_channels']
>>> sum(len(symbols) for symbols in SYMPTOM_MARKER_FAMILIES.values())
118

```

The motivating case is cerebral malaria, where GABA receptors and sodium
channels appear in the association tables because seizures are a
symptom. Inhibiting GABRA1 treats the seizure; it does nothing to the
parasite.

## Classification, and what overrides what

`classify_target` returns a label and the reasoning behind it. Two
conditions override the confidence score entirely.

```python
>>> from neorx.core.causal.scoring import classify_target
>>> label, _ = classify_target(
...     "GABRA1", causal_confidence=0.95, robustness=0.99,
...     is_identifiable=True, n_pathways=5, druggability=0.9,
...     target_type=TargetType.HOST_SYMPTOM,
...     tissue_relevant=True, evidence_streams=5,
... )
>>> label.value
'correlational'

```

Confidence 0.95, robust, identifiable, five evidence streams — and the
verdict is `correlational`, because a symptom marker is never a causal
drug target. A statistical score does not overrule the biology.

The second override is the tissue gate:

```python
>>> label, _ = classify_target(
...     "SOME_GENE", causal_confidence=0.95, robustness=0.99,
...     is_identifiable=True, n_pathways=5, druggability=0.9,
...     target_type=TargetType.HOST_INVASION,
...     tissue_relevant=False, evidence_streams=5,
... )
>>> label.value
'correlational'

```

For `causal`, all of the following must hold: at least two independent
evidence streams, confidence ≥ 0.6, robust, and identifiable.

```python
>>> label, reasoning = classify_target(
...     "CCR5", causal_confidence=0.72, robustness=0.9,
...     is_identifiable=True, n_pathways=5, druggability=0.6,
...     target_type=TargetType.HOST_INVASION,
...     tissue_relevant=True, evidence_streams=3,
... )
>>> label.value
'causal'
>>> "Backdoor criterion satisfied" in reasoning
True
>>> "Tissue gate: PASS" in reasoning
True

```

Below 0.4, or not robust, is `correlational`. Everything between is
`inconclusive` — and `inconclusive` is a real answer, not a failure to
decide. It is what the [worked example](worked-example.md) returns for
both of its identifiable genes.

## The tissue gate

The gate asks whether the gene is expressed in a tissue the disease
affects, using the Human Protein Atlas. It is a **boolean**, not a
modifier: it never adjusts `causal_confidence`. Confidence measures the
quality of causal evidence; expression is a separate question, and
blending them would make neither readable.

Unknown expression passes. The gate exists to catch a gene expressed
*only* in irrelevant tissues, not to penalise a gene the atlas has no
data for.

!!! warning "A dated build does not apply the gate"
    The HPA query is live and nothing pins it. On a graph built from a
    past release, today's HPA answer would decide `tissue_relevant`, and
    a failed gate forces `correlational` whatever the confidence — so an
    unpinned source would be setting a reported classification on a dated
    build. A dated build therefore skips the gate.

    **Not applying the gate is not the same as passing it.** The
    difference is recorded on every result rather than left to be
    inferred from a `tissue_relevant=True` that looks like a pass. A
    dated run is systematically more permissive here than a live one, and
    the two must not be pooled. See [Versioned snapshots](snapshots.md).

## Robustness

Robustness comes from a leave-one-source-out sensitivity analysis: for
each contributing source, remove its evidence and recompute. An estimate
that survives source removals is more credible than one carried by a
single database.

This is also where the associational edges earn their keep. They are not
admissible as arrows, but they still count toward evidence streams,
corroboration and the robustness check — being excluded from the causal
subgraph is not being thrown away.

## Evaluation and reporting are separate functions

`evaluate_all_targets` evaluates the whole candidate population.
`identify_causal_targets` ranks and truncates it. The split is not
stylistic:

```python
>>> import inspect
>>> from neorx.core.causal.ranking import rank_causal_targets
>>> params = inspect.signature(rank_causal_targets).parameters
>>> [(name, p.default) for name, p in params.items()]
[('results', <class 'inspect._empty'>), ('top_n', 10),
 ('min_causal_confidence', 0.3)]

```

Ranking is by `causal_confidence`, which awards a bonus for being
identifiable. So the reported slice is a sample selected partly by a
property one might want to measure over the candidates. **Any statistic
about the procedure — an identifiability rate, a failure breakdown —
belongs on `evaluate_all_targets`.** Computing it over the top-N measures
the ranking, not the method.

Ranking also splits its slots between human and pathogen pools, each
guaranteed at least half, so pathogen targets cannot crowd out human
ones (or the reverse) when one pool happens to score higher.

## The result object

Every evaluation returns a `NeoRxResult` carrying the verdict, the
adjustment set, the near-miss count, the truncation flag, the tissue
explanation, the per-source scores and the reasoning text. See the
[API reference](api.md#neorxcoregraphmodels).
