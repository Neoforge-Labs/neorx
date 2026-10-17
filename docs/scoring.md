# Ranking candidates

Once a molecule has been generated, screened and docked, the composite
scorer turns six measurements into one number. The design decision that
matters is which of them carries the most weight.

## The six dimensions

```python
>>> from neorx.core.scoring.scorer import DEFAULT_WEIGHTS
>>> for dimension, weight in DEFAULT_WEIGHTS.items():
...     print(f"{dimension:18} {weight}")
causal_confidence  0.3
binding_affinity   0.25
qed                0.15
sa                 0.1
admet              0.1
novelty            0.1

```

**Causal confidence is weighted highest, above binding affinity.** That
is the whole point: a moderately-binding molecule aimed at a target whose
effect on the disease is identifiable ranks above a tightly-binding
molecule aimed at a correlational bystander. Conventional virtual
screening has no term for the first thing at all.

## Normalisation

Raw scores live on different scales and in different directions, so each
is mapped to [0, 1] with 1 = best before weighting.

```python
>>> from neorx import normalise_affinity, normalise_sa
>>> normalise_affinity(-12.0), normalise_affinity(-6.0), normalise_affinity(0.0)
(1.0, 0.5, 0.0)
>>> normalise_sa(1.0), round(normalise_sa(5.0), 2), normalise_sa(10.0)
(1.0, 0.56, 0.0)

```

Binding affinity runs −12 to 0 kcal/mol with more negative better; SA
runs 1 to 10 with lower better. Both are clamped, so a −15 kcal/mol
outlier scores 1.0 rather than 1.25.

```python
>>> normalise_affinity(-15.0)
1.0

```

## Scoring one candidate

```python
>>> from neorx import score_candidate
>>> candidate = score_candidate(
...     smiles="CC(=O)Oc1ccccc1C(O)=O",
...     target_protein_id="CCR5",
...     target_protein_name="CCR5",
...     causal_confidence=0.72,
...     binding_affinity=-8.4,
...     qed_score=0.55,
...     sa_score=1.58,
...     admet_score=0.892,
...     novelty_score=0.8,
...     molecular_weight=180.04,
...     logp=1.31,
...     drug_likeness_class="Lead-like",
...     n_filters_passed=4,
... )
>>> candidate.composite_score
0.7363
>>> candidate.score_breakdown
{'causal_confidence': 0.216, 'binding_affinity': 0.175, 'qed': 0.0825,
 'sa': 0.0936, 'admet': 0.0892, 'novelty': 0.08}
>>> candidate.is_drug_like, candidate.is_novel
(True, True)

```

The breakdown is additive and per-dimension, so you can read off where a
score came from rather than taking the total on trust.

### Missing dimensions default to neutral, not zero

```python
>>> partial = score_candidate(
...     smiles="CCO", target_protein_id="X", target_protein_name="X",
...     causal_confidence=0.6,
... )
>>> partial.score_breakdown
{'causal_confidence': 0.18, 'binding_affinity': 0.125, 'qed': 0.075,
 'sa': 0.05, 'admet': 0.05, 'novelty': 0.05}

```

With no docking result, no QED and no SA score, those dimensions each
contribute `weight × 0.5`. This is what lets the pipeline degrade a
stage without failing the run — if docking is unavailable, candidates are
still scored, and the binding dimension says "unknown" rather than
"terrible".

!!! warning "A neutral default is not a measurement"
    A candidate scored without docking and a candidate that docked at
    −6.0 kcal/mol receive the same binding contribution. They are not the
    same claim. `ScoredCandidate.binding_affinity` is `None` in the first
    case and a number in the second, and that field — not the composite
    — is what distinguishes them. Do not rank a docked candidate against
    an undocked one on the composite alone.

## The derived flags

- `is_drug_like` — normalised QED ≥ 0.4, molecular weight within
  150–500 Da, and at least two filters passed.
- `is_novel` — novelty score ≥ 0.7.

## Ranking a set

```python
>>> from neorx import rank_candidates
>>> ranked = rank_candidates([partial, candidate])
>>> [(c.rank, c.composite_score) for c in ranked]
[(1, 0.7363), (2, 0.53)]

```

`rank_candidates` sorts best-first and sets `.rank` in place.

## Changing the weights

Three ways, in precedence order: a per-call `weights` argument, then the
`NEORX_WEIGHTS` environment variable holding a JSON object, then the
defaults.

```python
>>> binding_first = score_candidate(
...     smiles="CCO", target_protein_id="X", target_protein_name="X",
...     causal_confidence=0.6, binding_affinity=-10.0,
...     weights={"causal_confidence": 0.10, "binding_affinity": 0.45},
... )
>>> binding_first.score_breakdown["causal_confidence"]
0.06
>>> binding_first.score_breakdown["binding_affinity"]
0.375

```

Overrides are merged over the defaults, so naming two keys leaves the
other four alone. Note that nothing renormalises the weights to sum to
1 — if you override them, the totals are yours to keep consistent, and
composite scores computed under different weight sets are not
comparable.

A malformed `NEORX_WEIGHTS` value is ignored and the defaults stand.

## ADMET

The ADMET dimension is not a single rule but a composite over eight,
computed from the structure:

```python
>>> from neorx import predict_admet
>>> profile = predict_admet("CC(=O)Oc1ccccc1C(O)=O")
>>> profile.absorption, profile.metabolism, profile.toxicity
(1.0, 0.7, 0.8)
>>> profile.composite
0.892
>>> profile.flags
['Ghose filter: outside drug-like range']
>>> profile.details["bbb_permeable"], profile.details["lipinski_violations"]
(True, 0)

```

The eight rules: Lipinski (absorption), Veber (oral bioavailability),
Ghose (drug-like range), Egan (intestinal absorption), PAINS, a BBB
permeability estimate, an hERG liability estimate, and reactive-group
toxicophore alerts. `flags` names which fired and `details` carries the
underlying property values, so a low score is traceable to a rule rather
than opaque.

!!! note "These are rules, not measurements"
    ADMET here is medicinal-chemistry heuristics over computed
    properties. It is not an assay, not a trained model, and not a
    prediction with a confidence interval. It is named `predict_admet`,
    which overstates it; read `flags` and `details` rather than the
    composite if the number is going anywhere important.
