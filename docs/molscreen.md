# MolScreen — drug-likeness

MolScreen answers whether a molecule looks like a drug: it parses SMILES,
computes physico-chemical properties, runs six medicinal-chemistry
filters, and compares against known drugs for novelty. It is pure
RDKit — no network, no model weights.

## Properties

```python
>>> from neorx.molscreen import calculate_properties
>>> props = calculate_properties("CC(=O)Oc1ccccc1C(O)=O")
>>> props.smiles
'CC(=O)Oc1ccccc1C(=O)O'
>>> props.molecular_weight, props.logp
(180.04, 1.31)
>>> props.hbd, props.hba, props.tpsa
(1, 3, 63.6)
>>> props.rotatable_bonds, props.num_rings, props.num_aromatic_rings
(2, 1, 1)
>>> props.fsp3, props.molar_refractivity
(0.111, 44.71)

```

Note that `props.smiles` is the *canonical* form, not what was passed in:
the input `C(O)=O` came back as `C(=O)O`. Each field's docstring in the
model explains why the property matters for absorption or permeability —
see the [API reference](api.md#neorxmolscreen).

## Parsing

```python
>>> from neorx.molscreen import validate_smiles, canonicalise
>>> validate_smiles("CCO")
True
>>> validate_smiles("xyz!")
False
>>> canonicalise("C(C)O")
'CCO'

```

`parse_smiles` returns an RDKit `Mol` or `None`. `name_to_smiles`
resolves a common drug name, and `smart_parse` accepts either a name or a
SMILES string.

## The six filters

Each filter returns a result carrying both a verdict and the specific
violations, so a failure is actionable rather than a bare `False`.

```python
>>> from neorx.molscreen import parse_smiles, run_all_filters
>>> aspirin = parse_smiles("CC(=O)Oc1ccccc1C(O)=O")
>>> for result in run_all_filters(aspirin):
...     print(f"{result.name:9} {result.passed!s:6} {result.violations}")
Lipinski  True   []
Veber     True   []
Ghose     False  ['Atom count 13 outside [20, 70]']
Egan      True   []
PAINS     True   []
Brenk     False  ['phenol_ester']

```

| Filter | Asks |
|---|---|
| **Lipinski** | Rule of Five: MW, LogP, HBD, HBA — oral bioavailability |
| **Veber** | Rotatable bonds and TPSA — oral bioavailability from a different angle |
| **Ghose** | MW, LogP, atom count, molar refractivity within drug-like ranges |
| **Egan** | LogP and TPSA — absorption |
| **PAINS** | Pan-assay interference substructures: compounds that hit everything |
| **Brenk** | Unwanted functional groups: reactive, toxic, metabolically unstable |

Aspirin failing two of six is the right answer, not a bug. It is too
small for Ghose's 20-atom floor, and its phenol ester is on Brenk's list.
Aspirin is a real drug that would not survive a modern lead-like filter
set — which is a useful calibration for how to read these verdicts.

## Classification

```python
>>> from neorx.molscreen import classify_drug_likeness, DrugLikelihoodCategory
>>> [c.value for c in DrugLikelihoodCategory]
['Drug-like', 'Lead-like', 'Not drug-like']
>>> classify_drug_likeness(run_all_filters(aspirin))
'Lead-like'

```

`classify_drug_likeness` takes the **filter results**, not the molecule.

## Drug-likeness and synthetic accessibility

```python
>>> from neorx.molscreen import qed_score, sa_score
>>> round(qed_score(aspirin), 3)
0.55
>>> round(sa_score(aspirin), 3)
1.58

```

**QED** is RDKit's quantitative estimate of drug-likeness, 0 to 1,
combining MW, LogP, HBD, HBA, PSA, rotatable bonds, structural alerts and
aromaticity into one desirability score.

**SA score** is synthetic accessibility, 1 (trivial) to 10 (very hard).
Aspirin at 1.58 is about as easy as organic synthesis gets. The
[composite scorer](scoring.md) inverts and normalises this, since lower
is better.

## Similarity and novelty

```python
>>> from neorx.molscreen import tanimoto_similarity
>>> round(tanimoto_similarity("CCO", "CCCO"), 4)
0.5556

```

`find_similar_drugs` searches a cache of known drugs and is what supplies
the novelty dimension of the composite score — a candidate highly similar
to an approved drug is not novel. `build_cache` and `clear_cache` manage
that cache.

## From the command line

```bash
neorx molscreen screen "CC(=O)Oc1ccccc1C(O)=O"
```

## Reports

`ScreeningReport` and `ComparisonReport` bundle a molecule's properties,
filter results and similar drugs for reporting. See the
[API reference](api.md#neorxmolscreen).
