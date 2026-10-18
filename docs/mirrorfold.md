# MirrorFold — mirror-image proteins

MirrorFold predicts and compares the structures of natural (L-amino acid)
proteins against their mirror-image (D-amino acid) counterparts.

All known life uses L-amino acids. D-proteins are of therapeutic interest
for a specific reason: proteases evolved against L-substrates do not
cleave them, the immune system has not been trained on them, and the
consequence can be a dramatically extended half-life.

It is the most peripheral engine in NeoRx — the pipeline uses it only as
a stability signal inside the [RL environment](causalbiorl.md) — but it
is a complete tool on its own.

## Sequence transformation

The pure-sequence operations need nothing but the package:

```python
>>> from neorx.mirrorfold import mirror_sequence, is_valid_sequence
>>> mirror_sequence("YGGFL")
'LFGGY'
>>> is_valid_sequence("YGGFL")
True
>>> is_valid_sequence("YGGFLX")
False

```

`YGGFL` is Leu-enkephalin, which makes a convenient small example.

Note what `mirror_sequence` actually does: it reverses the string. That
is a *proxy*, and the module says so — the reversed sequence, folded by
an L-protein predictor, approximates the mirror-image fold, because
reversing the backbone direction is what preserves side-chain topology
under reflection (the retro-inverso relationship). The chirality
inversion itself lives in the coordinates and the SMILES, not in the
one-letter codes, so a reversed string on its own is not a D-protein.

## Per-residue chirality

```python
>>> from neorx.mirrorfold import get_l_smiles, get_d_smiles, verify_mirror_smiles
>>> get_l_smiles("A")
'N[C@@H](C)C(=O)O'
>>> get_d_smiles("A")
'N[C@H](C)C(=O)O'
>>> check = verify_mirror_smiles("A")
>>> check["is_mirror_pair"], check["same_formula"], check["valid"]
(True, True, True)

```

`@@H` against `@H` is the whole difference — the same formula, the
opposite stereocentre. `verify_mirror_smiles` checks it with RDKit rather
than asserting it, which is the right way round for a claim this easy to
get backwards.

## Biophysical properties

```python
>>> from neorx.mirrorfold import (molecular_weight, net_charge,
...                               isoelectric_point, hydrophobic_fraction)
>>> round(molecular_weight("YGGFL"), 2)
555.63
>>> round(isoelectric_point("YGGFL"), 2)
7.0
>>> round(hydrophobic_fraction("YGGFL"), 3)
0.4

```

These are sequence-level and chirality-independent: mass, charge and
hydrophobicity are identical for a protein and its mirror image. What
differs is how the thing folds, how proteases see it, and how the immune
system responds — which is why there is a structure step at all.

## Structure prediction

```python
>>> from neorx.mirrorfold import predict_pair
>>> l_pred, d_pred = predict_pair("YGGFL", use_cache=False,
...                               allow_api_fallback=False,
...                               allow_mock=True)
>>> l_pred.chirality, d_pred.chirality
('L', 'D')
>>> l_pred.prediction_method
'mock'

```

!!! danger "`prediction_method` is the field that matters"
    `predict_structure` has three paths, in this order: a local ESMFold
    (`uv sync --extra mirror`), the public ESM Atlas API, and a **mock**
    geometry. The mock exists so that the plumbing is testable offline.
    **It is not a structure.** It is a placeholder shape, and an RMSD or
    a TM-score computed from it is a number about the placeholder.

    The example above passes `allow_api_fallback=False, allow_mock=True`
    to force that path deliberately, and it is why no RMSD appears on
    this page. Always read `prediction_method` before reading anything
    else off a `StructurePrediction`, and treat `mock` as a failure to
    predict rather than a prediction.

    It also passes `use_cache=False`, which is worth knowing about for a
    different reason: with the default `use_cache=True`,
    `predict_structure` writes its cache to
    `src/neorx/mirrorfold/cache/` — *inside the installed package*. On a
    source checkout that leaves untracked files in the working tree, and
    a dirty tree marks every [experiment run non-citable](experiments.md).
    Pass `use_cache=False` from anything that should leave no trace.

With a real prediction in hand:

```python
from neorx.mirrorfold import (predict_pair, compare_structures,
                              compute_property_profile,
                              assess_therapeutic_potential)

l_pred, d_pred = predict_pair("YGGFL")      # needs ESMFold or the API
report = compare_structures(l_pred, d_pred)
print(report.rmsd, report.tm_score, report.aligned_length)
print(report.conserved_regions, report.divergent_regions)

l_props = compute_property_profile(l_pred)
d_props = compute_property_profile(d_pred)
assessment = assess_therapeutic_potential(
    "leu-enkephalin", report, l_props, d_props,
)
print(assessment.therapeutic_viability)
print(assessment.rationale)
```

`compare_structures` superimposes and reports RMSD, TM-score,
per-residue distances, secondary-structure agreement, and which regions
are conserved versus divergent. `superimpose`, `calculate_tm_score` and
`assign_secondary_structure` are available separately.

`compute_property_profile` takes a `StructurePrediction`, not a string —
it needs the coordinates for `n_surface_residues`, `n_buried_residues`
and `estimated_stability`. `compare_properties` diffs two profiles.

## Therapeutic assessment

`assess_therapeutic_potential` combines the structural comparison with
protease-resistance and immunogenicity estimates into a
`TherapeuticAssessment` carrying a viability verdict, whether the binding
pocket is conserved, and a prose rationale.

The two component estimates work on sequence alone:

```python
>>> from neorx.mirrorfold import (assess_protease_resistance,
...                               assess_immunogenicity)
>>> assess_protease_resistance("YGGFL")
'Normal (susceptible to natural proteases)'
>>> assess_immunogenicity("YGGFL")
'Normal (subject to standard immune surveillance)'

```

Those are the **L**-form answers, which is the point of comparison: the
D-form is where the resistance claim lives.

!!! note "Heuristics, not assays"
    Protease resistance and immunogenicity here are rule-based estimates
    over sequence features. They are not experimental measurements and
    not trained predictors. `TherapeuticAssessment.rationale` says what
    drove the verdict; read it rather than the verdict.

## Coordinate reflection

```python
>>> from neorx.mirrorfold import reflect_coordinates
>>> reflected = reflect_coordinates(l_pred.pdb_string, axis="x")
>>> len(reflected.splitlines()) == len(l_pred.pdb_string.splitlines())
True

```

`reflect_coordinates` takes a PDB string and returns one, negating the
chosen axis — the geometric half of the mirror operation.

## Fetching a known structure

`fetch_pdb` downloads from the RCSB PDB and
`extract_sequence_from_pdb` pulls the sequence back out, so a real
crystal structure can stand in for the L-form instead of a prediction.
Both reach the network.

## From the command line

```bash
neorx mirrorfold predict YGGFL
neorx mirrorfold compare YGGFL
neorx mirrorfold assess YGGFL
neorx mirrorfold info
```

`neorx mirrorfold info` prints the chirality reference for the amino
acids and needs nothing external.

## Visualisation

`neorx.mirrorfold.viz` has interactive 3D viewers (py3Dmol) and
Ramachandran plots for comparing the two backbones.
