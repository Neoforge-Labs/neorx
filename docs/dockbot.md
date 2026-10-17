# DockBot — docking

DockBot automates the molecular-docking workflow around AutoDock Vina:
fetch a receptor, clean it, find the pocket, prepare the ligand, dock,
score, and report.

Docking is the one stage of the pipeline that is genuinely expensive and
genuinely network-dependent, and most of this page cannot be run as a
doctest for that reason. What *is* offline is ligand preparation, and
that is shown below.

## The workflow

1. **Protein preparation** — fetch from the PDB, strip waters and
   co-crystallised ligands, convert to PDBQT.
2. **Ligand preparation** — SMILES → 3D conformer → MMFF94 minimisation
   → PDBQT.
3. **Binding-site detection** — from a co-crystal ligand, from fpocket,
   or specified manually.
4. **Docking** — AutoDock Vina, via its Python bindings or its CLI.
5. **Parallel screening** — a multiprocessing pool over a compound
   library, resumable.
6. **Composite scoring** — Vina affinity plus drug-likeness.
7. **Reporting** — a self-contained HTML report with embedded 3D viewers.

## Ligand preparation

This part needs nothing but RDKit and Meeko:

```python
>>> from neorx.dockbot import prepare_ligand_pdbqt
>>> mol, pdbqt = prepare_ligand_pdbqt("CC(=O)Oc1ccccc1C(O)=O",
...                                   name="aspirin")
>>> mol is not None
True
>>> pdbqt.splitlines()[0]
'REMARK SMILES CC(=O)Oc1ccccc1C(=O)O'
>>> "ATOM" in pdbqt
True

```

The five steps behind that one call: parse the SMILES to a 2D graph, add
explicit hydrogens, generate a 3D conformer with ETKDG
(Experimental-Torsion Knowledge Distance Geometry), relax it with the
MMFF94 force field to remove steric clashes, then add partial charges and
AutoDock atom types with Meeko. Explicit hydrogens are not optional —
the force field needs them.

`prepare_ligand` stops at the RDKit `Mol`, and takes `num_conformers`
and `max_iterations` if you want more than one conformer or a longer
minimisation.

## Protein preparation

!!! warning "Requires network"
    `prepare_protein` downloads from the RCSB PDB.

```python
from neorx.dockbot import prepare_protein

protein = prepare_protein("1BNA")
print(protein.pdb_id, protein.resolution, protein.num_residues)
print(protein.removed_ligands)
print(protein.pdbqt_path)
```

Raw PDB files are crystallographic snapshots and contain artefacts that
interfere with docking. Waters (`HOH`) clutter the pocket and are
removed. Co-crystallised ligands *occupy* the pocket and must come out —
but their coordinates are the best available evidence for where the
pocket is, so they are recorded in `removed_ligands` on the way out
rather than discarded.

`prepare_protein` caches, so a repeated call for the same PDB ID does not
re-download; pass `force=True` to override.

## Binding sites

```python
>>> from neorx.dockbot import BindingSite, BindingSiteMethod
>>> [m.value for m in BindingSiteMethod]
['ligand', 'fpocket', 'manual']
>>> site = BindingSite(
...     center_x=10.0, center_y=20.0, center_z=30.0,
...     size_x=20.0, size_y=20.0, size_z=20.0,
...     method=BindingSiteMethod.MANUAL,
... )
>>> site.method.value
'manual'

```

Vina needs a search box, and how you got the box matters for how much to
trust the result:

- **`ligand`** — a box centred on the co-crystal ligand's centre of
  mass, with padding. The gold standard when structural data exists.
- **`fpocket`** — geometric pocket detection, for an apo structure.
- **`manual`** — you supply the coordinates.

`BindingSite.confidence` and `residues` carry the detection's own
assessment and the lining residues.

## Docking

!!! warning "Requires AutoDock Vina"
    Install with `uv sync --extra docking`, or have the `vina` binary on
    `PATH` for the CLI path.

```python
from neorx.dockbot import dock

result = dock(protein.pdbqt_path, pdbqt, site)
best = result.poses[0]
print(best.rank, best.affinity_kcal_mol, best.rmsd_lb, best.rmsd_ub)
```

Vina places the ligand in the box using iterated local search with a
Monte Carlo component, and scores each pose with a function
approximating binding free energy in kcal/mol — **more negative is
tighter**. A `DockingResult` carries the ranked poses, the receptor path
and the binding site that produced them, so a pose is never separated
from the box it was found in.

Each `DockingPose` carries its own PDBQT, so a pose can be visualised or
re-scored without re-docking.

## Screening a library

`neorx.dockbot.parallel` docks a library across CPU cores with a
multiprocessing pool, with progress bars and resume capability — a
screen interrupted at compound 4,000 of 10,000 restarts at 4,000.

```bash
neorx dockbot prepare-protein 1BNA
neorx dockbot binding-site 1BNA
neorx dockbot dock-cmd 1BNA --ligand "CC(=O)Oc1ccccc1C(O)=O"
neorx dockbot screen 1BNA --library compounds.smi
```

## DockBot's own composite score

DockBot has a composite scorer of its own —
`neorx.dockbot.scorer.score_docking_result` and `rank_results`, with
`ScoringWeights` and a `CompositeScore` result — which combines Vina
affinity with QED, SA and filter pass-rate.

!!! note "Two different composite scorers"
    This is **not** the same scorer as
    [`neorx.core.scoring.scorer`](scoring.md). DockBot's knows nothing
    about causal confidence; it is for ranking a docking campaign on its
    own. The pipeline's scorer is the one that weights causal confidence
    highest. Both export `normalise_affinity` and `normalise_sa`, and
    they are separate functions — check which module you imported from
    before comparing numbers across them.

## Reports

`neorx.dockbot.report` writes a self-contained HTML report: binding-site
metadata, a ranked results table, composite scores, embedded py3Dmol
viewers for the top poses, and matplotlib summary charts.
`neorx.dockbot.viz` has the plotting and viewer helpers on their own.

## In the pipeline

`run_pipeline(..., run_docking=True)` prepares the receptor from the
target's PDB IDs, detects the site, and docks each generated candidate.
If any of that fails the pipeline continues and scores without a binding
affinity — the dimension drops to its neutral default rather than
failing the run. See [Ranking candidates](scoring.md).
