# Package Skeleton — Design

**Date:** 2026-09-02
**Status:** Approved for planning
**Sub-project:** 1 of 4

---

## Context

NeoRx is a six-module computational drug-discovery platform (28,000 LOC source,
4,092 LOC tests). It is not published on PyPI — `pypi.org/pypi/neorx/json`
returns 404, so the name is available and the `pip install neorx` instruction in
the manuscripts is currently false.

A manuscript audit found that the problems blocking a working release come from
packaging and scaffolding rather than from algorithm code:

- `[tool.hatch.build.targets.wheel] packages = ["neorx", "modules"]` would ship a
  top-level `modules` package to PyPI, claiming one of the most generic names in
  the index.
- `neorx/__init__.py` re-exports only `modules.neorx`. The other five modules —
  `genmol`, `causalbiorl`, `molscreen`, `dockbot`, `mirrorfold` — are absent from
  the public API. `modules/molscreen/__init__.py` is 0 bytes.
- No trained model weights ship. `*.pt` and `*.pth` are gitignored and
  `checkpoints/` is absent from the sdist include list, so an installed GenMol has
  no model. Because the generator substitutes a fallback rather than raising, this
  fails silently.
- `publish.yml`'s `test-install` job runs `import neorx` plus three functions. It
  passes on a package with all of the above defects, and it imports the source
  tree rather than the built wheel.

### Decomposition

This work is sub-project 1 of four. Each gets its own spec, plan, and
implementation cycle:

| # | Sub-project | Delivers |
|---|-------------|----------|
| **1** | **Package skeleton** | **Layout, packaging, public API, CLI, CI, and the GenMol→CausalBioRL generation wiring. This spec.** |
| 2 | Experiment tracking | A harness where a reported number cannot exist without a saved run record |
| 3 | Correctness | Split `identifier.py` and `drug_discovery.py`; fix backdoor logic; make the CEM inner loop decode |
| 4 | Test hardening | Behaviour tests; molscreen from zero coverage |

---

## Goals

1. `pip install neorx` produces a package that imports cleanly and generates real
   molecules with no additional setup.
2. All six modules are reachable through a documented public API.
3. `modules` never appears in a published artifact.
4. CI fails when any of the above regresses.

## Non-goals

Explicitly deferred, to keep this sub-project reviewable:

- **Correctness fixes** (sub-project 3). `nx.d_separated`, the backdoor
  adjustment loop, and the ablation-harness gating are untouched here.

  One correctness fix *is* in scope — the GenMol→CausalBioRL generation wiring,
  specified below — because it is the difference between the shipped environment
  producing real molecules and producing twelve hardcoded strings. Its boundary
  is drawn precisely in that section.

  Existing module internals change only where imports require it. New code
  written in this sub-project is limited to four places, and nowhere else:
  `__init__.py` export lists, the `neorx.cli` app, the GenMol asset-loading path
  (`load_pretrained` and `GenMolAssetError`), and the environment's generation
  path (`_init_genmol` / `_decode_latent`). Any change outside those four is out
  of scope and belongs to a later sub-project.
- **Experiment tracking** (sub-project 2). The `experiments/` directory is
  created as a landing zone but stays empty.
- **Splitting large files.** `identifier.py` (1,291 lines) and
  `drug_discovery.py` (893 lines) move whole. Their target directories are
  created now so sub-project 3 has somewhere to land.
- **Re-running benchmarks.** Handled by tagging; see *Versioning*.

---

## Design

### Layout

A `src/` layout. This is not cosmetic: under the current flat layout `import
neorx` from the repo root resolves to the source tree, never the built wheel,
which is why CI passed on a broken package. A `src/` layout makes that
impossible — code absent from the wheel cannot be imported by the tests.

```
src/neorx/
  __init__.py            public API facade
  py.typed  _version.py
  core/                  ← was modules/neorx
    sources/               monarch, opentargets, kegg, reactome,
                           string, uniprot, pdb, chembl
    graph/                 builder, schema, merge, persistence
    causal/                identifier.py (moved whole; split in SP3)
    bio/                   classifier, tissue_filter
    scoring/               scorer, admet
    pipeline.py  report.py
  genmol/
    assets/                molvae_chembl36.pt · tokenizer.json
    models/  data/  evaluation/  train.py  generate.py
  causalbiorl/           envs/  agents/  causal/
  molscreen/  dockbot/  mirrorfold/
  cli/                   unified Typer app
tests/                   mirrors src/; imports the installed package
experiments/             empty; sub-project 2 lands here
docs/superpowers/specs/
```

`modules/neorx` becomes `neorx.core` rather than `neorx.neorx`. The facade keeps
`from neorx import run_pipeline` working, so the rename is invisible to users.

Tests move from `modules/*/tests/` to a top-level `tests/` mirroring `src/`.
Under a src layout tests must import the installed package; leaving them beside
the source would defeat the purpose.

### Public API

Every subpackage gets an `__init__.py` with an explicit `__all__`.

```python
from neorx import run_pipeline, build_disease_graph, identify_causal_targets
from neorx.genmol      import MolVAE, generate, load_pretrained
from neorx.causalbiorl import DrugDiscoveryEnv, CausalAgent
from neorx.molscreen   import lipinski_filter, qed_score, pains_filter, sa_score
from neorx.dockbot     import prepare_protein, prepare_ligand, dock
from neorx.mirrorfold  import predict_pair, compare_structures
```

Two invariants apply to everything public:

1. Every name in `__all__` is covered by a CI import test.
2. **No public function silently substitutes a fallback.** `load_pretrained()`
   raises `GenMolAssetError` when assets are missing or the tokenizer vocabulary
   is empty, rather than returning an untrained model. This is the bug class that
   produced the twelve-scaffold problem, and closing it in the asset-loading path
   is in scope for this sub-project.

### GenMol → CausalBioRL wiring

`DrugDiscovery-v0` is documented as navigating GenMol's latent space, but the
running environment never reaches the generator. Four defects in one ~35-line
path, all confirmed by running it:

1. `_init_genmol` constructs `SmilesTokenizer()` without calling `build_vocab()`,
   leaving `vocab_size == 0`.
2. It constructs `MolVAE(vocab_size=max(vocab_size, 64))` with no
   `load_state_dict` — a randomly initialised model. The trained checkpoint is
   never loaded.
3. `_decode_latent` calls `self._genmol_model.decode_from_latent(...)`, which does
   not exist on `MolVAE`. The resulting `AttributeError` is caught by a bare
   `except Exception` and logged at debug level.
4. Every call therefore reaches `_fallback_generate`, which hashes the first four
   of 128 latent dimensions into a twelve-entry hardcoded scaffold list:
   `int(|z[:4].sum()| * 1000) % 12`.

A twenty-step episode returns eight distinct molecules, all verbatim from that
list, with plausible-looking rewards and no error. The failure is silent by
construction, which is why it survived to publication.

**The fix is smaller than it appears.** `MolVAE.decode(z, temperature, greedy)`
already exists; nothing new is added to the model. The path becomes:

```python
def _init_genmol(self) -> None:
    from neorx.genmol import load_pretrained     # raises GenMolAssetError
    self._genmol_model, self._genmol_tokenizer = load_pretrained()

def _decode_latent(self, z) -> str:
    ids = self._genmol_model.decode(torch.as_tensor(z).unsqueeze(0))
    return self._genmol_tokenizer.decode(ids[0])
```

`_fallback_generate` and the twelve-scaffold list are deleted outright rather
than made to raise. A fallback that cannot be reached is dead code; a fallback
that can be reached is the bug. Both `except Exception` clauses in this path go
with it — a failure to load or decode must surface, not degrade.

**Boundary.** In scope is the environment's generation path: the environment
decodes real molecules from its latent vector. Out of scope, and remaining in
sub-project 3, is the CEM planner's inner loop (`planner.py:319-325`), which
scores its 200 candidates through a learned reward MLP without constructing a
molecule at all. Making CEM decode changes what the planner optimises — an
algorithmic change with behavioural consequences, not plumbing.

Note that with the wiring fixed, the environment inherits GenMol's documented
partial posterior collapse. `decode` defaults to `greedy=False, temperature=1.0`,
so outputs vary, but they may cluster tightly. That is a known property of the
trained model, not a defect introduced here, and it shapes how the gate below is
written.

### CLI

Five console scripts consolidate into one app with subcommands. The existing
names (`genmol`, `dockbot`, `causalbiorl`, `mirrorfold`) remain as aliases for
one release, then are removed in 0.3.0.

```bash
neorx run HIV --top-n 5          neorx genmol sample --n 100
neorx identify HIV               neorx genmol train
neorx graph "Type 2 Diabetes"    neorx dock 1BNA --ligand "CC(=O)O"
```

`molscreen` gains a subcommand; it is currently the only module without one.

### Packaging

```toml
[project]
requires-python = ">=3.12"

[tool.hatch.build.targets.wheel]
packages = ["src/neorx"]

[tool.hatch.build]
artifacts = ["src/neorx/genmol/assets/*.pt"]
```

**The `artifacts` entry is load-bearing.** Hatchling honours `.gitignore` when
selecting files, and `*.pt` is ignored at `.gitignore:50`. Without this entry the
build produces a wheel with no weights, with no error and no warning, and the
failure surfaces on the user's machine. The single shipped asset is also
un-ignored explicitly and committed (16 MB, changing once per model release —
not enough churn to justify git-lfs).

Weights are exported from `checkpoints/genmol_paper4/final_model.pt` (47 MB,
containing optimizer state) as a weights-only file of approximately 16 MB,
together with its matching 34-token `tokenizer.json`. The tokenizer must be the
one the checkpoint was trained against; a mismatch is a load-time error, not a
warning.

Further packaging changes:

- **`requires-python`: `>=3.13` → `>=3.12`.** No 3.13-only syntax exists in the
  codebase — zero occurrences of PEP 695 type aliases, PEP 695 generics, or
  `itertools.batched`. The floor excluded most scientific environments for no
  reason. CI runs a 3.12 / 3.13 matrix.
- **Relax dependency lower bounds.** `pandas>=3.0.1` and `rdkit>=2025.9.6` are
  aggressive floors that exclude pandas 2.x and force upgrades in existing
  environments. Because all dependencies are mandatory (a deliberate choice — see
  *Decisions*), these floors make `neorx` hard to co-install.

  The new floors are determined empirically, not guessed: the implementation plan
  includes a task that bisects each candidate floor by installing it in a clean
  venv and running the smoke suite, and a permanent CI job that installs the
  declared floors and runs the same suite. A dependency whose floor cannot be
  lowered without failing that suite keeps its current pin, with a comment saying
  why.
- **`vina` remains an extra.** It is a native build that fails on many platforms;
  making it mandatory would break every install.

### Versioning

The correctness fixes in sub-project 3 will change reported numbers — fixing
`d_separated` makes a `dsep_factor` of 1.2 reachable for the first time, which
shifts `causal_confidence`, classifications, and F₁ in a direction nobody can
predict without running it.

To keep the manuscripts reproducible while the code moves on:

- Tag the current commit **`v0.1.0-paper`**. The manuscripts cite this tag.
- Release the restructured package as **0.2.0**.

### Migration

Ordered so the test suite stays green at every step:

1. `git mv modules/<pkg> src/neorx/<pkg>` for all six — preserves per-file history.
2. Rewrite the 194 `modules.*` import statements across 42 tracked files. One
   mechanical pass, reviewed as a single commit.
3. Move `modules/*/tests/` to `tests/`, mirroring `src/`.
4. Add a `modules/` shim: six thin modules re-exporting from `neorx.*` and
   emitting `DeprecationWarning`.
5. Replace `pyproject.toml`. Verify wheel contents before publishing anything.
6. Retarget `[project.scripts]` at `neorx.cli`; reduce root `main.py` to a shim.

**Step 4 protects 25 untracked scripts** at the repository root — every
benchmark, eval, and diagnostic, including those that produce the manuscripts'
evidence. They are not in git, so a bad rewrite is unrecoverable. The shim lets
them keep working untouched, and its warnings identify which still need
updating. It is deleted in 0.3.0.

**The shim is excluded from the wheel automatically, not by rule.** Because
`packages = ["src/neorx"]`, anything outside `src/` is unreachable by the build;
a `modules/` directory at the repository root cannot be packaged even by
accident. This is a second reason to prefer the src layout — the exclusion is
structural rather than something a future contributor must remember.

### CI gates

The current `test-install` job passes on a broken package. Its replacement
asserts the properties that were actually violated:

```
build wheel → install into a clean venv → cd OUT of the repo → then:
  ✓ import every name in every __all__, across all six subpackages
  ✓ genmol.generate() returns valid molecules from the shipped weights
  ✓ DrugDiscovery-v0 constructs and steps at 244-D obs / 130-D action
  ✓ its molecules come from the VAE, not the deleted scaffold list
  ✓ "modules" absent from the wheel; genmol assets present in the wheel
  ✓ ruff, mypy, and a coverage ratchet
```

**The scaffold-provenance gate asserts provenance, not diversity.** It runs a
twenty-step episode and requires that no returned molecule appears in the
twelve-string list that `_fallback_generate` used, pinned as a test fixture. It
does *not* assert that the molecules are diverse: with partial posterior collapse
they may cluster, and a diversity assertion would fail for a reason unrelated to
what the gate is checking. Diversity is a model-quality question, and belongs to
whatever work retrains the model — not to a packaging gate.

Two further details that decide whether these gates are meaningful:

- **Coverage is a ratchet, not an invented number.** The gate is set to the
  coverage measured on the first green build and may not decrease. No target
  percentage is asserted here; raising it is sub-project 4's job, and molscreen
  currently sits at zero.
- **No CI gate may depend on a live external API.** The eight upstream databases
  drift and rate-limit, and a flaky gate is worse than none. Anything requiring
  network lives in a nightly job whose failure opens an issue rather than
  blocking a merge. The environment gate satisfies this: it runs on a prebuilt
  synthetic graph and the shipped weights, entirely offline.

`cd`-ing out of the repository matters as much as the src layout: it is what
forces the test to exercise the installed artifact rather than the source tree.

**Every gate here is a hard gate.** This sub-project introduces no `xfail`.
Because the generation wiring is now in scope, the scaffold-provenance test is a
real gate from the first green build rather than a documented defect.

---

## Decisions

Settled during design, recorded so they are not relitigated:

| Decision | Choice | Rationale |
|---|---|---|
| Distribution shape | One package, subpackages | Single version and install; removes the `modules` namespace claim |
| Model weights | Ship in wheel, ~16 MB | Only option where `pip install neorx` generates real molecules offline |
| Missing assets | Raise `GenMolAssetError` | Silent substitution is the audit's most damaging bug class |
| Dependencies | All mandatory | Simplest to maintain and document; every documented example works |
| `vina` | Extra | Native build; mandatory would break installs |
| Python floor | 3.12 | Nothing needs 3.13; 3.12 widens reach materially |
| Fix scope | Plumbing here, plus the generation wiring; remaining correctness in SP3 | The wiring decides whether the shipped environment produces real molecules or twelve fixed strings — too central to defer |
| CEM inner-loop decoding | Deferred to SP3 | Changes what the planner optimises; algorithmic, not plumbing |
| `modules/` shim | Yes, repo-only, removed in 0.3.0 | Protects 25 untracked, unrecoverable scripts |

---

## Risks

| Risk | Mitigation |
|---|---|
| Wheel builds without weights because `.pt` is gitignored | `artifacts` entry, plus a CI assertion that the asset is present in the wheel |
| Import rewrite misses a call site | `modules/` shim keeps the old path working; `DeprecationWarning` surfaces stragglers rather than crashing |
| 16 MB binary in git | Acceptable one-time cost; changes once per model release. Revisit git-lfs only if additional checkpoints ship |
| Relaxed dependency floors break at runtime on older versions | CI matrix installs the floor versions and runs the smoke suite |
| Untracked root scripts break despite the shim | They are exercised manually before the shim is removed in 0.3.0 |
| Deleting `_fallback_generate` turns a silent degradation into a hard failure in code paths that relied on it | Intended. The scaffold list is pinned as a test fixture before deletion, so the gate can still detect a regression to it |
| Posterior collapse makes environment molecules cluster, and a future contributor reads that as the wiring being broken | The gate asserts provenance, not diversity, and the spec records why. Model quality is out of scope here |
| SP3's correctness fixes invalidate manuscript numbers | `v0.1.0-paper` tag pins a reproducible state |

---

## Success criteria

Verified in CI on every change:

1. `pip install dist/neorx-0.2.0-*.whl` in a clean venv, from outside the
   repository, followed by importing every public name — succeeds.
2. `neorx.genmol.generate(n=20)` loaded from the shipped weights returns at
   least one valid SMILES, and more than one distinct value across the batch.
   The bound is deliberately loose: `generate` defaults to `temperature=1.0` with
   `deduplicate=True`, so it returns *at most* `n` and the exact count varies by
   seed. Asserting "exactly five distinct from `n=5`" would be a flaky gate.
3. `unzip -l` on the wheel shows `neorx/genmol/assets/molvae_chembl36.pt` and
   no `modules/` entry.
4. A twenty-step `DrugDiscovery-v0` episode, on a synthetic graph and the shipped
   weights, returns molecules of which none appears in the twelve-string scaffold
   list (pinned as a fixture). Provenance only — diversity is not asserted.
5. The existing test suite passes with unchanged behaviour after the move.
6. Installing the declared dependency floors and running the smoke suite
   succeeds.

Verified out of band, because it depends on eight live external APIs and takes
several minutes per disease:

7. `neorx run HIV --top-n 5` completes end to end from the installed package.
   Run manually before each release and in a nightly job; a failure opens an
   issue rather than blocking a merge.
