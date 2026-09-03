# Package Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure NeoRx into a single `src/`-layout package that publishes to PyPI as `neorx`, exposes all six modules, and ships trained GenMol weights so an installed copy generates real molecules.

**Architecture:** `modules/*` moves to `src/neorx/*`, with `modules/neorx` becoming `neorx.core`. A repo-only `modules/` shim keeps 25 untracked scripts working. GenMol weights ship as package data loaded by `load_pretrained()`, which raises rather than substituting a fallback; `DrugDiscovery-v0` is rewired to use it, and its twelve-scaffold fallback is deleted. CI installs the built wheel from outside the repository, so tests exercise the artifact rather than the source tree.

**Tech Stack:** Python 3.12+, hatchling, pytest, Typer, PyTorch, ruff, mypy, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-02-package-skeleton-design.md`

## Global Constraints

- **Python floor:** `>=3.12`. CI matrix is 3.12 and 3.13. No PEP 695 syntax may be introduced.
- **New code is permitted in exactly four places:** `__init__.py` export lists, `neorx.cli`, the GenMol asset-loading path (`load_pretrained`, `GenMolAssetError`), and the environment generation path (`_init_genmol`, `_decode_latent`). Everything else moves unchanged except for import statements.
- **No public function may silently substitute a fallback.** Missing assets raise; they never degrade.
- **No CI gate may depend on a live external API.** Network-dependent checks go in a nightly job.
- **`modules` must never appear in a built artifact.**
- **Deferred to later sub-projects, do not touch:** `nx.d_separated` / backdoor logic, the CEM inner loop in `causal/planner.py`, the ablation harness, splitting `identifier.py` or `drug_discovery.py`.
- **Version:** release as `0.2.0`. Tag the pre-move commit `v0.1.0-paper`.
- **Platform note:** commands are written for macOS (BSD `sed` needs `-i ''`). On GNU/Linux use `sed -i` with no argument.

---

### Task 1: Move to `src/` layout and organise `neorx.core`

Moves all six packages, subdivides core by responsibility, rewrites every import, relocates tests, and updates `pyproject.toml` enough that the suite runs. The repository must be green at the end of this task.

This is one task rather than two because both moves share a single import rewrite. Splitting them would mean rewriting 194 import statements twice, doubling the chance of a missed call site.

**Files:**
- Move: `modules/neorx/` → `src/neorx/core/`, then subdivide into
  `core/graph/` (`models.py`, `graph_builder.py`, `persistence.py`),
  `core/causal/` (`identifier.py`, `counterfactual.py`),
  `core/bio/` (`classifier.py`, `tissue_filter.py`),
  `core/scoring/` (`scorer.py`, `admet.py`),
  `core/sources/` (renamed from `data_sources/`)
- Move: `modules/{genmol,causalbiorl,molscreen,dockbot,mirrorfold}/` → `src/neorx/{...}/`
- Move: `modules/*/tests/` → `tests/*/`
- Create: `src/neorx/__init__.py` (moved from `neorx/__init__.py`), `src/neorx/py.typed`
- Modify: `pyproject.toml` (wheel packages, sdist include, testpaths, coverage source, isort)
- Delete: `neorx/` (old two-file facade package)

**Interfaces:**
- Consumes: nothing (first task)
- Produces: the import root `neorx`, with subpackages `neorx.core`, `neorx.genmol`, `neorx.causalbiorl`, `neorx.molscreen`, `neorx.dockbot`, `neorx.mirrorfold`. Core internals are addressable as `neorx.core.graph.models`, `neorx.core.causal.identifier`, `neorx.core.bio.classifier`, `neorx.core.scoring.scorer`, `neorx.core.sources.*`. Every later task imports from these paths.

- [ ] **Step 1: Tag the pre-move state**

The manuscripts cite this commit; it must stay reachable after the move.

```bash
git tag -a v0.1.0-paper -m "State reproducing the manuscript results (F1 = 0.474)"
git tag -l v0.1.0-paper
```

- [ ] **Step 2: Record the baseline test result**

You need to know which failures pre-date your change.

```bash
pytest -q 2>&1 | tail -5 > /tmp/neorx-baseline.txt
cat /tmp/neorx-baseline.txt
```

- [ ] **Step 3: Create the src tree and move packages**

```bash
mkdir -p src/neorx tests
git mv modules/neorx src/neorx/core
for m in genmol causalbiorl molscreen dockbot mirrorfold; do
  git mv "modules/$m" "src/neorx/$m"
done
git mv neorx/__init__.py src/neorx/__init__.py
git mv neorx/py.typed src/neorx/py.typed
rmdir neorx
```

- [ ] **Step 4: Move tests out of the packages**

```bash
for m in core genmol causalbiorl dockbot mirrorfold; do
  if [ -d "src/neorx/$m/tests" ]; then
    git mv "src/neorx/$m/tests" "tests/$m"
  fi
done
touch tests/__init__.py
ls tests/
```

- [ ] **Step 5: Rewrite imports — `modules.neorx` first**

Order matters. `modules.neorx` must become `neorx.core`, so it is rewritten before the general rule; otherwise it would become `neorx.neorx`.

```bash
FILES=$(grep -rl "modules\." --include="*.py" src tests *.py 2>/dev/null)
echo "$FILES" | wc -l
sed -i '' -E \
  -e 's/\bfrom modules\.neorx\b/from neorx.core/g' \
  -e 's/\bimport modules\.neorx\b/import neorx.core/g' \
  $FILES
sed -i '' -E \
  -e 's/\bfrom modules\./from neorx./g' \
  -e 's/\bimport modules\./import neorx./g' \
  $FILES
```

- [ ] **Step 6: Verify no `modules.` imports remain in the package**

```bash
grep -rn "modules\." --include="*.py" src tests || echo "CLEAN"
```

Expected: `CLEAN`. Root-level untracked scripts still reference `modules.` — that is intentional and Task 2 handles them.

- [ ] **Step 7: Create the `core/` subpackages**

The spec's layout groups core by responsibility, and sub-project 3 splits
`identifier.py` into `core/causal/`. Create the landing zones now, in the same
mechanical pass as the move — doing it later would mean rewriting imports twice.

Files not listed here stay at `core/`: `pipeline.py`, `report.py`, `cache.py`,
`api.py`, `validator.py`, `literature_validator.py`, `__init__.py`, `__main__.py`.

```bash
cd src/neorx/core
mkdir -p graph causal bio scoring
git mv data_sources sources
git mv models.py graph_builder.py persistence.py graph/
git mv identifier.py counterfactual.py causal/
git mv classifier.py tissue_filter.py bio/
git mv scorer.py admet.py scoring/
for d in graph causal bio scoring; do touch "$d/__init__.py" && git add "$d/__init__.py"; done
cd -
```

- [ ] **Step 8: Convert intra-core relative imports to absolute**

There are 30 `from .X import` statements inside core. After the move their
correct relative depth varies by destination directory, so rewrite them to
absolute paths instead — unambiguous wherever the file ends up.

```bash
FILES=$(find src/neorx/core -name "*.py")
sed -i '' -E \
  -e 's/\bfrom \.models import/from neorx.core.graph.models import/g' \
  -e 's/\bfrom \.graph_builder import/from neorx.core.graph.graph_builder import/g' \
  -e 's/\bfrom \.persistence import/from neorx.core.graph.persistence import/g' \
  -e 's/\bfrom \.identifier import/from neorx.core.causal.identifier import/g' \
  -e 's/\bfrom \.counterfactual import/from neorx.core.causal.counterfactual import/g' \
  -e 's/\bfrom \.classifier import/from neorx.core.bio.classifier import/g' \
  -e 's/\bfrom \.tissue_filter import/from neorx.core.bio.tissue_filter import/g' \
  -e 's/\bfrom \.scorer import/from neorx.core.scoring.scorer import/g' \
  -e 's/\bfrom \.admet import/from neorx.core.scoring.admet import/g' \
  -e 's/\bfrom \.data_sources import/from neorx.core.sources import/g' \
  -e 's/\bfrom \.cache import/from neorx.core.cache import/g' \
  -e 's/\bfrom \.validator import/from neorx.core.validator import/g' \
  -e 's/\bfrom \.report import/from neorx.core.report import/g' \
  -e 's/\bfrom \.pipeline import/from neorx.core.pipeline import/g' \
  -e 's/\bfrom \.literature_validator import/from neorx.core.literature_validator import/g' \
  $FILES
```

Also rewrite any parent-relative imports that the move invalidated, and the old
`data_sources` path wherever it appears absolutely:

```bash
sed -i '' -E \
  -e 's/\bfrom \.\.models import/from neorx.core.graph.models import/g' \
  -e 's/\bfrom \.\.cache import/from neorx.core.cache import/g' \
  -e 's/\bneorx\.core\.data_sources\b/neorx.core.sources/g' \
  $FILES $(find src tests -name "*.py")
```

- [ ] **Step 9: Verify no stale intra-core paths remain**

```bash
grep -rn "data_sources" --include="*.py" src tests && echo "STALE PATH — fix it" || echo "CLEAN"
python -c "import neorx.core.graph.models, neorx.core.causal.identifier, neorx.core.bio.classifier, neorx.core.scoring.scorer, neorx.core.sources; print('core subpackages import OK')"
```

Expected: `CLEAN` then `core subpackages import OK`. An `ImportError` here names
the module whose import rule is missing from Step 8 — add it and re-run.

- [ ] **Step 10: Point `pyproject.toml` at the new layout**

Replace these five blocks. Leave everything else in the file alone for now.

The `sdist` block matters: it currently lists `neorx/`, `modules/` and `main.py`,
all of which are now wrong. A stale include silently produces a source
distribution missing the package.

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/neorx"]

[tool.hatch.build.targets.sdist]
include = [
    "src/neorx/",
    "tests/",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "pyproject.toml",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --tb=short"
filterwarnings = [
    "ignore::DeprecationWarning:pkg_resources",
    "ignore::UserWarning:chembl_webresource_client",
]

[tool.coverage.run]
source = ["src/neorx"]
omit = ["*/tests/*", "*/__main__.py", "*/templates/*"]

[tool.ruff.lint.isort]
known-first-party = ["neorx"]
```

- [ ] **Step 11: Install editable and confirm imports resolve**

```bash
pip install -e . --no-deps
python -c "import neorx, neorx.core, neorx.genmol, neorx.causalbiorl, neorx.molscreen, neorx.dockbot, neorx.mirrorfold; print('all six import OK')"
```

Expected: `all six import OK`

- [ ] **Step 12: Run the suite and compare against baseline**

```bash
pytest -q 2>&1 | tail -5
diff <(pytest -q 2>&1 | tail -1) <(tail -1 /tmp/neorx-baseline.txt) && echo "IDENTICAL TO BASELINE"
```

Expected: same pass/fail counts as `/tmp/neorx-baseline.txt`. Any new failure is an import the rewrite missed — fix it before committing.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "refactor: move modules/* to src/neorx/* (src layout)

modules/neorx becomes neorx.core, subdivided into graph/, causal/, bio/,
scoring/ and sources/ so sub-project 3 has somewhere to split identifier.py.
194 modules.* imports rewritten across 42 files; the 30 intra-core relative
imports become absolute, since their correct depth now varies by directory.

Tests move to a top-level tests/ mirroring src/, so they import the
installed package rather than the source tree -- which is how CI
previously passed on a broken wheel."
```

---

### Task 2: Add the `modules/` compatibility shim

25 untracked scripts at the repository root import `modules.*`. They are not in git, so breaking them is unrecoverable. This shim keeps them working and names the ones that still need updating.

**Files:**
- Create: `modules/__init__.py`, `modules/neorx.py`, `modules/genmol.py`, `modules/causalbiorl.py`, `modules/molscreen.py`, `modules/dockbot.py`, `modules/mirrorfold.py`
- Test: `tests/test_modules_shim.py`

**Interfaces:**
- Consumes: `neorx.core`, `neorx.genmol`, `neorx.causalbiorl`, `neorx.molscreen`, `neorx.dockbot`, `neorx.mirrorfold` (Task 1)
- Produces: nothing later tasks depend on. This shim is repo-only, excluded from the wheel by `packages = ["src/neorx"]`, and deleted in 0.3.0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modules_shim.py
"""The repo-only modules/ shim keeps untracked root scripts working."""

import importlib
import warnings

import pytest

SHIMMED = [
    ("modules.neorx", "neorx.core"),
    ("modules.genmol", "neorx.genmol"),
    ("modules.causalbiorl", "neorx.causalbiorl"),
    ("modules.molscreen", "neorx.molscreen"),
    ("modules.dockbot", "neorx.dockbot"),
    ("modules.mirrorfold", "neorx.mirrorfold"),
]


@pytest.mark.parametrize("old,new", SHIMMED)
def test_old_path_warns_and_forwards(old, new):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim = importlib.import_module(old)

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        f"{old} must emit DeprecationWarning"
    )
    assert new in [w.message.args[0] for w in caught][0]

    real = importlib.import_module(new)
    assert shim.__doc__ == real.__doc__ or shim is not None
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
pytest tests/test_modules_shim.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'modules'`

- [ ] **Step 3: Write the shim package**

```python
# modules/__init__.py
"""Deprecated import path. Use ``neorx.*`` instead.

This package exists only inside the repository, to keep untracked
benchmark and evaluation scripts working after the move to src/neorx.
It is excluded from built artifacts and is removed in 0.3.0.
"""
```

Each of the six shim modules follows this shape. Write all six — `modules/neorx.py` is shown; the other five are identical except for the two names.

```python
# modules/neorx.py
"""Deprecated alias for :mod:`neorx.core`."""

import warnings

warnings.warn(
    "modules.neorx is deprecated; import from neorx.core instead. "
    "This shim is removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from neorx.core import *  # noqa: F401,F403,E402
from neorx.core import __doc__  # noqa: F401,E402
```

For the remaining five, substitute:

| File | Warning text names | Import source |
|---|---|---|
| `modules/genmol.py` | `modules.genmol` → `neorx.genmol` | `neorx.genmol` |
| `modules/causalbiorl.py` | `modules.causalbiorl` → `neorx.causalbiorl` | `neorx.causalbiorl` |
| `modules/molscreen.py` | `modules.molscreen` → `neorx.molscreen` | `neorx.molscreen` |
| `modules/dockbot.py` | `modules.dockbot` → `neorx.dockbot` | `neorx.dockbot` |
| `modules/mirrorfold.py` | `modules.mirrorfold` → `neorx.mirrorfold` | `neorx.mirrorfold` |

- [ ] **Step 4: Run the test to confirm it passes**

```bash
pytest tests/test_modules_shim.py -q
```

Expected: 6 passed

- [ ] **Step 5: Smoke-test a real untracked script**

```bash
python -c "import _quick_bench" 2>&1 | head -3
```

Expected: either it runs, or it fails for a reason unrelated to imports (missing data, network). A `ModuleNotFoundError` means the shim is incomplete.

- [ ] **Step 6: Commit**

```bash
git add modules/ tests/test_modules_shim.py
git commit -m "feat: add repo-only modules/ shim with DeprecationWarning

Keeps 25 untracked root scripts working after the src/ move. Excluded
from the wheel by packages = [\"src/neorx\"]; removed in 0.3.0."
```

---

### Task 3: Public API exports for all six subpackages

`molscreen/__init__.py` is currently 0 bytes. The top-level facade re-exports only the core pipeline.

**Files:**
- Create: `src/neorx/molscreen/__init__.py`
- Modify: `src/neorx/__init__.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Consumes: all six subpackages (Task 1)
- Produces: `neorx.__all__`, and a populated `neorx.molscreen.__all__`. Task 8's CI gate iterates every `__all__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_public_api.py
"""Every advertised name must import, from every subpackage."""

import importlib

import pytest

SUBPACKAGES = [
    "neorx",
    "neorx.core",
    "neorx.genmol",
    "neorx.causalbiorl",
    "neorx.molscreen",
    "neorx.dockbot",
    "neorx.mirrorfold",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_declares_all(name):
    mod = importlib.import_module(name)
    assert hasattr(mod, "__all__"), f"{name} must declare __all__"
    assert mod.__all__, f"{name}.__all__ must not be empty"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_every_exported_name_resolves(name):
    mod = importlib.import_module(name)
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, f"{name} exports names it does not define: {missing}"


def test_molscreen_exports_the_documented_screening_api():
    import neorx.molscreen as ms

    for fn in ("lipinski_filter", "qed_score", "pains_filter", "sa_score"):
        assert callable(getattr(ms, fn)), f"molscreen.{fn} must be callable"
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
pytest tests/test_public_api.py -q
```

Expected: FAIL — `neorx.molscreen must declare __all__` (the file is empty)

- [ ] **Step 3: Write molscreen's `__init__.py`**

These are the real names in the module; verify with `grep -n "^def " src/neorx/molscreen/*.py` if you doubt any.

```python
# src/neorx/molscreen/__init__.py
"""MolScreen — drug-likeness screening and molecular property calculation.

>>> from neorx.molscreen import qed_score, lipinski_filter, parse_smiles
>>> mol = parse_smiles("CC(=O)Oc1ccccc1C(O)=O")
>>> round(qed_score(mol), 3)
0.55
"""

from .accessibility import qed_score, sa_score
from .filters import (
    brenk_filter,
    classify_drug_likeness,
    egan_filter,
    ghose_filter,
    lipinski_filter,
    pains_filter,
    run_all_filters,
    veber_filter,
)
from .models import (
    ComparisonReport,
    DrugLikelihoodCategory,
    FilterResult,
    MolecularProperties,
    ScreeningReport,
    SimilarDrug,
)
from .parser import canonicalise, name_to_smiles, parse_smiles, smart_parse, validate_smiles
from .properties import calculate_properties

__all__ = [
    # Parsing
    "parse_smiles",
    "validate_smiles",
    "canonicalise",
    "name_to_smiles",
    "smart_parse",
    # Properties
    "calculate_properties",
    "qed_score",
    "sa_score",
    # Filters
    "lipinski_filter",
    "veber_filter",
    "ghose_filter",
    "egan_filter",
    "pains_filter",
    "brenk_filter",
    "run_all_filters",
    "classify_drug_likeness",
    # Models
    "MolecularProperties",
    "FilterResult",
    "ScreeningReport",
    "ComparisonReport",
    "SimilarDrug",
    "DrugLikelihoodCategory",
]
```

- [ ] **Step 4: Widen the top-level facade**

`src/neorx/__init__.py` currently re-exports only core names. Change its import line and append the subpackage re-exports. Keep every existing name — removing one is a breaking change.

Replace the existing `from modules.neorx import (...)` line with `from neorx.core import (...)`, keeping the same name list, then append:

```python
# --- Subpackages -----------------------------------------------------
from neorx import causalbiorl, core, dockbot, genmol, mirrorfold, molscreen  # noqa: F401,E402

__version__ = "0.2.0"

__all__ = __all__ + [
    "core",
    "genmol",
    "causalbiorl",
    "molscreen",
    "dockbot",
    "mirrorfold",
]
```

- [ ] **Step 5: Run the test to confirm it passes**

```bash
pytest tests/test_public_api.py -q
```

Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add src/neorx/__init__.py src/neorx/molscreen/__init__.py tests/test_public_api.py
git commit -m "feat: expose all six subpackages through the public API

molscreen had a 0-byte __init__.py and exported nothing. The top-level
facade now re-exports every subpackage alongside the core pipeline."
```

---

### Task 4: Ship GenMol weights and add `load_pretrained`

Without shipped weights an installed GenMol has no model. Because generation substitutes a fallback rather than raising, that failure is currently silent.

**Files:**
- Create: `src/neorx/genmol/assets/molvae_chembl36.pt`, `src/neorx/genmol/assets/tokenizer.json`
- Create: `src/neorx/genmol/assets/__init__.py` (empty; makes the directory a package for wheel inclusion)
- Modify: `src/neorx/genmol/__init__.py`, `pyproject.toml`, `.gitignore`
- Create: `src/neorx/genmol/pretrained.py`
- Test: `tests/genmol/test_pretrained.py`

**Interfaces:**
- Consumes: `MolVAE`, `SmilesTokenizer`, `load_checkpoint` from `neorx.genmol` (Task 1)
- Produces: `load_pretrained(device: torch.device | None = None) -> tuple[MolVAE, SmilesTokenizer]` and `class GenMolAssetError(RuntimeError)`, both exported from `neorx.genmol`. Task 5 calls `load_pretrained()`.

- [ ] **Step 1: Export weights-only checkpoint**

The source checkpoint is 47 MB because it carries Adam optimizer state. Strip it.

```bash
mkdir -p src/neorx/genmol/assets
python - <<'PY'
import torch, json, pathlib
src = pathlib.Path("checkpoints/genmol_paper4/final_model.pt")
dst = pathlib.Path("src/neorx/genmol/assets/molvae_chembl36.pt")
ckpt = torch.load(src, map_location="cpu", weights_only=False)
torch.save(
    {"model_state_dict": ckpt["model_state_dict"], "config": ckpt.get("config", {})},
    dst,
)
print(f"{src.stat().st_size/1e6:.1f} MB -> {dst.stat().st_size/1e6:.1f} MB")
PY
cp checkpoints/genmol_paper4/tokenizer.json src/neorx/genmol/assets/tokenizer.json
touch src/neorx/genmol/assets/__init__.py
```

Expected: roughly `47.0 MB -> 16.6 MB`

- [ ] **Step 2: Verify the tokenizer matches the checkpoint**

A vocab-size mismatch between tokenizer and weights makes `load_state_dict` fail at the user's machine.

```bash
python -c "
import json
v = json.load(open('src/neorx/genmol/assets/tokenizer.json'))
t = v.get('token_to_idx') or v
print('vocab size:', len(t))
assert len(t) == 34, f'expected 34 tokens, got {len(t)}'
print('OK')
"
```

Expected: `vocab size: 34` then `OK`

- [ ] **Step 3: Un-ignore the shipped asset and force its inclusion**

`.gitignore:50` ignores `*.pt`, which excludes the asset from both git *and* the hatchling build. Both need an exception.

Append to `.gitignore`:

```gitignore
# Shipped model asset — must be committed and packaged (see pyproject artifacts)
!src/neorx/genmol/assets/*.pt
```

Add to `pyproject.toml`:

```toml
[tool.hatch.build]
artifacts = ["src/neorx/genmol/assets/*.pt"]
```

- [ ] **Step 4: Write the failing test**

```python
# tests/genmol/test_pretrained.py
"""Shipped weights load, and their absence raises rather than degrading."""

import pytest

from neorx.genmol import GenMolAssetError, load_pretrained


def test_load_pretrained_returns_model_and_tokenizer():
    model, tok = load_pretrained()
    assert tok.vocab_size == 34
    assert not model.training, "model must be returned in eval mode"


def test_generated_molecules_are_valid():
    from rdkit import Chem

    from neorx.genmol import generate

    model, tok = load_pretrained()
    smiles = generate(model, tok, n=20)
    assert smiles, "generate returned nothing"
    assert all(Chem.MolFromSmiles(s) is not None for s in smiles)
    assert len(set(smiles)) > 1, "expected more than one distinct molecule"


def test_missing_asset_raises_not_degrades(monkeypatch, tmp_path):
    import neorx.genmol.pretrained as p

    monkeypatch.setattr(p, "ASSET_DIR", tmp_path)
    with pytest.raises(GenMolAssetError, match="no trained checkpoint"):
        p.load_pretrained()


def test_empty_vocabulary_raises(monkeypatch, tmp_path):
    import json

    import neorx.genmol.pretrained as p

    (tmp_path / "molvae_chembl36.pt").write_bytes(b"stub")
    (tmp_path / "tokenizer.json").write_text(json.dumps({"token_to_idx": {}}))
    monkeypatch.setattr(p, "ASSET_DIR", tmp_path)
    with pytest.raises(GenMolAssetError, match="vocabulary is empty"):
        p.load_pretrained()
```

- [ ] **Step 5: Run it to confirm it fails**

```bash
pytest tests/genmol/test_pretrained.py -q
```

Expected: FAIL — `ImportError: cannot import name 'GenMolAssetError'`

- [ ] **Step 6: Write `pretrained.py`**

```python
# src/neorx/genmol/pretrained.py
"""Loading of the GenMol weights shipped with the package.

The loader raises on missing or inconsistent assets. It never returns an
untrained model: a randomly initialised VAE emits plausible-looking SMILES,
so silent degradation here is undetectable downstream.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .data.tokenizer import SmilesTokenizer
from .models.vae import MolVAE
from .train import load_checkpoint

ASSET_DIR = Path(__file__).parent / "assets"
CHECKPOINT_NAME = "molvae_chembl36.pt"
TOKENIZER_NAME = "tokenizer.json"


class GenMolAssetError(RuntimeError):
    """Raised when the shipped model assets are missing or unusable."""


def load_pretrained(
    device: torch.device | None = None,
) -> tuple[MolVAE, SmilesTokenizer]:
    """Load the packaged MolVAE and its matching tokenizer.

    Returns
    -------
    tuple[MolVAE, SmilesTokenizer]
        Model in eval mode, and the tokenizer it was trained against.

    Raises
    ------
    GenMolAssetError
        If either asset is absent, or the tokenizer carries no vocabulary.
    """
    ckpt_path = ASSET_DIR / CHECKPOINT_NAME
    tok_path = ASSET_DIR / TOKENIZER_NAME

    if not ckpt_path.exists() or not tok_path.exists():
        raise GenMolAssetError(
            f"no trained checkpoint found in {ASSET_DIR}. "
            "Reinstall neorx, or train one with `neorx genmol train`."
        )

    tokenizer = SmilesTokenizer.load(tok_path)
    if tokenizer.vocab_size == 0:
        raise GenMolAssetError(
            f"tokenizer at {tok_path} loaded but its vocabulary is empty. "
            "The asset is corrupt; reinstall neorx."
        )

    model = MolVAE(vocab_size=tokenizer.vocab_size)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    return model, tokenizer
```

- [ ] **Step 7: Export it from the subpackage**

Append to `src/neorx/genmol/__init__.py`:

```python
from .pretrained import GenMolAssetError, load_pretrained  # noqa: F401
```

and add `"load_pretrained"` and `"GenMolAssetError"` to its `__all__`.

- [ ] **Step 8: Run the tests to confirm they pass**

```bash
pytest tests/genmol/test_pretrained.py -q
```

Expected: 4 passed

- [ ] **Step 9: Commit**

```bash
git add -f src/neorx/genmol/assets/molvae_chembl36.pt
git add src/neorx/genmol/ tests/genmol/test_pretrained.py pyproject.toml .gitignore
git commit -m "feat: ship trained GenMol weights as package data

Weights-only export (47MB -> 16.6MB, optimizer state stripped) plus the
matching 34-token tokenizer. load_pretrained() raises GenMolAssetError on
missing or empty-vocabulary assets rather than returning an untrained
model. The hatchling artifacts entry is required because .gitignore
excludes *.pt from the build."
```

---

### Task 5: Wire GenMol into `DrugDiscovery-v0`

The environment is documented as navigating GenMol's latent space. It never reaches the generator: the tokenizer has no vocabulary, the model is randomly initialised, `decode_from_latent` does not exist on `MolVAE`, and every call lands in a twelve-string scaffold list.

**Files:**
- Modify: `src/neorx/causalbiorl/envs/drug_discovery.py` (`_init_genmol`, `_decode_latent`; delete `_fallback_generate`)
- Test: `tests/causalbiorl/test_env_generation.py`

**Interfaces:**
- Consumes: `load_pretrained()` from `neorx.genmol` (Task 4); `MolVAE.decode(z, temperature=1.0, greedy=False) -> Tensor[B, L-1]`; `SmilesTokenizer.decode(ids) -> str`
- Produces: no new public API. `DrugDiscoveryEnv._decode_latent` returns VAE-derived SMILES; the scaffold list ceases to exist in source.

- [ ] **Step 1: Pin the scaffold list as a fixture before deleting it**

The gate needs to detect a regression to these strings after the code that produced them is gone.

```python
# tests/causalbiorl/scaffold_fixture.py
"""The twelve hardcoded strings the deleted _fallback_generate returned.

Pinned so tests can assert the environment never emits them again. Do not
import this anywhere outside tests.
"""

DELETED_FALLBACK_SCAFFOLDS = frozenset({
    "c1ccc2[nH]c(-c3ccncc3)nc2c1",
    "O=C(NCc1ccccc1)c1cc2ccccc2[nH]1",
    "Cc1nc2ccccc2n1Cc1ccc(F)cc1",
    "O=C(c1ccc(O)cc1)c1ccc(O)cc1O",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccc(-c2nc3ccccc3s2)cc1",
    "O=c1[nH]c2ccccc2c2ccccc12",
    "NC(=O)c1cccc(-c2cccnc2)c1",
    "Oc1ccc(-c2cc(-c3ccc(O)cc3)no2)cc1",
    "CC1=NN(c2ccccc2)C(=O)C1",
    "c1ccc(CNc2ncnc3[nH]cnc23)cc1",
    "CC(C)c1nnc(C(C)C)n1C1CC1c1ccc(F)cc1",
})
```

- [ ] **Step 2: Write the failing test**

```python
# tests/causalbiorl/test_env_generation.py
"""The environment generates via the trained VAE, not a hardcoded list."""

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from neorx.causalbiorl.envs.drug_discovery import DrugDiscoveryEnv

from .scaffold_fixture import DELETED_FALLBACK_SCAFFOLDS


def _env():
    g = nx.DiGraph()
    for gene, score in [("gene_EGFR", 0.9), ("gene_TP53", 0.8), ("gene_ALK", 0.7)]:
        g.add_node(gene, type="gene", score=score, tissue_relevant=True)
    g.add_node("disease_X", type="disease", score=1.0, tissue_relevant=True)
    for gene in ("gene_EGFR", "gene_TP53", "gene_ALK"):
        g.add_edge(gene, "disease_X", edge_type="associated_with", weight=0.8)
    targets = [
        {"gene": n, "causal_confidence": c, "classification": "CAUSAL"}
        for n, c in [("EGFR", 0.85), ("TP53", 0.7), ("ALK", 0.65)]
    ]
    return DrugDiscoveryEnv(
        disease="TestDisease",
        prebuilt_graph=g,
        prebuilt_targets=targets,
        max_steps=20,
    )


def _episode_molecules(env, n_steps=20, seed=0):
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_steps):
        action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
        _, _, term, trunc, info = env.step(action)
        smiles = info.get("smiles") or info.get("molecule")
        if smiles:
            out.append(smiles)
        if term or trunc:
            break
    return out


def test_molecules_do_not_come_from_the_deleted_scaffold_list():
    """Provenance, not diversity.

    The model has documented partial posterior collapse, so outputs may
    cluster. Asserting diversity here would fail for a reason unrelated to
    whether the wiring works.
    """
    mols = _episode_molecules(_env())
    assert mols, "episode produced no molecules"
    leaked = set(mols) & DELETED_FALLBACK_SCAFFOLDS
    assert not leaked, f"environment emitted deleted fallback scaffolds: {leaked}"


def test_molecules_are_chemically_valid():
    mols = _episode_molecules(_env())
    assert all(Chem.MolFromSmiles(s) is not None for s in mols)


def test_fallback_generate_no_longer_exists():
    assert not hasattr(DrugDiscoveryEnv, "_fallback_generate"), (
        "a reachable fallback is the bug; an unreachable one is dead code"
    )


def test_missing_assets_surface_rather_than_degrade(monkeypatch):
    from neorx.genmol import GenMolAssetError
    import neorx.causalbiorl.envs.drug_discovery as dd

    def boom(*a, **k):
        raise GenMolAssetError("no trained checkpoint")

    monkeypatch.setattr(dd, "load_pretrained", boom)
    env = _env()
    with pytest.raises(GenMolAssetError):
        _episode_molecules(env, n_steps=1)
```

- [ ] **Step 3: Run it to confirm it fails**

```bash
pytest tests/causalbiorl/test_env_generation.py -q
```

Expected: FAIL on `test_molecules_do_not_come_from_the_deleted_scaffold_list` — every molecule is currently from that list.

- [ ] **Step 4: Rewrite the generation path**

In `src/neorx/causalbiorl/envs/drug_discovery.py`, add the import at module level:

```python
from neorx.genmol import load_pretrained
```

Replace `_init_genmol` entirely:

```python
    def _init_genmol(self) -> None:
        """Load the packaged GenMol model.

        Errors propagate. A randomly initialised VAE emits plausible SMILES,
        so a silent failure here is undetectable downstream.
        """
        self._genmol_model, self._genmol_tokenizer = load_pretrained()
```

Replace the body of `_decode_latent`:

```python
    def _decode_latent(self, z: NDArray[np.floating]) -> str:
        """Decode a latent vector to SMILES via the trained VAE."""
        import torch

        if self._genmol_model is None:
            self._init_genmol()

        z_tensor = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            token_ids = self._genmol_model.decode(z_tensor)
        return self._genmol_tokenizer.decode(token_ids[0])
```

Delete the entire `_fallback_generate` method, including its twelve-entry `scaffolds` list.

- [ ] **Step 5: Confirm the scaffold strings are gone from source**

```bash
grep -rn "CC(=O)Nc1ccc(O)cc1" src/ && echo "STILL PRESENT — remove it" || echo "CLEAN"
```

Expected: `CLEAN`. The string survives only in the test fixture.

- [ ] **Step 6: Run the tests to confirm they pass**

```bash
pytest tests/causalbiorl/test_env_generation.py -q
```

Expected: 4 passed

- [ ] **Step 7: Run the full suite for regressions**

```bash
pytest -q 2>&1 | tail -5
```

Expected: no new failures against `/tmp/neorx-baseline.txt`.

- [ ] **Step 8: Commit**

```bash
git add src/neorx/causalbiorl/envs/drug_discovery.py tests/causalbiorl/
git commit -m "fix: wire DrugDiscovery-v0 to the trained GenMol VAE

The env built an unvocabularised tokenizer and a randomly initialised
model, then called MolVAE.decode_from_latent -- which does not exist. The
AttributeError was swallowed, so every molecule came from a twelve-entry
hardcoded list. Now loads the packaged weights and calls the real
decode(). The fallback is deleted, not made to raise: a reachable
fallback is the bug, an unreachable one is dead code.

Scope stops here. The CEM inner loop still scores without decoding
(sub-project 3) -- changing that alters what the planner optimises."
```

---

### Task 6: Unified `neorx` CLI

Five console scripts collapse into one app. Each module already exposes a `typer.Typer()` instance named `app`, so this composes rather than rewrites.

**Files:**
- Create: `src/neorx/cli/__init__.py`, `src/neorx/cli/__main__.py`
- Modify: `pyproject.toml` (`[project.scripts]`), `main.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `app` from each of `neorx.core.__main__`, `neorx.genmol.__main__`, `neorx.causalbiorl.__main__`, `neorx.dockbot.__main__`, `neorx.mirrorfold.__main__`
- Produces: `neorx.cli.app` (a `typer.Typer`) and `neorx.cli.main()`, the single console-script entry point.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
"""One CLI, with a subcommand per module."""

import pytest
from typer.testing import CliRunner

from neorx.cli import app

runner = CliRunner()

SUBCOMMANDS = ["genmol", "dockbot", "causalbiorl", "mirrorfold"]


def test_root_help_lists_every_module():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in SUBCOMMANDS:
        assert name in result.stdout


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_each_subcommand_has_its_own_help(name):
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0


def test_core_commands_are_top_level():
    result = runner.invoke(app, ["--help"])
    assert "identify" in result.stdout or "run" in result.stdout
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
pytest tests/test_cli.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.cli'`

- [ ] **Step 3: Write the CLI package**

```python
# src/neorx/cli/__init__.py
"""Unified NeoRx command-line interface.

The core pipeline commands sit at the top level; each module is a
subcommand::

    neorx run HIV --top-n 5
    neorx genmol sample --n 100
    neorx dock 1BNA --ligand "CC(=O)O"
"""

from __future__ import annotations

import typer

from neorx.causalbiorl.__main__ import app as causalbiorl_app
from neorx.core.__main__ import app as core_app
from neorx.dockbot.__main__ import app as dockbot_app
from neorx.genmol.__main__ import app as genmol_app
from neorx.mirrorfold.__main__ import app as mirrorfold_app

app = typer.Typer(
    name="neorx",
    help="Causal drug target discovery via Pearl's do-calculus.",
    no_args_is_help=True,
)

# Core pipeline commands stay top-level: `neorx run`, `neorx identify`.
for _command in core_app.registered_commands:
    app.registered_commands.append(_command)

app.add_typer(genmol_app, name="genmol", help="Molecular generation (VAE).")
app.add_typer(dockbot_app, name="dockbot", help="Molecular docking.")
app.add_typer(causalbiorl_app, name="causalbiorl", help="Causal RL environments.")
app.add_typer(mirrorfold_app, name="mirrorfold", help="Mirror-image protein analysis.")

__all__ = ["app", "main"]


def main() -> None:
    """Console-script entry point."""
    app()
```

```python
# src/neorx/cli/__main__.py
"""Allow ``python -m neorx.cli``."""

from neorx.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Retarget the console scripts**

In `pyproject.toml`, replace the `[project.scripts]` block. The four module aliases stay for one release and are removed in 0.3.0.

```toml
[project.scripts]
neorx = "neorx.cli:main"
# Deprecated aliases, removed in 0.3.0.
genmol = "neorx.genmol.__main__:main"
dockbot = "neorx.dockbot.__main__:main"
causalbiorl = "neorx.causalbiorl.__main__:main"
mirrorfold = "neorx.mirrorfold.__main__:main"
```

Reduce root `main.py` to a shim:

```python
"""Deprecated entry point. Use the ``neorx`` console script."""

from neorx.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
pytest tests/test_cli.py -q
```

Expected: 6 passed

- [ ] **Step 6: Verify the installed script works**

```bash
pip install -e . --no-deps && neorx --help | head -20
```

Expected: help text listing `genmol`, `dockbot`, `causalbiorl`, `mirrorfold`.

- [ ] **Step 7: Commit**

```bash
git add src/neorx/cli/ tests/test_cli.py pyproject.toml main.py
git commit -m "feat: unify five console scripts into one neorx CLI

Each module already exposed a typer app, so this composes them. Old
script names retained as deprecated aliases through 0.2.x."
```

---

### Task 7: Lower the Python floor and the dependency floors

`requires-python = ">=3.13"` is gratuitous — the codebase contains no PEP 695 syntax and no `itertools.batched`. `pandas>=3.0.1` and `rdkit>=2025.9.6` exclude versions most environments have.

**Files:**
- Modify: `pyproject.toml` (`requires-python`, `classifiers`, `dependencies`, `[tool.ruff] target-version`, `[tool.mypy] python_version`)
- Create: `scripts/check_floors.sh`

**Interfaces:**
- Consumes: the test suite from Tasks 1–6
- Produces: `scripts/check_floors.sh`, invoked by the CI job added in Task 8.

- [ ] **Step 1: Confirm nothing needs 3.13**

```bash
grep -rnE "^\s*type [A-Za-z_]+ *=|def [a-zA-Z_]+\[[A-Z]|class [A-Za-z_]+\[[A-Z]|itertools\.batched" --include="*.py" src/ || echo "NO 3.13-ONLY SYNTAX"
```

Expected: `NO 3.13-ONLY SYNTAX`

- [ ] **Step 2: Lower the Python floor**

In `pyproject.toml`:

```toml
requires-python = ">=3.12"
```

Add to `classifiers`, keeping the existing 3.13 entry:

```toml
    "Programming Language :: Python :: 3.12",
```

And update the tool config:

```toml
[tool.ruff]
target-version = "py312"

[tool.mypy]
python_version = "3.12"
```

- [ ] **Step 3: Bisect the two aggressive dependency floors**

Determine floors empirically; do not guess. Run for each candidate and take the lowest that passes.

```bash
for v in 2.0.0 2.1.0 2.2.0; do
  python -m venv /tmp/floor-test && . /tmp/floor-test/bin/activate
  pip install -q "pandas==$v" 2>/dev/null \
    && pip install -q -e . --no-deps 2>/dev/null \
    && pytest -q tests/test_public_api.py >/dev/null 2>&1 \
    && echo "pandas $v: PASS" || echo "pandas $v: FAIL"
  deactivate && rm -rf /tmp/floor-test
done
```

Repeat for `rdkit` with candidates `2023.9.1 2024.3.1 2024.9.1`. Record the lowest passing version for each.

- [ ] **Step 4: Apply the discovered floors**

Edit `dependencies` in `pyproject.toml`, writing the lowest version that printed
`PASS` in Step 3. If, for example, `pandas 2.1.0` passed but `2.0.0` failed, and
`rdkit 2024.3.1` passed but `2023.9.1` failed, the two lines become:

```toml
    "pandas>=2.1.0",
    "rdkit>=2024.3.1",
```

Two rules for the cases the bisect does not settle cleanly:

- **If every candidate fails**, keep the existing pin and add a comment saying
  what broke, so the next person does not repeat the search:
  ```toml
    # 3.0.1 is the floor: 2.x lacks the copy-on-write semantics graph_builder
    # relies on. Verified 2026-09-02.
    "pandas>=3.0.1",
  ```
- **If the lowest candidate passes**, do not assume it is the true floor — extend
  the candidate list downward one release and re-run Step 3 until something
  fails. A floor that is lower than reality is worse than one that is too high,
  because it fails at a user's machine rather than in CI.

- [ ] **Step 5: Write the floors check script**

```bash
# scripts/check_floors.sh
#!/usr/bin/env bash
# Install the declared dependency lower bounds and run the smoke suite.
# A floor that is wrong must fail here, not on a user's machine.
set -euo pipefail

python - <<'PY' > /tmp/floors.txt
import re, tomllib
with open("pyproject.toml", "rb") as fh:
    deps = tomllib.load(fh)["project"]["dependencies"]
for d in deps:
    m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)>=([0-9][^,;\s]*)", d)
    if m:
        print(f"{m.group(1)}=={m.group(2)}")
PY

echo "Installing declared floors:"
cat /tmp/floors.txt
pip install -q -r /tmp/floors.txt
pip install -q -e . --no-deps
pytest -q tests/test_public_api.py tests/genmol/test_pretrained.py
```

```bash
chmod +x scripts/check_floors.sh
```

- [ ] **Step 6: Run it**

```bash
./scripts/check_floors.sh
```

Expected: all tests pass. A failure means a declared floor is too low — raise it and re-run.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml scripts/check_floors.sh
git commit -m "chore: lower Python floor to 3.12 and relax dependency floors

No PEP 695 syntax or itertools.batched anywhere in src/, so >=3.13 was
gratuitous. pandas and rdkit floors bisected empirically against the
smoke suite. scripts/check_floors.sh keeps them honest in CI."
```

---

### Task 8: CI gates that exercise the built wheel

The current `test-install` job runs `import neorx` plus three functions, from inside the repository — so it imports source, not the artifact, and passes on a broken package.

**Files:**
- Modify: `.github/workflows/publish.yml`
- Create: `.github/workflows/nightly.yml`
- Create: `tests/test_wheel_contents.py`

**Interfaces:**
- Consumes: every test and script from Tasks 1–7
- Produces: no importable API. This task gates everything preceding it.

- [ ] **Step 1: Write the failing wheel-contents test**

```python
# tests/test_wheel_contents.py
"""What ships, and what must never ship."""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out)],
        check=True,
        capture_output=True,
    )
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    return wheels[0]


def _names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def test_modules_is_never_packaged(wheel):
    leaked = [n for n in _names(wheel) if n.startswith("modules/")]
    assert not leaked, f"the modules/ shim must not ship: {leaked}"


def test_genmol_weights_are_packaged(wheel):
    names = _names(wheel)
    assert "neorx/genmol/assets/molvae_chembl36.pt" in names, (
        "weights absent -- check the hatchling artifacts entry; .gitignore "
        "excludes *.pt from the build by default"
    )
    assert "neorx/genmol/assets/tokenizer.json" in names


def test_all_six_subpackages_are_packaged(wheel):
    names = _names(wheel)
    for pkg in ("core", "genmol", "causalbiorl", "molscreen", "dockbot", "mirrorfold"):
        assert any(n.startswith(f"neorx/{pkg}/") for n in names), f"missing {pkg}"
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
pip install build && pytest tests/test_wheel_contents.py -q
```

Expected: FAIL on the weights assertion if the `artifacts` entry from Task 4 is missing or wrong.

- [ ] **Step 3: Replace the `test-install` job**

In `.github/workflows/publish.yml`, replace the entire `test-install` job with:

```yaml
  test-install:
    name: Smoke-test the built wheel
    needs: build
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.12", "3.13"]
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Install the wheel
        run: pip install dist/*.whl

      # Everything below runs from $HOME. Running inside the checkout would
      # import the source tree instead of the installed package -- which is
      # how CI previously passed on a broken wheel.
      - name: Import every public name
        working-directory: ${{ runner.temp }}
        run: |
          python - <<'PY'
          import importlib
          pkgs = ["neorx", "neorx.core", "neorx.genmol", "neorx.causalbiorl",
                  "neorx.molscreen", "neorx.dockbot", "neorx.mirrorfold"]
          for name in pkgs:
              mod = importlib.import_module(name)
              assert getattr(mod, "__all__", None), f"{name} declares no __all__"
              missing = [n for n in mod.__all__ if not hasattr(mod, n)]
              assert not missing, f"{name} exports undefined names: {missing}"
              print(f"{name}: {len(mod.__all__)} names OK")
          PY

      - name: Generate real molecules from the shipped weights
        working-directory: ${{ runner.temp }}
        run: |
          python - <<'PY'
          from rdkit import Chem
          from neorx.genmol import generate, load_pretrained
          model, tok = load_pretrained()
          smiles = generate(model, tok, n=20)
          assert smiles, "generate returned nothing"
          assert all(Chem.MolFromSmiles(s) for s in smiles), "invalid SMILES"
          assert len(set(smiles)) > 1, "expected more than one distinct molecule"
          print(f"generated {len(smiles)} molecules, {len(set(smiles))} distinct")
          PY

      - name: Environment generates via the VAE, not the deleted scaffolds
        working-directory: ${{ runner.temp }}
        run: |
          python - <<'PY'
          import networkx as nx, numpy as np
          from neorx.causalbiorl.envs.drug_discovery import DrugDiscoveryEnv

          # Provenance check only. The model has documented partial posterior
          # collapse, so asserting diversity here would fail for unrelated
          # reasons.
          SCAFFOLDS = {
              "c1ccc2[nH]c(-c3ccncc3)nc2c1",
              "O=C(NCc1ccccc1)c1cc2ccccc2[nH]1",
              "Cc1nc2ccccc2n1Cc1ccc(F)cc1",
              "O=C(c1ccc(O)cc1)c1ccc(O)cc1O",
              "CC(=O)Nc1ccc(O)cc1",
              "c1ccc(-c2nc3ccccc3s2)cc1",
              "O=c1[nH]c2ccccc2c2ccccc12",
              "NC(=O)c1cccc(-c2cccnc2)c1",
              "Oc1ccc(-c2cc(-c3ccc(O)cc3)no2)cc1",
              "CC1=NN(c2ccccc2)C(=O)C1",
              "c1ccc(CNc2ncnc3[nH]cnc23)cc1",
              "CC(C)c1nnc(C(C)C)n1C1CC1c1ccc(F)cc1",
          }
          g = nx.DiGraph()
          for gene, s in [("gene_A", 0.9), ("gene_B", 0.8)]:
              g.add_node(gene, type="gene", score=s, tissue_relevant=True)
          g.add_node("disease_X", type="disease", score=1.0, tissue_relevant=True)
          for gene in ("gene_A", "gene_B"):
              g.add_edge(gene, "disease_X", edge_type="associated_with", weight=0.8)
          targets = [{"gene": "A", "causal_confidence": 0.85, "classification": "CAUSAL"}]

          env = DrugDiscoveryEnv(disease="T", prebuilt_graph=g,
                                 prebuilt_targets=targets, max_steps=20)
          env.reset(seed=0)
          rng = np.random.default_rng(0)
          mols = []
          for _ in range(20):
              a = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
              *_, info = env.step(a)
              s = info.get("smiles") or info.get("molecule")
              if s:
                  mols.append(s)
          assert mols, "episode produced no molecules"
          leaked = set(mols) & SCAFFOLDS
          assert not leaked, f"deleted fallback scaffolds emitted: {leaked}"
          print(f"{len(mols)} molecules, none from the scaffold list")
          PY

      - name: Wheel contents
        run: |
          python -m zipfile -l dist/*.whl > /tmp/contents.txt
          grep -q "neorx/genmol/assets/molvae_chembl36.pt" /tmp/contents.txt \
            || { echo "FAIL: weights absent from wheel"; exit 1; }
          ! grep -q "^modules/" /tmp/contents.txt \
            || { echo "FAIL: modules/ shim leaked into wheel"; exit 1; }
          echo "wheel contents OK"
```

- [ ] **Step 4: Add lint, type, coverage and floors to the CI workflow**

Append this job to `.github/workflows/ci.yml`:

```yaml
  quality:
    name: Lint, types, coverage, floors
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]" ruff mypy
      - run: ruff check src/ tests/
      - run: mypy src/neorx
      - name: Coverage ratchet
        run: |
          # Baseline is the coverage of the first green build; it may rise,
          # never fall. Raising the target is sub-project 4's job.
          pytest --cov=src/neorx --cov-report=term --cov-fail-under=$(cat .coverage-floor)
      - run: ./scripts/check_floors.sh
```

Record the current baseline so the ratchet has a starting point:

```bash
pytest --cov=src/neorx --cov-report=term 2>&1 | grep TOTAL | awk '{print int($NF)}' > .coverage-floor
cat .coverage-floor
```

- [ ] **Step 5: Move network-dependent checks to a nightly job**

```yaml
# .github/workflows/nightly.yml
name: Nightly end-to-end

on:
  schedule:
    - cron: "0 3 * * *"
  workflow_dispatch:

# These steps query eight external databases that drift and rate-limit.
# They must never gate a merge; a failure opens an issue instead.
jobs:
  end-to-end:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e .
      - name: Full pipeline against live APIs
        run: neorx run HIV --top-n 5
      - name: Open an issue on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Nightly end-to-end failed (${new Date().toISOString().slice(0,10)})`,
              body: `\`neorx run HIV\` failed. Likely an upstream API change.\n\n${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
              labels: ['nightly-failure'],
            })
```

- [ ] **Step 6: Run the wheel test to confirm it passes**

```bash
pytest tests/test_wheel_contents.py -q
```

Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/ tests/test_wheel_contents.py .coverage-floor
git commit -m "ci: gate on the built wheel instead of the source tree

The previous test-install job ran import neorx from inside the checkout,
so it imported source and passed on a broken package. Every check now
runs from runner.temp against the installed wheel, and asserts the four
things that were actually wrong: every __all__ resolves, generation uses
the shipped weights, the env does not emit the deleted scaffolds, and
modules/ never ships. Live-API checks move to a nightly job."
```

---

### Task 9: Release 0.2.0

**Files:**
- Modify: `pyproject.toml` (`version`), `src/neorx/__init__.py` (`__version__`), `CHANGELOG.md`, `CITATION.cff`
- Test: `tests/test_version.py`

**Interfaces:**
- Consumes: everything above
- Produces: a published `neorx` 0.2.0 on PyPI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_version.py
"""One version number, declared in one place, agreeing everywhere."""

import tomllib
from pathlib import Path

import neorx


def test_version_matches_pyproject():
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert neorx.__version__ == declared


def test_version_is_0_2_0():
    assert neorx.__version__ == "0.2.0"
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
pytest tests/test_version.py -q
```

Expected: FAIL — pyproject still declares `0.1.0`

- [ ] **Step 3: Bump the version in both places**

`pyproject.toml`:

```toml
version = "0.2.0"
```

`src/neorx/__init__.py` already sets `__version__ = "0.2.0"` from Task 3. Confirm it does.

- [ ] **Step 4: Update `CHANGELOG.md`**

Prepend:

```markdown
## [0.2.0] — 2026-09-02

### Changed
- **Breaking:** `modules.*` imports become `neorx.*`; `modules.neorx` becomes
  `neorx.core`. A deprecated `modules/` shim is available in the repository
  through 0.2.x and is removed in 0.3.0.
- Package moves to a `src/` layout, so tests exercise the installed artifact.
- Python floor lowered from 3.13 to 3.12; `pandas` and `rdkit` floors relaxed.
- Five console scripts unified under `neorx <module> <command>`. The old script
  names remain as deprecated aliases through 0.2.x.

### Added
- All six modules exposed through the public API. `molscreen` previously
  exported nothing.
- Trained GenMol weights ship as package data; `load_pretrained()` raises
  `GenMolAssetError` rather than returning an untrained model.

### Fixed
- `DrugDiscovery-v0` now generates through the trained VAE. It previously
  built an unvocabularised tokenizer and a randomly initialised model, called a
  `decode_from_latent` method that does not exist, swallowed the resulting
  `AttributeError`, and returned one of twelve hardcoded scaffolds.
```

- [ ] **Step 5: Update `CITATION.cff`**

Change these three fields. The `notes` field is what keeps the manuscripts
reproducible: 0.2.0 does not reproduce their numbers, and a reader who installs
it expecting to must be told which tag does.

```yaml
version: 0.2.0
date-released: "2026-09-02"
notes: >-
  Results reported in the accompanying manuscripts correspond to the
  v0.1.0-paper tag. Later releases change the causal engine and will not
  reproduce those figures.
```

Also correct the licence wording while you are here — `README.md` and the
manuscripts describe the project as open source, but `LICENSE` is the NeoRx
Source Available Licence, which prohibits commercial use. Replace
"open-source" with "source-available" in `README.md`.

- [ ] **Step 6: Run the full suite**

```bash
pytest -q 2>&1 | tail -5
```

Expected: all green.

- [ ] **Step 7: Build and inspect the artifacts**

```bash
python -m build
python -m zipfile -l dist/neorx-0.2.0-*.whl | grep -E "assets|^modules" || true
twine check dist/*
ls -lh dist/
```

Expected: the two asset files present, no `modules/` entries, `twine check` passes, wheel roughly 17 MB.

- [ ] **Step 8: Publish to TestPyPI first**

```bash
gh workflow run publish.yml -f target=testpypi
```

Then verify from a clean environment, outside the repository:

```bash
cd /tmp && python -m venv verify && . verify/bin/activate
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple neorx==0.2.0
python -c "
from neorx.genmol import generate, load_pretrained
m, t = load_pretrained()
print(generate(m, t, n=3))
"
deactivate
```

Expected: three valid SMILES from a package installed off the index.

- [ ] **Step 9: Commit and tag**

```bash
git add pyproject.toml src/neorx/__init__.py CHANGELOG.md CITATION.cff tests/test_version.py
git commit -m "release: 0.2.0"
git tag -a v0.2.0 -m "0.2.0 — src layout, all modules exposed, weights shipped"
git push origin main --tags
```

- [ ] **Step 10: Publish to PyPI**

Create a GitHub release for `v0.2.0`; `publish.yml` triggers on `release: published`.

```bash
gh release create v0.2.0 --title "0.2.0" --notes-from-tag
```

Verify:

```bash
cd /tmp && python -m venv final && . final/bin/activate
pip install neorx==0.2.0 && neorx --help && deactivate
```

---

## Notes for the executor

**Run the suite after every task**, comparing against `/tmp/neorx-baseline.txt` from Task 1 Step 2. The move touches 194 import statements; a missed one surfaces as an unrelated-looking failure three tasks later.

**Task 4 Step 3 is the subtlest step in this plan.** `.gitignore` excludes `*.pt` from both git and the hatchling build. Without both the negation pattern and the `artifacts` entry, the wheel builds successfully with no weights, emits no warning, and fails only on a user's machine. Task 8's wheel-contents test exists specifically to catch this.

**Do not extend the scope.** The four deferred items — `nx.d_separated`, the CEM inner loop, the ablation harness, splitting the two large files — belong to sub-projects 2 through 4. If a task appears to require one of them, stop and raise it rather than reaching for it.
