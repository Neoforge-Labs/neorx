# Experiment Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runner that owns experiment persistence, so a reported number cannot exist without a saved, replayable run record.

**Architecture:** A `neorx.experiments` package ships with the library: `RunRecord` writes rows incrementally to `runs/<id>/rows.jsonl` as each cell completes, `capture.py` freezes HTTP inputs via vcrpy cassettes for exact replay, and `gates.py` provides CI checks that fail when a figure carries a numeric literal or a manuscript cites a run that does not exist. The four experiments that back the papers are then migrated onto it and their standalone scripts deleted.

**Tech Stack:** Python 3.12+, pytest, Typer, vcrpy, polars, hatchling.

**Spec:** `docs/superpowers/specs/2026-09-03-experiment-tracking-design.md`

## Global Constraints

- **Python floor `>=3.12`.** No PEP 695 syntax (`type X =`, `def f[T]`, `class C[T]`).
- **No shims, compatibility layers, or silent fallbacks.** Standing owner directive. A failure surfaces; it never degrades. vcrpy is the one sanctioned exception: imported only by `neorx.experiments`, active only inside a recording context, never in a library code path.
- **Do NOT touch** (sub-project 3 owns these): `nx.d_separated` / backdoor logic in `core/causal/identifier.py`, the CEM inner loop in `causalbiorl/causal/planner.py`, the ablation harness, and the internals of `identifier.py` / `drug_discovery.py`.
- **`runs/` is tracked.** This reverses `.gitignore:37`. The runner refuses to write a record exceeding **50 MB** without `--allow-large`.
- **Run ID format:** `YYYY-MM-DD-<experiment>-<6-char hash>`, hash over start timestamp + code SHA.
- **A run from a dirty tree is `citable: false`.** The figure command refuses non-citable runs.
- **ChEMBL version** comes from `SELECT chembl_release, creation_date FROM chembl_release ORDER BY chembl_release_id DESC LIMIT 1` — expected `('CHEMBL_36', '2025-07-28 00:00:00.000000')`. The `version` table is NOT the right source; it holds ontology versions.
- **No `Co-Authored-By` trailer** and no tool attribution in any commit message.
- **The venv is uv-managed and has NO pip.** Use `uv pip install --python .venv/bin/python <pkg>`, never `.venv/bin/python -m pip`.
- **Run tests as** `.venv/bin/python -m pytest`. Bare `pytest` is not on PATH.
- **Do NOT run the full suite** (~10 min, cannot be parallelised — xdist fails on nondeterministic collection). Run targeted tests; the controller runs the full suite.
- **Never `git add`:** `checkpoints/`, `.backup-scripts/`, `dist/`, `_diag_*.py`, `_check_*.py`, `_smoke_test.py`, `_fast_*.py`, `_run_genmol*.py`, `_train_genmol_cpu.py`, `_verify_tissue.py`, `_export_smiles.py`. Use explicit `git add <paths>` and check `git status --short` before each commit.
- **Platform is macOS.** BSD `sed` needs `-i ''`; zsh does NOT word-split unquoted variables — use arrays (`FILES=(${(f)"$(...)"})`).

---

### Task 1: RunRecord

The core primitive. Everything else depends on its exact interface.

**Files:**
- Create: `src/neorx/experiments/__init__.py`, `src/neorx/experiments/record.py`
- Test: `tests/experiments/__init__.py`, `tests/experiments/test_record.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `RunRecord.create(experiment: str, *, runs_dir: Path | None = None) -> RunRecord`
  - `record.run_id: str`, `record.path: Path`, `record.citable: bool`
  - `record.append_row(row: dict) -> None`
  - `record.finalise(status: str) -> None`  (`"complete" | "incomplete" | "failed"`)
  - `record.size_bytes() -> int`
  - `RunRecord.load(run_id: str, *, runs_dir: Path | None = None) -> RunRecord`
  - `RunRecord.rows() -> list[dict]`
  - `class RecordTooLargeError(RuntimeError)`
  - Module constants: `RUNS_DIR: Path`, `MAX_RECORD_BYTES: int = 50 * 1024 * 1024`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_record.py
"""RunRecord: incremental rows, honest env capture, explicit status."""

import json
import subprocess
from pathlib import Path

import pytest

from neorx.experiments.record import (
    MAX_RECORD_BYTES,
    RecordTooLargeError,
    RunRecord,
)


def test_run_id_shape(tmp_path):
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    parts = rec.run_id.split("-")
    assert len(parts) >= 5, rec.run_id
    assert parts[0].isdigit() and len(parts[0]) == 4      # year
    assert "demo-exp" in rec.run_id
    assert len(parts[-1]) == 6                             # short hash


def test_rows_are_written_as_they_are_appended(tmp_path):
    """The whole point: a crash must not cost completed cells."""
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.append_row({"disease": "HIV", "f1": 0.545})
    on_disk = (rec.path / "rows.jsonl").read_text().strip().splitlines()
    assert len(on_disk) == 1, "row must hit disk before the run ends"
    rec.append_row({"disease": "malaria", "f1": 0.333})
    on_disk = (rec.path / "rows.jsonl").read_text().strip().splitlines()
    assert len(on_disk) == 2
    assert json.loads(on_disk[0])["disease"] == "HIV"


def test_unfinalised_record_is_incomplete_not_missing(tmp_path):
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.append_row({"disease": "HIV", "f1": 0.545})
    # simulate a crash: never call finalise
    reloaded = RunRecord.load(rec.run_id, runs_dir=tmp_path)
    assert reloaded.status == "incomplete"
    assert len(reloaded.rows()) == 1
    assert reloaded.citable is False


def test_finalise_sets_status_and_completes(tmp_path):
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.append_row({"disease": "HIV", "f1": 0.545})
    rec.finalise("complete")
    data = json.loads((rec.path / "record.json").read_text())
    assert data["status"] == "complete"
    assert data["n_rows"] == 1


def test_env_records_code_sha_and_dirty_flag(tmp_path):
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    env = json.loads((rec.path / "env.json").read_text())
    assert len(env["git_sha"]) == 40
    assert isinstance(env["git_dirty"], bool)
    assert env["python"].startswith("3.")
    assert "platform" in env


def test_dirty_tree_makes_run_non_citable(tmp_path, monkeypatch):
    """A run nobody can reproduce must not be citable."""
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod, "_git_dirty", lambda: True)
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.finalise("complete")
    assert rec.citable is False
    assert json.loads((rec.path / "record.json").read_text())["citable"] is False


def test_clean_tree_complete_run_is_citable(tmp_path, monkeypatch):
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod, "_git_dirty", lambda: False)
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.finalise("complete")
    assert rec.citable is True


def test_oversized_record_is_refused(tmp_path, monkeypatch):
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod, "MAX_RECORD_BYTES", 100)
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    (rec.path / "inputs").mkdir(exist_ok=True)
    (rec.path / "inputs" / "big.json").write_bytes(b"x" * 500)
    with pytest.raises(RecordTooLargeError, match="50 MB|allow-large|exceeds"):
        rec.finalise("complete")


def test_allow_large_bypasses_the_refusal(tmp_path, monkeypatch):
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod, "MAX_RECORD_BYTES", 100)
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path, allow_large=True)
    (rec.path / "inputs").mkdir(exist_ok=True)
    (rec.path / "inputs" / "big.json").write_bytes(b"x" * 500)
    rec.finalise("complete")  # must not raise
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
.venv/bin/python -m pytest tests/experiments/test_record.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.experiments'`

- [ ] **Step 3: Write the implementation**

```python
# src/neorx/experiments/record.py
"""Run records: the artifact that makes a reported number traceable.

A record is written incrementally. Rows land on disk as each cell of an
experiment completes, so a crash costs the current cell rather than the
run -- the failure that lost 881 seconds of the original RL benchmark and
led to its numbers being retyped by hand.

Nothing here degrades on error. A record that cannot be written is a
failed run, not a quiet one.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
MAX_RECORD_BYTES = 50 * 1024 * 1024

_VALID_STATUS = ("complete", "incomplete", "failed")


class RecordTooLargeError(RuntimeError):
    """Raised when a finalised record exceeds MAX_RECORD_BYTES."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def _git_sha() -> str:
    return _git("rev-parse", "HEAD") or "0" * 40


def _git_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def _dep_versions() -> dict[str, str]:
    from importlib.metadata import distributions

    return {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}


class RunRecord:
    """One execution of one experiment."""

    def __init__(self, run_id: str, path: Path, *, allow_large: bool = False) -> None:
        self.run_id = run_id
        self.path = path
        self.status = "incomplete"
        self.citable = False
        self._allow_large = allow_large
        self._n_rows = 0

    # ── construction ────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        experiment: str,
        *,
        runs_dir: Path | None = None,
        allow_large: bool = False,
    ) -> RunRecord:
        started = datetime.now(timezone.utc)
        sha = _git_sha()
        digest = hashlib.sha256(f"{started.isoformat()}{sha}".encode()).hexdigest()[:6]
        run_id = f"{started:%Y-%m-%d}-{experiment}-{digest}"

        base = runs_dir if runs_dir is not None else RUNS_DIR
        path = base / run_id
        (path / "inputs").mkdir(parents=True, exist_ok=True)

        rec = cls(run_id, path, allow_large=allow_large)
        rec._started = started
        rec._experiment = experiment
        (path / "rows.jsonl").touch()
        (path / "env.json").write_text(
            json.dumps(
                {
                    "git_sha": sha,
                    "git_dirty": _git_dirty(),
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "started_utc": started.isoformat(),
                    "deps": _dep_versions(),
                },
                indent=2,
            )
        )
        return rec

    @classmethod
    def load(cls, run_id: str, *, runs_dir: Path | None = None) -> RunRecord:
        base = runs_dir if runs_dir is not None else RUNS_DIR
        path = base / run_id
        if not path.is_dir():
            raise FileNotFoundError(f"no run record at {path}")
        rec = cls(run_id, path)
        summary = path / "record.json"
        if summary.exists():
            data = json.loads(summary.read_text())
            rec.status = data["status"]
            rec.citable = data["citable"]
        return rec

    # ── writing ─────────────────────────────────────────────────

    def append_row(self, row: dict[str, Any]) -> None:
        """Append one result row, flushed to disk immediately."""
        with (self.path / "rows.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
        self._n_rows += 1

    def finalise(self, status: str) -> None:
        if status not in _VALID_STATUS:
            raise ValueError(f"status must be one of {_VALID_STATUS}, got {status!r}")

        size = self.size_bytes()
        if size > MAX_RECORD_BYTES and not self._allow_large:
            raise RecordTooLargeError(
                f"record {self.run_id} is {size / 1e6:.1f} MB, which exceeds the "
                f"{MAX_RECORD_BYTES / 1e6:.0f} MB limit. Re-run with --allow-large "
                f"if this is intentional, or prune inputs you do not need."
            )

        env = json.loads((self.path / "env.json").read_text())
        self.status = status
        self.citable = status == "complete" and not env["git_dirty"]
        (self.path / "record.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "experiment": getattr(self, "_experiment", self.run_id),
                    "status": status,
                    "citable": self.citable,
                    "n_rows": self._n_rows,
                    "size_bytes": size,
                    "finalised_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )

    # ── reading ─────────────────────────────────────────────────

    def rows(self) -> list[dict[str, Any]]:
        text = (self.path / "rows.jsonl").read_text().strip()
        return [json.loads(line) for line in text.splitlines()] if text else []

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())
```

```python
# src/neorx/experiments/__init__.py
"""Experiment runner: a number cannot exist without a run record.

The runner owns persistence. Experiments are declared in the registry and
executed by the runner, which writes the record itself -- there is no code
path by which an experiment reports a result without one.
"""

from neorx.experiments.record import (
    MAX_RECORD_BYTES,
    RUNS_DIR,
    RecordTooLargeError,
    RunRecord,
)

__all__ = [
    "RunRecord",
    "RecordTooLargeError",
    "RUNS_DIR",
    "MAX_RECORD_BYTES",
]
```

```bash
touch tests/experiments/__init__.py
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/experiments/test_record.py -q
```

Expected: 9 passed

- [ ] **Step 5: Verify RUNS_DIR resolves to the repo root**

`RUNS_DIR` is computed from `__file__` via `parents[3]`. Confirm it lands on the repository root and not inside `src/`:

```bash
.venv/bin/python -c "
from neorx.experiments.record import RUNS_DIR
from pathlib import Path
print('RUNS_DIR:', RUNS_DIR)
assert RUNS_DIR.name == 'runs', RUNS_DIR
assert (RUNS_DIR.parent / 'pyproject.toml').exists(), f'not repo root: {RUNS_DIR.parent}'
print('resolves to repo root: OK')"
```

If the assertion fails, adjust the `parents[N]` index rather than hardcoding a path — the file is at `src/neorx/experiments/record.py`, so the root is three parents up.

- [ ] **Step 6: Commit**

```bash
git add src/neorx/experiments/ tests/experiments/
git commit -m "feat: add RunRecord, the experiment provenance primitive

Rows are flushed to rows.jsonl as each cell completes, so a crash costs
the current cell rather than the run. env.json records the code SHA and
whether the tree was dirty; a run from a dirty tree is not citable,
because nobody can reproduce it."
```

---

### Task 2: Registry and the `neorx exp` CLI

**Files:**
- Create: `src/neorx/experiments/registry.py`, `src/neorx/experiments/__main__.py`
- Modify: `src/neorx/cli/__init__.py`
- Test: `tests/experiments/test_registry.py`, `tests/experiments/test_exp_cli.py`

**Interfaces:**
- Consumes: `RunRecord.create/append_row/finalise/load` (Task 1)
- Produces:
  - `@experiment(name: str, help: str = "")` decorator wrapping `fn(record: RunRecord) -> None`
  - `get_experiment(name: str) -> ExperimentDef` — raises `UnknownExperimentError`
  - `list_experiments() -> list[ExperimentDef]`
  - `ExperimentDef` with `.name`, `.help`, `.fn`
  - `run_experiment(name: str, *, allow_large: bool = False, runs_dir: Path | None = None) -> RunRecord`
  - `class UnknownExperimentError(KeyError)`
  - Typer app `app` exported from `neorx.experiments.__main__`, registered in the CLI as `neorx exp`

- [ ] **Step 1: Write the failing tests**

```python
# tests/experiments/test_registry.py
"""The registry is what makes the runner, not the script, own persistence."""

import pytest

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import (
    UnknownExperimentError,
    experiment,
    get_experiment,
    list_experiments,
    run_experiment,
)


def test_decorator_registers_by_name():
    @experiment(name="unit-demo", help="a demo")
    def _demo(record: RunRecord) -> None:
        record.append_row({"x": 1})

    assert get_experiment("unit-demo").help == "a demo"
    assert "unit-demo" in [e.name for e in list_experiments()]


def test_unknown_experiment_raises_with_the_known_names():
    with pytest.raises(UnknownExperimentError, match="unit-demo|available"):
        get_experiment("no-such-experiment")


def test_run_experiment_writes_a_finalised_record(tmp_path):
    @experiment(name="unit-writes", help="")
    def _w(record: RunRecord) -> None:
        record.append_row({"disease": "HIV", "f1": 0.5})

    rec = run_experiment("unit-writes", runs_dir=tmp_path)
    assert rec.status == "complete"
    assert len(rec.rows()) == 1


def test_a_raising_experiment_still_leaves_a_record(tmp_path):
    """A failed run is an artifact, not an absence."""

    @experiment(name="unit-raises", help="")
    def _r(record: RunRecord) -> None:
        record.append_row({"disease": "HIV", "f1": 0.5})
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_experiment("unit-raises", runs_dir=tmp_path)

    written = sorted(tmp_path.iterdir())
    assert len(written) == 1
    rec = RunRecord.load(written[0].name, runs_dir=tmp_path)
    assert rec.status == "failed"
    assert len(rec.rows()) == 1, "rows completed before the failure must survive"
    assert rec.citable is False
```

```python
# tests/experiments/test_exp_cli.py
"""`neorx exp` is the only supported way to run an experiment."""

from typer.testing import CliRunner

from neorx.cli import app

runner = CliRunner()


def test_exp_is_a_neorx_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "exp" in result.stdout


def test_exp_list_names_the_registered_experiments():
    result = runner.invoke(app, ["exp", "list"])
    assert result.exit_code == 0
    assert "neorx-7disease" in result.stdout


def test_exp_run_rejects_an_unknown_name():
    result = runner.invoke(app, ["exp", "run", "no-such-experiment"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run them to confirm they fail**

```bash
.venv/bin/python -m pytest tests/experiments/test_registry.py tests/experiments/test_exp_cli.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.experiments.registry'`

- [ ] **Step 3: Write the registry**

```python
# src/neorx/experiments/registry.py
"""Experiment registry.

An experiment is a function taking a RunRecord. It never opens a file and
never decides where results go -- the runner does both. That is what makes
"a number cannot exist without a record" structural rather than a habit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from neorx.experiments.record import RunRecord

ExperimentFn = Callable[[RunRecord], None]


class UnknownExperimentError(KeyError):
    """Raised when no experiment is registered under the given name."""


@dataclass(frozen=True)
class ExperimentDef:
    name: str
    help: str
    fn: ExperimentFn


_REGISTRY: dict[str, ExperimentDef] = {}


def experiment(*, name: str, help: str = "") -> Callable[[ExperimentFn], ExperimentFn]:
    """Register an experiment under ``name``."""

    def decorate(fn: ExperimentFn) -> ExperimentFn:
        if name in _REGISTRY:
            raise ValueError(f"experiment {name!r} is already registered")
        _REGISTRY[name] = ExperimentDef(name=name, help=help, fn=fn)
        return fn

    return decorate


def get_experiment(name: str) -> ExperimentDef:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise UnknownExperimentError(
            f"no experiment named {name!r}. Available: {known}"
        ) from None


def list_experiments() -> list[ExperimentDef]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def run_experiment(
    name: str,
    *,
    allow_large: bool = False,
    runs_dir: Path | None = None,
) -> RunRecord:
    """Execute an experiment, writing its record whatever the outcome."""
    defn = get_experiment(name)
    record = RunRecord.create(name, runs_dir=runs_dir, allow_large=allow_large)
    try:
        defn.fn(record)
    except BaseException:
        record.finalise("failed")
        raise
    record.finalise("complete")
    return record
```

- [ ] **Step 4: Write the CLI**

```python
# src/neorx/experiments/__main__.py
"""`neorx exp` -- run, list and inspect experiment records."""

from __future__ import annotations

import typer

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import list_experiments, run_experiment

app = typer.Typer(
    name="exp",
    help="Run recorded experiments and inspect their run records.",
    no_args_is_help=True,
)


@app.command("list")
def list_cmd() -> None:
    """List every registered experiment."""
    import experiments  # noqa: F401  -- import registers the definitions

    for defn in list_experiments():
        typer.echo(f"{defn.name:22} {defn.help}")


@app.command("run")
def run_cmd(
    name: str = typer.Argument(..., help="Experiment name, from `neorx exp list`."),
    allow_large: bool = typer.Option(
        False, "--allow-large", help="Permit a record larger than 50 MB."
    ),
) -> None:
    """Run an experiment and write its record."""
    import experiments  # noqa: F401

    record = run_experiment(name, allow_large=allow_large)
    typer.echo(f"{record.run_id}  status={record.status}  citable={record.citable}")


@app.command("show")
def show_cmd(run_id: str = typer.Argument(..., help="Run ID to inspect.")) -> None:
    """Print a run record's status and rows."""
    record = RunRecord.load(run_id)
    typer.echo(f"{record.run_id}  status={record.status}  citable={record.citable}")
    for row in record.rows():
        typer.echo(f"  {row}")


def main() -> None:
    app()
```

- [ ] **Step 5: Register it in the CLI**

In `src/neorx/cli/__init__.py`, add the import beside the other module apps and the `add_typer` call beside the others (the file already follows this exact pattern at lines 21 and 33-37):

```python
from neorx.experiments.__main__ import app as exp_app
```

```python
app.add_typer(exp_app, name="exp", help="Recorded experiments.")
```

- [ ] **Step 6: Run the tests to confirm they pass**

`test_exp_list_names_the_registered_experiments` asserts `neorx-7disease` is listed, which is not registered until Task 8. Until then it is expected to fail:

```bash
.venv/bin/python -m pytest tests/experiments/test_registry.py tests/experiments/test_exp_cli.py -q
```

Expected: 4 registry tests pass; 2 of 3 CLI tests pass; `test_exp_list_names_the_registered_experiments` FAILS.

Mark that one test with an explicit reason so the failure is intentional rather than ambient, and remove the marker in Task 8:

```python
@pytest.mark.xfail(
    reason="neorx-7disease is registered in Task 8; remove this marker then",
    strict=True,
)
def test_exp_list_names_the_registered_experiments():
```

`strict=True` means the marker itself fails once Task 8 lands, forcing its removal.

- [ ] **Step 7: Confirm the whole suite is green**

```bash
.venv/bin/python -m pytest tests/experiments/ tests/test_cli.py -q
```

Expected: all pass (the xfail counts as expected-failure, not failure).

- [ ] **Step 8: Commit**

```bash
git add src/neorx/experiments/ src/neorx/cli/__init__.py tests/experiments/
git commit -m "feat: add the experiment registry and neorx exp CLI

An experiment is a function taking a RunRecord; it never opens a file and
never chooses where results go. A run that raises still finalises its
record as failed, so rows completed before the failure survive."
```

---

### Task 3: HTTP capture and replay

**Files:**
- Create: `src/neorx/experiments/capture.py`
- Modify: `pyproject.toml` (add `vcrpy`), `src/neorx/core/sources/__init__.py` (docstring)
- Test: `tests/experiments/test_capture.py`

**Interfaces:**
- Consumes: `RunRecord` (Task 1)
- Produces:
  - `capture(record: RunRecord, *, mode: str = "record") -> ContextManager[None]` — `mode` is `"record"` or `"replay"`
  - `cassette_path(record: RunRecord) -> Path` → `record.path / "inputs" / "http.yaml"`
  - `interaction_count(record: RunRecord) -> int`
  - `class NoInteractionsRecordedError(RuntimeError)`

- [ ] **Step 1: Install vcrpy**

```bash
uv pip install --python .venv/bin/python vcrpy
.venv/bin/python -c "import vcr; print('vcrpy', vcr.VERSION)"
```

Then add it to `[dependency-groups].dev` in `pyproject.toml` (NOT to `[project].dependencies` — it is a development and experiment tool, not a runtime requirement of the library), and lock it:

```bash
uv lock && uv sync --group dev
grep -n "vcrpy" pyproject.toml uv.lock | head -3
```

- [ ] **Step 2: Write the failing test**

```python
# tests/experiments/test_capture.py
"""Record once, replay exactly. This is the property the subsystem claims."""

import json

import pytest
import requests

from neorx.experiments.capture import (
    NoInteractionsRecordedError,
    capture,
    cassette_path,
    interaction_count,
)
from neorx.experiments.record import RunRecord

URL = "https://httpbin.org/json"


@pytest.mark.network
def test_record_then_replay_reproduces_the_same_bytes(tmp_path):
    rec = RunRecord.create("cap-demo", runs_dir=tmp_path)
    with capture(rec, mode="record"):
        live = requests.get(URL, timeout=30).text
    assert cassette_path(rec).exists()
    assert interaction_count(rec) == 1

    with capture(rec, mode="replay"):
        replayed = requests.get(URL, timeout=30).text
    assert replayed == live


@pytest.mark.network
def test_replay_does_not_reach_the_network(tmp_path):
    """Replay must serve the cassette, not re-fetch."""
    rec = RunRecord.create("cap-offline", runs_dir=tmp_path)
    with capture(rec, mode="record"):
        requests.get(URL, timeout=30)

    with capture(rec, mode="replay"):
        # A URL absent from the cassette must fail rather than hit the network.
        with pytest.raises(Exception):
            requests.get("https://httpbin.org/uuid", timeout=30)


def test_a_recording_that_captured_nothing_is_an_error(tmp_path):
    """Silence here means the sources bypassed the recorder."""
    rec = RunRecord.create("cap-empty", runs_dir=tmp_path)
    with pytest.raises(NoInteractionsRecordedError, match="no HTTP interactions"):
        with capture(rec, mode="record"):
            pass  # make no requests at all
```

- [ ] **Step 3: Register the `network` marker**

Add to `[tool.pytest.ini_options]` in `pyproject.toml`, so the two network tests can be deselected in CI:

```toml
markers = [
    "network: test performs real network I/O; deselect with -m 'not network'",
]
```

- [ ] **Step 4: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/experiments/test_capture.py -q -m "not network"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.experiments.capture'`

- [ ] **Step 5: Write the implementation**

```python
# src/neorx/experiments/capture.py
"""Freeze an experiment's HTTP inputs so a run can be replayed exactly.

The seven HTTP data sources call module-level ``requests.get``/``post`` at
13 sites and construct a Session per call, so a transport adapter mounted
on a Session would intercept nothing. vcrpy patches at the connection
level and therefore captures all of them without editing a single source
module.

This module is the one sanctioned use of monkeypatching in the project:
it is imported only by the experiment runner, is active only inside the
``capture`` context, and never touches a library code path.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import vcr

from neorx.experiments.record import RunRecord


class NoInteractionsRecordedError(RuntimeError):
    """A recording session captured no HTTP traffic at all.

    This almost always means the code under test bypassed the recorder --
    for example a source that switched to httpx or aiohttp -- rather than
    that the experiment genuinely made no calls.
    """


def cassette_path(record: RunRecord) -> Path:
    return record.path / "inputs" / "http.yaml"


def interaction_count(record: RunRecord) -> int:
    path = cassette_path(record)
    if not path.exists():
        return 0
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    return len(data.get("interactions", []))


@contextmanager
def capture(record: RunRecord, *, mode: str = "record") -> Iterator[None]:
    """Record or replay the HTTP traffic of the enclosed block.

    Parameters
    ----------
    mode : {"record", "replay"}
        ``record`` performs live calls and writes them to the cassette.
        ``replay`` serves them back and refuses any request the cassette
        does not contain.
    """
    if mode not in ("record", "replay"):
        raise ValueError(f"mode must be 'record' or 'replay', got {mode!r}")

    path = cassette_path(record)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "replay" and not path.exists():
        raise FileNotFoundError(
            f"cannot replay {record.run_id}: no cassette at {path}. "
            f"Its inputs may have been pruned."
        )

    config = vcr.VCR(
        record_mode="all" if mode == "record" else "none",
        match_on=["method", "scheme", "host", "port", "path", "query"],
        decode_compressed_response=True,
    )

    with config.use_cassette(str(path)):
        yield

    if mode == "record" and interaction_count(record) == 0:
        raise NoInteractionsRecordedError(
            f"recording {record.run_id} captured no HTTP interactions. "
            f"The code under test did not route through the recorder -- check "
            f"whether a data source now uses a client other than requests."
        )
```

- [ ] **Step 6: Document the recording at the source**

Interception is invisible from the modules being recorded, so say so where a reader of those modules will see it. Add to the top of `src/neorx/core/sources/__init__.py`:

```python
"""Biomedical data sources.

All HTTP traffic in this package is subject to recording. When these
modules run inside ``neorx.experiments.capture``, every request and
response is frozen into the run's cassette so the run can be replayed
exactly. Adding a source that uses a client other than ``requests`` will
silently escape that recording -- see ``neorx/experiments/capture.py``.
"""
```

- [ ] **Step 7: Run the offline tests**

```bash
.venv/bin/python -m pytest tests/experiments/test_capture.py -q -m "not network"
```

Expected: 1 passed, 2 deselected

- [ ] **Step 8: Run the network round-trip once, by hand**

This is the claim the whole subsystem rests on, so verify it at least once locally:

```bash
.venv/bin/python -m pytest tests/experiments/test_capture.py -q -m network
```

Expected: 2 passed. If `httpbin.org` is unreachable, substitute any stable JSON endpoint and note the substitution in your report.

- [ ] **Step 9: Commit**

```bash
git add src/neorx/experiments/capture.py src/neorx/core/sources/__init__.py \
        tests/experiments/test_capture.py pyproject.toml uv.lock
git commit -m "feat: record and replay experiment HTTP inputs

vcrpy patches at the connection level, so all 13 module-level requests
call sites are captured without editing a source module. A recording that
captures nothing raises rather than producing an empty cassette -- silence
there means a source escaped the recorder."
```

---

### Task 4: ChEMBL provenance

**Files:**
- Create: `src/neorx/experiments/chembl.py`
- Test: `tests/experiments/test_chembl_provenance.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `chembl_provenance(db_path: Path) -> dict` → `{"release", "release_date", "size_bytes", "mtime_utc", "path"}`
  - `verify_chembl(recorded: dict, db_path: Path) -> None` — raises `ChEMBLMismatchError`
  - `class ChEMBLMismatchError(RuntimeError)`
  - `class ChEMBLProvenanceError(RuntimeError)`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_chembl_provenance.py
"""ChEMBL is a 28 GB immutable release -- record its version, not its bytes."""

import sqlite3

import pytest

from neorx.experiments.chembl import (
    ChEMBLMismatchError,
    ChEMBLProvenanceError,
    chembl_provenance,
    verify_chembl,
)


def _make_db(path, release="CHEMBL_36", date="2025-07-28 00:00:00.000000"):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE chembl_release "
        "(chembl_release_id INTEGER, chembl_release TEXT, creation_date TEXT)"
    )
    conn.execute(
        "INSERT INTO chembl_release VALUES (35, 'CHEMBL_35', '2024-12-01 00:00:00.000000')"
    )
    conn.execute("INSERT INTO chembl_release VALUES (36, ?, ?)", (release, date))
    conn.commit()
    conn.close()
    return path


def test_reads_the_latest_release_not_the_first(tmp_path):
    db = _make_db(tmp_path / "chembl.db")
    prov = chembl_provenance(db)
    assert prov["release"] == "CHEMBL_36"
    assert prov["release_date"].startswith("2025-07-28")
    assert prov["size_bytes"] > 0


def test_matching_release_verifies(tmp_path):
    db = _make_db(tmp_path / "chembl.db")
    verify_chembl(chembl_provenance(db), db)  # must not raise


def test_a_different_release_is_refused_by_name(tmp_path):
    db = _make_db(tmp_path / "chembl.db")
    recorded = chembl_provenance(db)
    other = _make_db(tmp_path / "other.db", release="CHEMBL_35")
    with pytest.raises(ChEMBLMismatchError, match="CHEMBL_36.*CHEMBL_35|CHEMBL_35"):
        verify_chembl(recorded, other)


def test_a_database_without_the_release_table_fails_loudly(tmp_path):
    """Never silently record 'no version'."""
    path = tmp_path / "bare.db"
    sqlite3.connect(path).execute("CREATE TABLE t (x INTEGER)").connection.commit()
    with pytest.raises(ChEMBLProvenanceError, match="chembl_release"):
        chembl_provenance(path)


def test_the_ontology_version_table_is_not_used(tmp_path):
    """`version` holds ontology versions, not the ChEMBL release."""
    db = _make_db(tmp_path / "chembl.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE version (name TEXT, creation_date TEXT, comments TEXT)")
    conn.execute("INSERT INTO version VALUES ('Bioassay Ontology 2.0', NULL, 'BAO')")
    conn.commit()
    conn.close()
    assert chembl_provenance(db)["release"] == "CHEMBL_36"
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/experiments/test_chembl_provenance.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.experiments.chembl'`

- [ ] **Step 3: Write the implementation**

```python
# src/neorx/experiments/chembl.py
"""Provenance for the local ChEMBL SQLite database.

ChEMBL is ~28 GB and its releases are versioned and immutable, so hashing
the file costs minutes and proves nothing the release string does not.
We record the release, the file size and its mtime.

The release comes from the ``chembl_release`` table. The similarly named
``version`` table is NOT the right source -- it holds ontology versions
such as "Bioassay Ontology 2.0".
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RELEASE_QUERY = (
    "SELECT chembl_release, creation_date "
    "FROM chembl_release ORDER BY chembl_release_id DESC LIMIT 1"
)


class ChEMBLProvenanceError(RuntimeError):
    """The database could not be interrogated for its release."""


class ChEMBLMismatchError(RuntimeError):
    """The database present is a different release from the one recorded."""


def chembl_provenance(db_path: Path) -> dict[str, Any]:
    """Describe the ChEMBL database backing a run."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise ChEMBLProvenanceError(f"no ChEMBL database at {db_path}")

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = conn.execute(_RELEASE_QUERY).fetchone()
    except sqlite3.Error as exc:
        raise ChEMBLProvenanceError(
            f"could not read the chembl_release table from {db_path}: {exc}"
        ) from exc
    finally:
        try:
            conn.close()
        except NameError:
            pass

    if not row:
        raise ChEMBLProvenanceError(
            f"chembl_release table in {db_path} is empty; cannot determine release"
        )

    stat = db_path.stat()
    return {
        "path": str(db_path),
        "release": row[0],
        "release_date": row[1],
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def verify_chembl(recorded: dict[str, Any], db_path: Path) -> None:
    """Raise unless the database at ``db_path`` is the recorded release."""
    current = chembl_provenance(db_path)
    if current["release"] != recorded["release"]:
        raise ChEMBLMismatchError(
            f"run recorded ChEMBL release {recorded['release']!r} but "
            f"{db_path} is {current['release']!r}. Replay would not reproduce "
            f"the recorded rows."
        )
```

- [ ] **Step 4: Run the tests**

```bash
.venv/bin/python -m pytest tests/experiments/test_chembl_provenance.py -q
```

Expected: 5 passed

- [ ] **Step 5: Verify against the real database**

```bash
.venv/bin/python -c "
from pathlib import Path
from neorx.experiments.chembl import chembl_provenance
p = Path('chembl_36.db')
if p.exists():
    prov = chembl_provenance(p)
    print(prov['release'], prov['release_date'], f\"{prov['size_bytes']/1e9:.1f} GB\")
    assert prov['release'] == 'CHEMBL_36', prov['release']
    print('real database: OK')
else:
    print('SKIP: chembl_36.db not present in this checkout')"
```

Expected: `CHEMBL_36 2025-07-28 00:00:00.000000 28.x GB` then `real database: OK`.

- [ ] **Step 6: Commit**

```bash
git add src/neorx/experiments/chembl.py tests/experiments/test_chembl_provenance.py
git commit -m "feat: record ChEMBL release provenance instead of hashing 28 GB

Releases are versioned and immutable, so the release string plus size and
mtime is sufficient. Reads chembl_release, not the similarly named version
table, which holds ontology versions."
```

---

### Task 5: Enforcement gates

**Files:**
- Create: `src/neorx/experiments/gates.py`, `docs/run-manifest.toml`
- Test: `tests/experiments/test_gates.py`

**Interfaces:**
- Consumes: `RunRecord.load` (Task 1)
- Produces:
  - `Finding` dataclass with `.path: str`, `.line: int`, `.message: str`
  - `find_hardcoded_metrics(path: Path, allowlist: set[float] | None = None) -> list[Finding]`
  - `check_records_wellformed(runs_dir: Path) -> list[Finding]`
  - `check_cited_runs(manifest: Path, runs_dir: Path) -> list[Finding]`
  - `METRIC_DECIMALS: int = 3`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_gates.py
"""The gates that make the guarantee enforceable rather than aspirational."""

import json

from neorx.experiments.gates import (
    check_cited_runs,
    check_records_wellformed,
    find_hardcoded_metrics,
)


def test_a_metric_literal_in_figure_code_is_found(tmp_path):
    src = tmp_path / "figures.py"
    src.write_text(
        "def fig3():\n"
        "    neorx_f1 = [0.545, 0.333, 0.556]\n"
        "    return neorx_f1\n"
    )
    findings = find_hardcoded_metrics(src)
    assert findings, "0.545 is a metric-shaped literal and must be flagged"
    assert findings[0].line == 2
    assert "0.545" in findings[0].message


def test_coarse_constants_are_not_flagged(tmp_path):
    """Axis limits and thresholds are not metrics."""
    src = tmp_path / "figures.py"
    src.write_text("def fig():\n    ax.set_ylim(0, 1.0)\n    alpha = 0.5\n")
    assert find_hardcoded_metrics(src) == []


def test_allowlisted_values_are_permitted(tmp_path):
    src = tmp_path / "figures.py"
    src.write_text("GOLDEN = 1.618\n")
    assert find_hardcoded_metrics(src) == []          # 3 dp, but allowlisted below
    assert find_hardcoded_metrics(src, allowlist=set()) != []


def test_a_record_missing_its_summary_is_malformed(tmp_path):
    (tmp_path / "2026-09-03-x-aaaaaa").mkdir()
    findings = check_records_wellformed(tmp_path)
    assert findings and "record.json" in findings[0].message


def test_row_count_must_match_the_summary(tmp_path):
    run = tmp_path / "2026-09-03-x-aaaaaa"
    run.mkdir()
    (run / "record.json").write_text(
        json.dumps({"run_id": "x", "status": "complete", "citable": True, "n_rows": 5})
    )
    (run / "rows.jsonl").write_text('{"a":1}\n')
    findings = check_records_wellformed(tmp_path)
    assert findings and "5" in findings[0].message and "1" in findings[0].message


def test_a_cited_run_that_does_not_exist_is_found(tmp_path):
    manifest = tmp_path / "run-manifest.toml"
    manifest.write_text(
        '[tables]\n"paper1.table2" = "2026-09-03-neorx-7disease-deadbe"\n'
    )
    findings = check_cited_runs(manifest, tmp_path / "runs")
    assert findings and "deadbe" in findings[0].message


def test_a_cited_run_that_is_not_citable_is_found(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "2026-09-03-x-aaaaaa"
    run.mkdir(parents=True)
    (run / "record.json").write_text(
        json.dumps(
            {"run_id": "2026-09-03-x-aaaaaa", "status": "complete",
             "citable": False, "n_rows": 0}
        )
    )
    (run / "rows.jsonl").write_text("")
    manifest = tmp_path / "run-manifest.toml"
    manifest.write_text('[tables]\n"paper1.table2" = "2026-09-03-x-aaaaaa"\n')
    findings = check_cited_runs(manifest, runs)
    assert findings and "citable" in findings[0].message.lower()
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/experiments/test_gates.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.experiments.gates'`

- [ ] **Step 3: Write the implementation**

```python
# src/neorx/experiments/gates.py
"""Checks that make "no number without a record" enforceable.

Each gate closes one hole the manuscript audit found:

* ``find_hardcoded_metrics``  -- _gen_figures.py held the F1 values as
  literals, so Figure 3 was a transcription of a table rather than a
  rendering of data.
* ``check_records_wellformed`` -- a record whose summary disagrees with
  its rows is not evidence.
* ``check_cited_runs``        -- a manuscript citing a run that does not
  exist, or one nobody can reproduce.
"""

from __future__ import annotations

import ast
import json
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

METRIC_DECIMALS = 3

#: Values with >=3 decimals that are legitimately constants, not metrics.
#: Every entry needs a comment saying why -- never widen the pattern instead.
DEFAULT_ALLOWLIST: set[float] = {
    1.618,   # golden ratio, used for figure aspect ratios
    0.001,   # p-value floor in the identifier's proxy
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str


def _decimals(text: str) -> int:
    exponent = Decimal(text).as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def find_hardcoded_metrics(
    path: Path, allowlist: set[float] | None = None
) -> list[Finding]:
    """Flag metric-shaped float literals (>= METRIC_DECIMALS decimals)."""
    allowed = DEFAULT_ALLOWLIST if allowlist is None else allowlist
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, float):
            continue
        literal = ast.get_source_segment(Path(path).read_text(), node) or repr(node.value)
        try:
            if _decimals(literal) < METRIC_DECIMALS:
                continue
        except Exception:
            continue
        if node.value in allowed:
            continue
        findings.append(
            Finding(
                path=str(path),
                line=node.lineno,
                message=(
                    f"metric-shaped literal {literal} -- render it from a run "
                    f"record instead, or add it to DEFAULT_ALLOWLIST with a reason"
                ),
            )
        )
    return findings


def check_records_wellformed(runs_dir: Path) -> list[Finding]:
    """Every run directory must have a summary that matches its rows."""
    findings: list[Finding] = []
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return findings

    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        summary = run / "record.json"
        if not summary.exists():
            findings.append(
                Finding(str(run), 0, f"{run.name}: no record.json -- run never finalised")
            )
            continue
        data = json.loads(summary.read_text())
        rows_file = run / "rows.jsonl"
        actual = len(
            [ln for ln in rows_file.read_text().splitlines() if ln.strip()]
        ) if rows_file.exists() else 0
        if data.get("n_rows") != actual:
            findings.append(
                Finding(
                    str(run),
                    0,
                    f"{run.name}: record.json claims {data.get('n_rows')} rows "
                    f"but rows.jsonl has {actual}",
                )
            )
    return findings


def check_cited_runs(manifest: Path, runs_dir: Path) -> list[Finding]:
    """Every run cited by a manuscript must exist and be citable."""
    manifest = Path(manifest)
    if not manifest.exists():
        return [Finding(str(manifest), 0, f"no run manifest at {manifest}")]

    data = tomllib.loads(manifest.read_text())
    findings: list[Finding] = []

    for section, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for citation, run_id in entries.items():
            summary = Path(runs_dir) / run_id / "record.json"
            if not summary.exists():
                findings.append(
                    Finding(
                        str(manifest),
                        0,
                        f"{section}.{citation} cites run {run_id!r}, which is not "
                        f"in {runs_dir}",
                    )
                )
                continue
            if not json.loads(summary.read_text()).get("citable"):
                findings.append(
                    Finding(
                        str(manifest),
                        0,
                        f"{section}.{citation} cites run {run_id!r}, which is not "
                        f"citable (incomplete, failed, or built from a dirty tree)",
                    )
                )
    return findings
```

- [ ] **Step 4: Create the manifest**

```bash
cat > docs/run-manifest.toml <<'EOF'
# Maps each manuscript claim to the run record backing it.
# `test_cited_runs_exist` fails if a cited run is missing or not citable,
# so a number cannot appear in a paper without an artifact behind it.
#
# Entries are added as each experiment is migrated (Tasks 6-9). Empty
# sections are valid: they assert nothing rather than asserting falsely.

[tables]

[figures]
EOF
```

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/python -m pytest tests/experiments/test_gates.py -q
```

Expected: 7 passed

- [ ] **Step 6: Point the gate at the real figure script and observe it fire**

The gate exists because of a real defect; confirm it detects that defect before the script is migrated in Task 9:

```bash
.venv/bin/python -c "
from pathlib import Path
from neorx.experiments.gates import find_hardcoded_metrics
p = Path('_gen_figures.py')
if p.exists():
    for f in find_hardcoded_metrics(p)[:5]:
        print(f'{f.path}:{f.line}  {f.message}')
else:
    print('SKIP: _gen_figures.py not in this checkout (migrated in Task 9)')"
```

Expected: findings at the `neorx_f1` / `corr_f1` lines. Record the count in your report — Task 9 must bring it to zero.

- [ ] **Step 7: Commit**

```bash
git add src/neorx/experiments/gates.py docs/run-manifest.toml \
        tests/experiments/test_gates.py
git commit -m "feat: add the experiment enforcement gates

Flags metric-shaped float literals in figure code, records whose summary
disagrees with their rows, and manuscript citations pointing at runs that
are missing or not citable."
```

---

### Task 6: Migrate `genmol-eval`

The cheapest migration, chosen first because it is offline and already produces a real artifact — so it validates the runner before the runner meets a hard case.

**Files:**
- Create: `experiments/__init__.py`, `experiments/genmol_eval.py`
- Test: `tests/experiments/test_genmol_eval.py`
- Delete (last step): `_eval_genmol_final.py`

**Interfaces:**
- Consumes: `@experiment` (Task 2), `RunRecord.append_row` (Task 1)
- Produces: an experiment registered as `genmol-eval`

- [ ] **Step 1: Read what the current script measures**

```bash
grep -nE "validity|uniqueness|novelty|diversity|reconstruction|per_molecule_ms|n_params|MW_mean" _eval_genmol_final.py | head -20
```

The existing artifact `results/genmol_paper4_final.json` shows the shape the paper reports: `n_params`, `vocab_size`, `validity`, `uniqueness`, `novelty`, `diversity`, `reconstruction_accuracy`, `per_molecule_ms`, and the MW/logP/QED means and standard deviations.

**Note two values in that file are hardcoded literals, not measurements**: `train_time_s: 2911.2` and `n_smiles_raw: 27572` at `_eval_genmol_final.py:170-173`. Do not carry them forward as if measured. Either measure them or omit them; if omitted, say so in your report.

- [ ] **Step 2: Write the failing test**

```python
# tests/experiments/test_genmol_eval.py
"""genmol-eval is offline and deterministic enough to assert on shape."""

import pytest

from neorx.experiments.registry import get_experiment, run_experiment


def test_the_experiment_is_registered():
    import experiments  # noqa: F401

    assert get_experiment("genmol-eval").name == "genmol-eval"


@pytest.mark.slow
def test_run_records_the_metrics_the_paper_reports(tmp_path):
    import experiments  # noqa: F401

    rec = run_experiment("genmol-eval", runs_dir=tmp_path)
    assert rec.status == "complete"
    rows = rec.rows()
    assert len(rows) == 1, "genmol-eval reports one summary row"

    row = rows[0]
    for key in (
        "n_params", "vocab_size", "n_generated", "validity",
        "uniqueness", "novelty", "diversity", "per_molecule_ms",
        "mw_mean", "mw_std", "logp_mean", "qed_mean",
    ):
        assert key in row, f"missing {key}"

    assert 0.0 <= row["validity"] <= 1.0
    assert row["vocab_size"] == 34
    assert row["n_params"] == 4_140_322
```

- [ ] **Step 3: Register the `slow` marker**

Add to `markers` in `[tool.pytest.ini_options]`, beside the `network` marker from Task 3:

```toml
    "slow: test runs a real experiment; deselect with -m 'not slow'",
```

- [ ] **Step 4: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/experiments/test_genmol_eval.py -q -m "not slow"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'experiments'`

- [ ] **Step 5: Write the experiment**

```python
# experiments/__init__.py
"""Experiment definitions for this project.

Importing this package registers every experiment. It is deliberately not
part of the installed wheel: these are this repository's experiments, not
a library feature.
"""

from experiments import causalbiorl_bench, figures, genmol_eval, neorx_7disease  # noqa: F401
```

Note: that import line references all four modules. Create the other three as empty placeholders now so the import succeeds, and fill them in Tasks 7–9:

```bash
mkdir -p experiments
for m in causalbiorl_bench figures neorx_7disease; do
  printf '"""Placeholder; implemented in a later task."""\n' > "experiments/$m.py"
done
```

```python
# experiments/genmol_eval.py
"""GenMol generation quality, as reported in the GenMol manuscript section 4.3.

Offline: loads the shipped weights and generates locally. No HTTP, so no
cassette is recorded.
"""

from __future__ import annotations

import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

N_GENERATE = 1000
TEMPERATURE = 0.8


@experiment(name="genmol-eval", help="GenMol generation quality metrics.")
def genmol_eval(record: RunRecord) -> None:
    from neorx.genmol import generate, load_pretrained

    model, tokenizer = load_pretrained()
    n_params = sum(p.numel() for p in model.parameters())

    started = time.perf_counter()
    smiles = generate(model, tokenizer, n=N_GENERATE, temperature=TEMPERATURE)
    elapsed = time.perf_counter() - started

    mols = [(s, Chem.MolFromSmiles(s)) for s in smiles]
    valid = [(s, m) for s, m in mols if m is not None]
    unique = {s for s, _ in valid}

    mw = [Descriptors.MolWt(m) for _, m in valid]
    logp = [Crippen.MolLogP(m) for _, m in valid]
    qed = [QED.qed(m) for _, m in valid]

    record.append_row(
        {
            "n_params": n_params,
            "vocab_size": tokenizer.vocab_size,
            "temperature": TEMPERATURE,
            "n_requested": N_GENERATE,
            "n_generated": len(smiles),
            "n_valid": len(valid),
            "validity": len(valid) / len(smiles) if smiles else 0.0,
            "uniqueness": len(unique) / len(valid) if valid else 0.0,
            "novelty": _novelty(unique),
            "diversity": _diversity([m for _, m in valid]),
            "per_molecule_ms": 1000 * elapsed / len(smiles) if smiles else 0.0,
            "mw_mean": float(np.mean(mw)) if mw else 0.0,
            "mw_std": float(np.std(mw)) if mw else 0.0,
            "logp_mean": float(np.mean(logp)) if logp else 0.0,
            "logp_std": float(np.std(logp)) if logp else 0.0,
            "qed_mean": float(np.mean(qed)) if qed else 0.0,
            "qed_std": float(np.std(qed)) if qed else 0.0,
        }
    )


def _novelty(generated: set[str]) -> float:
    """Fraction of generated molecules absent from the training corpus."""
    from neorx.genmol.data.download import load_smiles

    training = set(load_smiles())
    if not generated:
        return 0.0
    return len(generated - training) / len(generated)


def _diversity(mols: list) -> float:
    """1 - mean pairwise Tanimoto over Morgan fingerprints, full pairwise.

    The original script sampled a 50-neighbour sliding window over the first
    200 molecules; this computes the real statistic over a capped sample so
    the number means what its name says.
    """
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    sample = mols[:500]
    if len(sample) < 2:
        return 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in sample]
    sims = [
        s
        for i, fp in enumerate(fps[:-1])
        for s in DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])
    ]
    return 1.0 - (sum(sims) / len(sims)) if sims else 0.0
```

- [ ] **Step 6: Run the fast test**

```bash
.venv/bin/python -m pytest tests/experiments/test_genmol_eval.py -q -m "not slow"
```

Expected: 1 passed, 1 deselected

- [ ] **Step 7: Run the experiment for real and inspect the record**

```bash
.venv/bin/python -m neorx.cli exp run genmol-eval
.venv/bin/python -m neorx.cli exp show $(ls -t runs | head -1)
```

Expected: a `runs/<id>/` with `record.json` (`status=complete`), one row in `rows.jsonl`, and `env.json`. Compare the reported `validity` and `n_params` against `results/genmol_paper4_final.json` (0.97 and 4,140,322) and note any divergence in your report — a difference is informative, not automatically wrong.

- [ ] **Step 8: Delete the superseded script**

```bash
rm -f _eval_genmol_final.py _eval_genmol.py
```

These are untracked, so this is not a git operation. Copies remain in `.backup-scripts-preMerge/` in the main checkout.

- [ ] **Step 9: Commit**

```bash
git add experiments/ tests/experiments/test_genmol_eval.py pyproject.toml
git commit -m "feat: migrate genmol-eval onto the experiment runner

Diversity is now the real full-pairwise statistic over a 500-molecule
sample rather than a 50-neighbour sliding window over the first 200, so
the number means what its name says. train_time_s and n_smiles_raw are
dropped: they were hardcoded literals in the old script, not measurements."
```

---

### Task 7: Migrate `causalbiorl-bench`

Subsumes both `_bench_paper2.py` and `_bench_fast_baselines.py`, and fixes the incremental-write failure that lost the original 881-second run.

**Files:**
- Modify: `experiments/causalbiorl_bench.py`
- Test: `tests/experiments/test_causalbiorl_bench.py`
- Delete (last step): `_bench_paper2.py`, `_bench_fast_baselines.py`, `_fast_benchmark.py`

**Interfaces:**
- Consumes: `@experiment`, `RunRecord.append_row`
- Produces: an experiment registered as `causalbiorl-bench`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_causalbiorl_bench.py
"""One row per env x agent cell, written as each completes."""

import pytest

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import get_experiment


def test_the_experiment_is_registered():
    import experiments  # noqa: F401

    assert get_experiment("causalbiorl-bench").name == "causalbiorl-bench"


def test_each_cell_is_written_before_the_next_begins(tmp_path, monkeypatch):
    """The failure this migration exists to fix: a timeout must not lose
    the cells that already finished."""
    import experiments.causalbiorl_bench as mod

    seen: list[int] = []

    def fake_run_single(env_id, agent_type, *, seed, n_episodes, difficulty, verbose):
        if len(seen) == 2:
            raise TimeoutError("simulated timeout on the third cell")
        seen.append(1)

        class _R:
            episode_rewards = [-100.0, -200.0]

        return _R()

    monkeypatch.setattr(mod, "run_single", fake_run_single)
    monkeypatch.setattr(mod, "ENVS", ["GeneticToggle-v0"])
    monkeypatch.setattr(mod, "AGENTS", ["random", "ppo", "sac"])
    monkeypatch.setattr(mod, "N_SEEDS", 1)

    rec = RunRecord.create("causalbiorl-bench", runs_dir=tmp_path)
    with pytest.raises(TimeoutError):
        mod.causalbiorl_bench(rec)
    rec.finalise("failed")

    rows = rec.rows()
    assert len(rows) == 2, "cells completed before the timeout must be on disk"
    assert {r["agent"] for r in rows} == {"random", "ppo"}
    assert rec.citable is False


def test_row_carries_the_episode_count_it_was_measured_over(tmp_path, monkeypatch):
    """The original benchmark compared a 30-episode mean against 50-episode
    means without recording either. The count is part of the result."""
    import experiments.causalbiorl_bench as mod

    def fake_run_single(env_id, agent_type, *, seed, n_episodes, difficulty, verbose):
        class _R:
            episode_rewards = [-1.0] * n_episodes

        return _R()

    monkeypatch.setattr(mod, "run_single", fake_run_single)
    monkeypatch.setattr(mod, "ENVS", ["GeneticToggle-v0"])
    monkeypatch.setattr(mod, "AGENTS", ["random"])
    monkeypatch.setattr(mod, "N_SEEDS", 1)

    rec = RunRecord.create("causalbiorl-bench", runs_dir=tmp_path)
    mod.causalbiorl_bench(rec)
    row = rec.rows()[0]
    assert row["n_episodes"] == mod.EPISODES["random"]
    assert row["n_seeds"] == 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/experiments/test_causalbiorl_bench.py -q
```

Expected: FAIL — `AttributeError: module 'experiments.causalbiorl_bench' has no attribute 'causalbiorl_bench'`

- [ ] **Step 3: Write the experiment**

```python
# experiments/causalbiorl_bench.py
"""CausalBioRL agent comparison across the three control environments.

Replaces _bench_paper2.py and _bench_fast_baselines.py. The former wrote
its results only after all twelve env x agent cells finished, timed out on
the fourth, and lost 881 seconds of measurement that was then retyped by
hand into the latter as a literal dict. Here every cell is written the
moment it completes.

Offline: the environments are simulated locally, so no cassette is recorded.
"""

from __future__ import annotations

import time

import numpy as np

from neorx.causalbiorl.benchmark import run_single
from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

ENVS = ["GeneticToggle-v0", "MetabolicPathway-v0", "CellGrowth-v0"]
AGENTS = ["causal", "ppo", "sac", "random"]
N_SEEDS = 2
DIFFICULTY = "medium"

#: Episode budget per agent. Recorded on every row: the original benchmark
#: compared a 30-episode causal mean against 50-episode baseline means
#: without disclosing it, which is not a like-for-like comparison.
EPISODES = {"causal": 30, "ppo": 50, "sac": 50, "random": 50}


@experiment(name="causalbiorl-bench", help="RL agent comparison across control envs.")
def causalbiorl_bench(record: RunRecord) -> None:
    for env_id in ENVS:
        for agent in AGENTS:
            n_episodes = EPISODES[agent]
            started = time.perf_counter()
            per_seed: list[list[float]] = []

            for seed in range(N_SEEDS):
                result = run_single(
                    env_id,
                    agent,
                    seed=seed,
                    n_episodes=n_episodes,
                    difficulty=DIFFICULTY,
                    verbose=False,
                )
                per_seed.append(list(result.episode_rewards))

            matrix = np.array(per_seed)
            record.append_row(
                {
                    "env": env_id,
                    "agent": agent,
                    "n_episodes": n_episodes,
                    "n_seeds": N_SEEDS,
                    "difficulty": DIFFICULTY,
                    "mean_episode_reward": float(np.mean(matrix)),
                    "std_across_seeds": float(np.std(matrix.mean(axis=1))),
                    "wall_clock_s": round(time.perf_counter() - started, 1),
                }
            )
```

- [ ] **Step 4: Run the tests**

```bash
.venv/bin/python -m pytest tests/experiments/test_causalbiorl_bench.py -q
```

Expected: 3 passed

- [ ] **Step 5: Delete the superseded scripts**

```bash
rm -f _bench_paper2.py _bench_fast_baselines.py _fast_benchmark.py
```

- [ ] **Step 6: Commit**

```bash
git add experiments/causalbiorl_bench.py tests/experiments/test_causalbiorl_bench.py
git commit -m "feat: migrate causalbiorl-bench onto the experiment runner

Every env x agent cell is written the moment it completes, so a timeout
costs one cell rather than the run. Each row records the episode count it
was measured over: the original compared a 30-episode causal mean against
50-episode baseline means without disclosing it."
```

---

### Task 8: Migrate `neorx-7disease`

The hard one: eight live APIs, needs capture, ~25 minutes per run. This is the experiment behind Tables 2–5, none of which has an artifact today.

**Files:**
- Modify: `experiments/neorx_7disease.py`, `tests/experiments/test_exp_cli.py` (remove the xfail)
- Test: `tests/experiments/test_neorx_7disease.py`
- Delete (last step): `benchmark_paper.py`, `_quick_bench.py`

**Interfaces:**
- Consumes: `@experiment`, `RunRecord.append_row`, `capture` (Task 3), `chembl_provenance` (Task 4)
- Produces: an experiment registered as `neorx-7disease`

- [ ] **Step 1: Read the row shape the current script produces**

```bash
sed -n '13,25p;232,246p' benchmark_paper.py
```

`all_results[disease]` carries: `found`, `causal`, `tp`, `fp`, `missed`, `P`, `R`, `F1`, `grade`, `corr_P`, `corr_R`, `corr_F1`, `corr_fp`, `demoted`, `fp_ranks`, `t_graph`, `t_identify`. That is the row schema; keep every field.

**Two known defects in that script, which this task does NOT fix** (sub-project 3 owns them, and fixing them here would conflate a provenance change with a methodology change):
- `t_graph` reads `graph._build_time`, an attribute never set anywhere, so it is always `0.0`.
- `correlation_only()` filters to node types `gene` and `protein`, excluding `pathogen_gene`, so the baseline cannot see POL, DHFR-TS, PPPK-DHPS or GP.

Record both faithfully and note them in your report. Recording a defect accurately is the point of this sub-project.

- [ ] **Step 2: Write the failing test**

```python
# tests/experiments/test_neorx_7disease.py
"""The seven-disease benchmark, recorded per disease as each completes."""

import pytest

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import get_experiment


def test_the_experiment_is_registered():
    import experiments  # noqa: F401

    assert get_experiment("neorx-7disease").name == "neorx-7disease"


def test_all_seven_diseases_are_configured():
    import experiments.neorx_7disease as mod

    assert len(mod.DISEASES) == 7
    lowered = [d.lower() for d in mod.DISEASES]
    for expected in ("hiv", "malaria", "ebola"):
        assert any(expected in d for d in lowered), expected


def test_a_disease_row_is_written_before_the_next_starts(tmp_path, monkeypatch):
    import experiments.neorx_7disease as mod

    calls: list[str] = []

    def fake_evaluate(disease):
        calls.append(disease)
        if len(calls) == 3:
            raise RuntimeError("simulated failure on the third disease")
        return {"found": 20, "causal": 7, "tp": ["CCR5"], "fp": [], "missed": [],
                "P": 0.4, "R": 0.75, "F1": 0.5, "grade": "B",
                "corr_P": 0.1, "corr_R": 0.5, "corr_F1": 0.17, "corr_fp": [],
                "demoted": [], "fp_ranks": {}, "t_graph": 0.0, "t_identify": 1.0}

    monkeypatch.setattr(mod, "_evaluate_disease", fake_evaluate)
    monkeypatch.setattr(mod, "_capture_enabled", lambda: False)

    rec = RunRecord.create("neorx-7disease", runs_dir=tmp_path)
    with pytest.raises(RuntimeError):
        mod.neorx_7disease(rec)
    rec.finalise("failed")

    assert len(rec.rows()) == 2, "completed diseases must survive the failure"
    assert rec.rows()[0]["disease"] == mod.DISEASES[0]


def test_every_row_carries_the_full_schema(tmp_path, monkeypatch):
    import experiments.neorx_7disease as mod

    monkeypatch.setattr(
        mod, "_evaluate_disease",
        lambda d: {"found": 20, "causal": 7, "tp": [], "fp": [], "missed": [],
                   "P": 0.4, "R": 0.75, "F1": 0.5, "grade": "B",
                   "corr_P": 0.1, "corr_R": 0.5, "corr_F1": 0.17, "corr_fp": [],
                   "demoted": [], "fp_ranks": {}, "t_graph": 0.0, "t_identify": 1.0},
    )
    monkeypatch.setattr(mod, "_capture_enabled", lambda: False)

    rec = RunRecord.create("neorx-7disease", runs_dir=tmp_path)
    mod.neorx_7disease(rec)

    required = {"disease", "found", "causal", "tp", "fp", "missed", "P", "R", "F1",
                "grade", "corr_P", "corr_R", "corr_F1", "corr_fp", "demoted",
                "fp_ranks", "t_graph", "t_identify"}
    for row in rec.rows():
        assert required <= set(row), sorted(required - set(row))
    assert len(rec.rows()) == 7
```

- [ ] **Step 3: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/experiments/test_neorx_7disease.py -q
```

Expected: FAIL — `AttributeError: module 'experiments.neorx_7disease' has no attribute 'DISEASES'`

- [ ] **Step 4: Write the experiment**

Port the logic from `benchmark_paper.py` — `run_pipeline`, `validate`, `correlation_only` and the per-disease loop — into `_evaluate_disease`, changing only where results go. Do not alter the scoring, the classifier wiring, or the baseline's node-type filter.

```python
# experiments/neorx_7disease.py
"""The seven-disease causal target benchmark behind Tables 2-5.

Ported from benchmark_paper.py, which computed these numbers and never
persisted any of them -- no json.dump, no write_text. Every reported value
existed only in terminal scrollback.

The evaluation logic is carried across unchanged. Two known defects are
recorded faithfully rather than fixed here, because fixing them is
sub-project 3's job and conflating the two would make neither reviewable:

* ``t_graph`` reads ``graph._build_time``, an attribute nothing sets, so
  it is always 0.0.
* the correlation-only baseline filters to node types ``gene`` and
  ``protein``, so it cannot see pathogen targets that NeoRx can.

Online: eight external APIs, so the run is recorded to a cassette and can
be replayed exactly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from neorx.experiments.capture import capture
from neorx.experiments.chembl import chembl_provenance
from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

DISEASES = [
    "HIV",
    "malaria",
    "type 2 diabetes",
    "alzheimer disease",
    "lung cancer",
    "breast cancer",
    "ebola",
]

TOP_N = 20
CHEMBL_DB = Path("chembl_36.db")


def _capture_enabled() -> bool:
    """Recording is on unless explicitly disabled (tests disable it)."""
    return os.environ.get("NEORX_EXP_CAPTURE", "1") != "0"


@experiment(name="neorx-7disease", help="Causal target benchmark across 7 diseases.")
def neorx_7disease(record: RunRecord) -> None:
    if CHEMBL_DB.exists():
        prov = chembl_provenance(CHEMBL_DB)
        (record.path / "inputs" / "chembl.json").write_text(
            __import__("json").dumps(prov, indent=2)
        )

    if _capture_enabled():
        with capture(record, mode="record"):
            _run_all(record)
    else:
        _run_all(record)


def _run_all(record: RunRecord) -> None:
    for disease in DISEASES:
        started = time.perf_counter()
        result = _evaluate_disease(disease)
        record.append_row(
            {"disease": disease, "wall_clock_s": round(time.perf_counter() - started, 1),
             **result}
        )


def _evaluate_disease(disease: str) -> dict:
    """Evaluate one disease. Logic ported verbatim from benchmark_paper.py."""
    from neorx.core.graph.graph_builder import build_disease_graph
    from neorx.core.causal.identifier import identify_causal_targets
    from neorx.core.validator import KnownTargetValidator

    t0 = time.perf_counter()
    graph = build_disease_graph(disease, use_cache=False)
    t_graph = round(time.perf_counter() - t0, 1)

    t1 = time.perf_counter()
    targets = identify_causal_targets(graph, top_n=TOP_N, disease_name=disease)
    t_identify = round(time.perf_counter() - t1, 1)

    validator = KnownTargetValidator()
    report = validator.validate(disease, targets)
    corr = _correlation_only(graph, disease, validator)

    causal = [t for t in targets if t.classification.value == "CAUSAL"]
    return {
        "found": len(targets),
        "causal": len(causal),
        "tp": sorted(report.true_positives),
        "fp": sorted(report.false_positives_known or []),
        "missed": sorted(report.missed_targets),
        "P": report.precision,
        "R": report.recall,
        "F1": report.f1,
        "grade": report.grade,
        "corr_P": corr["P"],
        "corr_R": corr["R"],
        "corr_F1": corr["F1"],
        "corr_fp": corr["fp"],
        "demoted": sorted(
            t.gene_symbol for t in targets if t.classification.value != "CAUSAL"
        ),
        "fp_ranks": {},
        "t_graph": t_graph,
        "t_identify": t_identify,
    }


def _correlation_only(graph, disease: str, validator) -> dict:
    """Rank by raw association score, no causal analysis.

    NOTE: filters to node types ``gene`` and ``protein``, so pathogen
    targets are invisible to this baseline. That asymmetry is a known
    defect carried across unchanged; sub-project 3 owns it.
    """
    nodes = []
    for node in graph.nodes:
        if node.node_type.value in ("gene", "protein"):
            name = node.node_id.split(":", 1)[1] if ":" in node.node_id else node.node_id
            nodes.append((name, node.score or 0.0))
    nodes.sort(key=lambda x: x[1], reverse=True)
    top = nodes[:TOP_N]

    truth = validator.ground_truth(disease)
    known_tp = {t.upper() for t in truth["targets"]}
    known_fp = {t.upper() for t in truth["false_targets"]}
    found = {n.upper() for n, _ in top}

    tp = found & known_tp
    P = len(tp) / len(top) if top else 0.0
    R = len(tp) / len(known_tp) if known_tp else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"P": P, "R": R, "F1": F1, "fp": sorted(found & known_fp)}
```

If `KnownTargetValidator` exposes the ground truth under a different method name than `ground_truth`, read `src/neorx/core/validator.py` and use the real one — do not invent an accessor.

- [ ] **Step 5: Remove the xfail from Task 2**

`tests/experiments/test_exp_cli.py::test_exp_list_names_the_registered_experiments` carries a `strict=True` xfail marker that now fails because the test passes. Delete the marker and its import if unused.

- [ ] **Step 6: Run the tests**

```bash
.venv/bin/python -m pytest tests/experiments/test_neorx_7disease.py tests/experiments/test_exp_cli.py -q
```

Expected: 7 passed, 0 xfailed

- [ ] **Step 7: Do a single-disease smoke run against live APIs**

The full run is ~25 minutes. Verify the recording path works on one disease first by temporarily narrowing `DISEASES`:

```bash
.venv/bin/python -c "
import experiments.neorx_7disease as m
m.DISEASES = ['HIV']
from neorx.experiments.registry import run_experiment
rec = run_experiment('neorx-7disease')
print('run:', rec.run_id, 'rows:', len(rec.rows()), 'citable:', rec.citable)
from neorx.experiments.capture import interaction_count
print('http interactions captured:', interaction_count(rec))
print('record size: %.1f MB' % (rec.size_bytes() / 1e6))"
```

Expected: one row, a non-zero interaction count, and a size well under 50 MB. **If the interaction count is zero, stop and report** — that means the sources bypassed the recorder and the whole capture design needs revisiting.

- [ ] **Step 8: Delete the superseded scripts**

```bash
rm -f benchmark_paper.py _quick_bench.py
```

- [ ] **Step 9: Commit**

```bash
git add experiments/neorx_7disease.py tests/experiments/test_neorx_7disease.py \
        tests/experiments/test_exp_cli.py
git commit -m "feat: migrate neorx-7disease onto the experiment runner

benchmark_paper.py computed Tables 2-5 and persisted nothing; every value
existed only in terminal scrollback. Rows are now written per disease as
each completes, with the eight external APIs frozen to a cassette so the
run can be replayed exactly.

The evaluation logic is carried across unchanged, including two known
defects recorded rather than fixed: t_graph reads an attribute nothing
sets, and the correlation-only baseline cannot see pathogen targets.
Sub-project 3 owns both."
```

---

### Task 9: Migrate `figures`

**Files:**
- Modify: `experiments/figures.py`
- Create: `tests/experiments/test_figures.py`
- Delete (last step): `_gen_figures.py`

**Interfaces:**
- Consumes: `RunRecord.load` (Task 1), `find_hardcoded_metrics` (Task 5)
- Produces: an experiment registered as `figures`, taking `--from <run-id>`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_figures.py
"""Figures render from run records. A literal in this file is a test failure."""

import json
from pathlib import Path

import pytest

from neorx.experiments.gates import find_hardcoded_metrics


def test_the_figure_module_contains_no_metric_literals():
    """The defect this whole sub-project exists to prevent."""
    findings = find_hardcoded_metrics(Path("experiments/figures.py"))
    assert findings == [], [f"{f.path}:{f.line} {f.message}" for f in findings]


def test_fig3_reads_its_values_from_a_record(tmp_path):
    import experiments.figures as mod

    run = tmp_path / "2026-09-03-neorx-7disease-aaaaaa"
    (run / "inputs").mkdir(parents=True)
    (run / "rows.jsonl").write_text(
        "\n".join(
            json.dumps({"disease": d, "F1": f1, "corr_F1": c, "P": 0.4, "R": 0.7})
            for d, f1, c in [("HIV", 0.545, 0.167), ("malaria", 0.333, 0.074)]
        )
        + "\n"
    )
    (run / "record.json").write_text(
        json.dumps({"run_id": run.name, "status": "complete", "citable": True,
                    "n_rows": 2})
    )
    series = mod.fig3_series(run.name, runs_dir=tmp_path)
    assert series["diseases"] == ["HIV", "malaria"]
    assert series["neorx_f1"] == [0.545, 0.333]
    assert series["corr_f1"] == [0.167, 0.074]


def test_a_non_citable_run_is_refused(tmp_path):
    import experiments.figures as mod

    run = tmp_path / "2026-09-03-neorx-7disease-bbbbbb"
    (run / "inputs").mkdir(parents=True)
    (run / "rows.jsonl").write_text('{"disease":"HIV","F1":0.5,"corr_F1":0.1}\n')
    (run / "record.json").write_text(
        json.dumps({"run_id": run.name, "status": "complete", "citable": False,
                    "n_rows": 1})
    )
    with pytest.raises(ValueError, match="citable"):
        mod.fig3_series(run.name, runs_dir=tmp_path)
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/experiments/test_figures.py -q
```

Expected: FAIL — `AttributeError: module 'experiments.figures' has no attribute 'fig3_series'`

- [ ] **Step 3: Write the module**

Port the plotting code from `_gen_figures.py` unchanged — `fig1_pipeline` and `fig2_hiv_graph` are diagrams with no data dependency and move across as they are. Only `fig3_pr_comparison` and `fig4_disease_context` change: they take their series from a record instead of literals.

```python
# experiments/figures.py
"""Manuscript figures, rendered from run records.

_gen_figures.py held the F1 values as literals at lines 226-227, so
Figure 3 was a transcription of the paper's table rather than a rendering
of data. Every data-bearing figure here takes a run ID and reads its
series from that run's rows.

`test_the_figure_module_contains_no_metric_literals` fails if a literal
reappears in this file.
"""

from __future__ import annotations

from pathlib import Path

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

OUT = Path("figures")


def fig3_series(run_id: str, *, runs_dir: Path | None = None) -> dict:
    """Extract Figure 3's series from a run record."""
    record = RunRecord.load(run_id, runs_dir=runs_dir)
    if not record.citable:
        raise ValueError(
            f"run {run_id} is not citable (status={record.status}); a figure "
            f"may not be drawn from a run nobody can reproduce"
        )
    rows = record.rows()
    return {
        "diseases": [r["disease"] for r in rows],
        "neorx_f1": [r["F1"] for r in rows],
        "corr_f1": [r["corr_F1"] for r in rows],
    }


@experiment(name="figures", help="Render manuscript figures from a run record.")
def figures(record: RunRecord) -> None:
    """Registered so figure generation is itself a recorded run.

    The source run is read from NEORX_FIGURE_RUN; `neorx exp figure` sets it.
    """
    import os

    source = os.environ.get("NEORX_FIGURE_RUN")
    if not source:
        raise ValueError(
            "set NEORX_FIGURE_RUN to the run ID the figures should render from, "
            "or use `neorx exp figure --from <run-id>`"
        )

    OUT.mkdir(exist_ok=True)
    series = fig3_series(source)
    _render_fig3(series)
    record.append_row(
        {"figure": "fig3_pr_comparison", "source_run": source,
         "n_series_points": len(series["diseases"])}
    )
```

Add `_render_fig3(series)` by moving the body of `_gen_figures.py:223-295` across, replacing the two literal lists with `series["neorx_f1"]` and `series["corr_f1"]`, and the disease labels with `series["diseases"]`.

- [ ] **Step 4: Add the `figure` CLI command**

In `src/neorx/experiments/__main__.py`:

```python
@app.command("figure")
def figure_cmd(
    from_run: str = typer.Option(..., "--from", help="Run ID to render from."),
) -> None:
    """Render manuscript figures from a run record."""
    import os

    import experiments  # noqa: F401

    os.environ["NEORX_FIGURE_RUN"] = from_run
    record = run_experiment("figures")
    typer.echo(f"rendered from {from_run}; figure run {record.run_id}")
```

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/python -m pytest tests/experiments/test_figures.py -q
```

Expected: 3 passed

- [ ] **Step 6: Confirm the gate now reports zero on the migrated module**

```bash
.venv/bin/python -c "
from pathlib import Path
from neorx.experiments.gates import find_hardcoded_metrics
print('findings in experiments/figures.py:', len(find_hardcoded_metrics(Path('experiments/figures.py'))))"
```

Expected: `0`. Compare against the count you recorded in Task 5 Step 6 for `_gen_figures.py` and report both.

- [ ] **Step 7: Delete the superseded script**

```bash
rm -f _gen_figures.py
```

- [ ] **Step 8: Commit**

```bash
git add experiments/figures.py tests/experiments/test_figures.py \
        src/neorx/experiments/__main__.py
git commit -m "feat: render figures from run records instead of literals

_gen_figures.py held the F1 values as hardcoded lists, so Figure 3 was a
transcription of the paper's table rather than a rendering of data. Every
data-bearing figure now takes a run ID, and a non-citable run is refused.
A test fails if a metric literal reappears in the module."
```

---

### Task 10: Wire the gates into CI and track `runs/`

**Files:**
- Modify: `.gitignore`, `.github/workflows/ci.yml`
- Create: `tests/test_experiment_gates.py`
- Test: as listed

**Interfaces:**
- Consumes: every gate from Task 5
- Produces: no importable API; this task makes the guarantee enforced

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment_gates.py
"""The gates, wired to the real repository."""

from pathlib import Path

from neorx.experiments.gates import (
    check_cited_runs,
    check_records_wellformed,
    find_hardcoded_metrics,
)

REPO = Path(__file__).resolve().parent.parent


def test_no_experiment_module_carries_a_metric_literal():
    offenders = []
    for path in sorted((REPO / "experiments").glob("*.py")):
        offenders += find_hardcoded_metrics(path)
    assert offenders == [], [f"{o.path}:{o.line} {o.message}" for o in offenders]


def test_every_stored_run_is_wellformed():
    findings = check_records_wellformed(REPO / "runs")
    assert findings == [], [f.message for f in findings]


def test_every_cited_run_exists_and_is_citable():
    findings = check_cited_runs(REPO / "docs" / "run-manifest.toml", REPO / "runs")
    assert findings == [], [f.message for f in findings]
```

- [ ] **Step 2: Run to see the current state**

```bash
.venv/bin/python -m pytest tests/test_experiment_gates.py -q
```

Expected: all pass. `runs/` may be empty and the manifest has no entries yet — both gates return no findings on empty input, which is correct: they assert nothing rather than asserting falsely.

- [ ] **Step 3: Track `runs/`**

`.gitignore:37` ignores `results/`. Add an explicit un-ignore for `runs/` so records are committed:

```bash
cat >> .gitignore <<'EOF'

# Experiment run records are deliverables, not scratch -- they are the
# evidence behind every reported number. See docs/run-manifest.toml.
!runs/
EOF
grep -n "runs/" .gitignore
```

Confirm `runs/` is not covered by an earlier pattern:

```bash
git check-ignore -v runs/ 2>/dev/null && echo "STILL IGNORED -- fix the pattern" || echo "runs/ is tracked"
```

- [ ] **Step 4: Add the gates to CI**

In `.github/workflows/ci.yml`, add to the existing `quality` job's hard-gated section (it already runs `ruff check` and `mypy` over an 8-file whitelist):

```yaml
      - name: Experiment provenance gates
        run: |
          .venv/bin/python -m pytest tests/test_experiment_gates.py -q
```

If the `quality` job uses a system Python rather than `.venv`, match whatever invocation the neighbouring steps use rather than introducing a second convention.

- [ ] **Step 5: Add the new source files to the lint whitelist**

Task 5's gates and the Task 1–4 modules are new code this branch owns, so they belong in the hard-gated whitelist that both the `lint` and `quality` jobs share. Add these paths to **both** lists, keeping them byte-identical:

```
src/neorx/experiments/__init__.py
src/neorx/experiments/record.py
src/neorx/experiments/registry.py
src/neorx/experiments/capture.py
src/neorx/experiments/chembl.py
src/neorx/experiments/gates.py
src/neorx/experiments/__main__.py
experiments/genmol_eval.py
experiments/causalbiorl_bench.py
experiments/neorx_7disease.py
experiments/figures.py
```

Then confirm they pass both checks:

```bash
.venv/bin/python -m ruff check src/neorx/experiments/ experiments/
.venv/bin/python -m ruff format --check src/neorx/experiments/ experiments/
```

Fix anything reported — new code must be clean, per the ratchet policy this repo already follows.

- [ ] **Step 6: Verify the whitelists still match**

```bash
grep -A14 "ruff check" .github/workflows/ci.yml | grep -c "src/neorx/experiments"
```

Both jobs must show the same entries. Two lists that drift are worse than one.

- [ ] **Step 7: Run the targeted suite**

```bash
.venv/bin/python -m pytest tests/experiments/ tests/test_experiment_gates.py tests/test_cli.py -q -m "not network and not slow"
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add .gitignore .github/workflows/ci.yml tests/test_experiment_gates.py
git commit -m "ci: enforce experiment provenance and track run records

runs/ is now tracked: the records are the evidence behind every reported
number, not scratch. CI fails if an experiment module carries a metric
literal, if a stored record's summary disagrees with its rows, or if the
manuscript manifest cites a run that is missing or not citable."
```

---

### Task 11: Replay and prune

Closes the spec's headline claim — that a run can be re-executed against its frozen inputs and reproduce its numbers. May be done any time after Task 3; it is placed last only because it is easiest to demonstrate once real records exist.

**Files:**
- Create: `src/neorx/experiments/replay.py`
- Modify: `src/neorx/experiments/__main__.py`
- Test: `tests/experiments/test_replay.py`

**Interfaces:**
- Consumes: `RunRecord.load/rows/path` (Task 1), `run_experiment` (Task 2), `capture(mode="replay")` and `cassette_path` (Task 3)
- Produces:
  - `RowDiff` dataclass with `.index: int`, `.key: str`, `.recorded`, `.replayed`
  - `ReplayResult` dataclass with `.identical: bool`, `.diffs: list[RowDiff]`, `.replay_run_id: str`
  - `replay_experiment(run_id: str, *, runs_dir: Path | None = None) -> ReplayResult`
  - `prune_record(run_id: str, *, runs_dir: Path | None = None) -> int` → bytes freed
  - `class NotReplayableError(RuntimeError)`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_replay.py
"""Replay is the property the whole subsystem claims. Test it directly."""

import json

import pytest

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment, run_experiment
from neorx.experiments.replay import (
    NotReplayableError,
    prune_record,
    replay_experiment,
)


@experiment(name="replay-fixture", help="")
def _fixture(record: RunRecord) -> None:
    """Deterministic and offline, so any diff is the replay machinery's fault."""
    for i, disease in enumerate(["HIV", "malaria"]):
        record.append_row({"disease": disease, "F1": 0.5 + i / 10})


def test_replay_of_a_deterministic_run_is_identical(tmp_path):
    original = run_experiment("replay-fixture", runs_dir=tmp_path)
    result = replay_experiment(original.run_id, runs_dir=tmp_path)
    assert result.identical, result.diffs
    assert result.diffs == []
    assert result.replay_run_id != original.run_id


def test_a_changed_row_is_reported_by_field(tmp_path):
    original = run_experiment("replay-fixture", runs_dir=tmp_path)
    rows = original.rows()
    rows[1]["F1"] = 0.999
    (original.path / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    )

    result = replay_experiment(original.run_id, runs_dir=tmp_path)
    assert not result.identical
    assert len(result.diffs) == 1
    diff = result.diffs[0]
    assert diff.index == 1 and diff.key == "F1"
    assert diff.recorded == 0.999 and diff.replayed == 0.6


def test_replaying_a_pruned_run_refuses(tmp_path):
    original = run_experiment("replay-fixture", runs_dir=tmp_path)
    (original.path / "inputs" / "http.yaml").write_text("interactions: []\n")
    prune_record(original.run_id, runs_dir=tmp_path)
    with pytest.raises(NotReplayableError, match="pruned|inputs"):
        replay_experiment(original.run_id, runs_dir=tmp_path)


def test_prune_keeps_the_numbers_and_drops_only_the_inputs(tmp_path):
    original = run_experiment("replay-fixture", runs_dir=tmp_path)
    (original.path / "inputs" / "http.yaml").write_text("x" * 4096)

    freed = prune_record(original.run_id, runs_dir=tmp_path)
    assert freed >= 4096
    assert not (original.path / "inputs").exists()
    assert (original.path / "rows.jsonl").exists()

    reloaded = RunRecord.load(original.run_id, runs_dir=tmp_path)
    assert len(reloaded.rows()) == 2, "pruning must never destroy results"
    summary = json.loads((original.path / "record.json").read_text())
    assert summary["replayable"] is False
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/experiments/test_replay.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.experiments.replay'`

- [ ] **Step 3: Write the implementation**

```python
# src/neorx/experiments/replay.py
"""Re-execute a recorded run against its frozen inputs and diff the result.

This is the claim the subsystem rests on: given a run record, anyone can
reproduce its numbers without reaching the network. A replay that differs
is not a failure of the tool -- it names exactly which values moved, which
is the information a reviewer actually needs.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neorx.experiments.capture import capture, cassette_path
from neorx.experiments.record import RunRecord
from neorx.experiments.registry import get_experiment


class NotReplayableError(RuntimeError):
    """The run cannot be replayed -- typically its inputs were pruned."""


@dataclass(frozen=True)
class RowDiff:
    index: int
    key: str
    recorded: Any
    replayed: Any


@dataclass
class ReplayResult:
    replay_run_id: str
    diffs: list[RowDiff] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not self.diffs


def _experiment_name(record: RunRecord) -> str:
    return json.loads((record.path / "record.json").read_text())["experiment"]


def replay_experiment(run_id: str, *, runs_dir: Path | None = None) -> ReplayResult:
    """Re-run a recorded experiment against its cassette and diff the rows."""
    original = RunRecord.load(run_id, runs_dir=runs_dir)
    summary = json.loads((original.path / "record.json").read_text())
    if summary.get("replayable") is False:
        raise NotReplayableError(
            f"run {run_id} was pruned: its frozen inputs are gone, so it cannot "
            f"be replayed. Its recorded rows remain in rows.jsonl."
        )

    defn = get_experiment(_experiment_name(original))
    replay_rec = RunRecord.create(f"{defn.name}-replay", runs_dir=runs_dir)

    has_cassette = cassette_path(original).exists()
    if has_cassette:
        shutil.copy2(cassette_path(original), cassette_path(replay_rec))
        with capture(replay_rec, mode="replay"):
            defn.fn(replay_rec)
    else:
        defn.fn(replay_rec)
    replay_rec.finalise("complete")

    return ReplayResult(
        replay_run_id=replay_rec.run_id,
        diffs=_diff_rows(original.rows(), replay_rec.rows()),
    )


def _diff_rows(recorded: list[dict], replayed: list[dict]) -> list[RowDiff]:
    diffs: list[RowDiff] = []
    for i in range(max(len(recorded), len(replayed))):
        old = recorded[i] if i < len(recorded) else None
        new = replayed[i] if i < len(replayed) else None
        if old is None or new is None:
            diffs.append(RowDiff(i, "<row>", old, new))
            continue
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                diffs.append(RowDiff(i, key, old.get(key), new.get(key)))
    return diffs


def prune_record(run_id: str, *, runs_dir: Path | None = None) -> int:
    """Drop a run's frozen inputs, keeping its results. Returns bytes freed."""
    record = RunRecord.load(run_id, runs_dir=runs_dir)
    inputs = record.path / "inputs"
    freed = (
        sum(p.stat().st_size for p in inputs.rglob("*") if p.is_file())
        if inputs.exists()
        else 0
    )
    if inputs.exists():
        shutil.rmtree(inputs)

    summary_path = record.path / "record.json"
    summary = json.loads(summary_path.read_text())
    summary["replayable"] = False
    summary["pruned_bytes"] = freed
    summary_path.write_text(json.dumps(summary, indent=2))
    return freed
```

- [ ] **Step 4: Add the CLI commands**

In `src/neorx/experiments/__main__.py`:

```python
@app.command("replay")
def replay_cmd(run_id: str = typer.Argument(..., help="Run ID to replay.")) -> None:
    """Re-execute a recorded run against its frozen inputs and diff the rows."""
    import experiments  # noqa: F401

    from neorx.experiments.replay import replay_experiment

    result = replay_experiment(run_id)
    if result.identical:
        typer.echo(f"IDENTICAL  ({run_id} reproduced by {result.replay_run_id})")
        return
    typer.echo(f"DIFFERS  ({len(result.diffs)} field(s))")
    for d in result.diffs:
        typer.echo(f"  row {d.index}  {d.key}: recorded={d.recorded!r} replayed={d.replayed!r}")
    raise typer.Exit(code=1)


@app.command("prune")
def prune_cmd(run_id: str = typer.Argument(..., help="Run ID to prune.")) -> None:
    """Drop a run's frozen inputs, keeping its recorded results."""
    from neorx.experiments.replay import prune_record

    freed = prune_record(run_id)
    typer.echo(f"pruned {run_id}: freed {freed / 1e6:.1f} MB; results retained")
```

- [ ] **Step 5: Record `experiment` in the summary**

`replay_experiment` reads `record.json["experiment"]`. Task 1's `finalise` writes it via `getattr(self, "_experiment", self.run_id)`. Confirm it is the plain experiment name, not the run ID:

```bash
.venv/bin/python -c "
import tempfile, json
from pathlib import Path
from neorx.experiments.record import RunRecord
with tempfile.TemporaryDirectory() as d:
    r = RunRecord.create('demo-exp', runs_dir=Path(d)); r.finalise('complete')
    got = json.loads((r.path / 'record.json').read_text())['experiment']
    assert got == 'demo-exp', got
    print('experiment name recorded correctly:', got)"
```

If it records the run ID instead, fix `RunRecord.create` to set `self._experiment` before `finalise` can be called — `replay_experiment` depends on it.

- [ ] **Step 6: Run the tests**

```bash
.venv/bin/python -m pytest tests/experiments/test_replay.py -q
```

Expected: 4 passed

- [ ] **Step 7: Demonstrate replay end to end**

```bash
.venv/bin/python -m neorx.cli exp run genmol-eval
RUN=$(ls -t runs | head -1)
.venv/bin/python -m neorx.cli exp replay "$RUN"
```

Expected: `IDENTICAL`. `genmol-eval` samples stochastically, so if it reports `DIFFERS` on the generation metrics that is the model's sampling, not a replay bug — report which fields moved rather than treating it as a failure, and note that a seeded generator is what would make this experiment bit-reproducible.

- [ ] **Step 8: Commit**

```bash
git add src/neorx/experiments/replay.py src/neorx/experiments/__main__.py \
        tests/experiments/test_replay.py
git commit -m "feat: add exp replay and exp prune

Replay re-executes a recorded run against its frozen cassette and names
exactly which fields moved, which is what a reviewer needs. Prune drops a
run's inputs while keeping its rows, marking it replayable: false rather
than deleting the evidence."
```

---

## Notes for the executor

**Run the targeted tests after every task**, and the full suite only when the controller asks. The suite takes ~10 minutes and cannot be parallelised.

**Task 3 Step 8 is the load-bearing verification in this plan.** If the record/replay round-trip does not reproduce byte-identical responses, nothing downstream is trustworthy — stop and report rather than continuing to Task 4.

**Task 8 Step 7 is the second one.** If the single-disease smoke run captures zero HTTP interactions, the sources are bypassing the recorder and the capture design needs revisiting before the full benchmark is worth running.

**Do not fix methodology defects you encounter.** Task 8 names two (`t_graph` always zero; the baseline's node-type filter). Recording a defect faithfully is this sub-project's job; fixing it is sub-project 3's. If you find a third, record it and report it — do not repair it.

**Task 11 may be pulled earlier.** It depends only on Tasks 1-3, and doing it before the migrations means each migrated experiment can be replay-checked as it lands rather than all at the end.

**Do not add a shim.** If an experiment does not fit the runner, change the runner or say so. The previous sub-project deleted a compatibility shim that corrupted the shipped package, and the owner's directive since is explicit.
