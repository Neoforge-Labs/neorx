"""Checks that make "no number without a record" enforceable.

Each gate closes one hole the manuscript audit found:

* ``find_hardcoded_metrics``  -- _gen_figures.py held the F1 values as
  literals, so Figure 3 was a transcription of a table rather than a
  rendering of data.
* ``check_records_wellformed`` -- a record whose summary disagrees with
  its rows is not evidence.
* ``check_cited_runs``        -- a manuscript citing a run that does not
  exist, or one nobody can reproduce.
* ``check_committed_records_are_citable`` -- a non-citable run record
  (dirty tree, incomplete, or failed) committed to git anyway, adding
  ~MB of weight to the repository while backing no claim.

Known limit of ``find_hardcoded_metrics``: it only matches bare
``ast.Constant`` float literals. A metric assembled by computation --
``545 / 1000``, ``round(x, 3)``, an f-string embedding a number, a value
read from a dict -- is invisible to an AST literal check by construction,
not by an oversight fixable here. A clean run of this gate means "no
metric was typed in as a literal"; it does NOT mean "no computed metric
literal exists anywhere in this file". Treat "0 findings" accordingly.
"""

from __future__ import annotations

import ast
import decimal
import json
import subprocess
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

METRIC_DECIMALS = 3

#: Values with >=3 decimals that are legitimately constants, not metrics.
#: Every entry needs a comment saying why -- never widen the pattern instead.
DEFAULT_ALLOWLIST: set[float] = {
    1.618,  # golden ratio, used for figure aspect ratios
    0.001,  # p-value floor in the identifier's proxy
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str


def _decimals(text: str) -> int:
    exponent = Decimal(text).as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def find_hardcoded_metrics(path: Path, allowlist: set[float] | None = None) -> list[Finding]:
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
        except decimal.InvalidOperation:
            # The gate's entire job is catching metric literals -- a
            # literal it cannot classify must surface, not vanish. This is
            # not expected to fire on ordinary Python float syntax; if it
            # does, someone needs to look at it.
            findings.append(
                Finding(
                    path=str(path),
                    line=node.lineno,
                    message=(
                        f"could not classify literal {literal!r} as a decimal -- "
                        f"unable to check whether it is a metric; inspect it by hand"
                    ),
                )
            )
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
        actual = (
            len([ln for ln in rows_file.read_text().splitlines() if ln.strip()])
            if rows_file.exists()
            else 0
        )
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


def check_committed_records_are_citable(
    runs_dir: Path, tracked: set[str] | None = None
) -> list[Finding]:
    """Every run record committed to git must be citable.

    ``runs/`` is tracked on the theory that its records are the evidence
    behind every reported number. A record with ``citable: false`` (built
    from a dirty tree, or left incomplete or failed) is not evidence --
    the figure command already refuses to render from one -- so committing
    it adds weight to the repository without adding anything that backs a
    claim. ``tracked`` is the set of run-directory names (run IDs) known
    to be committed; when omitted, it is read from ``git ls-files``.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []

    if tracked is None:
        result = subprocess.run(
            ["git", "ls-files", "."],
            cwd=str(runs_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return [
                Finding(
                    str(runs_dir),
                    0,
                    f"git ls-files failed (exit {result.returncode}): "
                    f"{result.stderr.strip()} -- could not determine which "
                    f"records are committed",
                )
            ]
        tracked = {
            Path(line).parts[0]
            for line in result.stdout.splitlines()
            if line.strip()
        }

    findings: list[Finding] = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if run.name not in tracked:
            continue
        summary = run / "record.json"
        if not summary.exists():
            continue  # check_records_wellformed already flags this
        data = json.loads(summary.read_text())
        if not data.get("citable"):
            findings.append(
                Finding(
                    str(summary),
                    0,
                    f"{run.name}: committed to git but citable=false -- a "
                    f"non-citable record backs no claim, so committing it "
                    f"adds weight to the repository without evidence",
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
                        f"{section}.{citation} cites run {run_id!r}, which is not in {runs_dir}",
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
