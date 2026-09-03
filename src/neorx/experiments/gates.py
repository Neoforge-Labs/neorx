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
