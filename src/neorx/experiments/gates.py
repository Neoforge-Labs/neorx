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
* ``check_record_integrity`` -- rows.jsonl edited after the fact (e.g. an
  F1 changed from 0.474 to 0.999, row count left untouched) with no other
  gate noticing. This detects accidental edits and drift by recomputing
  the SHA-256 stored at finalise time; it does NOT detect a determined
  forger who edits rows.jsonl and recomputes/rewrites rows_sha256 to
  match -- that requires signing the record, which this gate is not.

Known limit of ``find_hardcoded_metrics``: it matches bare ``ast.Constant``
float literals, and ``ast.Constant`` string literals that contain a
metric-shaped number (a digit, a decimal point, and >= METRIC_DECIMALS
digits after it -- e.g. ``"C=0.990"``). A metric assembled by computation --
``545 / 1000``, ``round(x, 3)``, an f-string embedding a number, a value
read from a dict -- is invisible to an AST literal check by construction,
not by an oversight fixable here. A clean run of this gate means "no
metric was typed in as a float literal or spelled out in a string
literal"; it does NOT mean "no computed metric literal exists anywhere in
this file". Treat "0 findings" accordingly.
"""

from __future__ import annotations

import ast
import decimal
import hashlib
import json
import re
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

#: A digit, a decimal point, and >= METRIC_DECIMALS digits after it --
#: matches "0.990" inside a string like "C=0.990" the same way the float
#: check matches the literal 0.990. Deliberately unanchored: a metric can
#: sit anywhere inside a larger label string.
_METRIC_SHAPED_NUMBER_IN_STRING = re.compile(r"\d\.\d{" + str(METRIC_DECIMALS) + r",}")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str


def _decimals(text: str) -> int:
    exponent = Decimal(text).as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def find_hardcoded_metrics(path: Path, allowlist: set[float] | None = None) -> list[Finding]:
    """Flag metric-shaped float literals (>= METRIC_DECIMALS decimals) and

    metric-shaped numbers spelled inside string literals (e.g. "C=0.990").
    A measured value transcribed as a string evades a float-only check just
    as completely as one assembled by computation -- this closes that hole
    for the string case specifically.
    """
    allowed = DEFAULT_ALLOWLIST if allowlist is None else allowlist
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, str):
            match = _METRIC_SHAPED_NUMBER_IN_STRING.search(node.value)
            if match is not None:
                findings.append(
                    Finding(
                        path=str(path),
                        line=node.lineno,
                        message=(
                            f"string literal {node.value!r} spells out a "
                            f"metric-shaped number ({match.group()}) -- render it "
                            f"from a run record instead of transcribing it as text"
                        ),
                    )
                )
            continue
        if not isinstance(node.value, float):
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
        tracked = {Path(line).parts[0] for line in result.stdout.splitlines() if line.strip()}

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


def check_record_integrity(runs_dir: Path) -> list[Finding]:
    """Every run record's stored rows_sha256 must match its rows.jsonl.

    ``RunRecord.finalise`` computes a SHA-256 over rows.jsonl's bytes at
    finalise time and stores it in record.json as ``rows_sha256``. A record
    edited afterwards -- a metric changed, a row dropped, in either case
    with the row count left alone so ``check_records_wellformed`` sees
    nothing wrong -- will disagree with that stored digest.

    This catches accidental edits and drift, not a forger who edits
    rows.jsonl and also recomputes and rewrites rows_sha256 to match; that
    is a signing problem, not a hashing problem, and out of scope here.
    """
    findings: list[Finding] = []
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return findings

    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        summary = run / "record.json"
        if not summary.exists():
            continue  # check_records_wellformed already flags this
        data = json.loads(summary.read_text())
        stored = data.get("rows_sha256")
        if stored is None:
            findings.append(
                Finding(
                    str(summary),
                    0,
                    f"{run.name}: record.json has no rows_sha256 -- "
                    f"finalised by an older version of RunRecord, or tampered",
                )
            )
            continue
        rows_file = run / "rows.jsonl"
        actual = hashlib.sha256(rows_file.read_bytes()).hexdigest() if rows_file.exists() else None
        if actual != stored:
            findings.append(
                Finding(
                    str(rows_file),
                    0,
                    f"{run.name}: rows.jsonl digest {actual!r} does not match "
                    f"record.json's rows_sha256 {stored!r} -- rows.jsonl was "
                    f"edited after the run finalised",
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


# Names whose presence in the causal package means a fabricated statistic
# has come back. Each was removed for a specific reason:
#   _bootstrap_confidence_interval / _bootstrap_ci -- percentiles of
#       hand-chosen Gaussian noise added to deterministic scores, so the
#       interval width was a readout of two constants
#   p_value -- a rescaling of a heuristic score, with no null distribution
#   d_separated -- removed from NetworkX; every call raised, and the
#       except branch continued as though it had passed
_FABRICATED_STATISTIC_NAMES = (
    "_bootstrap_confidence_interval",
    "_bootstrap_ci",
    "p_value",
    "d_separated",
)


def check_no_fabricated_statistics(causal_dir: Path) -> list[Finding]:
    """Assert no fabricated statistic has been reintroduced.

    Scoped to a directory by argument, never to ``src/`` as a whole:
    ``neorx/genmol/evaluation/distribution.py`` reports a genuine
    Kolmogorov-Smirnov p-value, and a repository-wide search would flag
    it. Point this at ``neorx/core/causal/`` only.
    """
    findings: list[Finding] = []

    for path in sorted(causal_dir.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for name in _FABRICATED_STATISTIC_NAMES:
                if name in line:
                    findings.append(
                        Finding(
                            path=str(path),
                            line=lineno,
                            message=(
                                f"{name} is a fabricated statistic removed in "
                                f"sub-project 3; it must not return. See "
                                f"docs/superpowers/specs/2026-09-03-correctness-design.md"
                            ),
                        )
                    )

    return findings
