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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
# ~7 MB per disease measured on a live neorx-7disease run; seven diseases
# ~= 49 MB, so 250 MB leaves genuine headroom while still catching a
# runaway snapshot. (The original 50 MB figure was a spec estimate of
# "5-6 MB per seven-disease run" -- about eight times too low; a full run
# would have hit refusal after ~25 minutes of live API cost were sunk.)
MAX_RECORD_BYTES = 250 * 1024 * 1024

_VALID_STATUS = ("complete", "incomplete", "failed")


class RecordTooLargeError(RuntimeError):
    """Raised when a finalised record exceeds MAX_RECORD_BYTES."""


class ProvenanceError(RuntimeError):
    """Raised when the code provenance of a run cannot be established.

    A run whose git SHA or dirty state cannot be determined has no
    provenance. It is a failed run, not one that quietly records a
    fabricated SHA or a false "clean" state.
    """


class RunIDCollisionError(RuntimeError):
    """Raised when a run ID collides with an existing run directory."""


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProvenanceError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_sha() -> str:
    return _git("rev-parse", "HEAD")


def _git_dirty() -> bool:
    # ':(top,exclude)runs' anchors the exclusion to the repo root and
    # excludes the runs/ directory itself, regardless of the process's
    # current working directory. runs/ holds the run records this module
    # writes -- it is tracked evidence, not scratch, but its own act of
    # being written must never make the run that wrote it look dirty.
    # This checks whether the CODE is dirty, not whether artifacts exist.
    return bool(_git("status", "--porcelain", "--", ":(top,exclude)runs"))


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
        # Set by create() only -- a record produced by load() never learns
        # its start time or experiment name from these fields (finalise()
        # falls back to run_id via getattr for the latter).
        self._started: datetime | None = None
        self._experiment: str | None = None

    # -- construction --------------------------------------------------

    @classmethod
    def create(
        cls,
        experiment: str,
        *,
        runs_dir: Path | None = None,
        allow_large: bool = False,
    ) -> RunRecord:
        started = datetime.now(UTC)
        sha = _git_sha()
        digest = hashlib.sha256(f"{started.isoformat()}{sha}".encode()).hexdigest()[:6]
        run_id = f"{started:%Y-%m-%d}-{experiment}-{digest}"

        base = runs_dir if runs_dir is not None else RUNS_DIR
        path = base / run_id
        try:
            (path / "inputs").mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise RunIDCollisionError(
                f"run id {run_id!r} already exists at {path}; refusing to "
                "write into an existing run's directory"
            ) from None

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

    # -- writing ----------------------------------------------------------

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
        rows_path = self.path / "rows.jsonl"
        rows_sha256 = (
            hashlib.sha256(rows_path.read_bytes()).hexdigest() if rows_path.exists() else None
        )
        (self.path / "record.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "experiment": getattr(self, "_experiment", self.run_id),
                    "status": status,
                    "citable": self.citable,
                    "n_rows": self._n_rows,
                    "size_bytes": size,
                    "rows_sha256": rows_sha256,
                    "finalised_utc": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )

    # -- reading ----------------------------------------------------------

    def rows(self) -> list[dict[str, Any]]:
        text = (self.path / "rows.jsonl").read_text().strip()
        return [json.loads(line) for line in text.splitlines()] if text else []

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())
