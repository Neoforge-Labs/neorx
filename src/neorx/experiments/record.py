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

    # -- construction --------------------------------------------------

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

    # -- reading ----------------------------------------------------------

    def rows(self) -> list[dict[str, Any]]:
        text = (self.path / "rows.jsonl").read_text().strip()
        return [json.loads(line) for line in text.splitlines()] if text else []

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())
