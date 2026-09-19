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
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neorx.snapshots.manifest import digest_file, read_manifest
from neorx.snapshots.reader import SnapshotStore

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
# Where `neorx snapshot build` writes in this repository. Anchored the same
# way as RUNS_DIR: a cwd-relative path let a run launched from a
# subdirectory read one store while the runner cited another's manifest.
SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "snapshots"
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
        # (source, release) -> (manifest describing it, extract actually read)
        self._snapshot_reads: dict[tuple[str, str], tuple[Path, Path]] = {}

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
                    # Filled by settle_snapshot_citations once the run has
                    # read what it reads -- a citation is a statement about
                    # what produced the numbers, so it cannot be written
                    # before any were produced.
                    "snapshots": {},
                },
                indent=2,
            )
        )
        return rec

    # -- snapshot provenance --------------------------------------------

    def snapshot_store(self, root: Path) -> SnapshotStore:
        """A snapshot store whose every read is recorded against this run.

        Experiments open stores through this rather than constructing one
        directly, so what gets cited is what was read -- not the whole
        manifest, and not a manifest found by a second path that could
        name a different store.
        """
        manifest = SnapshotStore(root).manifest_path

        def note(source: str, release: str, path: Path) -> None:
            key = (source, release)
            seen = self._snapshot_reads.get(key)
            if seen is not None and seen[0] != manifest:
                raise ProvenanceError(
                    f"{source}/{release} was read from two stores in one run "
                    f"({seen[0].parent} and {manifest.parent}); a citation "
                    f"keyed by source and release cannot say which produced "
                    f"the numbers."
                )
            self._snapshot_reads[key] = (manifest, path)

        return SnapshotStore(root, on_read=note)

    @property
    def snapshot_reads(self) -> list[tuple[str, str]]:
        """Every (source, release) this run read, in a stable order."""
        return sorted(self._snapshot_reads)

    def settle_snapshot_citations(self) -> list[str]:
        """Cite exactly the extracts this run read; return any it could not.

        Each citation is the whole manifest entry, not a chosen subset. A
        digest pins which bytes an extract was, not what they meant, and
        the manifest that says what they meant is replaced in place when an
        extract is rebuilt -- so a run record that kept only the digest
        would, months later, cite something nothing can interpret.
        `synthesised_consensus` is what makes that bite: it says whether
        OmniPath's consensus direction was read or derived, which changes
        what every directed edge in a dated graph means. `source` and
        `release` are dropped only because they are the key.

        A read whose extract has no manifest entry is not silently left
        out: it is written under `snapshots_uncited` and returned, so the
        caller decides whether that is fatal and the record says so either
        way.
        """
        manifests: dict[Path, dict] = {}
        cited: dict[str, dict[str, Any]] = {}
        uncited: list[str] = []
        mismatched: list[dict[str, str]] = []
        for (source, release), (manifest, path) in sorted(
            self._snapshot_reads.items()
        ):
            if manifest not in manifests:
                manifests[manifest] = read_manifest(manifest)
            entry = manifests[manifest].get((source, release))
            if entry is None:
                uncited.append(f"{source}/{release}")
                continue
            # The manifest describes bytes; this run read bytes. Until now
            # nothing checked they were the same bytes. `snapshot build`
            # writes the extract and its manifest entry as two steps, so an
            # interrupted rebuild of an already-built release leaves the
            # manifest describing the old file and the disk holding the new
            # -- and the run would cite a digest of bytes it never read,
            # which is the defect this whole layer exists to prevent.
            actual = digest_file(path)
            if actual != entry.sha256:
                mismatched.append(
                    {
                        "extract": f"{source}/{release}",
                        "path": str(path),
                        "manifest_sha256": entry.sha256,
                        "actual_sha256": actual,
                    }
                )
                continue
            fields = asdict(entry)
            del fields["source"], fields["release"]
            cited[f"{source}/{release}"] = fields

        env_path = self.path / "env.json"
        env = json.loads(env_path.read_text(encoding="utf-8"))
        env["snapshots"] = cited
        env["snapshots_uncited"] = uncited
        env["snapshots_mismatched"] = mismatched
        env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")
        return uncited + [f"{m['extract']} (digest mismatch)" for m in mismatched]

    def note_settlement_failure(self, error: BaseException) -> None:
        """Record that citations could not be settled, and why.

        Only reached when the run was already failing. Losing the record
        to a second fault would leave the operator reading the manifest
        error instead of the one that stopped the run.
        """
        env_path = self.path / "env.json"
        env = json.loads(env_path.read_text(encoding="utf-8"))
        env["snapshots_settlement_error"] = f"{type(error).__name__}: {error}"
        env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")

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
