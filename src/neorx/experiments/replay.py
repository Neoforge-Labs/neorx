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
    #: Diffs on fields the experiment declared ``volatile_fields`` -- e.g.
    #: wall_clock_s. Never counted toward ``identical``, but always
    #: reported here rather than silently dropped: a gate must never
    #: swallow information, only decide what does and doesn't fail a
    #: verdict.
    volatile_diffs: list[RowDiff] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not self.diffs


def _experiment_name(record: RunRecord) -> str:
    data: dict[str, Any] = json.loads((record.path / "record.json").read_text())
    return str(data["experiment"])


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
    if defn.captures_http and has_cassette:
        shutil.copy2(cassette_path(original), cassette_path(replay_rec))
        with capture(replay_rec, mode="replay"):
            defn.fn(replay_rec)
    else:
        defn.fn(replay_rec)
    replay_rec.finalise("complete")

    diffs, volatile_diffs = _diff_rows(
        original.rows(), replay_rec.rows(), volatile_fields=defn.volatile_fields
    )
    return ReplayResult(
        replay_run_id=replay_rec.run_id,
        diffs=diffs,
        volatile_diffs=volatile_diffs,
    )


def _diff_rows(
    recorded: list[dict], replayed: list[dict], *, volatile_fields: tuple[str, ...] = ()
) -> tuple[list[RowDiff], list[RowDiff]]:
    """Diff two runs' rows, splitting out fields declared volatile.

    Volatile fields (e.g. wall_clock_s) are expected to differ on every
    replay -- they are excluded from the returned verdict-bearing diffs so
    a real experiment can ever report IDENTICAL, but they are never
    dropped: they come back in the second list, still visible to whoever
    is looking.
    """
    volatile = set(volatile_fields)
    diffs: list[RowDiff] = []
    volatile_diffs: list[RowDiff] = []
    for i in range(max(len(recorded), len(replayed))):
        old = recorded[i] if i < len(recorded) else None
        new = replayed[i] if i < len(replayed) else None
        if old is None or new is None:
            diffs.append(RowDiff(i, "<row>", old, new))
            continue
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                target = volatile_diffs if key in volatile else diffs
                target.append(RowDiff(i, key, old.get(key), new.get(key)))
    return diffs, volatile_diffs


def prune_record(run_id: str, *, runs_dir: Path | None = None) -> int:
    """Drop a run's frozen inputs, keeping its results. Returns bytes freed."""
    record = RunRecord.load(run_id, runs_dir=runs_dir)
    inputs = record.path / "inputs"
    freed = (
        sum(p.stat().st_size for p in inputs.rglob("*") if p.is_file()) if inputs.exists() else 0
    )
    if inputs.exists():
        shutil.rmtree(inputs)

    summary_path = record.path / "record.json"
    summary = json.loads(summary_path.read_text())
    summary["replayable"] = False
    summary["pruned_bytes"] = freed
    summary_path.write_text(json.dumps(summary, indent=2))
    return freed
