"""Experiment registry.

An experiment is a function taking a RunRecord. It never opens a file and
never decides where results go -- the runner does both. That is what makes
"a number cannot exist without a record" structural rather than a habit.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from neorx.experiments.capture import capture
from neorx.experiments.record import ProvenanceError, RunRecord

ExperimentFn = Callable[[RunRecord], None]


class UnknownExperimentError(KeyError):
    """Raised when no experiment is registered under the given name."""


@dataclass(frozen=True)
class ExperimentDef:
    name: str
    help: str
    fn: ExperimentFn
    captures_http: bool = False
    #: Row keys whose value is expected to differ between a run and its
    #: replay -- wall-clock timings and the like -- and so must not count
    #: toward the replay's IDENTICAL/DIFFERS verdict. They are still
    #: reported (see ReplayResult.volatile_diffs), never silently dropped.
    volatile_fields: tuple[str, ...] = ()
    #: Declares that this experiment reads pinned snapshots, so its run
    #: record must cite which extracts produced its numbers. The
    #: experiment declares the need; the runner supplies the manifest
    #: path, for the same reason it chooses the HTTP capture mode -- only
    #: the caller knows where the store lives.
    reads_snapshots: bool = False


_REGISTRY: dict[str, ExperimentDef] = {}


def experiment(
    *,
    name: str,
    help: str = "",
    captures_http: bool = False,
    volatile_fields: tuple[str, ...] = (),
    reads_snapshots: bool = False,
) -> Callable[[ExperimentFn], ExperimentFn]:
    """Register an experiment under ``name``.

    ``captures_http`` declares that the experiment needs its HTTP traffic
    recorded. An experiment never chooses the capture *mode* -- only the
    caller (a live run vs. a replay) knows whether that should be record
    or replay -- so it only declares the need; the runner decides the mode.

    ``volatile_fields`` declares row keys that are expected to differ on
    every replay (wall-clock timings and similar) -- see
    ``ExperimentDef.volatile_fields``.

    ``reads_snapshots`` declares that the experiment's numbers come from
    pinned extracts, so the runner records which ones. Without it, an
    experiment reading snapshots writes an ``env.json`` whose
    ``snapshots`` object is empty -- a run citing no inputs, which is the
    provenance failure this project already found in four published
    papers.
    """

    def decorate(fn: ExperimentFn) -> ExperimentFn:
        if name in _REGISTRY:
            raise ValueError(f"experiment {name!r} is already registered")
        _REGISTRY[name] = ExperimentDef(
            name=name,
            help=help,
            fn=fn,
            captures_http=captures_http,
            volatile_fields=tuple(volatile_fields),
            reads_snapshots=reads_snapshots,
        )
        return fn

    return decorate


def get_experiment(name: str) -> ExperimentDef:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise UnknownExperimentError(f"no experiment named {name!r}. Available: {known}") from None


def list_experiments() -> list[ExperimentDef]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def _cited_snapshots(record: RunRecord) -> dict:
    """What the record's env.json actually cites."""
    env = json.loads((record.path / "env.json").read_text(encoding="utf-8"))
    return env.get("snapshots") or {}


# Where `neorx snapshot build` writes by default, so a run reads the
# manifest the CLI wrote without a second place to configure one path.
SNAPSHOT_MANIFEST = Path("snapshots") / "manifest.toml"


def run_experiment(
    name: str,
    *,
    allow_large: bool = False,
    runs_dir: Path | None = None,
    snapshot_manifest: Path | None = SNAPSHOT_MANIFEST,
) -> RunRecord:
    """Execute an experiment, writing its record whatever the outcome.

    An experiment that declares ``reads_snapshots`` gets its extracts
    cited in ``env.json``. One that does not gets an empty ``snapshots``
    object, which is accurate: it read none.
    """
    defn = get_experiment(name)
    record = RunRecord.create(
        name,
        runs_dir=runs_dir,
        allow_large=allow_large,
        snapshot_manifest=(
            snapshot_manifest if defn.reads_snapshots else None
        ),
    )
    if defn.reads_snapshots and not _cited_snapshots(record):
        # The declaration buys a path, not a guarantee. A store whose
        # extracts are present but whose manifest is missing produced a
        # run that read those extracts, wrote "snapshots": {}, and
        # finalised `complete` -- a number with no traceable derivation,
        # which is the exact failure this project found in four published
        # papers. Refuse instead: a run that cannot say what produced it
        # is a failed run, not a quiet one.
        record.finalise("failed")
        raise ProvenanceError(
            f"experiment {name!r} declares reads_snapshots, but no snapshot "
            f"was cited: {snapshot_manifest} is missing or has no entries. "
            f"Build the extracts with `neorx snapshot build`, which writes "
            f"the manifest beside them. Running without it would report "
            f"numbers derived from extracts the record cannot name."
        )
    try:
        if defn.captures_http:
            with capture(record, mode="record"):
                defn.fn(record)
        else:
            defn.fn(record)
    except BaseException:
        record.finalise("failed")
        raise
    record.finalise("complete")
    return record
