"""Experiment registry.

An experiment is a function taking a RunRecord. It never opens a file and
never decides where results go -- the runner does both. That is what makes
"a number cannot exist without a record" structural rather than a habit.
"""

from __future__ import annotations

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
    #: Declares that this experiment reads pinned snapshots. It must open
    #: them with ``record.snapshot_store(root)``; the runner then cites
    #: exactly what was read and refuses a declared experiment that
    #: recorded no reads, since that means a store was opened some other
    #: way and nothing it read is cited.
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


def settle_or_refuse(record: RunRecord, name: str, *, declared: bool) -> None:
    """Cite what the run read, and refuse a run whose inputs cannot be named.

    Called after a run completes, by both the runner and replay, so the two
    cannot drift apart -- they did once, and a replay cited nothing while
    reporting `identical: True`.

    Two refusals. A read with no manifest entry: the extracts were there
    and the manifest describing them was not, so the run produced numbers
    from inputs it cannot identify -- the failure this project found in
    four published papers. And an experiment that declares
    `reads_snapshots` but recorded no reads: it opened a store some way
    other than `RunRecord.snapshot_store`, so nothing it read was tracked.

    What it deliberately does NOT accept is "some citation exists". The
    earlier check did, and a run over the 18.06 release completed while
    citing only a 25.06 entry that happened to be in the manifest.
    """
    uncited = record.settle_snapshot_citations()
    if uncited:
        record.finalise("failed")
        raise ProvenanceError(
            f"{name!r} read {', '.join(uncited)}, which the store's manifest "
            f"does not describe. The run produced numbers from extracts it "
            f"cannot identify. Rebuild them with `neorx snapshot build`, "
            f"which writes the manifest entry beside each extract."
        )
    if declared and not record.snapshot_reads:
        record.finalise("failed")
        raise ProvenanceError(
            f"{name!r} declares reads_snapshots but recorded no reads. Open "
            f"stores with `record.snapshot_store(root)` rather than "
            f"constructing SnapshotStore directly, or nothing read is cited."
        )


def run_experiment(
    name: str,
    *,
    allow_large: bool = False,
    runs_dir: Path | None = None,
) -> RunRecord:
    """Execute an experiment, writing its record whatever the outcome.

    ``env.json`` cites exactly the extracts the run read, taken from the
    manifest of the store it read them from.
    """
    defn = get_experiment(name)
    record = RunRecord.create(name, runs_dir=runs_dir, allow_large=allow_large)
    try:
        if defn.captures_http:
            with capture(record, mode="record"):
                defn.fn(record)
        else:
            defn.fn(record)
    except BaseException:
        # Cite what was read before the failure. The original error is the
        # one that propagates; the record still says what the partial rows
        # came from, and lists any read it could not cite.
        #
        # Settlement itself can fail -- a malformed manifest.toml, or an
        # entry carrying a field this version does not know. That must not
        # cost the run record: without finalise() there is no record.json
        # at all, and RunRecord.load would later report a failed run as
        # merely incomplete while the operator saw the manifest error
        # instead of the one that actually stopped the run. So the
        # settlement error is recorded and the original is re-raised.
        try:
            record.settle_snapshot_citations()
        except Exception as settlement_error:
            record.note_settlement_failure(settlement_error)
        record.finalise("failed")
        raise
    settle_or_refuse(record, name, declared=defn.reads_snapshots)
    record.finalise("complete")
    return record
