"""Experiment registry.

An experiment is a function taking a RunRecord. It never opens a file and
never decides where results go -- the runner does both. That is what makes
"a number cannot exist without a record" structural rather than a habit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
        raise UnknownExperimentError(f"no experiment named {name!r}. Available: {known}") from None


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
