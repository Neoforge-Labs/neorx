"""Experiment runner: a number cannot exist without a run record.

The runner owns persistence. Experiments are declared in the registry and
executed by the runner, which writes the record itself -- there is no code
path by which an experiment reports a result without one.
"""

from neorx.experiments.record import (
    MAX_RECORD_BYTES,
    RUNS_DIR,
    RecordTooLargeError,
    RunRecord,
)

__all__ = [
    "RunRecord",
    "RecordTooLargeError",
    "RUNS_DIR",
    "MAX_RECORD_BYTES",
]
