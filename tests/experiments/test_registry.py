"""The registry is what makes the runner, not the script, own persistence."""

import pytest

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import (
    UnknownExperimentError,
    experiment,
    get_experiment,
    list_experiments,
    run_experiment,
)


def test_decorator_registers_by_name():
    @experiment(name="unit-demo", help="a demo")
    def _demo(record: RunRecord) -> None:
        record.append_row({"x": 1})

    assert get_experiment("unit-demo").help == "a demo"
    assert "unit-demo" in [e.name for e in list_experiments()]


def test_unknown_experiment_raises_with_the_known_names():
    with pytest.raises(UnknownExperimentError, match="unit-demo|available"):
        get_experiment("no-such-experiment")


def test_run_experiment_writes_a_finalised_record(tmp_path):
    @experiment(name="unit-writes", help="")
    def _w(record: RunRecord) -> None:
        record.append_row({"disease": "HIV", "f1": 0.5})

    rec = run_experiment("unit-writes", runs_dir=tmp_path)
    assert rec.status == "complete"
    assert len(rec.rows()) == 1


def test_a_raising_experiment_still_leaves_a_record(tmp_path):
    """A failed run is an artifact, not an absence."""

    @experiment(name="unit-raises", help="")
    def _r(record: RunRecord) -> None:
        record.append_row({"disease": "HIV", "f1": 0.5})
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_experiment("unit-raises", runs_dir=tmp_path)

    written = sorted(tmp_path.iterdir())
    assert len(written) == 1
    rec = RunRecord.load(written[0].name, runs_dir=tmp_path)
    assert rec.status == "failed"
    assert len(rec.rows()) == 1, "rows completed before the failure must survive"
    assert rec.citable is False
