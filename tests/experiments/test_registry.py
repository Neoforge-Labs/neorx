"""The registry is what makes the runner, not the script, own persistence."""

import pytest
import requests

from neorx.experiments.capture import cassette_path
from neorx.experiments.record import RunRecord
from neorx.experiments.registry import (
    UnknownExperimentError,
    experiment,
    get_experiment,
    list_experiments,
    run_experiment,
)

HTTPBIN_URL = "https://httpbin.org/json"


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


def test_captures_http_false_is_not_wrapped_in_a_capture_context(tmp_path):
    """An offline experiment must never be able to trip NoInteractionsRecordedError.

    If ``run_experiment`` wrapped every experiment in a capture context
    regardless of declaration, an offline experiment that makes zero HTTP
    calls would blow up on completion. It must not be wrapped at all.
    """

    @experiment(name="unit-offline", help="", captures_http=False)
    def _offline(record: RunRecord) -> None:
        record.append_row({"disease": "HIV", "f1": 0.5})

    rec = run_experiment("unit-offline", runs_dir=tmp_path)
    assert rec.status == "complete"
    assert not cassette_path(rec).exists(), "an undeclared experiment must not open a cassette"


@pytest.mark.network
def test_captures_http_true_wraps_the_run_in_record_mode(tmp_path):
    @experiment(name="unit-http-capture", help="", captures_http=True)
    def _cap(record: RunRecord) -> None:
        requests.get(HTTPBIN_URL, timeout=30)
        record.append_row({"disease": "HIV", "f1": 0.5})

    rec = run_experiment("unit-http-capture", runs_dir=tmp_path)
    assert rec.status == "complete"
    assert cassette_path(rec).exists(), "a declared experiment must have its traffic recorded"


@pytest.mark.network
def test_a_raising_captures_http_experiment_still_leaves_a_record_with_rows(tmp_path):
    """The capture context must not swallow the exception or block finalisation.

    Ordering matters: the experiment's exception has to propagate through
    the capture context and reach run_experiment's except clause so that
    finalise("failed") still runs and the rows recorded before the raise
    still land on disk -- and the cassette is still written.
    """

    @experiment(name="unit-raises-http", help="", captures_http=True)
    def _r(record: RunRecord) -> None:
        requests.get(HTTPBIN_URL, timeout=30)
        record.append_row({"disease": "HIV", "f1": 0.5})
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_experiment("unit-raises-http", runs_dir=tmp_path)

    written = sorted(tmp_path.iterdir())
    assert len(written) == 1
    rec = RunRecord.load(written[0].name, runs_dir=tmp_path)
    assert rec.status == "failed"
    assert len(rec.rows()) == 1, "rows completed before the failure must survive"
