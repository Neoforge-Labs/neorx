"""Replay is the property the whole subsystem claims. Test it directly."""

import json
import socket

import pytest
import requests

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment, run_experiment
from neorx.experiments.replay import (
    NotReplayableError,
    prune_record,
    replay_experiment,
)

HTTPBIN_URL = "https://httpbin.org/json"


@experiment(name="replay-fixture", help="")
def _fixture(record: RunRecord) -> None:
    """Deterministic and offline, so any diff is the replay machinery's fault."""
    for i, disease in enumerate(["HIV", "malaria"]):
        record.append_row({"disease": disease, "F1": 0.5 + i / 10})


@experiment(name="replay-timing-fixture", help="", volatile_fields=("wall_clock_s",))
def _timing_fixture(record: RunRecord) -> None:
    """Deterministic except for a timing field, like the three real,

    migrated experiments (neorx-7disease, causalbiorl-bench, genmol-eval)
    which all write wall-clock timing fields on every row. Without
    volatile_fields, this run could never replay as IDENTICAL -- exactly
    the defect under test.
    """
    import time

    for i, disease in enumerate(["HIV", "malaria"]):
        record.append_row(
            {"disease": disease, "F1": 0.5 + i / 10, "wall_clock_s": time.perf_counter()}
        )


@experiment(name="replay-http-fixture", help="", captures_http=True)
def _http_fixture(record: RunRecord) -> None:
    """Declares captures_http=True, like neorx-7disease -- the fix under test.

    Deliberately does NOT open its own capture context (the pre-fix bug in
    neorx_7disease.py was exactly that: an experiment opening its own inner
    ``capture(..., mode="record")``, which won by nesting, so the runner's
    outer replay-mode capture was shadowed and replay reissued live HTTP
    calls). This fixture relies entirely on the runner to own capture, the
    behaviour the fix establishes.
    """
    body = requests.get(HTTPBIN_URL, timeout=30).json()
    record.append_row({"disease": "HIV", "slideshow_title": body["slideshow"]["title"]})


def test_replay_of_a_deterministic_run_is_identical(tmp_path):
    original = run_experiment("replay-fixture", runs_dir=tmp_path)
    result = replay_experiment(original.run_id, runs_dir=tmp_path)
    assert result.identical, result.diffs
    assert result.diffs == []
    assert result.replay_run_id != original.run_id


def test_a_run_with_a_timing_field_still_replays_as_identical(tmp_path):
    """The headline claim, exercised against a fixture that actually has a

    timing field -- like every one of the three migrated experiments does.
    Before volatile_fields, this replay would always report DIFFERS.
    """
    original = run_experiment("replay-timing-fixture", runs_dir=tmp_path)
    result = replay_experiment(original.run_id, runs_dir=tmp_path)

    assert result.identical, result.diffs
    assert result.diffs == []
    # The timing delta is real and must not be hidden -- it is reported
    # separately, just excluded from the identical/differs verdict.
    assert len(result.volatile_diffs) == 2
    assert {d.key for d in result.volatile_diffs} == {"wall_clock_s"}


def test_a_changed_row_is_reported_by_field(tmp_path):
    original = run_experiment("replay-fixture", runs_dir=tmp_path)
    rows = original.rows()
    rows[1]["F1"] = 0.999
    (original.path / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    )

    result = replay_experiment(original.run_id, runs_dir=tmp_path)
    assert not result.identical
    assert len(result.diffs) == 1
    diff = result.diffs[0]
    assert diff.index == 1 and diff.key == "F1"
    assert diff.recorded == 0.999 and diff.replayed == 0.6


@pytest.mark.network
def test_replay_of_a_captures_http_experiment_serves_the_cassette_not_the_network(
    tmp_path, monkeypatch
):
    """The defect: replay must reproduce a captures_http=True run from its
    cassette, never re-issue live HTTP calls.

    After recording, the real network path (socket.create_connection, which
    sits below vcrpy's http.client-level patching) is monkeypatched to
    blow up. If replay correctly stays inside the cassette, that patched
    function is never called and the socket layer is never touched. If the
    runner fails to wrap the replay in ``capture(mode="replay")`` -- the
    exact defect this task closes -- the request escapes to the real
    network and this test fails loudly instead of silently passing.
    """
    original = run_experiment("replay-http-fixture", runs_dir=tmp_path)
    assert original.rows()[0]["slideshow_title"]

    def _network_forbidden(*args, **kwargs):
        raise AssertionError("replay reached the real network instead of the cassette")

    monkeypatch.setattr(socket, "create_connection", _network_forbidden)

    result = replay_experiment(original.run_id, runs_dir=tmp_path)

    assert result.identical, result.diffs
    assert result.diffs == []


def test_replaying_a_pruned_run_refuses(tmp_path):
    original = run_experiment("replay-fixture", runs_dir=tmp_path)
    (original.path / "inputs" / "http.yaml").write_text("interactions: []\n")
    prune_record(original.run_id, runs_dir=tmp_path)
    with pytest.raises(NotReplayableError, match="pruned|inputs"):
        replay_experiment(original.run_id, runs_dir=tmp_path)


def test_prune_keeps_the_numbers_and_drops_only_the_inputs(tmp_path):
    original = run_experiment("replay-fixture", runs_dir=tmp_path)
    (original.path / "inputs" / "http.yaml").write_text("x" * 4096)

    freed = prune_record(original.run_id, runs_dir=tmp_path)
    assert freed >= 4096
    assert not (original.path / "inputs").exists()
    assert (original.path / "rows.jsonl").exists()

    reloaded = RunRecord.load(original.run_id, runs_dir=tmp_path)
    assert len(reloaded.rows()) == 2, "pruning must never destroy results"
    summary = json.loads((original.path / "record.json").read_text())
    assert summary["replayable"] is False


# ── A replay must cite its extracts, or refuse ──────────────────────
#
# run_experiment refuses a run that reads pinned extracts and cannot name
# them; replay_experiment did not, so replaying a dated run produced a
# record whose env.json said "snapshots": {} while reporting
# identical: True over rows derived from those extracts. That is worse
# than a run that cannot cite: it asserts agreement between two sets of
# numbers whose inputs it cannot identify.


def test_replaying_a_snapshot_reading_experiment_without_a_manifest_is_refused(
    tmp_path, monkeypatch
):
    import neorx.experiments.replay as replay_mod
    from neorx.experiments.record import ProvenanceError, RunRecord
    from neorx.experiments.registry import ExperimentDef

    record = RunRecord.create("citer", runs_dir=tmp_path / "runs")
    record.append_row({"n": 1})
    record.finalise("complete")

    defn = ExperimentDef(
        name="citer",
        help="",
        fn=lambda rec: rec.append_row({"n": 1}),
        reads_snapshots=True,
    )
    monkeypatch.setattr(replay_mod, "get_experiment", lambda _name: defn)
    # Point the manifest somewhere that does not exist, which is exactly
    # the case that used to pass silently.
    monkeypatch.setattr(
        replay_mod, "SNAPSHOT_MANIFEST", tmp_path / "absent" / "manifest.toml"
    )

    with pytest.raises(ProvenanceError) as excinfo:
        replay_mod.replay_experiment(record.run_id, runs_dir=tmp_path / "runs")
    assert "reads_snapshots" in str(excinfo.value)


def test_replaying_an_experiment_that_reads_no_snapshots_is_unaffected(
    tmp_path, monkeypatch
):
    # The guard is about experiments that declare the need. Everything
    # else replays exactly as before.
    import neorx.experiments.replay as replay_mod
    from neorx.experiments.record import RunRecord
    from neorx.experiments.registry import ExperimentDef

    record = RunRecord.create("plain", runs_dir=tmp_path / "runs")
    record.append_row({"n": 1})
    record.finalise("complete")

    defn = ExperimentDef(
        name="plain",
        help="",
        fn=lambda rec: rec.append_row({"n": 1}),
        reads_snapshots=False,
    )
    monkeypatch.setattr(replay_mod, "get_experiment", lambda _name: defn)
    result = replay_mod.replay_experiment(record.run_id, runs_dir=tmp_path / "runs")
    assert result is not None
