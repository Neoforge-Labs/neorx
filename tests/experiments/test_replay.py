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
