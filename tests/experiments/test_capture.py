"""Record once, replay exactly. This is the property the subsystem claims."""

import json

import pytest
import requests

from neorx.experiments.capture import (
    NoInteractionsRecordedError,
    capture,
    cassette_path,
    interaction_count,
)
from neorx.experiments.record import RunRecord

URL = "https://httpbin.org/json"


@pytest.mark.network
def test_record_then_replay_reproduces_the_same_bytes(tmp_path):
    rec = RunRecord.create("cap-demo", runs_dir=tmp_path)
    with capture(rec, mode="record"):
        live = requests.get(URL, timeout=30).text
    assert cassette_path(rec).exists()
    assert interaction_count(rec) == 1

    with capture(rec, mode="replay"):
        replayed = requests.get(URL, timeout=30).text
    assert replayed == live


@pytest.mark.network
def test_replay_does_not_reach_the_network(tmp_path):
    """Replay must serve the cassette, not re-fetch."""
    rec = RunRecord.create("cap-offline", runs_dir=tmp_path)
    with capture(rec, mode="record"):
        requests.get(URL, timeout=30)

    with capture(rec, mode="replay"):
        # A URL absent from the cassette must fail rather than hit the network.
        with pytest.raises(Exception):
            requests.get("https://httpbin.org/uuid", timeout=30)


def test_a_recording_that_captured_nothing_is_an_error(tmp_path):
    """Silence here means the sources bypassed the recorder."""
    rec = RunRecord.create("cap-empty", runs_dir=tmp_path)
    with pytest.raises(NoInteractionsRecordedError, match="no HTTP interactions"):
        with capture(rec, mode="record"):
            pass  # make no requests at all
