"""RunRecord: incremental rows, honest env capture, explicit status."""

import json
import subprocess
from pathlib import Path

import pytest

from neorx.experiments.record import (
    MAX_RECORD_BYTES,
    RecordTooLargeError,
    RunRecord,
)


def test_run_id_shape(tmp_path):
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    parts = rec.run_id.split("-")
    assert len(parts) >= 5, rec.run_id
    assert parts[0].isdigit() and len(parts[0]) == 4      # year
    assert "demo-exp" in rec.run_id
    assert len(parts[-1]) == 6                             # short hash


def test_rows_are_written_as_they_are_appended(tmp_path):
    """The whole point: a crash must not cost completed cells."""
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.append_row({"disease": "HIV", "f1": 0.545})
    on_disk = (rec.path / "rows.jsonl").read_text().strip().splitlines()
    assert len(on_disk) == 1, "row must hit disk before the run ends"
    rec.append_row({"disease": "malaria", "f1": 0.333})
    on_disk = (rec.path / "rows.jsonl").read_text().strip().splitlines()
    assert len(on_disk) == 2
    assert json.loads(on_disk[0])["disease"] == "HIV"


def test_unfinalised_record_is_incomplete_not_missing(tmp_path):
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.append_row({"disease": "HIV", "f1": 0.545})
    # simulate a crash: never call finalise
    reloaded = RunRecord.load(rec.run_id, runs_dir=tmp_path)
    assert reloaded.status == "incomplete"
    assert len(reloaded.rows()) == 1
    assert reloaded.citable is False


def test_finalise_sets_status_and_completes(tmp_path):
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.append_row({"disease": "HIV", "f1": 0.545})
    rec.finalise("complete")
    data = json.loads((rec.path / "record.json").read_text())
    assert data["status"] == "complete"
    assert data["n_rows"] == 1


def test_env_records_code_sha_and_dirty_flag(tmp_path):
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    env = json.loads((rec.path / "env.json").read_text())
    assert len(env["git_sha"]) == 40
    assert isinstance(env["git_dirty"], bool)
    assert env["python"].startswith("3.")
    assert "platform" in env


def test_dirty_tree_makes_run_non_citable(tmp_path, monkeypatch):
    """A run nobody can reproduce must not be citable."""
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod, "_git_dirty", lambda: True)
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.finalise("complete")
    assert rec.citable is False
    assert json.loads((rec.path / "record.json").read_text())["citable"] is False


def test_clean_tree_complete_run_is_citable(tmp_path, monkeypatch):
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod, "_git_dirty", lambda: False)
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.finalise("complete")
    assert rec.citable is True


def test_oversized_record_is_refused(tmp_path, monkeypatch):
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod, "MAX_RECORD_BYTES", 100)
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    (rec.path / "inputs").mkdir(exist_ok=True)
    (rec.path / "inputs" / "big.json").write_bytes(b"x" * 500)
    with pytest.raises(RecordTooLargeError, match="50 MB|allow-large|exceeds"):
        rec.finalise("complete")


def test_allow_large_bypasses_the_refusal(tmp_path, monkeypatch):
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod, "MAX_RECORD_BYTES", 100)
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path, allow_large=True)
    (rec.path / "inputs").mkdir(exist_ok=True)
    (rec.path / "inputs" / "big.json").write_bytes(b"x" * 500)
    rec.finalise("complete")  # must not raise
