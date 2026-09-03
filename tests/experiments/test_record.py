"""RunRecord: incremental rows, honest env capture, explicit status."""

import json
import subprocess
from pathlib import Path

import pytest

from neorx.experiments.record import (
    MAX_RECORD_BYTES,
    ProvenanceError,
    RecordTooLargeError,
    RunIDCollisionError,
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


def _fake_git_run(*, dirty: bool, sha: str = "a" * 40):
    """Build a subprocess.run stand-in for git rev-parse / git status.

    Patching subprocess.run (rather than _git_dirty itself) exercises the
    real code path -- the one that produced C2, where a failed git
    invocation used to be silently swallowed into a fake SHA or a false
    "clean" reading.
    """

    def fake_run(argv, capture_output=True, text=True, check=False):
        assert argv[0] == "git"
        if argv[1] == "rev-parse":
            return subprocess.CompletedProcess(argv, 0, stdout=sha + "\n", stderr="")
        if argv[1] == "status":
            stdout = "?? some_dirty_file.py\n" if dirty else ""
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        raise AssertionError(f"unexpected git invocation: {argv}")

    return fake_run


def test_dirty_tree_makes_run_non_citable(tmp_path, monkeypatch):
    """A run nobody can reproduce must not be citable."""
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod.subprocess, "run", _fake_git_run(dirty=True))
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.finalise("complete")
    assert rec.citable is False
    assert json.loads((rec.path / "record.json").read_text())["citable"] is False


def test_clean_tree_complete_run_is_citable(tmp_path, monkeypatch):
    import neorx.experiments.record as mod

    monkeypatch.setattr(mod.subprocess, "run", _fake_git_run(dirty=False))
    rec = RunRecord.create("demo-exp", runs_dir=tmp_path)
    rec.finalise("complete")
    assert rec.citable is True


def test_git_failure_raises_provenance_error_not_a_fake_sha(tmp_path, monkeypatch):
    """A git invocation that fails must fail the run, not fabricate a SHA

    or a false "clean" reading (C2). This is the failure path the two
    tests above never touched when they monkeypatched _git_dirty directly.
    """
    import neorx.experiments.record as mod

    def fake_run(argv, capture_output=True, text=True, check=False):
        return subprocess.CompletedProcess(
            argv, 128, stdout="", stderr="fatal: not a git repository"
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(ProvenanceError, match="not a git repository"):
        RunRecord.create("demo-exp", runs_dir=tmp_path)

    # nothing should have been fabricated or left half-written
    assert list(tmp_path.iterdir()) == []


def test_real_run_on_default_runs_dir_is_citable_on_clean_tree(tmp_path, monkeypatch):
    """C1: citability must be reachable in real use, not just in a tmp_path

    test that never touches RUNS_DIR. Builds an isolated git repo (this
    worktree has pre-existing untracked scratch files that would otherwise
    contaminate the check), points the module's default RUNS_DIR at a
    runs/ directory inside it, and runs a real subprocess-backed git
    check end to end.
    """
    import neorx.experiments.record as mod

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "source.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    monkeypatch.chdir(repo)
    monkeypatch.setattr(mod, "RUNS_DIR", repo / "runs")

    rec = RunRecord.create("demo-exp")  # default runs_dir -> patched RUNS_DIR
    rec.append_row({"disease": "HIV", "f1": 0.545})
    rec.finalise("complete")

    assert rec.citable is True, "runs/ itself being untracked must not poison citability"


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


def test_colliding_run_id_raises_instead_of_corrupting(tmp_path, monkeypatch):
    """I2: a colliding run ID must fail loudly, not write into the

    existing run's directory and corrupt both records with no signal.
    """
    import datetime as dt_module

    import neorx.experiments.record as mod

    fixed = dt_module.datetime(2024, 1, 1, tzinfo=dt_module.timezone.utc)

    class FixedDatetime(dt_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(mod, "datetime", FixedDatetime)
    monkeypatch.setattr(mod, "_git_sha", lambda: "a" * 40)

    first = RunRecord.create("demo-exp", runs_dir=tmp_path)
    first.append_row({"disease": "HIV", "f1": 0.545})

    with pytest.raises(RunIDCollisionError, match=first.run_id):
        RunRecord.create("demo-exp", runs_dir=tmp_path)

    # the original run's data must be untouched
    on_disk = (first.path / "rows.jsonl").read_text().strip().splitlines()
    assert len(on_disk) == 1
