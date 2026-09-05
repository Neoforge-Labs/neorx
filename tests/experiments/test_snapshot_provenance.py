"""A dated run records which extracts produced it.

Sub-project 2 captures the code SHA because a number is not reproducible
without the code that made it. A dated run's inputs are equally part of
that, and a derived extract needs its derivation pinned too: two extracts
of the same release built by different extractor code are different
inputs.
"""

import json

from neorx.experiments.record import RunRecord
from neorx.snapshots.manifest import SnapshotEntry, write_entry


def _manifest(tmp_path):
    m = tmp_path / "manifest.toml"
    write_entry(
        m,
        SnapshotEntry(
            source="opentargets",
            release="18.06",
            url="https://example.invalid/18.06",
            sha256="b" * 64,
            extractor_version=1,
            rows=42,
        ),
    )
    return m


def test_env_records_the_snapshots_a_dated_run_used(tmp_path):
    record = RunRecord.create(
        "census",
        runs_dir=tmp_path / "runs",
        snapshot_manifest=_manifest(tmp_path),
    )
    env = json.loads((record.path / "env.json").read_text())
    assert env["snapshots"]["opentargets/18.06"]["sha256"] == "b" * 64
    assert env["snapshots"]["opentargets/18.06"]["extractor_version"] == 1


def test_an_undated_run_records_no_snapshots(tmp_path):
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    env = json.loads((record.path / "env.json").read_text())
    assert env.get("snapshots", {}) == {}
