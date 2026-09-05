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
    # An OmniPath archive extract, which is where synthesised_consensus is
    # True in practice: the archived TSV has no consensus-direction column,
    # so the reader derives one from is_directed. That changes what every
    # directed edge built from this extract means.
    write_entry(
        m,
        SnapshotEntry(
            source="omnipath",
            release="20180614",
            url="https://example.invalid/omnipath-20180614",
            sha256="c" * 64,
            extractor_version=2,
            rows=7,
            synthesised_consensus=True,
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


def test_env_records_every_field_of_the_entry(tmp_path):
    """Not a chosen subset.

    A digest identifies which bytes an extract was; it does not say what
    they meant, and the manifest that would say is replaced in place when
    an extract is rebuilt. So the run record carries the entry, and
    ``source``/``release`` are dropped only because they are the key.
    """
    record = RunRecord.create(
        "census",
        runs_dir=tmp_path / "runs",
        snapshot_manifest=_manifest(tmp_path),
    )
    env = json.loads((record.path / "env.json").read_text())
    assert env["snapshots"]["opentargets/18.06"] == {
        "url": "https://example.invalid/18.06",
        "sha256": "b" * 64,
        "extractor_version": 1,
        "rows": 42,
        "synthesised_consensus": False,
    }


def test_a_synthesised_consensus_survives_into_the_run_record(tmp_path):
    # The field the subset was losing. An OmniPath extract whose consensus
    # direction was derived rather than read must be legible as such from
    # env.json alone, without going back to a manifest that may since have
    # been overwritten.
    record = RunRecord.create(
        "census",
        runs_dir=tmp_path / "runs",
        snapshot_manifest=_manifest(tmp_path),
    )
    env = json.loads((record.path / "env.json").read_text())
    assert env["snapshots"]["omnipath/20180614"] == {
        "url": "https://example.invalid/omnipath-20180614",
        "sha256": "c" * 64,
        "extractor_version": 2,
        "rows": 7,
        "synthesised_consensus": True,
    }


def test_an_undated_run_records_no_snapshots(tmp_path):
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    env = json.loads((record.path / "env.json").read_text())
    assert env["snapshots"] == {}


def test_a_manifest_that_does_not_exist_records_no_snapshots(tmp_path):
    record = RunRecord.create(
        "census",
        runs_dir=tmp_path / "runs",
        snapshot_manifest=tmp_path / "absent.toml",
    )
    env = json.loads((record.path / "env.json").read_text())
    assert env["snapshots"] == {}
