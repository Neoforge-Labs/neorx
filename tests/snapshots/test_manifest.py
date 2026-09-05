"""What a snapshot records about itself.

A release id alone does not identify a derived extract: two extracts of
OpenTargets 18.06 produced by different extractor code are different
inputs to every downstream number. So the extractor version is recorded
beside the release, and the digest is of the extract we actually wrote.
"""

import pytest

from neorx.snapshots.manifest import (
    SnapshotEntry,
    digest_file,
    read_manifest,
    write_entry,
)
from neorx.snapshots.schema import EXTRACTOR_VERSION


def _entry(source="opentargets", release="18.06", rows=100):
    return SnapshotEntry(
        source=source, release=release,
        url=f"https://example.invalid/{release}",
        sha256="a" * 64, extractor_version=EXTRACTOR_VERSION, rows=rows,
    )


def test_an_entry_round_trips(tmp_path):
    m = tmp_path / "manifest.toml"
    write_entry(m, _entry())
    got = read_manifest(m)[("opentargets", "18.06")]
    assert got == _entry()


def test_entries_from_different_sources_coexist(tmp_path):
    m = tmp_path / "manifest.toml"
    write_entry(m, _entry())
    write_entry(m, _entry(source="omnipath", release="20180614"))
    assert set(read_manifest(m)) == {
        ("opentargets", "18.06"), ("omnipath", "20180614"),
    }


def test_rewriting_the_same_key_replaces_rather_than_duplicates(tmp_path):
    m = tmp_path / "manifest.toml"
    write_entry(m, _entry(rows=100))
    write_entry(m, _entry(rows=250))
    entries = read_manifest(m)
    assert len(entries) == 1
    assert entries[("opentargets", "18.06")].rows == 250


def test_reading_an_absent_manifest_yields_no_entries(tmp_path):
    assert read_manifest(tmp_path / "nope.toml") == {}


def test_digest_is_stable_and_content_dependent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"snapshot"); b.write_bytes(b"snapshot")
    assert digest_file(a) == digest_file(b)
    b.write_bytes(b"different")
    assert digest_file(a) != digest_file(b)


def test_entry_is_immutable():
    with pytest.raises(Exception):
        _entry().rows = 1


def test_synthesised_consensus_round_trips_with_true(tmp_path):
    """An entry with synthesised_consensus=True must round-trip as True."""
    m = tmp_path / "manifest.toml"
    entry_with_true = SnapshotEntry(
        source="omnipath", release="20180614",
        url="https://example.invalid/20180614",
        sha256="b" * 64, extractor_version=EXTRACTOR_VERSION, rows=200,
        synthesised_consensus=True,
    )
    write_entry(m, entry_with_true)
    got = read_manifest(m)[("omnipath", "20180614")]
    assert got.synthesised_consensus is True


def test_synthesised_consensus_defaults_to_false(tmp_path):
    """An entry written without synthesised_consensus must read back as False."""
    m = tmp_path / "manifest.toml"
    write_entry(m, _entry())
    got = read_manifest(m)[("opentargets", "18.06")]
    assert got.synthesised_consensus is False
