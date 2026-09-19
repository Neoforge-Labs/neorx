"""A dated run records which extracts produced it -- and only those.

Sub-project 2 captures the code SHA because a number is not reproducible
without the code that made it. A dated run's inputs are equally part of
that, and a derived extract needs its derivation pinned too: two extracts
of the same release built by different extractor code are different
inputs.

What gets cited is what the run READ, through a store opened by its record.
An earlier version cited the whole manifest at creation time, from a path
configured separately from the store the experiment read. Two defects
followed: a run launched from a subdirectory read one store and cited
another's manifest, and a run over 18.06 completed while citing only a
25.06 entry, because the check accepted any citation at all.
"""

import json
from dataclasses import replace

import polars as pl

from neorx.experiments.record import RunRecord
from neorx.snapshots.manifest import SnapshotEntry, digest_file, write_entry
from neorx.snapshots.schema import ASSOCIATION_COLUMNS, INTERACTION_COLUMNS

OT_ENTRY = SnapshotEntry(
    source="opentargets",
    release="18.06",
    url="https://example.invalid/18.06",
    sha256="b" * 64,
    extractor_version=1,
    rows=42,
)
# An OmniPath archive extract, which is where synthesised_consensus is True
# in practice: the archived TSV has no consensus-direction column, so the
# reader derives one from is_directed. That changes what every directed
# edge built from this extract means.
OMNI_ENTRY = SnapshotEntry(
    source="omnipath",
    release="20180614",
    url="https://example.invalid/omnipath-20180614",
    sha256="c" * 64,
    extractor_version=2,
    rows=7,
    synthesised_consensus=True,
)
UNREAD_ENTRY = SnapshotEntry(
    source="opentargets",
    release="25.06",
    url="https://example.invalid/25.06",
    sha256="d" * 64,
    extractor_version=2,
    rows=9,
)


_FILES = {"opentargets": "associations.parquet", "omnipath": "interactions.parquet"}


def _store_root(tmp_path, entries):
    """A real store: extracts on disk, and a manifest describing `entries`.

    Each entry's digest is taken from the extract actually written, because
    a citation is now checked against the bytes the run read. An entry
    whose extract does not exist keeps its literal digest -- it describes
    something this store does not hold, which is the point of those cases.
    """
    root = tmp_path / "snapshots"
    ot = root / "opentargets" / "18.06"
    ot.mkdir(parents=True)
    pl.DataFrame(schema=ASSOCIATION_COLUMNS).write_parquet(ot / "associations.parquet")
    omni = root / "omnipath" / "20180614"
    omni.mkdir(parents=True)
    pl.DataFrame(schema=INTERACTION_COLUMNS).write_parquet(omni / "interactions.parquet")
    for entry in entries:
        extract = root / entry.source / entry.release / _FILES[entry.source]
        if extract.exists():
            entry = replace(entry, sha256=digest_file(extract))
        write_entry(root / "manifest.toml", entry)
    return root


def _digest_of(root, source, release):
    return digest_file(root / source / release / _FILES[source])


def _env(record):
    return json.loads((record.path / "env.json").read_text())


def test_a_new_record_cites_nothing_yet(tmp_path):
    # A citation is a statement about what produced the numbers, so it
    # cannot be written before any were produced.
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    assert _env(record)["snapshots"] == {}


def test_only_what_was_read_is_cited(tmp_path):
    root = _store_root(tmp_path, [OT_ENTRY, OMNI_ENTRY, UNREAD_ENTRY])
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(root).associations("18.06")

    assert record.settle_snapshot_citations() == []
    cited = _env(record)["snapshots"]
    # The manifest also describes omnipath and a 25.06 extract. Neither was
    # read, so neither is cited: over-citing names inputs that did not
    # produce the numbers.
    assert set(cited) == {"opentargets/18.06"}


def test_env_records_every_field_of_the_entry(tmp_path):
    """Not a chosen subset.

    A digest identifies which bytes an extract was; it does not say what
    they meant, and the manifest that would say is replaced in place when
    an extract is rebuilt. So the run record carries the entry, and
    ``source``/``release`` are dropped only because they are the key.
    """
    root = _store_root(tmp_path, [OT_ENTRY, OMNI_ENTRY])
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(root).associations("18.06")
    record.settle_snapshot_citations()

    assert _env(record)["snapshots"]["opentargets/18.06"] == {
        "url": "https://example.invalid/18.06",
        "sha256": _digest_of(root, "opentargets", "18.06"),
        "extractor_version": 1,
        "rows": 42,
        "synthesised_consensus": False,
    }


def test_a_synthesised_consensus_survives_into_the_run_record(tmp_path):
    # The field a subset once lost. An OmniPath extract whose consensus
    # direction was derived rather than read must be legible as such from
    # env.json alone, without going back to a manifest that may since have
    # been overwritten.
    root = _store_root(tmp_path, [OT_ENTRY, OMNI_ENTRY])
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(root).interactions("20180614")
    record.settle_snapshot_citations()

    assert _env(record)["snapshots"]["omnipath/20180614"] == {
        "url": "https://example.invalid/omnipath-20180614",
        "sha256": _digest_of(root, "omnipath", "20180614"),
        "extractor_version": 2,
        "rows": 7,
        "synthesised_consensus": True,
    }


def test_a_read_the_manifest_does_not_describe_is_reported_not_dropped(tmp_path):
    """An unrelated entry does not make a read citable.

    The manifest describes a 25.06 extract only; the run read 18.06. The
    earlier check was satisfied by any citation at all, and a run in
    exactly this state completed. Now the uncited read is returned to the
    caller and written into the record under its own key.
    """
    root = _store_root(tmp_path, [UNREAD_ENTRY])
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(root).associations("18.06")

    assert record.settle_snapshot_citations() == ["opentargets/18.06"]
    env = _env(record)
    assert env["snapshots"] == {}
    assert env["snapshots_uncited"] == ["opentargets/18.06"]


def test_a_store_with_no_manifest_cites_nothing_and_says_what_it_read(tmp_path):
    root = _store_root(tmp_path, [])
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(root).associations("18.06")

    assert record.settle_snapshot_citations() == ["opentargets/18.06"]
    assert _env(record)["snapshots"] == {}


def test_the_cited_manifest_is_the_read_stores_own(tmp_path):
    """The extracts read and the manifest cited cannot come from two places.

    Two stores, same release, different digests. The run reads from the
    second; the citation must carry the second's digest. When the manifest
    path was configured separately from the store, a run from a
    subdirectory read one store and cited another.
    """
    first = _store_root(tmp_path / "a", [OT_ENTRY])
    other = SnapshotEntry(
        source="opentargets",
        release="18.06",
        url="https://example.invalid/other",
        sha256="e" * 64,
        extractor_version=1,
        rows=1,
    )
    second = _store_root(tmp_path / "b", [other])
    assert first != second

    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(second).associations("18.06")
    record.settle_snapshot_citations()
    # The second store's own bytes, not the first store's entry.
    assert _env(record)["snapshots"]["opentargets/18.06"]["url"] == (
        "https://example.invalid/other"
    )


def test_reading_one_release_from_two_stores_is_refused(tmp_path):
    # A citation keyed by source and release cannot say which store
    # produced the numbers, so the ambiguity is refused where it happens.
    import pytest

    from neorx.experiments.record import ProvenanceError

    first = _store_root(tmp_path / "a", [OT_ENTRY])
    second = _store_root(tmp_path / "b", [OT_ENTRY])
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(first).associations("18.06")
    with pytest.raises(ProvenanceError):
        record.snapshot_store(second).associations("18.06")


# ── The bytes, not just the manifest ────────────────────────────────


def test_a_manifest_describing_different_bytes_is_refused(tmp_path):
    """`snapshot build` writes the extract and its entry as two steps.

    An interrupted rebuild of an already-built release leaves the manifest
    describing the old file and the disk holding the new one. Nothing
    checked the two matched, so a run cited a digest of bytes it never
    read -- the defect this layer exists to prevent, arriving through the
    one door still open.
    """
    root = _store_root(tmp_path, [])
    # Written by hand so the digest is deliberately not the file's.
    write_entry(root / "manifest.toml", OT_ENTRY)  # sha256 is "b" * 64
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(root).associations("18.06")

    problems = record.settle_snapshot_citations()
    assert problems == ["opentargets/18.06 (digest mismatch)"]

    env = _env(record)
    assert env["snapshots"] == {}
    (mismatch,) = env["snapshots_mismatched"]
    assert mismatch["extract"] == "opentargets/18.06"
    assert mismatch["manifest_sha256"] == "b" * 64
    # And it says what was actually there, so the operator can tell a
    # stale manifest from a corrupted extract.
    assert mismatch["actual_sha256"] != "b" * 64
    assert len(mismatch["actual_sha256"]) == 64


def test_a_matching_digest_is_cited(tmp_path):
    """The check must not refuse a correct store.

    The digest is taken over the extract on disk, so the entry is written
    from that same file rather than from a literal.
    """
    from neorx.snapshots.manifest import SnapshotEntry, digest_file, write_entry

    root = _store_root(tmp_path, [])
    extract = root / "opentargets" / "18.06" / "associations.parquet"
    write_entry(
        root / "manifest.toml",
        SnapshotEntry(
            source="opentargets",
            release="18.06",
            url="https://example.invalid/18.06",
            sha256=digest_file(extract),
            extractor_version=2,
            rows=0,
        ),
    )
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    record.snapshot_store(root).associations("18.06")

    assert record.settle_snapshot_citations() == []
    assert set(_env(record)["snapshots"]) == {"opentargets/18.06"}
    assert _env(record)["snapshots_mismatched"] == []
