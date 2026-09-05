"""Reading derived extracts back.

This layer knows the on-disk layout and nothing else -- by the time a
frame reaches it, which upstream era produced it is invisible. A missing
snapshot raises a named error rather than returning an empty frame,
because an empty frame reads downstream as "this release had no evidence"
rather than "this release was never extracted".
"""

import polars as pl
import pytest

from neorx.snapshots.reader import SnapshotMissingError, SnapshotStore
from neorx.snapshots.schema import ASSOCIATION_COLUMNS, empty_associations


def _store(tmp_path, release="18.06", rows=None):
    d = tmp_path / "opentargets" / release
    d.mkdir(parents=True)
    df = pl.DataFrame(rows or [{
        "target_id": "ENSG1", "target_symbol": "PIK3CA",
        "disease_id": "EFO_1", "datatype": "genetic_association", "score": 0.6,
    }], schema=ASSOCIATION_COLUMNS)
    df.write_parquet(d / "associations.parquet")
    return SnapshotStore(tmp_path)


def test_reads_back_what_was_written(tmp_path):
    store = _store(tmp_path)
    df = store.associations("18.06")
    assert df.height == 1
    assert df.row(0, named=True)["target_symbol"] == "PIK3CA"


def test_read_back_matches_the_canonical_schema(tmp_path):
    assert dict(_store(tmp_path).associations("18.06").schema) == ASSOCIATION_COLUMNS


def test_a_missing_release_raises_and_names_it(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(SnapshotMissingError, match="21.11"):
        store.associations("21.11")


def test_the_error_names_the_source_too(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(SnapshotMissingError, match="opentargets"):
        store.associations("21.11")


def test_has_reports_presence_without_raising(tmp_path):
    store = _store(tmp_path)
    assert store.has("opentargets", "18.06") is True
    assert store.has("opentargets", "21.11") is False
    assert store.has("omnipath", "18.06") is False


def test_an_empty_extract_is_distinguishable_from_a_missing_one(tmp_path):
    d = tmp_path / "opentargets" / "99.99"
    d.mkdir(parents=True)
    empty_associations().write_parquet(d / "associations.parquet")
    store = SnapshotStore(tmp_path)
    assert store.has("opentargets", "99.99") is True
    assert store.associations("99.99").height == 0
