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
from neorx.snapshots.schema import (
    ASSOCIATION_COLUMNS,
    INTERACTION_COLUMNS,
    empty_associations,
    empty_interactions,
)


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


def test_interactions_reads_back_what_was_written(tmp_path):
    d = tmp_path / "omnipath" / "2024-01-15"
    d.mkdir(parents=True)
    df = pl.DataFrame(
        [{
            "source_symbol": "EGFR",
            "target_symbol": "TP53",
            "is_directed": True,
            "consensus_direction": True,
            "is_stimulation": False,
            "is_inhibition": True,
            "primary_sources": "SIGNOR",
            "references": "PMID:12345",
        }],
        schema=INTERACTION_COLUMNS,
    )
    df.write_parquet(d / "interactions.parquet")
    store = SnapshotStore(tmp_path)
    result = store.interactions("2024-01-15")
    assert result.height == 1
    assert result.row(0, named=True)["source_symbol"] == "EGFR"


def test_interactions_schema_matches_canonical(tmp_path):
    d = tmp_path / "omnipath" / "2024-01-15"
    d.mkdir(parents=True)
    empty_interactions().write_parquet(d / "interactions.parquet")
    store = SnapshotStore(tmp_path)
    assert dict(store.interactions("2024-01-15").schema) == INTERACTION_COLUMNS


def test_missing_interactions_raises_and_names_source_and_release(tmp_path):
    d = tmp_path / "omnipath" / "2024-01-15"
    d.mkdir(parents=True)
    empty_interactions().write_parquet(d / "interactions.parquet")
    store = SnapshotStore(tmp_path)
    with pytest.raises(SnapshotMissingError, match="2024-02-01"):
        store.interactions("2024-02-01")
    with pytest.raises(SnapshotMissingError, match="omnipath"):
        store.interactions("2024-02-01")


def test_dispatch_to_interactions_not_associations(tmp_path):
    """Verify that interactions() dispatch resolves to omnipath, not opentargets.

    If _FILES["omnipath"] were incorrectly mapped to "associations.parquet",
    this test would fail because an opentargets extract exists but should not
    be visible to the omnipath source.
    """
    # Create only an opentargets extract in the same release
    ot_d = tmp_path / "opentargets" / "2024-01-15"
    ot_d.mkdir(parents=True)
    ot_df = pl.DataFrame(
        [{
            "target_id": "ENSG1",
            "target_symbol": "EGFR",
            "disease_id": "EFO_1",
            "datatype": "genetic_association",
            "score": 0.5,
        }],
        schema=ASSOCIATION_COLUMNS,
    )
    ot_df.write_parquet(ot_d / "associations.parquet")

    store = SnapshotStore(tmp_path)
    # Omnipath should not be found for this release
    assert store.has("omnipath", "2024-01-15") is False
    # And interactions() should raise, not return the associations
    with pytest.raises(SnapshotMissingError):
        store.interactions("2024-01-15")
