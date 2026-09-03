"""ChEMBL is a 28 GB immutable release -- record its version, not its bytes."""

import sqlite3

import pytest

from neorx.experiments.chembl import (
    ChEMBLMismatchError,
    ChEMBLProvenanceError,
    chembl_provenance,
    verify_chembl,
)


def _make_db(path, release="CHEMBL_36", date="2025-07-28 00:00:00.000000"):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE chembl_release "
        "(chembl_release_id INTEGER, chembl_release TEXT, creation_date TEXT)"
    )
    conn.execute(
        "INSERT INTO chembl_release VALUES (35, 'CHEMBL_35', '2024-12-01 00:00:00.000000')"
    )
    conn.execute("INSERT INTO chembl_release VALUES (36, ?, ?)", (release, date))
    conn.commit()
    conn.close()
    return path


def test_reads_the_latest_release_not_the_first(tmp_path):
    db = _make_db(tmp_path / "chembl.db")
    prov = chembl_provenance(db)
    assert prov["release"] == "CHEMBL_36"
    assert prov["release_date"].startswith("2025-07-28")
    assert prov["size_bytes"] > 0


def test_matching_release_verifies(tmp_path):
    db = _make_db(tmp_path / "chembl.db")
    verify_chembl(chembl_provenance(db), db)  # must not raise


def test_a_different_release_is_refused_by_name(tmp_path):
    db = _make_db(tmp_path / "chembl.db")
    recorded = chembl_provenance(db)
    other = _make_db(tmp_path / "other.db", release="CHEMBL_35")
    with pytest.raises(ChEMBLMismatchError, match="CHEMBL_36.*CHEMBL_35|CHEMBL_35"):
        verify_chembl(recorded, other)


def test_a_database_without_the_release_table_fails_loudly(tmp_path):
    """Never silently record 'no version'."""
    path = tmp_path / "bare.db"
    sqlite3.connect(path).execute("CREATE TABLE t (x INTEGER)").connection.commit()
    with pytest.raises(ChEMBLProvenanceError, match="chembl_release"):
        chembl_provenance(path)


def test_the_ontology_version_table_is_not_used(tmp_path):
    """`version` holds ontology versions, not the ChEMBL release."""
    db = _make_db(tmp_path / "chembl.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE version (name TEXT, creation_date TEXT, comments TEXT)")
    conn.execute("INSERT INTO version VALUES ('Bioassay Ontology 2.0', NULL, 'BAO')")
    conn.commit()
    conn.close()
    assert chembl_provenance(db)["release"] == "CHEMBL_36"
