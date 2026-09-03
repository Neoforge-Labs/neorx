"""Provenance for the local ChEMBL SQLite database.

ChEMBL is ~28 GB and its releases are versioned and immutable, so hashing
the file costs minutes and proves nothing the release string does not.
We record the release, the file size and its mtime.

The release comes from the ``chembl_release`` table. The similarly named
``version`` table is NOT the right source -- it holds ontology versions
such as "Bioassay Ontology 2.0".
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RELEASE_QUERY = (
    "SELECT chembl_release, creation_date "
    "FROM chembl_release ORDER BY chembl_release_id DESC LIMIT 1"
)


class ChEMBLProvenanceError(RuntimeError):
    """The database could not be interrogated for its release."""


class ChEMBLMismatchError(RuntimeError):
    """The database present is a different release from the one recorded."""


def chembl_provenance(db_path: Path) -> dict[str, Any]:
    """Describe the ChEMBL database backing a run."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise ChEMBLProvenanceError(f"no ChEMBL database at {db_path}")

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = conn.execute(_RELEASE_QUERY).fetchone()
    except sqlite3.Error as exc:
        raise ChEMBLProvenanceError(
            f"could not read the chembl_release table from {db_path}: {exc}"
        ) from exc
    finally:
        try:
            conn.close()
        except NameError:
            pass

    if not row:
        raise ChEMBLProvenanceError(
            f"chembl_release table in {db_path} is empty; cannot determine release"
        )

    stat = db_path.stat()
    return {
        "path": str(db_path),
        "release": row[0],
        "release_date": row[1],
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def verify_chembl(recorded: dict[str, Any], db_path: Path) -> None:
    """Raise unless the database at ``db_path`` is the recorded release."""
    current = chembl_provenance(db_path)
    if current["release"] != recorded["release"]:
        raise ChEMBLMismatchError(
            f"run recorded ChEMBL release {recorded['release']!r} but "
            f"{db_path} is {current['release']!r}. Replay would not reproduce "
            f"the recorded rows."
        )
