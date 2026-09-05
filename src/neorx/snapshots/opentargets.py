"""
Turning an OpenTargets release into canonical association rows.

Three eras, three readers, one output schema. Each reader is a pure
function of already-fetched data so it can be unit-tested against a
literal fixture; downloading is the CLI's job.

Era differences, all verified against the real releases:

  18.06   one gzipped JSON object per line. Datatype scores nested under
          association_score.datatypes; gene symbol inline at
          target.gene_info.symbol.
  21.11   ETL era, Parquet under output/.
  25.06   Parquet under output/association_by_datasource_direct/ with
          columns datatypeId, datasourceId, diseaseId, targetId, score,
          evidenceCount -- and NO symbol. Symbols come from output/target/
          as id -> approvedSymbol and must be joined on.
"""

from __future__ import annotations

import json
from typing import Iterable

import polars as pl

from neorx.snapshots.schema import ASSOCIATION_COLUMNS, empty_associations

__all__ = ["read_1806"]


def read_1806(lines: Iterable[str]) -> pl.DataFrame:
    """Read 18.06's line-delimited JSON into canonical association rows.

    One output row per (target, disease, datatype) with a non-zero score.
    A zero score is not evidence, and carrying it would inflate every
    downstream count of "how many datatypes support this pair".

    Records without a gene symbol are dropped: the candidate frame is
    built from symbols, so a record we cannot name is one we cannot place
    in a frame, and inventing a name would corrupt it. Malformed JSON
    raises -- a truncated download must fail loudly rather than silently
    yield fewer rows.
    """
    rows: list[dict[str, object]] = []

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)

        target = record.get("target") or {}
        symbol = ((target.get("gene_info") or {}).get("symbol") or "").strip()
        target_id = (target.get("id") or "").strip()
        disease_id = ((record.get("disease") or {}).get("id") or "").strip()
        if not symbol or not target_id or not disease_id:
            continue

        datatypes = (record.get("association_score") or {}).get("datatypes") or {}
        for datatype, score in datatypes.items():
            if not isinstance(score, (int, float)) or score <= 0.0:
                continue
            rows.append({
                "target_id": target_id,
                "target_symbol": symbol,
                "disease_id": disease_id,
                "datatype": datatype,
                "score": float(score),
            })

    if not rows:
        return empty_associations()
    return pl.DataFrame(rows, schema=ASSOCIATION_COLUMNS)
