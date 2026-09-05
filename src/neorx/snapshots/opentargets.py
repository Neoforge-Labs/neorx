"""
Turning an OpenTargets release into canonical association rows.

Three eras, three readers, one output schema. Each reader is a pure
function of already-fetched data so it can be unit-tested against a
literal fixture; downloading is the CLI's job.

Era differences, all verified against the real releases:

  18.06   one gzipped JSON object per line. Datatype scores nested under
          association_score.datatypes; gene symbol inline at
          target.gene_info.symbol.
  21.11   ETL era, Parquet under output/etl/parquet/associationByDatasourceDirect/
          with columns datatypeId, datasourceId, diseaseId, targetId,
          score, evidenceCount -- and NO symbol. Symbols come from
          output/etl/parquet/targets/ as id -> approvedSymbol and must be
          joined on, same as 25.06.
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

__all__ = ["read_1806", "read_2111", "read_modern"]


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


def read_modern(
    associations: pl.DataFrame,
    targets: pl.DataFrame,
) -> pl.DataFrame:
    """Read the current-era association and target tables into canonical rows.

    ``associations`` is ``output/association_by_datasource_direct``:
    datatypeId, datasourceId, diseaseId, targetId, score, evidenceCount.
    ``targets`` is ``output/target``: id, approvedSymbol, biotype.

    The association table carries no symbol, so this joins one on. A
    target present in associations but absent from -- or blank in -- the
    target table raises: a null symbol would drop out of every candidate
    frame silently, and read as an absence of evidence rather than as a
    broken release.
    """
    if associations.height == 0:
        return empty_associations()

    scored = associations.filter(pl.col("score") > 0.0)
    if scored.height == 0:
        return empty_associations()

    symbols = (
        targets
        .select(
            pl.col("id").alias("target_id"),
            pl.col("approvedSymbol").str.strip_chars().alias("target_symbol"),
        )
        .filter(pl.col("target_symbol") != "")
    )

    joined = (
        scored
        .select(
            pl.col("targetId").alias("target_id"),
            pl.col("diseaseId").alias("disease_id"),
            pl.col("datatypeId").alias("datatype"),
            pl.col("score").cast(pl.Float64),
        )
        .join(symbols, on="target_id", how="left")
    )

    unmapped = joined.filter(pl.col("target_symbol").is_null())
    if unmapped.height:
        missing = sorted(set(unmapped["target_id"].to_list()))[:5]
        raise ValueError(
            f"{unmapped.height} association rows reference target ids absent "
            f"from the target table, e.g. {missing}. A release whose target "
            f"table does not cover its own associations is incomplete; "
            f"joining a null symbol would remove these silently from every "
            f"candidate frame."
        )

    return joined.select(list(ASSOCIATION_COLUMNS)).cast(ASSOCIATION_COLUMNS)


def read_2111(
    associations: pl.DataFrame,
    targets: pl.DataFrame,
) -> pl.DataFrame:
    """Read the 21.11 ETL-era association and target tables into canonical rows.

    ``associations`` is ``output/etl/parquet/associationByDatasourceDirect``:
    datatypeId, datasourceId, diseaseId, targetId, score, evidenceCount.
    ``targets`` is ``output/etl/parquet/targets``: id, approvedSymbol, biotype
    (among many other fields, ignored here).

    Verified against the real 21.11 release: the association table carries
    no symbol, exactly as in 25.06, so this takes two arguments and joins
    one on. A target present in associations but absent from -- or blank
    in -- the target table raises: a null symbol would drop out of every
    candidate frame silently, and read as an absence of evidence rather
    than as a broken release. Same failure mode as read_modern, same
    wording.
    """
    if associations.height == 0:
        return empty_associations()

    scored = associations.filter(pl.col("score") > 0.0)
    if scored.height == 0:
        return empty_associations()

    symbols = (
        targets
        .select(
            pl.col("id").alias("target_id"),
            pl.col("approvedSymbol").str.strip_chars().alias("target_symbol"),
        )
        .filter(pl.col("target_symbol") != "")
    )

    joined = (
        scored
        .select(
            pl.col("targetId").alias("target_id"),
            pl.col("diseaseId").alias("disease_id"),
            pl.col("datatypeId").alias("datatype"),
            pl.col("score").cast(pl.Float64),
        )
        .join(symbols, on="target_id", how="left")
    )

    unmapped = joined.filter(pl.col("target_symbol").is_null())
    if unmapped.height:
        missing = sorted(set(unmapped["target_id"].to_list()))[:5]
        raise ValueError(
            f"{unmapped.height} association rows reference target ids absent "
            f"from the target table, e.g. {missing}. A release whose target "
            f"table does not cover its own associations is incomplete; "
            f"joining a null symbol would remove these silently from every "
            f"candidate frame."
        )

    return joined.select(list(ASSOCIATION_COLUMNS)).cast(ASSOCIATION_COLUMNS)
