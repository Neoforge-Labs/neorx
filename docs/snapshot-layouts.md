# OpenTargets snapshot layouts

Field mapping for each OpenTargets release era consumed by
`neorx.snapshots.opentargets`. Downstream code never learns which era a
row came from -- each era's reader absorbs the format drift and emits the
canonical `ASSOCIATION_COLUMNS` schema (`target_id`, `target_symbol`,
`disease_id`, `datatype`, `score`). This file records what was actually
found in each release, so a later era doesn't have to be re-verified from
scratch.

## 18.06

Base: `https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/18.06/`

Flat gzipped line-delimited JSON, one association record per line, at:

```
<release>/18.06_association_data.json.gz
```

Field mapping:

| canonical field | source field |
|---|---|
| `target_id` | `target.id` |
| `target_symbol` | `target.gene_info.symbol` |
| `disease_id` | `disease.id` |
| `datatype` | key under `association_score.datatypes` (one row emitted per key) |
| `score` | value under `association_score.datatypes.<name>` |

Also present but unused by the reader: `is_direct`.

There is no separate target table in this era -- the symbol is inline on
every association record.

Reader: `read_1806(lines: Iterable[str]) -> pl.DataFrame`.

## 21.11 (ETL era)

Base: `https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.11/`

Characterised directly against the release (not inferred) by listing
`output/` and downloading one Parquet part file from each of the
association and target tables.

`output/` for this release is *not* the flat `output/<table>/` layout
used later -- it contains only `etl/` and `literature/`. The tables live
one level deeper, under `output/etl/parquet/`:

```
<release>/output/etl/parquet/associationByDatasourceDirect/
<release>/output/etl/parquet/targets/
```

(camelCase directory names, unlike 25.06's snake_case
`association_by_datasource_direct` / `target`.)

**Association table** (`associationByDatasourceDirect`, 200 Parquet
parts, ~144 KB each), verified schema:

| column | dtype |
|---|---|
| `datatypeId` | String |
| `datasourceId` | String |
| `diseaseId` | String |
| `targetId` | String |
| `score` | Float64 |
| `evidenceCount` | Int64 |

This is byte-for-byte the same shape as 25.06's
`association_by_datasource_direct` -- same six columns, same names, same
types. It carries **no gene symbol**. Datatype values observed in the
sampled part: `somatic_mutation`, `genetic_association`,
`affected_pathway`, `literature`, `known_drug`, `animal_model`,
`rna_expression`. Sampled scores were strictly positive (no non-positive
scores in the inspected part; the reader still filters defensively, as
every era's reader does).

**Target table** (`targets`, 200 Parquet parts, ~360 KB each), relevant
columns verified:

| column | dtype |
|---|---|
| `id` | String |
| `approvedSymbol` | String |
| `biotype` | String |

(plus many nested fields -- `transcriptIds`, `genomicLocation`, `go`,
`hallmarks`, `synonyms`, `constraint`, `tractability`, etc. -- not needed
here). No nulls or blanks observed in `id` or `approvedSymbol` in the
sampled part.

Field mapping:

| canonical field | source field | table |
|---|---|---|
| `target_id` | `targetId` (association) / `id` (target) | join key |
| `target_symbol` | `approvedSymbol` | targets |
| `disease_id` | `diseaseId` | associations |
| `datatype` | `datatypeId` | associations |
| `score` | `score` | associations |

Because the association table carries no symbol, `read_2111` takes two
arguments -- `associations` and `targets` -- and joins them exactly as
`read_modern` does, with the same raise-on-unmapped-id behaviour and
wording: a null symbol would drop silently out of every candidate frame
and read as absent evidence rather than as a broken release.

Reader: `read_2111(associations: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame`.

## 25.06 (current era)

Base: `https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.06/`

Parquet at:

```
<release>/output/association_by_datasource_direct/
<release>/output/target/
```

Association table columns: `datatypeId`, `datasourceId`, `diseaseId`,
`targetId`, `score`, `evidenceCount`.

Target table columns (relevant subset): `id`, `approvedSymbol`,
`biotype`.

Same shape as 21.11's ETL-era tables, under a flatter, snake_case path
and with no `etl/` indirection. No symbol in the association table;
symbols are joined from the target table on `id` == `targetId`.

Field mapping:

| canonical field | source field | table |
|---|---|---|
| `target_id` | `targetId` (association) / `id` (target) | join key |
| `target_symbol` | `approvedSymbol` | target |
| `disease_id` | `diseaseId` | association |
| `datatype` | `datatypeId` | association |
| `score` | `score` | association |

Reader: `read_modern(associations: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame`.
