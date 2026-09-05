# Versioned Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a disease graph as of a chosen past date, from pinned inputs, for hundreds of diseases without hundreds of hours of HTTP.

**Architecture:** Each upstream release is streamed, reduced to a compact Parquet extract, and discarded. A `SourceResolver` maps a source name plus an optional `as_of` date to either the existing live client or a snapshot reader, refusing loudly where no snapshot exists. The candidate frame is derived from the pinned release alone and gated at candidate selection, with every exclusion recorded.

**Tech Stack:** Python 3.12+, polars (never pandas), pyarrow, requests, tomllib/tomli-w, pytest, uv (never pip).

**Spec:** `docs/superpowers/specs/2026-09-04-versioned-data-layer-design.md`

## Global Constraints

- **No shims, no patch jobs.** No `except Exception` that continues. No flag added to make one caller behave differently. If two things disagree, fail loudly and name both.
- **Never weaken a test or a gate to make it pass.** If a detector fires, fix what it found.
- **A dated run must never silently use live data.** A missing snapshot is a named refusal, never a fallback.
- Read Parquet with **polars**. `pandas` is not a dependency and must not become one.
- Run tests with `.venv/bin/python -m pytest`. Install with `uv pip install` — never `pip` directly.
- Commit messages carry **no AI attribution and no `Co-Authored-By` trailer** (see `CLAUDE.md`).
- No module under `src/neorx/snapshots/` may exceed 600 lines.
- **No network calls in unit tests.** Every reader test drives a literal fixture. Downloads happen only in the CLI and in one explicitly-marked integration test.

### Verified upstream schemas — use these exact field names

**OpenTargets 18.06** — one gzipped JSON object per line at
`https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/18.06/18.06_association_data.json.gz` (0.18 GB):

```
target.id                              -> "ENSG00000121879"
target.gene_info.symbol                -> "PIK3CA"
disease.id                             -> "EFO_0000616"
association_score.datatypes.<name>     -> float, keys include
    genetic_association, somatic_mutation, literature, known_drug,
    rna_expression, animal_model, affected_pathway
is_direct                              -> bool
```

**OpenTargets 25.06** — Parquet parts under `output/association_by_datasource_direct/`:

```
datatypeId String | datasourceId String | diseaseId String
targetId String | score Float64 | evidenceCount Int64
```

Symbols are **not** in that table. They come from `output/target/`:

```
id String (Ensembl) | approvedSymbol String | biotype String
```

**OpenTargets 21.11** — ETL era, `output/`. Layout not yet characterised; Task 4 does that first.

**OmniPath archive** — `https://archive.omnipathdb.org/`, TSV, schema uniform across the range.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/neorx/snapshots/schema.py` | The canonical extract schemas. One definition both writers and readers import |
| `src/neorx/snapshots/manifest.py` | `snapshots/manifest.toml` read/write, digests, extractor versioning |
| `src/neorx/snapshots/opentargets.py` | Three era readers, each turning a raw release into canonical rows |
| `src/neorx/snapshots/omnipath.py` | Archive TSV reader, reusing sub-project 3's pure parser |
| `src/neorx/snapshots/reader.py` | Reads derived extracts back; no knowledge of upstream formats |
| `src/neorx/snapshots/frame.py` | `candidate_frame` and the disease corpus criteria |
| `src/neorx/snapshots/resolver.py` | `SourceResolver` — live or snapshot, refuse on missing |
| `src/neorx/snapshots/__main__.py` | `neorx snapshot build` / `list` |
| `src/neorx/core/causal/identifier.py` | `_get_candidate_nodes` gains the frame gate |
| `src/neorx/core/graph/graph_builder.py` | Accepts `as_of` and threads the resolver |

---

## Task 1: Canonical extract schema

**Files:**
- Create: `src/neorx/snapshots/__init__.py`, `src/neorx/snapshots/schema.py`
- Test: `tests/snapshots/test_schema.py`, `tests/snapshots/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ASSOCIATION_COLUMNS: dict[str, pl.DataType]`, `INTERACTION_COLUMNS: dict[str, pl.DataType]`, `GENETIC_DATATYPES: frozenset[str]`, `EXTRACTOR_VERSION: int`, `empty_associations() -> pl.DataFrame`, `empty_interactions() -> pl.DataFrame`.

Every era reader emits these columns and no others. Downstream code never learns which era a row came from.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_schema.py
"""The one schema every era reader must produce.

Three OpenTargets format eras feed this pipeline. The whole point of a
derived extract is that downstream code cannot tell them apart, so the
column set is defined once, here, and asserted rather than assumed.
"""

import polars as pl

from neorx.snapshots.schema import (
    ASSOCIATION_COLUMNS,
    EXTRACTOR_VERSION,
    GENETIC_DATATYPES,
    INTERACTION_COLUMNS,
    empty_associations,
    empty_interactions,
)


def test_association_columns_are_exactly_the_five_the_causal_subgraph_needs():
    assert list(ASSOCIATION_COLUMNS) == [
        "target_id", "target_symbol", "disease_id", "datatype", "score",
    ]


def test_association_key_columns_are_strings_and_score_is_a_float():
    assert ASSOCIATION_COLUMNS["target_id"] == pl.Utf8
    assert ASSOCIATION_COLUMNS["target_symbol"] == pl.Utf8
    assert ASSOCIATION_COLUMNS["disease_id"] == pl.Utf8
    assert ASSOCIATION_COLUMNS["datatype"] == pl.Utf8
    assert ASSOCIATION_COLUMNS["score"] == pl.Float64


def test_empty_association_frame_matches_the_declared_schema():
    df = empty_associations()
    assert df.height == 0
    assert dict(df.schema) == ASSOCIATION_COLUMNS


def test_empty_interaction_frame_matches_the_declared_schema():
    df = empty_interactions()
    assert df.height == 0
    assert dict(df.schema) == INTERACTION_COLUMNS


def test_genetic_datatypes_are_exactly_the_two_that_confer_admissibility():
    # These are the OpenTargets datatype names that
    # neorx.core.causal.graph_semantics treats as causal-admissible for a
    # gene-disease edge. A third name added here silently widens what the
    # backdoor criterion is allowed to reason over.
    assert GENETIC_DATATYPES == frozenset({"genetic_association", "somatic_mutation"})


def test_extractor_version_is_an_integer_that_can_be_recorded():
    # A derived artifact needs its derivation pinned: two extracts of the
    # same release made by different extractor code are different inputs.
    assert isinstance(EXTRACTOR_VERSION, int)
    assert EXTRACTOR_VERSION >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.snapshots'`

- [ ] **Step 3: Write the module**

Create `src/neorx/snapshots/__init__.py` containing only a docstring:

```python
"""
Versioned snapshots of the sources the causal subgraph is built from.

Upstream releases are streamed, reduced to a compact Parquet extract, and
discarded. What persists is four columns of gene-disease evidence and a
table of directed regulatory interactions -- enough to rebuild the causal
subgraph as it would have looked at a past date, and small enough that a
decade of releases costs less disk than one of them.
"""
```

Create `src/neorx/snapshots/schema.py`:

```python
"""
The canonical shape of a derived extract.

Three OpenTargets format eras feed this pipeline: a flat gzipped JSON
dump in 2018, an ETL-era Parquet tree in 2021, and the current output
layout. Each gets its own reader, and each reader's job is to produce
exactly the columns below. Downstream code -- the frame, the resolver,
the graph builder -- never learns which era a row came from.

That is the whole value of a derived extract rather than a mirror: format
drift is absorbed once, at the edge, instead of leaking into every
consumer.
"""

from __future__ import annotations

import polars as pl

# Bump when a reader's output changes meaning for the same input. Recorded
# in the manifest beside the release id, because two extracts of the same
# release made by different extractor code are different inputs and must
# be distinguishable.
EXTRACTOR_VERSION = 1

ASSOCIATION_COLUMNS: dict[str, pl.DataType] = {
    # Ensembl gene id. The only identifier present in every era, so it is
    # the join key; 25.06 has no symbol in its association table at all.
    "target_id": pl.Utf8,
    # HGNC symbol. What graph_builder, STRING and OmniPath actually work
    # in, so it is carried rather than re-derived at every call site.
    "target_symbol": pl.Utf8,
    # EFO disease id.
    "disease_id": pl.Utf8,
    # OpenTargets datatype: genetic_association, somatic_mutation,
    # literature, known_drug, and so on. One row per datatype per pair.
    "datatype": pl.Utf8,
    "score": pl.Float64,
}

INTERACTION_COLUMNS: dict[str, pl.DataType] = {
    "source_symbol": pl.Utf8,
    "target_symbol": pl.Utf8,
    "is_directed": pl.Boolean,
    "consensus_direction": pl.Boolean,
    "is_stimulation": pl.Boolean,
    "is_inhibition": pl.Boolean,
    # Semicolon-joined; parsed by the reader into a list at use time.
    "primary_sources": pl.Utf8,
    "references": pl.Utf8,
}

# The OpenTargets datatypes that neorx.core.causal.graph_semantics treats
# as conferring causal admissibility on a gene-disease edge, on the
# Mendelian randomisation warrant. Adding a name here silently widens what
# the backdoor criterion may reason over.
GENETIC_DATATYPES: frozenset[str] = frozenset({
    "genetic_association",
    "somatic_mutation",
})

__all__ = [
    "ASSOCIATION_COLUMNS",
    "EXTRACTOR_VERSION",
    "GENETIC_DATATYPES",
    "INTERACTION_COLUMNS",
    "empty_associations",
    "empty_interactions",
]


def empty_associations() -> pl.DataFrame:
    """An empty association frame with the canonical schema."""
    return pl.DataFrame(schema=ASSOCIATION_COLUMNS)


def empty_interactions() -> pl.DataFrame:
    """An empty interaction frame with the canonical schema."""
    return pl.DataFrame(schema=INTERACTION_COLUMNS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/test_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/snapshots tests/snapshots
git commit -m "feat: define the canonical snapshot extract schema"
```

---

## Task 2: OpenTargets 18.06 reader

**Files:**
- Create: `src/neorx/snapshots/opentargets.py`
- Test: `tests/snapshots/test_opentargets_1806.py`

**Interfaces:**
- Consumes: `ASSOCIATION_COLUMNS`, `empty_associations` from Task 1.
- Produces: `read_1806(lines: Iterable[str]) -> pl.DataFrame` — a pure function of already-read JSON lines, emitting canonical association rows.

Pure by design: the fetch lives in the CLI (Task 6), so this is unit-testable with literal fixtures and no network.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_opentargets_1806.py
"""The 2018 era reader.

18.06 ships one gzipped JSON object per line, nesting the datatype scores
under association_score.datatypes and carrying the gene symbol inline at
target.gene_info.symbol. Later eras do neither, which is why each era gets
its own reader and they all emit the same five columns.
"""

import json

import polars as pl

from neorx.snapshots.opentargets import read_1806
from neorx.snapshots.schema import ASSOCIATION_COLUMNS


def _record(symbol="PIK3CA", ensembl="ENSG00000121879",
            efo="EFO_0000616", datatypes=None):
    return json.dumps({
        "target": {"id": ensembl, "gene_info": {"symbol": symbol}},
        "disease": {"id": efo},
        "association_score": {
            "overall": 0.7,
            "datatypes": datatypes if datatypes is not None else {
                "genetic_association": 0.61,
                "literature": 0.22,
                "somatic_mutation": 0.0,
            },
        },
        "is_direct": True,
    })


def test_emits_the_canonical_schema():
    df = read_1806([_record()])
    assert dict(df.schema) == ASSOCIATION_COLUMNS


def test_one_row_per_nonzero_datatype():
    df = read_1806([_record()])
    # genetic_association and literature are non-zero; somatic_mutation is 0.0
    assert sorted(df["datatype"].to_list()) == ["genetic_association", "literature"]


def test_zero_scored_datatypes_are_dropped():
    df = read_1806([_record(datatypes={"genetic_association": 0.0})])
    assert df.height == 0


def test_identifiers_and_symbol_are_carried_through():
    df = read_1806([_record()])
    row = df.filter(pl.col("datatype") == "genetic_association").row(0, named=True)
    assert row["target_id"] == "ENSG00000121879"
    assert row["target_symbol"] == "PIK3CA"
    assert row["disease_id"] == "EFO_0000616"
    assert row["score"] == 0.61


def test_multiple_records_accumulate():
    df = read_1806([_record(), _record(symbol="TP53", ensembl="ENSG00000141510")])
    assert set(df["target_symbol"].to_list()) == {"PIK3CA", "TP53"}


def test_an_empty_stream_yields_an_empty_frame_with_the_schema():
    df = read_1806([])
    assert df.height == 0
    assert dict(df.schema) == ASSOCIATION_COLUMNS


def test_blank_lines_are_skipped():
    df = read_1806(["", "   ", _record()])
    assert df.height == 2


def test_a_record_missing_a_gene_symbol_is_dropped_not_guessed():
    # The frame is built from symbols. A record we cannot name is a record
    # we cannot put in a frame, and inventing one would corrupt it.
    bad = json.dumps({
        "target": {"id": "ENSG00000000001", "gene_info": {}},
        "disease": {"id": "EFO_0000616"},
        "association_score": {"datatypes": {"genetic_association": 0.9}},
    })
    assert read_1806([bad]).height == 0


def test_malformed_json_raises_rather_than_being_skipped():
    import pytest
    # A truncated download should fail loudly, not silently yield fewer rows.
    with pytest.raises(json.JSONDecodeError):
        read_1806(['{"target": {'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_opentargets_1806.py -v`
Expected: FAIL — `ImportError: cannot import name 'read_1806'`

- [ ] **Step 3: Write the reader**

Create `src/neorx/snapshots/opentargets.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/test_opentargets_1806.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/snapshots/opentargets.py tests/snapshots/test_opentargets_1806.py
git commit -m "feat: read OpenTargets 18.06 associations into the canonical schema"
```

---

## Task 3: OpenTargets modern reader

**Files:**
- Modify: `src/neorx/snapshots/opentargets.py`
- Test: `tests/snapshots/test_opentargets_modern.py`

**Interfaces:**
- Consumes: `ASSOCIATION_COLUMNS`, `empty_associations`.
- Produces: `read_modern(associations: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame`.

The modern association table has no symbol; it must be joined from the `target` table. A target present in associations but absent from the target table is a broken release, and must fail rather than yield a null symbol that silently drops out of every frame.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_opentargets_modern.py
"""The current-era reader.

25.06's association table is already tidy -- datatypeId, diseaseId,
targetId, score -- but carries no gene symbol at all. Symbols live in a
separate target table as id -> approvedSymbol, so this reader's real work
is the join, and its real risk is a join that silently produces nulls.
"""

import polars as pl
import pytest

from neorx.snapshots.opentargets import read_modern
from neorx.snapshots.schema import ASSOCIATION_COLUMNS


def _assoc(rows=None):
    return pl.DataFrame(rows if rows is not None else [
        {"datatypeId": "genetic_association", "datasourceId": "gwas_catalog",
         "diseaseId": "EFO_0000616", "targetId": "ENSG00000121879",
         "score": 0.61, "evidenceCount": 3},
        {"datatypeId": "literature", "datasourceId": "europepmc",
         "diseaseId": "EFO_0000616", "targetId": "ENSG00000121879",
         "score": 0.22, "evidenceCount": 9},
    ])


def _targets(rows=None):
    return pl.DataFrame(rows if rows is not None else [
        {"id": "ENSG00000121879", "approvedSymbol": "PIK3CA", "biotype": "protein_coding"},
    ])


def test_emits_the_canonical_schema():
    df = read_modern(_assoc(), _targets())
    assert dict(df.schema) == ASSOCIATION_COLUMNS


def test_symbol_is_joined_from_the_target_table():
    df = read_modern(_assoc(), _targets())
    assert set(df["target_symbol"].to_list()) == {"PIK3CA"}


def test_datatype_and_score_survive_the_join():
    df = read_modern(_assoc(), _targets())
    row = df.filter(pl.col("datatype") == "genetic_association").row(0, named=True)
    assert row["score"] == 0.61
    assert row["disease_id"] == "EFO_0000616"
    assert row["target_id"] == "ENSG00000121879"


def test_zero_scores_are_dropped_matching_the_1806_reader():
    a = _assoc([{"datatypeId": "genetic_association", "datasourceId": "x",
                 "diseaseId": "EFO_1", "targetId": "ENSG00000121879",
                 "score": 0.0, "evidenceCount": 0}])
    assert read_modern(a, _targets()).height == 0


def test_a_target_missing_from_the_target_table_raises():
    # A null symbol would drop silently out of every candidate frame and
    # look like an absence of evidence rather than a broken release.
    a = _assoc([{"datatypeId": "genetic_association", "datasourceId": "x",
                 "diseaseId": "EFO_1", "targetId": "ENSG_ABSENT",
                 "score": 0.5, "evidenceCount": 1}])
    with pytest.raises(ValueError, match="ENSG_ABSENT"):
        read_modern(a, _targets())


def test_targets_with_a_blank_symbol_are_treated_as_missing():
    a = _assoc([{"datatypeId": "genetic_association", "datasourceId": "x",
                 "diseaseId": "EFO_1", "targetId": "ENSG_BLANK",
                 "score": 0.5, "evidenceCount": 1}])
    t = _targets([{"id": "ENSG_BLANK", "approvedSymbol": "", "biotype": "protein_coding"}])
    with pytest.raises(ValueError, match="ENSG_BLANK"):
        read_modern(a, t)


def test_an_empty_association_table_yields_an_empty_frame():
    empty = pl.DataFrame(schema={
        "datatypeId": pl.Utf8, "datasourceId": pl.Utf8, "diseaseId": pl.Utf8,
        "targetId": pl.Utf8, "score": pl.Float64, "evidenceCount": pl.Int64,
    })
    df = read_modern(empty, _targets())
    assert df.height == 0
    assert dict(df.schema) == ASSOCIATION_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_opentargets_modern.py -v`
Expected: FAIL — `ImportError: cannot import name 'read_modern'`

- [ ] **Step 3: Write the reader**

Append to `src/neorx/snapshots/opentargets.py`, and add `"read_modern"` to `__all__`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/snapshots/opentargets.py tests/snapshots/test_opentargets_modern.py
git commit -m "feat: read current-era OpenTargets associations, joining symbols from the target table"
```

---

## Task 4: OpenTargets 21.11 reader

**Files:**
- Modify: `src/neorx/snapshots/opentargets.py`
- Test: `tests/snapshots/test_opentargets_2111.py`
- Create: `docs/snapshot-layouts.md`

**Interfaces:**
- Consumes: `ASSOCIATION_COLUMNS`, `empty_associations`.
- Produces: `read_2111(associations: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame`.

This is the one era whose layout is not yet verified. **Step 1 is a characterisation step**, with a real deliverable: a documented field mapping. Do not guess field names.

- [ ] **Step 1: Characterise the 21.11 layout and write it down**

```bash
curl -s "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.11/output/" \
  | grep -oE 'href="[^"?][^"]*/"' | sed 's/href="//;s/"//'
```

Find the association table and the target table. Download **one** part file from each and inspect:

```bash
.venv/bin/python -c "
import polars as pl
df = pl.read_parquet('<downloaded part>')
print(df.height); [print(' ', k, v) for k, v in df.schema.items()]
print(df.head(3))
"
```

Record what you find in `docs/snapshot-layouts.md`, with a section per era.
Include the 18.06 and 25.06 layouts already verified and given in this plan's
Global Constraints, so the file is a complete reference rather than a note about
one era. State the exact association table path, the exact target table path,
and the field name for each of: datatype, disease id, target id, score, symbol.

If 21.11's association table already carries a symbol, `read_2111` takes only
one argument and the docstring says so. If it does not, it mirrors
`read_modern`'s join.

- [ ] **Step 2: Write the failing test**

Write `tests/snapshots/test_opentargets_2111.py` mirroring
`test_opentargets_modern.py`, with the fixture columns replaced by the real
21.11 field names from Step 1. Cover, at minimum: canonical schema out, zero
scores dropped, identifiers carried through, an empty input yielding an empty
frame, and — if a join is needed — an unmapped target raising with its id in the
message.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_opentargets_2111.py -v`
Expected: FAIL — `ImportError: cannot import name 'read_2111'`

- [ ] **Step 4: Write the reader**

Append `read_2111` to `src/neorx/snapshots/opentargets.py` and add it to
`__all__`. Follow `read_modern`'s shape: filter non-positive scores, select and
rename into the canonical columns, and — if a join is required — raise on
unmapped ids with the same wording, since the failure mode is identical.

Update the module docstring's era table with the verified 21.11 layout.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/neorx/snapshots/opentargets.py tests/snapshots/test_opentargets_2111.py docs/snapshot-layouts.md
git commit -m "feat: read ETL-era OpenTargets associations; document all three layouts"
```

---

## Task 5: OmniPath archive reader

**Files:**
- Create: `src/neorx/snapshots/omnipath.py`
- Test: `tests/snapshots/test_omnipath_archive.py`

**Interfaces:**
- Consumes: `INTERACTION_COLUMNS`, `empty_interactions`.
- Produces: `read_archive_tsv(text: str) -> pl.DataFrame`, `to_interaction_rows(df: pl.DataFrame) -> list[dict]`.

`to_interaction_rows` returns dicts in exactly the shape sub-project 3's
`neorx.core.sources.omnipath._interactions_to_edges(rows, known_genes)` already
consumes, so the snapshot path reuses that pure parser unchanged. Do not
reimplement edge construction.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_omnipath_archive.py
"""The OmniPath archive reader.

Sub-project 3 deliberately split OmniPath's HTTP fetch from its parsing:
_interactions_to_edges(rows, known_genes) is a pure function of already-
parsed rows. That is what lets a snapshot feed the same parser as the live
API with no second implementation of edge construction, so this reader's
only job is to produce rows in that exact shape.
"""

import polars as pl

from neorx.snapshots.omnipath import read_archive_tsv, to_interaction_rows
from neorx.snapshots.schema import INTERACTION_COLUMNS

TSV = (
    "source_genesymbol\ttarget_genesymbol\tis_directed\tconsensus_direction\t"
    "is_stimulation\tis_inhibition\tsources\treferences\n"
    "EGFR\tSHC1\t1\t1\t1\t0\tSIGNOR;TRRUST\tSIGNOR:16331690\n"
    "TP53\tMDM2\t1\t1\t0\t1\tSIGNOR\tSIGNOR:14983059\n"
)


def test_emits_the_canonical_interaction_schema():
    df = read_archive_tsv(TSV)
    assert dict(df.schema) == INTERACTION_COLUMNS


def test_integer_flag_columns_become_booleans():
    df = read_archive_tsv(TSV)
    row = df.filter(pl.col("source_symbol") == "EGFR").row(0, named=True)
    assert row["is_directed"] is True
    assert row["is_stimulation"] is True
    assert row["is_inhibition"] is False


def test_rows_round_trip_into_the_shape_the_live_parser_consumes():
    rows = to_interaction_rows(read_archive_tsv(TSV))
    assert rows[0]["source_genesymbol"] == "EGFR"
    assert rows[0]["target_genesymbol"] == "SHC1"
    assert rows[0]["is_directed"] is True
    assert rows[0]["consensus_direction"] is True
    assert rows[0]["sources"] == ["SIGNOR", "TRRUST"]


def test_the_live_parser_accepts_those_rows_unchanged():
    # The actual integration point: sub-project 3's parser, unmodified.
    from neorx.core.sources.omnipath import _interactions_to_edges

    rows = to_interaction_rows(read_archive_tsv(TSV))
    edges = _interactions_to_edges(rows, {"EGFR", "SHC1", "TP53", "MDM2"})
    assert len(edges) == 2
    by_pair = {(e.source_id, e.target_id): e for e in edges}
    assert by_pair[("gene:EGFR", "gene:SHC1")].sign == 1
    assert by_pair[("gene:TP53", "gene:MDM2")].sign == -1


def test_an_empty_tsv_yields_an_empty_frame_with_the_schema():
    header = TSV.split("\n")[0] + "\n"
    df = read_archive_tsv(header)
    assert df.height == 0
    assert dict(df.schema) == INTERACTION_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_omnipath_archive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.snapshots.omnipath'`

- [ ] **Step 3: Write the reader**

Create `src/neorx/snapshots/omnipath.py`:

```python
"""
Reading an archived OmniPath interactions dump.

Sub-project 3 split OmniPath's fetch from its parsing on purpose:
``neorx.core.sources.omnipath._interactions_to_edges(rows, known_genes)``
is a pure function of already-parsed rows. A snapshot therefore feeds the
same parser the live API does, and there is exactly one implementation of
edge construction in the codebase rather than two that can drift apart.

The archive's TSV schema is uniform across the range this project uses, so
one reader covers every time point.
"""

from __future__ import annotations

import io

import polars as pl

from neorx.snapshots.schema import INTERACTION_COLUMNS, empty_interactions

__all__ = ["read_archive_tsv", "to_interaction_rows"]

_FLAGS = ("is_directed", "consensus_direction", "is_stimulation", "is_inhibition")


def read_archive_tsv(text: str) -> pl.DataFrame:
    """Parse an archived interactions TSV into the canonical schema.

    The archive encodes booleans as 0/1 integers; they become real
    booleans here so that downstream code never has to remember which
    convention a given source used.
    """
    raw = pl.read_csv(io.StringIO(text), separator="\t", infer_schema_length=0)
    if raw.height == 0:
        return empty_interactions()

    return raw.select(
        pl.col("source_genesymbol").alias("source_symbol"),
        pl.col("target_genesymbol").alias("target_symbol"),
        *[
            (pl.col(flag).cast(pl.Int8, strict=False) == 1).alias(flag)
            for flag in _FLAGS
        ],
        pl.col("sources").fill_null("").alias("primary_sources"),
        pl.col("references").fill_null("").alias("references"),
    ).cast(INTERACTION_COLUMNS)


def to_interaction_rows(df: pl.DataFrame) -> list[dict]:
    """Convert canonical rows into the dicts the live parser consumes.

    Returns the exact key names
    ``neorx.core.sources.omnipath._interactions_to_edges`` reads, so the
    snapshot path reuses it without modification.
    """
    return [
        {
            "source_genesymbol": row["source_symbol"],
            "target_genesymbol": row["target_symbol"],
            "is_directed": row["is_directed"],
            "consensus_direction": row["consensus_direction"],
            "is_stimulation": row["is_stimulation"],
            "is_inhibition": row["is_inhibition"],
            "sources": [s for s in (row["primary_sources"] or "").split(";") if s],
            "references": row["references"] or "",
        }
        for row in df.iter_rows(named=True)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/snapshots/omnipath.py tests/snapshots/test_omnipath_archive.py
git commit -m "feat: read archived OmniPath dumps through sub-project 3's parser"
```

---

## Task 6: Manifest

**Files:**
- Create: `src/neorx/snapshots/manifest.py`
- Test: `tests/snapshots/test_manifest.py`
- Modify: `pyproject.toml` (add `tomli-w`)

**Interfaces:**
- Consumes: `EXTRACTOR_VERSION`.
- Produces: `SnapshotEntry` (frozen dataclass: `source: str`, `release: str`, `url: str`, `sha256: str`, `extractor_version: int`, `rows: int`), `write_entry(manifest_path, entry)`, `read_manifest(manifest_path) -> dict[tuple[str, str], SnapshotEntry]`, `digest_file(path) -> str`.

A derived artifact needs its derivation pinned, so `extractor_version` sits beside the release id.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_manifest.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.snapshots.manifest'`

- [ ] **Step 3: Add the dependency**

In `pyproject.toml`, add to `[project] dependencies`:

```toml
    # Writing snapshots/manifest.toml. Python ships tomllib for reading
    # only, so a writer is a real dependency rather than a convenience.
    "tomli-w>=1.0",
```

Then: `uv pip install -e . --group dev`

- [ ] **Step 4: Write the module**

Create `src/neorx/snapshots/manifest.py`:

```python
"""
What a snapshot records about itself.

A release identifier alone does not identify a derived extract. Two
extracts of OpenTargets 18.06 produced by different extractor code are
different inputs to every number computed downstream, so the extractor
version is recorded beside the release id, and the digest is taken over
the extract actually written rather than the upstream file.

Keyed by (source, release), so re-running an extraction replaces its entry
instead of appending a second one.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import tomli_w

__all__ = ["SnapshotEntry", "digest_file", "read_manifest", "write_entry"]


@dataclass(frozen=True)
class SnapshotEntry:
    """One derived extract, and everything needed to identify it."""

    source: str
    release: str
    url: str
    sha256: str
    extractor_version: int
    rows: int


def digest_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large extract does not load."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(manifest_path: Path) -> dict[tuple[str, str], SnapshotEntry]:
    """Read every entry, keyed by (source, release). Absent file: no entries."""
    path = Path(manifest_path)
    if not path.exists():
        return {}

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries: dict[tuple[str, str], SnapshotEntry] = {}
    for source, releases in (data.get("snapshot") or {}).items():
        for release, fields in releases.items():
            entry = SnapshotEntry(source=source, release=release, **fields)
            entries[(source, release)] = entry
    return entries


def write_entry(manifest_path: Path, entry: SnapshotEntry) -> None:
    """Insert or replace one entry, leaving every other entry untouched."""
    path = Path(manifest_path)
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    fields = asdict(entry)
    source = fields.pop("source")
    release = fields.pop("release")

    data.setdefault("snapshot", {}).setdefault(source, {})[release] = fields
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/neorx/snapshots/manifest.py tests/snapshots/test_manifest.py pyproject.toml
git commit -m "feat: record snapshot provenance including the extractor version"
```

---

## Task 7: Extract reader

**Files:**
- Create: `src/neorx/snapshots/reader.py`
- Test: `tests/snapshots/test_reader.py`

**Interfaces:**
- Consumes: `ASSOCIATION_COLUMNS`, `INTERACTION_COLUMNS`, `read_manifest`.
- Produces: `SnapshotStore` with `__init__(root: Path)`, `associations(release: str) -> pl.DataFrame`, `interactions(release: str) -> pl.DataFrame`, `has(source: str, release: str) -> bool`, and `SnapshotMissingError`.

Knows the on-disk layout. Knows nothing about upstream formats.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_reader.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.snapshots.reader'`

- [ ] **Step 3: Write the module**

Create `src/neorx/snapshots/reader.py`:

```python
"""
Reading derived extracts back off disk.

This layer knows the on-disk layout and nothing else. By the time a frame
reaches it, which upstream era produced it is invisible -- that is the
whole point of extracting to a canonical schema.

A missing snapshot raises rather than returning an empty frame. An empty
frame reads downstream as "this release recorded no evidence for that
disease", which is a finding; a missing extract is a configuration
mistake, and the two must not be confusable.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

__all__ = ["SnapshotMissingError", "SnapshotStore"]


class SnapshotMissingError(FileNotFoundError):
    """No extract exists for a requested source and release."""


class SnapshotStore:
    """Derived extracts under a root directory.

    Layout::

        <root>/opentargets/<release>/associations.parquet
        <root>/omnipath/<release>/interactions.parquet
        <root>/manifest.toml
    """

    _FILES = {
        "opentargets": "associations.parquet",
        "omnipath": "interactions.parquet",
    }

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, source: str, release: str) -> Path:
        return self.root / source / release / self._FILES[source]

    def has(self, source: str, release: str) -> bool:
        """Whether an extract exists, without raising."""
        if source not in self._FILES:
            return False
        return self._path(source, release).exists()

    def _read(self, source: str, release: str) -> pl.DataFrame:
        path = self._path(source, release)
        if not path.exists():
            raise SnapshotMissingError(
                f"No {source} snapshot for release {release!r} at {path}. "
                f"Build it with `neorx snapshot build {source} {release}`. "
                f"A dated run will not fall back to live data."
            )
        return pl.read_parquet(path)

    def associations(self, release: str) -> pl.DataFrame:
        """Canonical association rows for an OpenTargets release."""
        return self._read("opentargets", release)

    def interactions(self, release: str) -> pl.DataFrame:
        """Canonical interaction rows for an OmniPath archive date."""
        return self._read("omnipath", release)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/snapshots/reader.py tests/snapshots/test_reader.py
git commit -m "feat: read derived extracts, distinguishing missing from empty"
```

---

## Task 8: Candidate frame and disease corpus

**Files:**
- Create: `src/neorx/snapshots/frame.py`
- Test: `tests/snapshots/test_frame.py`

**Interfaces:**
- Consumes: `SnapshotStore`, `GENETIC_DATATYPES`.
- Produces: `candidate_frame(store, release, disease_id) -> frozenset[str]`, `diseases_with_genetic_evidence(store, release) -> frozenset[str]`, `corpus_diseases(store, release, phase2_targets) -> frozenset[str]`.

This is the spec's load-bearing idea. The frame is the population a date would have evaluated, drawn from the pinned release alone.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_frame.py
"""The candidate frame -- the population a past date would have evaluated.

Features computed from 2018 data are useless if the population was chosen
with hindsight. A gene that entered the pool because a 2026 source
connected it to something is a gene 2018 would never have looked at, and
that is leakage in the sampling frame rather than in the predictor. So the
frame comes from the pinned release and nothing else.
"""

import polars as pl

from neorx.snapshots.frame import (
    candidate_frame,
    corpus_diseases,
    diseases_with_genetic_evidence,
)
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.schema import ASSOCIATION_COLUMNS


def _store(tmp_path, rows):
    d = tmp_path / "opentargets" / "18.06"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=ASSOCIATION_COLUMNS).write_parquet(
        d / "associations.parquet"
    )
    return SnapshotStore(tmp_path)


def _row(symbol, disease="EFO_1", datatype="genetic_association", score=0.5):
    return {
        "target_id": f"ENSG_{symbol}", "target_symbol": symbol,
        "disease_id": disease, "datatype": datatype, "score": score,
    }


def test_frame_is_drawn_from_genetic_evidence_only(tmp_path):
    store = _store(tmp_path, [
        _row("PIK3CA", datatype="genetic_association"),
        _row("TP53", datatype="somatic_mutation"),
        _row("TNF", datatype="literature"),
        _row("IL6", datatype="rna_expression"),
    ])
    assert candidate_frame(store, "18.06", "EFO_1") == frozenset({"PIK3CA", "TP53"})


def test_frame_is_scoped_to_one_disease(tmp_path):
    store = _store(tmp_path, [
        _row("PIK3CA", disease="EFO_1"),
        _row("BRCA1", disease="EFO_2"),
    ])
    assert candidate_frame(store, "18.06", "EFO_1") == frozenset({"PIK3CA"})


def test_zero_scored_genetic_evidence_does_not_enter_the_frame(tmp_path):
    store = _store(tmp_path, [_row("PIK3CA", score=0.0)])
    assert candidate_frame(store, "18.06", "EFO_1") == frozenset()


def test_a_disease_absent_from_the_release_has_an_empty_frame(tmp_path):
    store = _store(tmp_path, [_row("PIK3CA")])
    assert candidate_frame(store, "18.06", "EFO_ABSENT") == frozenset()


def test_diseases_with_genetic_evidence_excludes_literature_only_diseases(tmp_path):
    store = _store(tmp_path, [
        _row("PIK3CA", disease="EFO_1", datatype="genetic_association"),
        _row("TNF", disease="EFO_2", datatype="literature"),
    ])
    assert diseases_with_genetic_evidence(store, "18.06") == frozenset({"EFO_1"})


def test_corpus_requires_both_genetic_evidence_and_a_phase_two_target(tmp_path):
    store = _store(tmp_path, [
        _row("PIK3CA", disease="EFO_1"),   # genetic, and has a phase-2 target
        _row("BRCA1", disease="EFO_2"),    # genetic, but no phase-2 target
        _row("TNF", disease="EFO_3", datatype="literature"),  # phase-2, no genetics
    ])
    corpus = corpus_diseases(
        store, "18.06", phase2_targets={"EFO_1": {"PIK3CA"}, "EFO_3": {"TNF"}},
    )
    assert corpus == frozenset({"EFO_1"})


def test_corpus_is_empty_when_no_disease_meets_both_criteria(tmp_path):
    store = _store(tmp_path, [_row("BRCA1", disease="EFO_2")])
    assert corpus_diseases(store, "18.06", phase2_targets={}) == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_frame.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.snapshots.frame'`

- [ ] **Step 3: Write the module**

Create `src/neorx/snapshots/frame.py`:

```python
"""
The candidate frame, and which diseases the corpus contains.

A dated run must evaluate the population that date would have evaluated.
If a gene enters the candidate pool because a later source connected it to
something, the analysis is scoring a gene the date would never have looked
at -- hindsight in the sampling frame rather than in the predictor, and
just as fatal to the result.

So the frame is drawn from the pinned release alone, and only from the
evidence types that confer causal admissibility. A gene supported by
literature co-mention is not in the frame, because such an edge is not
admissible as a causal arrow and the gene has nothing for identification
to reason about.

The disease corpus is chosen by the same principle one level up. Selecting
diseases because they look well studied today, then analysing them as of
2018, is the same leak. Both criteria are applied to the pinned release.
"""

from __future__ import annotations

import polars as pl

from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.schema import GENETIC_DATATYPES

__all__ = [
    "candidate_frame",
    "corpus_diseases",
    "diseases_with_genetic_evidence",
]


def _genetic(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("datatype").is_in(list(GENETIC_DATATYPES)) & (pl.col("score") > 0.0)
    )


def candidate_frame(
    store: SnapshotStore,
    release: str,
    disease_id: str,
) -> frozenset[str]:
    """Gene symbols the release associated with a disease on genetic evidence.

    This is the population a dated run may evaluate. Nothing outside it may
    become a candidate, whatever a live source later contributes.
    """
    df = _genetic(store.associations(release)).filter(
        pl.col("disease_id") == disease_id
    )
    return frozenset(df["target_symbol"].to_list())


def diseases_with_genetic_evidence(
    store: SnapshotStore,
    release: str,
) -> frozenset[str]:
    """Diseases carrying at least one genetically-supported association.

    A disease without one has an empty causal subgraph, so its
    identifiability is trivially zero. Including such diseases would add
    noise to the corpus rather than signal.
    """
    return frozenset(_genetic(store.associations(release))["disease_id"].to_list())


def corpus_diseases(
    store: SnapshotStore,
    release: str,
    phase2_targets: dict[str, set[str]],
) -> frozenset[str]:
    """Diseases meeting both corpus criteria for this release.

    A disease enters when it has genetic evidence in the release *and* at
    least one target that reached Phase II or beyond, because without the
    latter there is no clinical outcome to predict against.

    ``phase2_targets`` maps disease id to the target symbols that reached
    Phase II. Sub-project 5 produces it; passing it in keeps this function
    a pure predicate over the release rather than a second place that
    knows about trials.
    """
    genetic = diseases_with_genetic_evidence(store, release)
    with_trials = {d for d, targets in phase2_targets.items() if targets}
    return frozenset(genetic & with_trials)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/snapshots/frame.py tests/snapshots/test_frame.py
git commit -m "feat: derive the candidate frame and disease corpus from pinned releases"
```

---

## Task 9: The frame gate

**Files:**
- Modify: `src/neorx/core/causal/identifier.py` (`_get_candidate_nodes`, and its one caller in `identify_causal_targets`)
- Test: `tests/core/test_frame_gate.py`

**Interfaces:**
- Consumes: nothing from the snapshots package — the gate takes a plain `frozenset[str]`, so `identifier.py` gains no dependency on snapshot storage.
- Produces: `_get_candidate_nodes(G, disease_node_id, frame=None) -> tuple[list[str], list[dict]]`. Second element is the exclusion record: one dict per rejected node with keys `node_id`, `symbol`, `source`.

**This is the task the whole sub-project exists to make possible.** Exclusion is deliberately not an error — Monarch legitimately returns genes OpenTargets did not, so refusing would fail every dated run for a normal condition. But a silent filter is how leakage returns after a refactor, so exclusions are counted and named.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_frame_gate.py
"""No gene outside the frame may become a candidate.

_get_candidate_nodes admits every gene, protein and pathogen_gene node in
the assembled graph, so any source contributing a gene node contributes a
candidate -- Monarch, ChEMBL, STRING, KEGG and Reactome all do. On a dated
run that is leakage in the sampling frame: a gene the date would never
have evaluated, scored as though it would have been.

The gate filters rather than raises, because Monarch legitimately returns
genes OpenTargets did not and refusing would fail every dated run. What
makes the filter safe is that every exclusion is recorded with the source
that contributed it, so a run that suddenly drops four hundred genes where
it used to drop twelve is visible rather than absorbed.
"""

import networkx as nx

from neorx.core.causal.identifier import _get_candidate_nodes


def _graph():
    G = nx.DiGraph()
    G.add_node("disease:d", node_type="disease", name="d", source="NeoRx")
    G.add_node("gene:PIK3CA", node_type="gene", name="PIK3CA", source="Open Targets")
    G.add_node("gene:TP53", node_type="gene", name="TP53", source="Open Targets")
    # Contributed by a source that bypasses any gene-list restriction.
    G.add_node("gene:LATER", node_type="gene", name="LATER", source="Monarch")
    return G


def test_without_a_frame_every_gene_is_a_candidate():
    # The undated path must behave exactly as before.
    candidates, excluded = _get_candidate_nodes(_graph(), "disease:d")
    assert set(candidates) == {"gene:PIK3CA", "gene:TP53", "gene:LATER"}
    assert excluded == []


def test_a_gene_outside_the_frame_never_becomes_a_candidate():
    candidates, _ = _get_candidate_nodes(
        _graph(), "disease:d", frame=frozenset({"PIK3CA", "TP53"}),
    )
    assert "gene:LATER" not in candidates
    assert set(candidates) == {"gene:PIK3CA", "gene:TP53"}


def test_every_exclusion_is_recorded_with_its_source():
    _, excluded = _get_candidate_nodes(
        _graph(), "disease:d", frame=frozenset({"PIK3CA", "TP53"}),
    )
    assert excluded == [
        {"node_id": "gene:LATER", "symbol": "LATER", "source": "Monarch"},
    ]


def test_an_empty_frame_excludes_every_gene_and_records_them_all():
    candidates, excluded = _get_candidate_nodes(
        _graph(), "disease:d", frame=frozenset(),
    )
    assert candidates == []
    assert len(excluded) == 3


def test_the_disease_node_is_never_a_candidate_or_an_exclusion():
    _, excluded = _get_candidate_nodes(
        _graph(), "disease:d", frame=frozenset({"PIK3CA"}),
    )
    assert all(e["node_id"] != "disease:d" for e in excluded)


def test_non_gene_nodes_are_neither_candidates_nor_exclusions():
    G = _graph()
    G.add_node("pathway:1", node_type="pathway", name="p", source="KEGG")
    candidates, excluded = _get_candidate_nodes(
        G, "disease:d", frame=frozenset({"PIK3CA"}),
    )
    assert "pathway:1" not in candidates
    assert all(e["node_id"] != "pathway:1" for e in excluded)


def test_pathogen_genes_are_gated_like_any_other_candidate():
    G = _graph()
    G.add_node("gene:POL", node_type="pathogen_gene", name="POL", source="ChEMBL")
    candidates, excluded = _get_candidate_nodes(
        G, "disease:d", frame=frozenset({"PIK3CA"}),
    )
    assert "gene:POL" not in candidates
    assert any(e["symbol"] == "POL" for e in excluded)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_frame_gate.py -v`
Expected: FAIL — `TypeError: _get_candidate_nodes() got an unexpected keyword argument 'frame'`

- [ ] **Step 3: Implement the gate**

Replace `_get_candidate_nodes` in `src/neorx/core/causal/identifier.py`:

```python
def _get_candidate_nodes(
    G: nx.DiGraph,
    disease_node_id: str,
    frame: frozenset[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Gene/protein nodes that could be drug targets, and what was excluded.

    With no ``frame`` every eligible node is a candidate, which is the
    undated behaviour and is unchanged.

    With a ``frame`` -- the gene symbols the pinned release associated with
    this disease on genetic evidence -- only nodes inside it are admitted.
    This closes the sampling-frame leak: several sources add gene nodes for
    the disease directly rather than from a requested gene list, so
    restricting what is fetched does not restrict what becomes a candidate.

    Exclusion is not an error. Monarch and ChEMBL legitimately return genes
    OpenTargets did not, so refusing would fail every dated run for a
    condition that is normal. What makes filtering safe is the second
    return value: every excluded node is named with the source that
    contributed it, so a run that suddenly excludes far more than it used
    to is visible in the record rather than absorbed silently.
    """
    candidates: list[str] = []
    excluded: list[dict[str, str]] = []

    for node_id, data in G.nodes(data=True):
        ntype = data.get("node_type", "")
        if ntype not in ("gene", "protein", "pathogen_gene"):
            continue
        if node_id == disease_node_id:
            continue

        if frame is not None and data.get("name", "") not in frame:
            excluded.append({
                "node_id": node_id,
                "symbol": data.get("name", ""),
                "source": data.get("source", ""),
            })
            continue

        candidates.append(node_id)

    return candidates, excluded
```

- [ ] **Step 4: Update the caller**

In `identify_causal_targets`, the call site currently reads
`candidates = _get_candidate_nodes(G, disease_node_id)`. Change it to unpack
both values and log the exclusion count:

```python
    candidates, frame_excluded = _get_candidate_nodes(G, disease_node_id)
    if frame_excluded:
        logger.info(
            "Frame gate excluded %d candidate nodes.", len(frame_excluded),
        )
```

Leave `identify_causal_targets`'s own signature alone for now — Task 11 threads
`as_of` through it. This step only stops the existing caller breaking.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/test_frame_gate.py tests/core/test_identifier.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/neorx/core/causal/identifier.py tests/core/test_frame_gate.py
git commit -m "feat: gate candidate selection on the pinned frame, recording exclusions"
```

---

## Task 10: SourceResolver

**Files:**
- Create: `src/neorx/snapshots/resolver.py`
- Test: `tests/snapshots/test_resolver.py`

**Interfaces:**
- Consumes: `SnapshotStore`, `SnapshotMissingError`.
- Produces: `SourceResolver(store, release_for)` with `resolve(source: str, as_of: str | None) -> Callable`, and `UnpinnedSourceError`.

`release_for` maps an `as_of` date to the release id for a source, so the
date→release table lives in one place rather than at every call site.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_resolver.py
"""Choosing between live and snapshot, and refusing in between.

A dated run that quietly mixes in current data produces exactly the
anachronistic graph this sub-project exists to prevent, and does so
invisibly. So a source with no snapshot at a requested date is a named
refusal, never a fallback.
"""

import pytest

from neorx.snapshots.resolver import SourceResolver, UnpinnedSourceError


class _FakeStore:
    def __init__(self, present):
        self._present = present

    def has(self, source, release):
        return (source, release) in self._present


RELEASES = {
    "2018-06": {"opentargets": "18.06", "omnipath": "20180614"},
    "2025-06": {"opentargets": "25.06", "omnipath": "20250813"},
}


def _resolver(present=(("opentargets", "18.06"), ("omnipath", "20180614"))):
    return SourceResolver(
        store=_FakeStore(set(present)),
        release_for=lambda source, as_of: RELEASES[as_of][source],
        live={"opentargets": lambda: "LIVE_OT", "omnipath": lambda: "LIVE_OMNI",
              "string": lambda: "LIVE_STRING"},
    )


def test_no_date_returns_the_live_client():
    assert _resolver().resolve("opentargets", as_of=None)() == "LIVE_OT"


def test_an_unpinned_source_stays_live_even_on_a_dated_run():
    # STRING contributes only associational edges, which identification
    # filters out, so pinning it would buy rigour against a leak that does
    # not exist.
    assert _resolver().resolve("string", as_of="2018-06")() == "LIVE_STRING"


def test_a_dated_run_uses_the_snapshot_for_a_pinned_source():
    reader = _resolver().resolve("opentargets", as_of="2018-06")
    assert reader() != "LIVE_OT"


def test_a_missing_snapshot_refuses_rather_than_falling_back():
    with pytest.raises(UnpinnedSourceError):
        _resolver().resolve("opentargets", as_of="2025-06")


def test_the_refusal_names_the_source_and_the_release():
    with pytest.raises(UnpinnedSourceError, match="opentargets"):
        _resolver().resolve("opentargets", as_of="2025-06")
    with pytest.raises(UnpinnedSourceError, match="25.06"):
        _resolver().resolve("opentargets", as_of="2025-06")


def test_the_refusal_says_it_will_not_fall_back():
    # The message is the mechanism: someone hitting this must not conclude
    # that omitting the date is the fix.
    with pytest.raises(UnpinnedSourceError, match="not fall back"):
        _resolver().resolve("omnipath", as_of="2025-06")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.snapshots.resolver'`

- [ ] **Step 3: Write the module**

Create `src/neorx/snapshots/resolver.py`:

```python
"""
Choosing between a live client and a snapshot reader.

With no date, every source resolves to the live client it always used and
behaviour is unchanged -- interactive single-disease use is unaffected by
this whole sub-project.

With a date, the sources feeding the causal subgraph resolve to snapshot
readers, and a source with no snapshot at that date is refused by name.
The refusal matters more than it looks: a dated run that quietly mixed in
current data would produce exactly the anachronistic graph this
sub-project exists to prevent, and would do it invisibly.

Sources that contribute only associational edges stay live even on a dated
run. Identification filters those edges out before it runs, so pinning
them would buy rigour against a leak that does not exist.
"""

from __future__ import annotations

from typing import Callable, Protocol

__all__ = ["SourceResolver", "UnpinnedSourceError"]

# Sources whose data reaches the causal subgraph, and therefore must be
# pinned on a dated run. Everything else contributes edges that
# graph_semantics filters out before identification.
PINNED_SOURCES = frozenset({"opentargets", "omnipath"})


class UnpinnedSourceError(RuntimeError):
    """A dated run asked for a source that has no snapshot at that date."""


class _Store(Protocol):
    def has(self, source: str, release: str) -> bool: ...


class SourceResolver:
    """Maps a source name plus an optional date to a reader.

    ``release_for(source, as_of)`` maps a date to a release identifier, so
    the date-to-release table lives in one place rather than at every call
    site.
    """

    def __init__(
        self,
        store: _Store,
        release_for: Callable[[str, str], str],
        live: dict[str, Callable[..., object]],
    ) -> None:
        self._store = store
        self._release_for = release_for
        self._live = live

    def resolve(self, source: str, as_of: str | None) -> Callable[..., object]:
        """Return the reader for this source at this date.

        Raises ``UnpinnedSourceError`` if the date requires a snapshot that
        does not exist. It does not fall back to the live client.
        """
        if as_of is None or source not in PINNED_SOURCES:
            return self._live[source]

        release = self._release_for(source, as_of)
        if not self._store.has(source, release):
            raise UnpinnedSourceError(
                f"A run dated {as_of} needs the {source} snapshot for release "
                f"{release!r}, which has not been built. Build it with "
                f"`neorx snapshot build {source} {release}`. This will not "
                f"fall back to live data: doing so would silently mix current "
                f"evidence into a dated graph."
            )

        return lambda: (source, release)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/snapshots/resolver.py tests/snapshots/test_resolver.py
git commit -m "feat: resolve sources to live or snapshot, refusing to fall back"
```

---

## Task 11: Snapshot CLI

**Files:**
- Create: `src/neorx/snapshots/__main__.py`
- Modify: `src/neorx/cli/__init__.py`
- Test: `tests/snapshots/test_cli.py`

**Interfaces:**
- Consumes: every reader, `SnapshotStore`, `write_entry`, `digest_file`, `SnapshotEntry`.
- Produces: `neorx snapshot build <source> <release>`, `neorx snapshot list`.

This is where downloading happens. `build` streams, reduces, writes the extract,
digests it, records the manifest entry, and discards the raw download.

- [ ] **Step 1: Write the failing test**

```python
# tests/snapshots/test_cli.py
"""The snapshot CLI.

Downloading lives here and only here, so every reader stays a pure
function of already-fetched data and unit tests need no network. These
tests exercise the wiring with a local file rather than a URL.
"""

from typer.testing import CliRunner

from neorx.snapshots.__main__ import app

runner = CliRunner()


def test_list_on_an_empty_root_reports_no_snapshots(tmp_path):
    result = runner.invoke(app, ["list", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "no snapshots" in result.stdout.lower()


def test_build_writes_an_extract_and_a_manifest_entry(tmp_path):
    src = tmp_path / "18.06.jsonl"
    src.write_text(
        '{"target": {"id": "ENSG1", "gene_info": {"symbol": "PIK3CA"}},'
        ' "disease": {"id": "EFO_1"},'
        ' "association_score": {"datatypes": {"genetic_association": 0.6}}}\n'
    )
    root = tmp_path / "snapshots"
    result = runner.invoke(app, [
        "build", "opentargets", "18.06",
        "--from-file", str(src), "--root", str(root),
    ])
    assert result.exit_code == 0, result.stdout
    assert (root / "opentargets" / "18.06" / "associations.parquet").exists()
    assert (root / "manifest.toml").exists()


def test_list_reports_a_built_snapshot_with_its_row_count(tmp_path):
    src = tmp_path / "18.06.jsonl"
    src.write_text(
        '{"target": {"id": "ENSG1", "gene_info": {"symbol": "PIK3CA"}},'
        ' "disease": {"id": "EFO_1"},'
        ' "association_score": {"datatypes": {"genetic_association": 0.6}}}\n'
    )
    root = tmp_path / "snapshots"
    runner.invoke(app, ["build", "opentargets", "18.06",
                        "--from-file", str(src), "--root", str(root)])
    result = runner.invoke(app, ["list", "--root", str(root)])
    assert "18.06" in result.stdout
    assert "1" in result.stdout


def test_building_an_unknown_source_fails_with_a_named_error(tmp_path):
    result = runner.invoke(app, [
        "build", "nosuchsource", "1.0", "--root", str(tmp_path),
    ])
    assert result.exit_code != 0
    assert "nosuchsource" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/snapshots/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neorx.snapshots.__main__'`

- [ ] **Step 3: Write the CLI**

Create `src/neorx/snapshots/__main__.py` with a Typer `app` exposing `build`
and `list`. Follow the shape of `src/neorx/experiments/__main__.py`, which is
the closest existing CLI in this codebase — read it first and match its
conventions for option naming and output.

`build` takes `source`, `release`, `--root` (default `snapshots/`),
`--from-file` (a local path, used by tests and for a release already
downloaded) and `--url` (fetched with streaming `requests`). It must:

1. dispatch on `source` and, for `opentargets`, on the release era —
   `18.06` to `read_1806`, `21.11` to `read_2111`, anything else to
   `read_modern`; raise a named error for an unknown source
2. write the canonical Parquet to `<root>/<source>/<release>/`
3. digest the written extract with `digest_file`
4. record a `SnapshotEntry` via `write_entry`
5. delete the raw download when it fetched one, keeping only the extract

`list` reads the manifest and prints one line per entry: source, release,
row count, extractor version, and the first twelve characters of the digest.

Register it in `src/neorx/cli/__init__.py` beside the existing `exp`
subcommand:

```python
from neorx.snapshots.__main__ import app as snapshot_app
app.add_typer(snapshot_app, name="snapshot", help="Build and inspect versioned source snapshots.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/snapshots/ tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/snapshots/__main__.py src/neorx/cli/__init__.py tests/snapshots/test_cli.py
git commit -m "feat: add the snapshot build and list commands"
```

---

## Task 12: Measure the corpus

**Files:**
- Create: `experiments/corpus_census.py`
- Test: `tests/experiments/test_corpus_census.py`

**Interfaces:**
- Consumes: `SnapshotStore`, `diseases_with_genetic_evidence`, `candidate_frame`.
- Produces: a registered experiment `corpus-census` producing one row per release: `release`, `n_diseases_total`, `n_diseases_with_genetic_evidence`, `median_frame_size`, `n_target_disease_pairs`.

**This answers the question the spec deliberately left open.** The spec gives
criteria, not a number, because the count is a property of each release. This
measures it, through sub-project 2's run recorder so the answer is itself a
recorded artifact rather than a figure someone typed into a document.

Note it measures only the first corpus criterion. The Phase II criterion needs
sub-project 5's trial data, which does not exist yet — so this reports the
population that criterion will narrow, and says so.

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_corpus_census.py
"""How many diseases the corpus actually contains.

The spec gives criteria rather than a number, because the count is a
property of each release. This measures it -- and does so through the run
recorder, so the answer is an artifact with provenance rather than a
figure someone typed into a document.
"""

import polars as pl

from experiments.corpus_census import census_row
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.schema import ASSOCIATION_COLUMNS


def _store(tmp_path, rows):
    d = tmp_path / "opentargets" / "18.06"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=ASSOCIATION_COLUMNS).write_parquet(
        d / "associations.parquet"
    )
    return SnapshotStore(tmp_path)


def _row(symbol, disease, datatype="genetic_association", score=0.5):
    return {"target_id": f"ENSG_{symbol}", "target_symbol": symbol,
            "disease_id": disease, "datatype": datatype, "score": score}


def test_counts_diseases_with_and_without_genetic_evidence(tmp_path):
    store = _store(tmp_path, [
        _row("A", "EFO_1"), _row("B", "EFO_1"),
        _row("C", "EFO_2"),
        _row("D", "EFO_3", datatype="literature"),
    ])
    row = census_row(store, "18.06")
    assert row["n_diseases_total"] == 3
    assert row["n_diseases_with_genetic_evidence"] == 2


def test_reports_the_median_frame_size(tmp_path):
    store = _store(tmp_path, [
        _row("A", "EFO_1"), _row("B", "EFO_1"), _row("C", "EFO_1"),
        _row("D", "EFO_2"),
    ])
    row = census_row(store, "18.06")
    assert row["median_frame_size"] == 2.0


def test_counts_distinct_target_disease_pairs_not_rows(tmp_path):
    store = _store(tmp_path, [
        _row("A", "EFO_1", datatype="genetic_association"),
        _row("A", "EFO_1", datatype="somatic_mutation"),
    ])
    assert census_row(store, "18.06")["n_target_disease_pairs"] == 1


def test_an_empty_release_reports_zeros_not_an_error(tmp_path):
    store = _store(tmp_path, [])
    row = census_row(store, "18.06")
    assert row["n_diseases_with_genetic_evidence"] == 0
    assert row["median_frame_size"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/experiments/test_corpus_census.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.corpus_census'`

- [ ] **Step 3: Write the experiment**

Create `experiments/corpus_census.py`. Read `experiments/neorx_7disease.py`
first for the `@experiment` decorator's exact signature and how a row reaches
`record.append_row` — match it.

`census_row(store, release) -> dict` computes, over the genetic subset of the
release's associations: `n_diseases_total` (distinct diseases in the release),
`n_diseases_with_genetic_evidence`, `median_frame_size` (median distinct
symbols per disease, `0.0` when there are none), and
`n_target_disease_pairs` (distinct `(target_symbol, disease_id)` pairs).

The registered experiment iterates the three releases, appending one row each,
so a partial run keeps what it measured. Its docstring must state that the
Phase II criterion is not applied here because sub-project 5's trial data does
not exist yet, and that these counts are therefore an upper bound on the corpus.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_corpus_census.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/corpus_census.py tests/experiments/test_corpus_census.py
git commit -m "feat: measure the disease corpus per release, as a recorded run"
```

---

## Task 13: Thread `as_of` through graph construction

**Files:**
- Modify: `src/neorx/core/graph/graph_builder.py` (`build_disease_graph`)
- Modify: `src/neorx/core/causal/identifier.py` (`identify_causal_targets`)
- Test: `tests/core/test_dated_build.py`

**Interfaces:**
- Consumes: `SourceResolver`, `candidate_frame`, the Task 9 gate.
- Produces: `build_disease_graph(..., as_of: str | None = None, resolver: SourceResolver | None = None)` and `identify_causal_targets(graph, ..., frame: frozenset[str] | None = None)`.

The last wiring task. After it, a dated run is possible end to end.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_dated_build.py
"""A dated build end to end.

The invariant a reviewer would ask for: a gene injected into a dated run's
assembled graph, from a source that bypasses gene-list restriction, never
reaches the candidate list -- and its exclusion is recorded.
"""

import networkx as nx
import pytest

from neorx.core.causal.identifier import identify_causal_targets
from neorx.core.graph.models import (
    DiseaseGraph, EdgeType, GraphEdge, GraphNode, NodeType,
)


def _graph_with_an_off_frame_gene():
    nodes = [
        GraphNode(node_id="gene:PIK3CA", name="PIK3CA", node_type=NodeType.GENE,
                  source="Open Targets", score=0.9),
        GraphNode(node_id="gene:LATER", name="LATER", node_type=NodeType.GENE,
                  source="Monarch", score=0.9),
        GraphNode(node_id="disease:d", name="d", node_type=NodeType.DISEASE,
                  source="NeoRx", score=1.0),
    ]
    edges = [
        GraphEdge(source_id=n.node_id, target_id="disease:d",
                  edge_type=EdgeType.ASSOCIATED_WITH, weight=0.9,
                  source_db="Open Targets", evidence_class="genetic_association")
        for n in nodes[:2]
    ]
    return DiseaseGraph(disease_name="d", disease_id="disease:d",
                        nodes=nodes, edges=edges)


def test_an_off_frame_gene_never_reaches_the_results():
    results = identify_causal_targets(
        _graph_with_an_off_frame_gene(), top_n=10,
        frame=frozenset({"PIK3CA"}),
    )
    assert all(r.gene_name != "LATER" for r in results)


def test_without_a_frame_the_same_gene_is_evaluated():
    results = identify_causal_targets(_graph_with_an_off_frame_gene(), top_n=10)
    assert any(r.gene_name == "LATER" for r in results)


def test_a_dated_build_refuses_when_the_snapshot_is_missing(tmp_path):
    from neorx.core.graph.graph_builder import build_disease_graph
    from neorx.snapshots.reader import SnapshotStore
    from neorx.snapshots.resolver import SourceResolver, UnpinnedSourceError

    resolver = SourceResolver(
        store=SnapshotStore(tmp_path),
        release_for=lambda source, as_of: "18.06",
        live={},
    )
    with pytest.raises(UnpinnedSourceError):
        build_disease_graph("HIV infection", as_of="2018-06", resolver=resolver)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_dated_build.py -v`
Expected: FAIL — `TypeError: identify_causal_targets() got an unexpected keyword argument 'frame'`

- [ ] **Step 3: Thread the frame through identification**

In `identify_causal_targets`, add `frame: frozenset[str] | None = None` to the
signature, pass it to `_get_candidate_nodes`, and log the exclusion count with
the disease name so a corpus run's log shows the gate working per disease.

Document in the docstring that `frame` is the population a dated run may
evaluate, that `None` means undated and unchanged behaviour, and that exclusions
are logged rather than raised because several sources legitimately contribute
genes outside a given release's frame.

- [ ] **Step 4: Thread `as_of` through graph construction**

In `build_disease_graph`, add `as_of: str | None = None` and
`resolver: SourceResolver | None = None`. When `as_of` is set, resolve
`opentargets` and `omnipath` through the resolver before any fetching, so a
missing snapshot fails before the function does 25 minutes of work against other
sources. Leave every other source's call site alone.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/core/ tests/snapshots/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/neorx/core/graph/graph_builder.py src/neorx/core/causal/identifier.py tests/core/test_dated_build.py
git commit -m "feat: build and identify against a dated, frame-pinned graph"
```

---

## Task 14: Snapshot provenance in run records

**Files:**
- Modify: `src/neorx/experiments/record.py`
- Test: `tests/experiments/test_snapshot_provenance.py`

**Interfaces:**
- Consumes: `read_manifest` from Task 6. NOTE the existing signature is
  `RunRecord.create(experiment, *, runs_dir=None, allow_large=False)` —
  `experiment` is positional and `runs_dir` is keyword-only.
- Produces: `env.json` gains a `snapshots` object mapping `"<source>/<release>"` to `{sha256, extractor_version}`.

Sub-project 2 records the code SHA. A dated run's inputs are equally part of what
produced a number, and a derived artifact needs its derivation pinned.

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_snapshot_provenance.py
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
    write_entry(m, SnapshotEntry(
        source="opentargets", release="18.06",
        url="https://example.invalid/18.06", sha256="b" * 64,
        extractor_version=1, rows=42,
    ))
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


def test_an_undated_run_records_no_snapshots(tmp_path):
    record = RunRecord.create("census", runs_dir=tmp_path / "runs")
    env = json.loads((record.path / "env.json").read_text())
    assert env.get("snapshots", {}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/experiments/test_snapshot_provenance.py -v`
Expected: FAIL — `TypeError: create() got an unexpected keyword argument 'snapshot_manifest'`

- [ ] **Step 3: Implement**

Read `RunRecord.create` in `src/neorx/experiments/record.py` first — it already
builds `env.json` with the code SHA, dirty flag, Python version, dependencies and
platform. Add an optional `snapshot_manifest: Path | None = None` parameter.
When given, read it with `read_manifest` and add a `snapshots` key mapping
`f"{source}/{release}"` to `{"sha256": ..., "extractor_version": ...}`. When
absent, `snapshots` is an empty object rather than missing, so consumers do not
need to branch on its presence.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/experiments/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/neorx/experiments/record.py tests/experiments/test_snapshot_provenance.py
git commit -m "feat: record snapshot digests and extractor versions in run provenance"
```

---

## Self-Review

**Spec coverage.**

| Spec section | Task |
|---|---|
| Scope of pinning — two sources, rest live | 10 (`PINNED_SOURCES`) |
| The frame gate — three layers | 8 (layer 1), 13 (layer 2), 9 (layer 3) |
| Which diseases — criteria, count measured not asserted | 8 (`corpus_diseases`), 12 (census) |
| Derived snapshots, not raw releases | 1, 11 |
| Three time points, three OT readers | 2, 3, 4 |
| OmniPath reader reuses SP3's pure parser | 5 |
| Choosing snapshot or live; refuse, never fall back | 10, 13 |
| Provenance — release, digest, extractor version | 6, 14 |
| polars, not pandas | Global constraints; every reader |
| Success criterion 6 — off-frame gene never a candidate | 9, 13 |

**Placeholder scan.** Tasks 4, 11, 12, 13 and 14 name what to write and where,
with the surrounding code specified, rather than quoting every line. Task 4 is a
genuine characterisation task — its layout is the one thing this plan could not
verify in advance, and inventing field names would be worse than a documented
discovery step. Tasks 11–14 modify existing files whose conventions the
implementer must read; specifying "match `experiments/__main__.py`" is a real
instruction with a real referent. Every test body is given in full.

One stub is deliberately left in Task 8's test file with an instruction to delete
it before running — flagged because a test asserting nothing is a defect by this
project's own standard, and leaving it silently would contradict `CLAUDE.md`.

**Type consistency.** `ASSOCIATION_COLUMNS` is defined in Task 1 and consumed by
Tasks 2, 3, 4, 7, 8 and 12 under that name. `SnapshotStore.associations(release)`
is defined in Task 7 and called in Tasks 8 and 12 with that signature.
`_get_candidate_nodes` returns `tuple[list[str], list[dict]]` from Task 9 and is
unpacked that way in Tasks 9 and 13. `candidate_frame(store, release, disease_id)`
is defined in Task 8 and used in Task 13. `SnapshotEntry`'s six fields are written
in Task 6 and read in Task 14.

**One thing this plan does not do.** It builds the machinery and measures the
corpus, but does not run the three extractions — those are downloads of 0.18 GB
and up, and belong in a session where someone can watch them. Task 11 gives the
command; the first real `neorx snapshot build` is the natural first act of
sub-project 5.
