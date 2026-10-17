# Versioned snapshots

A dated run must read what that date would have read. Otherwise the
analysis is using hindsight — and not the obvious kind, in the predictor,
but the subtler kind in the *sampling frame*, where a gene is scored
because a later source connected it to something.

The snapshot layer exists so that a run over a 2018 release reads 2018.

## Derived extracts, not mirrors

Upstream releases are streamed, reduced to a compact Parquet extract, and
discarded. What persists is four columns of gene–disease evidence and a
table of directed regulatory interactions — enough to rebuild the causal
subgraph as it would have looked at a past date, and small enough that a
decade of releases costs less disk than one of them.

```python
>>> from neorx.snapshots.schema import ASSOCIATION_COLUMNS, INTERACTION_COLUMNS
>>> list(ASSOCIATION_COLUMNS)
['target_id', 'target_symbol', 'disease_id', 'datatype', 'score']
>>> list(INTERACTION_COLUMNS)
['source_symbol', 'target_symbol', 'is_directed', 'consensus_direction',
 'is_stimulation', 'is_inhibition', 'primary_sources', 'references']

```

Three Open Targets format eras feed this: a flat gzipped JSON dump in
2018, an ETL-era Parquet tree in 2021, and the current output layout.
Each gets its own reader, and each reader's job is to emit exactly those
columns. **Downstream code never learns which era a row came from** —
that is the whole value of a derived extract over a mirror. Format drift
is absorbed once, at the edge, instead of leaking into every consumer.

The per-era field mappings are recorded in
[OpenTargets release layouts](snapshot-layouts.md), characterised against
the live archive rather than inferred.

## Only genetic evidence reaches the frame

The candidate frame for a dated run is drawn from the pinned release
alone, and only from the datatypes that confer causal admissibility:

```python
>>> from neorx.snapshots.schema import GENETIC_DATATYPES
>>> sorted(GENETIC_DATATYPES)
['genetic_association', 'somatic_mutation']

```

That is the same set as
[`GENETIC_EVIDENCE_CLASSES`](admissibility.md#gene-to-disease-needs-genetics),
and for the same reason. A gene supported only by literature co-mention
is not in the frame, because such an edge is not admissible as a causal
arrow and the gene has nothing for identification to reason about.

Adding a name to `GENETIC_DATATYPES` silently widens both the frame and
what the backdoor criterion may reason over.

The same principle applies one level up, to the disease corpus. Selecting
diseases because they look well studied *today* and then analysing them
as of 2018 is the same leak, so both criteria are applied to the pinned
release.

## Building one

```bash
neorx snapshot build opentargets 21.11
neorx snapshot build omnipath 20211113-20220114
neorx snapshot list
```

`build` fetches the release, reduces it to the canonical schema, writes
the Parquet extract, and writes the manifest entry beside it.

## The manifest

A release identifier alone does not identify an extract. Two extracts of
Open Targets 18.06 produced by different extractor code are **different
inputs** to every number computed downstream, so the extractor version is
recorded next to the release id, and the SHA-256 digest is taken over the
extract actually written rather than over the upstream file.

```python
>>> from neorx.snapshots.manifest import SnapshotEntry
>>> from neorx.snapshots.schema import EXTRACTOR_VERSION
>>> EXTRACTOR_VERSION
2
>>> entry = SnapshotEntry(
...     source="opentargets", release="21.11",
...     url="https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.11/",
...     sha256="0" * 64, extractor_version=EXTRACTOR_VERSION, rows=1234,
... )
>>> entry.source, entry.release, entry.rows
('opentargets', '21.11', 1234)

```

`EXTRACTOR_VERSION` is bumped when a reader's output changes meaning for
the same input. Version 2 exists because the OmniPath archive reader now
filters to human–human interactions: 49% of the 2018 dump is mouse and
rat (242,338 and 73,989 rows of 644,845), and nothing downstream marked
them. An extract built before that change and one built after are
different inputs for the same release, and the manifest has to be able to
say which you have.

Entries are keyed by `(source, release)`, so re-running an extraction
replaces its entry rather than appending a second one.

### What a reader had to do, that the rows no longer show

```python
>>> from neorx.snapshots.schema import ExtractNotes
>>> notes = ExtractNotes(consensus_direction_synthesised=True,
...                      non_human_rows_dropped=316327)
>>> notes.consensus_direction_synthesised
True
>>> notes.non_human_rows_dropped
316327

```

Both facts are recorded rather than defaulted away, because both change
what the extract *means* and neither is visible in the extract itself.
`consensus_direction_synthesised` says the 2018 archive had no such
column and `is_directed` was used in its place — a weaker criterion than
later eras publish. And a `non_human_rows_dropped` of zero on a real
OmniPath archive means the filter did not fire, not that the dump was
human-only.

## Reading them back

```python
>>> import tempfile
>>> from pathlib import Path
>>> from neorx.snapshots.reader import SnapshotStore, SnapshotMissingError
>>> with tempfile.TemporaryDirectory() as tmp:
...     store = SnapshotStore(Path(tmp), on_read=None)
...     store.has("opentargets", "21.11")
...     try:
...         store.associations("21.11")
...     except SnapshotMissingError as exc:
...         print("refused:", "will not fall back to live data" in str(exc))
False
refused: True

```

**A missing snapshot raises rather than returning an empty frame.** An
empty frame reads downstream as *"this release recorded no evidence for
that disease"*, which is a finding. A missing extract is a configuration
mistake. The two must not be confusable, and the error names the exact
`neorx snapshot build` command that fixes it.

`on_read` is how a [run record](experiments.md) learns what a run
actually read, so it can cite exactly that — from this store's own
manifest, rather than from a second path that might name a different
store. It has no default, deliberately: when it defaulted to `None` an
experiment could read one extract through `RunRecord.snapshot_store` and
another through a store it built itself, and finish `complete` and
`citable` while citing only the first.

## Resolving a date

`SourceResolver` maps a source plus an optional date to a reader. With no
date, every source resolves to the live client it always used, so
interactive single-disease use is unaffected by any of this.

```python
>>> from neorx.snapshots.resolver import PINNED_SOURCES
>>> sorted(PINNED_SOURCES)
['omnipath', 'opentargets']

```

Only those two are pinned, because only those two reach the causal
subgraph. Sources contributing associational edges stay live even on a
dated run: identification filters those edges out before it runs, so
pinning them would buy rigour against a leak that does not exist.

A dated run that asks for a source with no snapshot at that date is
**refused by name**:

```python
>>> from neorx.snapshots.resolver import SourceResolver, UnpinnedSourceError
>>> from neorx.snapshots.releases import release_for
>>> import tempfile
>>> from pathlib import Path
>>> with tempfile.TemporaryDirectory() as tmp:
...     store = SnapshotStore(Path(tmp), on_read=None)
...     resolver = SourceResolver(store=store, release_for=release_for, live={})
...     try:
...         resolver.resolve("opentargets", "2021-11")
...     except UnpinnedSourceError as exc:
...         print("21.11" in str(exc), "fall back to live data" in str(exc))
True True

```

The refusal matters more than it looks. A dated run that quietly mixed in
current data would produce exactly the anachronistic graph this layer
exists to prevent, and would do it invisibly.

## A limitation recorded rather than smoothed over

The pinned release table pairs each date's Open Targets release with the
OmniPath dump whose window contains it. For 2018 and 2021 the two are
within days of each other. For 2025 they are not:

```python
>>> from neorx.snapshots.releases import TIME_POINTS
>>> TIME_POINTS["2025-06"]["opentargets"]
'25.06'
>>> TIME_POINTS["2025-06"]["omnipath_data_as_of"]
'2023-07-28'

```

OmniPath published no refresh between 2023-07-28 and 2025-08-13, so the
2025 time point pairs a 2025 association release with regulatory data
roughly 23 months older. `omnipath_data_as_of` is recorded separately
from the archive name precisely so that this is visible, and it belongs
in the limitations section of anything that reports a 2025 result.

Every entry in the table was verified against the live archive on
2026-09-06 by listing it, not by reading a spec.
