# Experiments and run records

This is the part of NeoRx that exists because of a failure, and the
failure is worth stating plainly, because the design only makes sense
against it.

## Why the runner owns persistence

A manuscript audit of this codebase found that **no reported number in
four papers could be traced to an execution artifact.** The defects were
all in scaffolding rather than algorithm code:

- `benchmark_paper.py` produced the seven-disease results behind four
  tables and **persisted nothing** — no `json.dump`, no `write_text`.
  Every number existed only in terminal scrollback.
- `_bench_paper2.py` saved only after all twelve environment × agent
  cells finished. It timed out on the fourth, so 881 seconds of real
  measurement was lost and later **retyped by hand** into another script
  as a literal dict. That transcribed −619.0 is what a published table
  reports; the one surviving raw file gives −587.8, from a different run.
- `_gen_figures.py` held the F₁ values as hardcoded literals, so a
  published figure was a transcription of a table rather than a rendering
  of data.
- `results/` and `reports/` were both gitignored, so even what *was*
  saved was invisible to version control.

The common thread: **a number could exist without a record of the run
that produced it.** This subsystem makes that impossible going forward.

It does not make the published numbers reproducible retroactively.
Nothing can; a `v0.1.0-paper` tag preserves the state the manuscripts
describe.

## The enforcement model

The **runner** owns persistence. An experiment is a function taking a
`RunRecord`; it never opens a file and never decides where results go.
There is no code path by which an experiment reports a result without a
record.

This is deliberately stronger than a context manager an author opts into.
A voluntary mechanism is exactly what the earlier state already had
available and did not use.

```python
>>> import tempfile
>>> from pathlib import Path
>>> from neorx.experiments.registry import experiment, run_experiment
>>>
>>> @experiment(name="docs-demo", help="Two cells, one record.")
... def demo(record):
...     for n in (1, 2):
...         record.append_row({"cell": n, "value": n * 0.5})
...
>>> with tempfile.TemporaryDirectory() as tmp:
...     record = run_experiment("docs-demo", runs_dir=Path(tmp))
...     status = record.status
...     rows = record.rows()
...     files = sorted(p.name for p in record.path.iterdir())
>>> status
'complete'
>>> rows
[{'cell': 1, 'value': 0.5}, {'cell': 2, 'value': 1.0}]
>>> files
['env.json', 'inputs', 'record.json', 'rows.jsonl']

```

## What a record contains

```
runs/2026-09-03-neorx-7disease-a3f9c1/
  record.json   summary, status, citable flag, rows digest
  rows.jsonl    appended per cell as it completes
  env.json      code SHA, dirty flag, python, deps, platform, snapshots
  inputs/       frozen API responses (vcrpy cassettes)
```

Run directories default to **not** tracked, so a routine `git add -A`
cannot sweep in an ad-hoc or smoke run. A record that actually backs a
claim is committed deliberately with `git add -f runs/<run-id>`, and the
`check_committed_records_are_citable` gate then holds it to being
citable.

```python
>>> import json
>>> with tempfile.TemporaryDirectory() as tmp:
...     record = run_experiment("docs-demo", runs_dir=Path(tmp))
...     env = json.loads((record.path / "env.json").read_text())
...     summary = json.loads((record.path / "record.json").read_text())
>>> sorted(env)
['deps', 'git_dirty', 'git_sha', 'platform', 'python', 'snapshots',
 'snapshots_mismatched', 'snapshots_uncited', 'started_utc']
>>> sorted(summary)
['citable', 'experiment', 'finalised_utc', 'n_rows', 'rows_sha256',
 'run_id', 'size_bytes', 'status']

```

Three properties matter more than the format.

### Rows append as they complete

`rows.jsonl` gains a line the moment each cell finishes. This is the
direct fix for the lost-881-seconds failure: a crash now costs the
current cell, not the run. `record.json` is written at finalisation.

### A dirty tree marks a run non-citable

`env.json` records the code SHA **and** whether the working tree carried
uncommitted changes. A run from a dirty tree cannot be reproduced by
anyone, so `record.json` sets `citable: false`.

```python
>>> with tempfile.TemporaryDirectory() as tmp:
...     record = run_experiment("docs-demo", runs_dir=Path(tmp))
...     env = json.loads((record.path / "env.json").read_text())
...     # citable requires status 'complete' AND a clean tree
...     env["git_dirty"] is False or record.citable is False
True

```

Provenance that cannot be established is a **failure**, not a blank
field:

```python
>>> from neorx.experiments import ProvenanceError
>>> issubclass(ProvenanceError, RuntimeError)
True

```

A run whose git SHA or dirty state cannot be determined raises rather
than recording a fabricated SHA or a false "clean".

### A record has a size ceiling

```python
>>> from neorx.experiments import MAX_RECORD_BYTES
>>> MAX_RECORD_BYTES // (1024 * 1024)
250

```

250 MB, from a measurement: ~7 MB per disease on a live seven-disease
run, so seven diseases is about 49 MB. The original spec estimate was
"5–6 MB per seven-disease run" — roughly eight times too low, which
would have meant hitting refusal after ~25 minutes of live API cost was
already sunk. `--allow-large` overrides it deliberately.

```python
>>> from neorx.experiments import RecordTooLargeError, RunIDCollisionError
>>> issubclass(RecordTooLargeError, RuntimeError)
True
>>> issubclass(RunIDCollisionError, RuntimeError)
True

```

A run ID collision refuses to write into an existing run's directory
rather than merging into it.

## Snapshot citations

If an experiment reads [pinned snapshots](snapshots.md), the runner
cites exactly what was read — from that store's own manifest.

```python
>>> from neorx.experiments.registry import experiment
>>> @experiment(name="docs-pinned", reads_snapshots=True,
...             help="Declares that its numbers come from extracts.")
... def pinned(record):
...     store = record.snapshot_store(Path("snapshots"))
...     # ... read through `store`, never a SnapshotStore built directly
...
>>> from neorx.experiments.registry import get_experiment
>>> get_experiment("docs-pinned").reads_snapshots
True

```

Two refusals, both of which this project has actually hit:

- **A read with no manifest entry.** The extracts were there and the
  manifest describing them was not, so the run produced numbers from
  inputs it cannot identify. This is the failure found in four papers.
- **`reads_snapshots` declared but no reads recorded.** The experiment
  opened a store some way other than `record.snapshot_store`, so nothing
  it read was tracked.

What the check deliberately does **not** accept is "some citation
exists". An earlier version did, and a run over the 18.06 release
completed while citing only a 25.06 entry that happened to be in the
manifest.

Each citation is the **whole** manifest entry, not a digest. A digest
pins which bytes an extract was, not what they meant — and the manifest
that says what they meant is replaced in place when an extract is
rebuilt, so a record keeping only the digest would, months later, cite
something nothing can interpret.

## Replay

A record can be re-executed against its frozen inputs:

```bash
neorx exp replay 2026-09-03-neorx-7disease-a3f9c1
```

The claim the subsystem rests on is that anyone can reproduce a run's
numbers without reaching the network. A replay that differs is not a
failure of the tool — it names exactly which values moved, which is what
a reviewer needs.

HTTP traffic is frozen with vcrpy, which patches at the connection level.
That is not a stylistic choice: the source clients call module-level
`requests.get`/`post` at thirteen sites and construct a `Session` per
call, so a transport adapter mounted on a Session would intercept
nothing.

### Volatile fields

Wall-clock timings differ on every replay. Each experiment declares them,
and they are excluded from the IDENTICAL/DIFFERS verdict but still
**reported**:

```python
>>> import sys
>>> sys.path.insert(0, ".")           # the repository's own experiments/
>>> import experiments                # registers the definitions
>>> from neorx.experiments.registry import get_experiment
>>> get_experiment("neorx-7disease").volatile_fields
('wall_clock_s', 't_graph', 't_identify')
>>> get_experiment("neorx-7disease").captures_http
True

```

A gate must never swallow information, only decide what does and does
not fail a verdict.

## The registered experiments

```python
>>> from neorx.experiments.registry import list_experiments
>>> this_repository = {"neorx-7disease", "causalbiorl-bench", "genmol-eval",
...                    "dated-build", "corpus-census", "figures"}
>>> this_repository <= {d.name for d in list_experiments()}
True

```

(A subset check rather than an exact list: the registry is process-global,
so anything else registered in the same interpreter — this page's own
examples above, the test suite's fixtures — would be in it too.)

| Experiment | Measures | Freezes HTTP | Reads snapshots |
|---|---|---|---|
| `neorx-7disease` | Causal target benchmark across 7 diseases | yes | no |
| `causalbiorl-bench` | RL agent comparison across control envs | no | no |
| `genmol-eval` | GenMol generation quality metrics | no | no |
| `dated-build` | Graphs from pinned releases; exclusions and timing | no | yes |
| `corpus-census` | Per-release disease counts | no | yes |
| `figures` | Renders manuscript figures *from a run record* | no | no |

`figures` is the one that closes the hardcoded-literals hole: it takes a
run ID and renders from that record's rows, so a figure is a rendering of
data by construction.

!!! warning "The definitions are not in the wheel"
    `neorx.experiments` — the runner — ships, so an installing user can
    record their own runs. The experiment *definitions* live in the
    repository's `experiments/` package and are deliberately **not**
    shipped: they are this project's experiments, not a library feature.

    On a clean install `neorx exp list` says so by name, rather than
    listing zero experiments — an empty list looks like an answer, and
    the honest answer is that the definitions live somewhere the
    installation cannot see.

## Using it

```bash
neorx exp list
neorx exp run neorx-7disease
neorx exp show 2026-09-03-neorx-7disease-a3f9c1
neorx exp figure --from 2026-09-03-neorx-7disease-a3f9c1
neorx exp replay 2026-09-03-neorx-7disease-a3f9c1
neorx exp prune 2026-09-03-neorx-7disease-a3f9c1
```

`prune` drops a run's frozen inputs and keeps its recorded results, for
when the cassettes are no longer worth the disk. A pruned run is marked
unreplayable and says so when you try.

## The CI gates

`neorx.experiments.gates` holds the checks that make "no number without a
record" enforceable:

| Gate | Closes |
|---|---|
| `find_hardcoded_metrics` | A figure script holding metric values as literals |
| `check_records_wellformed` | A record whose summary disagrees with its rows |
| `check_cited_runs` | A manuscript citing a run that does not exist or nobody can reproduce |
| `check_committed_records_are_citable` | A non-citable record committed anyway, adding weight while backing no claim |
| `check_record_integrity` | `rows.jsonl` edited after the fact, by recomputing the SHA-256 stored at finalise |
| `check_no_fabricated_statistics` | The specific fabricated-statistic names this project has previously shipped |

!!! note "What these gates do not claim"
    `find_hardcoded_metrics` matches bare float literals and
    metric-shaped string literals. A metric assembled by computation —
    `545 / 1000`, `round(x, 3)`, an f-string, a dict lookup — is
    invisible to an AST literal check **by construction**, not by an
    oversight fixable there. A clean run means "no metric was typed in as
    a literal"; it does not mean "no computed metric literal exists".

    `check_record_integrity` detects accidental edits and drift. It does
    **not** detect a determined forger who edits `rows.jsonl` and
    rewrites `rows_sha256` to match. That would require signing, and this
    gate is not that.

    Both limits are written into the module's own docstring, where
    someone reading a green gate will find them.
