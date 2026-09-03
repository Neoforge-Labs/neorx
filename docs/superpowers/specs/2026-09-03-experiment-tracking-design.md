# Experiment Tracking — Design

**Date:** 2026-09-03
**Status:** Approved for planning
**Sub-project:** 2 of 4

---

## Context

A manuscript audit found that no reported number in the four NeoRx papers could
be traced to an execution artifact. The failures were not subtle, and they were
all in scaffolding rather than algorithm code:

- `benchmark_paper.py` produces the seven-disease results behind Tables 2–5 and
  **never persists anything** — no `json.dump`, no `write_text`. Every number
  existed only in terminal scrollback.
- `_bench_paper2.py` saves only after all twelve environment × agent cells
  finish. It timed out on the fourth, so 881 seconds of real measurement was
  lost and later retyped by hand into `_bench_fast_baselines.py:24-31` as a
  literal dict. That transcribed `−619.0` is what Table 4.3.2 reports; the one
  surviving raw file gives `−587.8`, from a different run.
- `_gen_figures.py:226-228` holds the F₁ values as hardcoded literals. Figure 3
  is a transcription of the paper's table, not a rendering of data.
- `results/` and `reports/` are both gitignored, so even what *is* saved is
  invisible to version control.
- The experiment scripts themselves live untracked at the repository root.

The common thread: **a number could exist without a record of the run that
produced it.** This sub-project makes that impossible going forward.

### Decomposition

Sub-project 2 of four. Sub-project 1 (package skeleton) is complete and merged.

| # | Sub-project | Status |
|---|-------------|--------|
| 1 | Package skeleton | Complete — merged at `43d3ad2` |
| **2** | **Experiment tracking** | **This spec** |
| 3 | Correctness | Split `identifier.py` / `drug_discovery.py`; fix backdoor logic; make the CEM inner loop decode |
| 4 | Test hardening | Behaviour tests; molscreen from zero coverage |

---

## Goals

1. An experiment cannot report a number without writing a run record.
2. A run can be replayed against frozen inputs and reproduce its numbers exactly.
3. Figures and tables are rendered from run records, never from literals.
4. CI fails when any of the above regresses.

## Non-goals

- **Fixing the benchmark's methodology.** The causal-engine defects (`d_separated`
  never executing, the no-op backdoor loop, the ablation harness that does not
  ablate) belong to sub-project 3. This sub-project records what the code does
  today; it does not make it correct.
- **Making the published numbers reproducible retroactively.** Nothing can. The
  `v0.1.0-paper` tag preserves the state the manuscripts describe; see
  *Consequences* below.
- **Migrating diagnostic scripts.** `_diag_*`, `_check_*`, `_smoke_test`, and the
  root `test_*` scripts (~11 files) inform work but back no published claim.
  They stay as they are.
- **A general-purpose experiment framework.** This runs four named experiments
  for one project. Anything resembling MLflow is out of scope.

---

## Design

### Enforcement model

The runner owns persistence. Experiments are declared in a registry and executed
by the runner, which writes the record itself — an experiment has no code path
that reports a result without one. This is deliberately stronger than a
`run_record()` context manager an author opts into: a voluntary mechanism is
what the current state already had available and did not use.

### Layout

```
src/neorx/experiments/          ← ships with the package
  registry.py    experiment definitions, discovered by name
  record.py      RunRecord: create, append rows, finalise
  capture.py     HTTP record/replay via vcrpy cassettes
  chembl.py      SQLite provenance (version + queries, not a file hash)
  gates.py       the checks CI runs
  __main__.py    `neorx exp` subcommands

experiments/                    ← tracked, not shipped
  neorx_7disease.py  causalbiorl_bench.py
  genmol_eval.py     figures.py

runs/                           ← tracked; this is the point
  2026-09-03-neorx-7disease-a3f9c1/
    record.json   summary, status, citable flag, input digest
    rows.jsonl    appended per cell as it completes
    env.json      code SHA, dirty flag, python, deps, platform
    inputs/       frozen API responses (vcrpy cassettes)
    stdout.log
```

The runner library ships so an installing user can record their own runs. The
experiment *definitions* stay out of the wheel — they are this project's
experiments, not a library feature.

`runs/` becomes tracked, reversing `.gitignore:37`. See *Record size* below.

### The run record

Three properties matter more than the file format.

**Rows append as they complete.** `rows.jsonl` gains one line the moment each
cell finishes. This is the direct fix for the `_bench_paper2.py` failure: a
crash now costs the current cell, not the run. `record.json` is written at
finalisation with the summary and status.

**A dirty tree marks a run non-citable.** `env.json` records the code SHA *and*
whether the working tree carried uncommitted changes. A run from a dirty tree
cannot be reproduced by anyone, so `record.json` sets `citable: false` and the
figure gate refuses it.

**Status is explicit.** `record.json` carries
`status: complete | incomplete | failed`, set at finalisation. An incomplete run
remains a valid artifact — it records what was measured before the failure — but
it is not citable.

Run IDs are `YYYY-MM-DD-<experiment>-<6-char hash>`, the hash taken over start
timestamp and code SHA.

### Capturing inputs

The seven HTTP data sources call module-level `requests.get`/`requests.post` at
13 sites; none route through `cached_api_call`, which is exported but never
called. Mounting a transport adapter on a `Session` would therefore intercept
nothing, because `requests.get` constructs its own session per call.

**We use vcrpy**, which patches at the connection level and so captures all 13
sites with no edits to the source modules. It already implements record, replay,
and once semantics — redirects, error responses, header handling — which are
tedious and easy to get subtly wrong by hand, and its cassette is a documented
format rather than a bespoke one.

vcrpy is monkeypatching, which brushes against the project's standing "no
patch jobs" directive. It is acceptable here because of scope: it is imported
only by `neorx.experiments`, active only inside a recording context, and never
present in a library code path. A module docstring in `src/neorx/core/sources/`
states that HTTP there is subject to recording, so the behaviour is discoverable
from the code being recorded rather than only from the recorder.

**ChEMBL is handled separately.** It is a 28 GB local SQLite file; hashing it
costs minutes and buys nothing, because ChEMBL releases are versioned and
immutable. The version is read from the database's own `chembl_release` table,
verified against the shipped `chembl_36.db`:

```sql
SELECT chembl_release, creation_date
FROM chembl_release ORDER BY chembl_release_id DESC LIMIT 1;
-- ('CHEMBL_36', '2025-07-28 00:00:00.000000')
```

The record captures that release string and date, the file size and mtime, and
every SQL query with its parameters and row counts. Replay verifies the release
matches and refuses with a named mismatch otherwise. (Note the `version` table is
*not* the right source — it holds ontology versions such as `Bioassay Ontology
2.0`, not the ChEMBL release.)

**Graph-level caching is left alone.** `graph_builder.py:99,276` caches the
assembled graph under `GRAPH_TTL`. That cache holds an *output*; replaying from
it would skip the code the replay exists to re-execute. Recording operates at
input granularity and the runner disables the graph cache during a run.

### Replay

`neorx exp replay <run-id>` re-executes the experiment against the frozen
cassettes and diffs the resulting rows against the recorded ones. It reports
either identical, or exactly which rows moved and by how much. This is the
property the subsystem claims, so it is also its primary test: record a run,
replay it, assert the rows are byte-identical. If that round trip does not hold,
nothing else here means anything.

### Enforcement gates

| Gate | Closes |
|---|---|
| `test_no_hardcoded_metrics` | Float literals in figure/table code — `_gen_figures.py:226` |
| Figures require `--from <run-id>` | A figure drawn from nothing |
| Runner refuses non-citable runs | Dirty tree or incomplete status — the transcribed `−619.0` |
| `test_cited_runs_exist` | A manuscript citing a run ID absent from `runs/` |

The last gate is the actual closure. `docs/run-manifest.toml` maps each paper
table and figure to the run ID backing it, and a test asserts every cited run
exists and is citable. A number in a manuscript then cannot exist without a
record, because CI fails if it does.

`test_no_hardcoded_metrics` flags float literals with three or more decimal
places in `experiments/figures.py` and any table-generating code. Legitimate
constants (axis limits, thresholds) are excluded by an explicit allowlist with a
comment per entry, never by loosening the pattern.

### Record size

Frozen inputs are roughly 5–6 MB per seven-disease run. The runner reports
record size on finalisation and refuses to write beyond 50 MB without
`--allow-large`, so a runaway snapshot cannot quietly add hundreds of megabytes
to the repository. `neorx exp prune <run-id>` removes a superseded run's inputs
while keeping its `record.json` and `rows.jsonl`, which marks it `replayable:
false` but preserves the numbers and their provenance.

### Migration order

Cheapest and safest first, so the runner is validated before it meets the hard
case.

1. **`genmol_eval`** — offline, and already produces a real artifact
   (`results/genmol_paper4_final.json`). Lowest risk; proves the runner.
2. **`causalbiorl_bench`** — offline (Gymnasium envs). Subsumes both
   `_bench_paper2.py` and `_bench_fast_baselines.py`, and fixes the
   incremental-write failure that lost the original 881-second run.
3. **`neorx_7disease`** — eight live APIs, needs capture, ~25 minutes per run.
4. **`figures`** — last, because it reads the records the first three produce.

---

## Consequences

**Re-running `neorx_7disease` will not reproduce F₁ = 0.474.** Upstream data has
drifted since March, and the causal-engine defects remain unfixed until
sub-project 3. This is expected and must not be treated as a regression. The
`v0.1.0-paper` tag preserves the state the manuscripts describe; from this
sub-project forward, every number carries a record, and the existing ones carry
a tag and a caveat.

**Two manuscript corrections fall out of this work.** Appendix B states that
graph construction "caches API responses with a configurable TTL" — it caches
the assembled graph, not the responses, and the two API-response helpers are
dead code. Table 5's "Cached" column is explained by the graph cache, not by
response caching. Both should be reworded when the papers are next revised.

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Enforcement | Runner owns persistence | A voluntary API is what already existed and went unused |
| Scope | Five paper-backing scripts, as four experiments | They produce every untraceable number; diagnostics back no claim |
| Inputs | Snapshot + replay | Provenance alone leaves numbers unverifiable by anyone else |
| Capture mechanism | vcrpy at connection level | 13 call sites, zero source edits; sessions would intercept nothing |
| ChEMBL | Version + queries, not a file hash | 28 GB; releases are versioned and immutable |
| Graph cache | Disabled during runs | It caches an output; replaying from it skips the code under test |
| `runs/` | Tracked, with a size guard | Records are the deliverable; unbounded growth is the risk |

---

## Risks

| Risk | Mitigation |
|---|---|
| vcrpy fails to intercept a source that later switches to `httpx` or `aiohttp` | A gate asserts the cassette recorded ≥1 interaction per configured source; a source contributing zero interactions fails the run |
| Frozen inputs balloon the repository | 50 MB refusal threshold; `exp prune` drops inputs while keeping numbers |
| Replay drifts because a dependency changed behaviour, not the inputs | `env.json` pins dependency versions; replay reports a version delta alongside any row diff |
| Authors bypass the runner and run a script directly | All five migrated scripts are deleted, not left beside the runner. `test_no_hardcoded_metrics` catches the figure path |
| A future ChEMBL build lacks or renames `chembl_release` | The runner fails loudly naming the missing table rather than silently recording no version; file size and mtime still recorded |

---

## Success criteria

Verified in CI on every change:

1. Record a run of a fixture experiment, replay it, and the rows are identical.
2. A run made from a dirty tree is marked `citable: false` and the figure
   command refuses it.
3. An experiment killed mid-run leaves `rows.jsonl` containing every completed
   cell and `record.json` with `status: incomplete`.
4. `test_no_hardcoded_metrics` fails when a metric-shaped literal is introduced
   into figure code.
5. `test_cited_runs_exist` fails when the manifest cites a run absent from
   `runs/`.
6. `neorx exp run <name>` refuses to write a record exceeding 50 MB without
   `--allow-large`.

Verified out of band, because it takes ~25 minutes and touches eight live APIs:

7. A full `neorx_7disease` run records, and its replay reproduces its own rows.
   Run manually before each manuscript revision.
