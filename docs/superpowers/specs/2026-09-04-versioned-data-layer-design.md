# Versioned Data Layer — Design

**Date:** 2026-09-04
**Status:** Approved for planning
**Sub-project:** 4 of 8

---

## Context

The research goal changed after sub-project 2. NeoRx is no longer four
manuscripts describing a platform; it is one paper aimed at Nature
Communications, asking a question the platform is now instrumented to answer:

> Does causal *structure* — the identifiability verdict, the adjustment set,
> the confounding exposure — predict clinical trial outcomes beyond the
> presence of genetic support alone?

That framing sets the bar. "Genetic evidence predicts drug approval" is
established (Nelson et al., *Nat Genet* 2015; King, Davis & Degner, *PLoS
Genet* 2019, ~2× enrichment). It is the baseline, not the contribution. The
contribution is whether structure adds signal on top of it.

Two things follow, and neither is possible against live APIs.

**Scale.** Seven diseases cannot support an outcome-prediction result.
Hundreds of target–disease pairs are needed for any power. At ~25 minutes per
disease against eight live endpoints, that is not a waiting problem.

**Time.** The validation requires computing identifiability as it would have
been known at a past date, then checking it against what happened afterwards.
A live API cannot return its 2018 state at any price.

### Decomposition

| # | Sub-project | Status |
|---|-------------|--------|
| 1 | Package skeleton | Merged — `43d3ad2` |
| 2 | Experiment tracking | Merged — `9117816` |
| 3 | Correctness | Spec and plan written |
| **4** | **Versioned data layer** | **This spec** |
| 5 | Outcome labels | Curated efficacy-failure dataset |
| 6 | Trial-outcome validation | The prediction result |
| 7 | Selectivity-constrained generation | Adjustment sets as design constraints |
| 8 | Test hardening | molscreen from zero coverage |

---

## Goals

1. Build a disease graph as of a chosen past date, from pinned inputs.
2. Do it for hundreds of diseases without hundreds of hours of HTTP.
3. Record which release every number came from.

## Non-goals

- **Replacing the live path.** Interactive single-disease use keeps calling
  live APIs. Snapshots serve the corpus run. Both exist.
- **Pinning every source.** See *Scope of pinning* — only the sources feeding
  the causal subgraph and the candidate frame are time-pinned.
- **Mirroring upstream releases.** We store derived extracts, not copies of
  OpenTargets.

---

## Design

### Scope of pinning

The prediction uses identifiability: the verdict, the adjustment set, the
near-miss count, the cyclic status. Every one of those is computed from the
causal subgraph, which by the sub-project 3 design is built from exactly two
sources — OpenTargets genetic associations and OmniPath directed regulatory
interactions. The associational sources (STRING, Reactome, KEGG, Monarch)
contribute edges that identification filters out before it runs.

So those two sources are pinned, and the rest stay current. Pinning STRING
would buy rigour against a leak that does not exist.

**The candidate frame is pinned with them.** This is the subtle half. Features
computed from 2018 data are useless if the *population* was chosen with
hindsight: a gene that entered the candidate pool because a 2026 source
connected it to something is a gene 2018 would never have evaluated. That is
leakage in the sampling frame rather than in the predictor, and it is just as
fatal. `_get_candidate_nodes` must therefore draw candidates from the pinned
sources only, for any dated run.

### Derived snapshots, not raw releases

A modern OpenTargets release does not fit on this machine, and mirroring one
would be wasteful even if it did: the causal subgraph needs four columns.

Each release is therefore **streamed, reduced, and discarded**. What persists
is a compact Parquet extract:

```
snapshots/
  opentargets/
    18.06/associations.parquet     target, disease, datatype, score
    21.11/associations.parquet
    25.06/associations.parquet
  omnipath/
    20180614/interactions.parquet  source, target, is_directed,
    20211113/interactions.parquet  consensus_direction, is_stimulation,
    20250813/interactions.parquet  is_inhibition, sources, references
  manifest.toml                    release IDs, digests, extraction version
```

This buys three things at once. Disk: many time points in a fraction of what
one raw release costs. Format drift: each era's reader is a small extraction
function with one job, and everything downstream sees one schema whatever the
input was. Publication: a versioned, multi-year, minimal extract of genetic
gene–disease evidence is a citable dataset in its own right — and a hedge, so
that a null prediction result still leaves something the field can use.

Parquet is read with polars, per the project's standing preference.

### Time points

Three, chosen so each OpenTargets format era is represented and each has a
matching OmniPath dump. OmniPath's interaction archive begins 2018-06-14 and
OpenTargets 18.06 is June 2018, so the earliest usable pairing is nearly exact.

| Time point | OpenTargets | OmniPath archive | OT layout |
|---|---|---|---|
| 2018-06 | `18.06` | `20180614-20181114` | flat `18.06_association_data.json.gz` (0.18 GB) |
| 2021-11 | `21.11` | `20211113-20220114` | ETL era — `output/` Parquet |
| 2025-06 | `25.06` | latest ≤ 2025-06 | current — `output/association_by_datasource_direct` |

Three readers on the OpenTargets side. OmniPath's archived TSV schema is
uniform across the range, so one reader covers all three.

Three points rather than one is what turns "it worked at our cut date" into "it
holds across three", which is the difference between a result a reviewer probes
and one they accept.

### Choosing snapshot or live

A `SourceResolver` maps a source name plus an optional `as_of` date to a
reader. With no date, it returns the existing live client and behaviour is
unchanged. With a date, it returns a snapshot reader for the pinned sources and
refuses — loudly, naming the source — for any source that has no snapshot at
that date.

It must refuse rather than silently fall back to live. A dated run that quietly
mixes in current data produces exactly the anachronistic graph this
sub-project exists to prevent, and it would do so invisibly.

The sub-project 3 OmniPath module is already factored for this:
`_interactions_to_edges(rows, known_genes)` is a pure function of parsed rows,
separate from the HTTP fetch, so the snapshot reader reuses it unchanged. No
change to sub-project 3 is required.

### Provenance

Each snapshot's release identifier, source URL, byte digest, and the version of
the extraction code that produced it are recorded in `snapshots/manifest.toml`.
A run against pinned inputs records those identifiers in its `env.json`
alongside the code SHA that sub-project 2 already captures.

A derived artifact needs its derivation pinned, not just its source: two
extracts of OT 18.06 made by different extractor versions are different inputs
and must be distinguishable.

---

## Consequences

**The corpus run replaces the seven-disease run as the headline experiment.**
`neorx_7disease` remains as an integration exercise over live APIs; the paper's
numbers come from a dated corpus run.

**Identifiability becomes computable at scale**, which is what makes the
Result 1 audit — the identifiability rate and its failure taxonomy across
hundreds of pairs — a measurement rather than an anecdote.

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope of pinning | OpenTargets + OmniPath, plus the candidate frame | Only these feed the causal subgraph; frame leakage is the subtle failure |
| Storage | Derived Parquet extracts, raw discarded | A release does not fit; the subgraph needs four columns |
| Time points | Three: 2018-06, 2021-11, 2025-06 | One per OT format era, each with a matching OmniPath dump |
| Snapshot vs live | `SourceResolver` on an optional `as_of` | Keeps the interactive path unchanged |
| Missing snapshot | Refuse, naming the source | A silent fallback to live rebuilds the anachronism invisibly |
| Reader | polars | Project standard |

---

## Risks

| Risk | Mitigation |
|---|---|
| An archived release is withdrawn or its URL changes | The extract, not the release, is the durable artifact; digests in the manifest detect a changed re-download |
| OT 18.06's schema differs more than expected from the modern one | The extraction target is four columns; a per-era reader absorbs the difference, and the 0.18 GB file makes iteration cheap |
| The candidate frame differs enough across eras that populations are not comparable | Report population size and overlap per time point; a shifting frame is a finding to state, not to hide |
| Derived extracts drift from what the raw release said | Extraction is deterministic and versioned; the manifest records the extractor version, so any extract can be regenerated and compared |
| Streaming passes exhaust working disk | The machine now has ~99 GB free; each pass reduces to Parquet and deletes its download before the next begins |

---

## Success criteria

1. `SourceResolver` with no `as_of` returns the live client and existing
   behaviour is unchanged.
2. `SourceResolver` with an `as_of` for which no snapshot exists raises,
   naming the source and the date, and never falls back to live.
3. Each of the three OpenTargets readers extracts the same four-column schema
   from its era's layout, verified on a fixture per era.
4. A disease graph built with `as_of=2018-06` contains no edge whose source
   release postdates that date.
5. The candidate frame for a dated run is drawn only from pinned sources.
6. `snapshots/manifest.toml` records a release ID, digest, and extractor
   version for every extract, and a run's `env.json` cites them.
7. Building one disease from snapshots is faster than the live path by at
   least an order of magnitude, measured and recorded.
