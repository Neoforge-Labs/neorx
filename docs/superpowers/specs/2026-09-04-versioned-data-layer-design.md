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
interactions. The associational sources (STRING, Reactome, KEGG, Monarch,
ChEMBL) contribute edges that identification filters out before it runs.

So those two sources are pinned, and the rest stay current.

**Corrected 2026-09-06.** This section originally continued: "Pinning STRING
would buy rigour against a leak that does not exist." That was wrong, and the
error was found by executing a dated build rather than by reading it. Their
*edges* are filtered. Their **nodes** are not, and an unpinned node does three
things a filtered edge cannot.

It wins the graph builder's score merge, which keeps the highest score.
Monarch assigns a flat 0.85 to every causal association, above most
OpenTargets genetic scores; ChEMBL's score is 60% today's clinical phase —
which is the outcome sub-project 6 exists to predict, entering a dated graph
as a predictor. A dated node was observed carrying Monarch's live 0.85 while
its own metadata still asserted the pinned provenance.

It draws causal-admissible arrows out of the *pinned* OmniPath extract,
because the gene list handed to that reader was built from every node
collected so far. On identical snapshots, adding one live node flipped a
verdict from `identifiable_by_adjustment` to `cyclic_component`.

And it competes for the `max_genes` cap, displacing frame genes from the
regulatory query entirely.

The first correction, made 2026-09-06, was: **on a dated build the gene node
population is the frame.** Unpinned sources may enrich nodes already in the
frame — pathways, structures, protein metadata, associational edges among
frame genes — but may not introduce gene or protein nodes, and may not
overwrite a pinned node's score or provenance.

**Corrected again, later the same day.** That rule was still too narrow,
and the sentence permitting "enrichment" was carrying the same false premise
one level down. Three further channels were measured:

*Provenance is an input, not a label.* The builder appends a live source's
name to a pinned node's `source` string, on the reasoning that recording
corroboration is not overwriting a score. But that string is *split* to
build `collect_source_scores`, so an unpinned source becomes an evidence
stream, and `n_active_sources` is the denominator of the consensus term.
Every target in the graph moves. The one target ChEMBL has a drug for was
the only one left unchanged, and the others fell by up to 0.0375 — 2.5×
the druggability leak already ruled a defect, in the same direction.

*A derived flag was blocked while its raw input stayed open.* UniProt
computes `is_druggable` partly from `len(pdb_ids) > 0`. Blocking the flag
left `pdb_ids` (+0.20), `uniprot_id` (+0.10) and `description` keywords
(+0.15) reaching the druggability score directly: +0.045 on
`causal_confidence`, three times the leak whose fix prompted it.

*Associational edges reach the score even though they never reach
identification.* STRING `interacts_with` edges between two frame genes
moved `causal_confidence` 0.5342 → 0.5819 and `robustness` 0.5404 → 0.6142.

So the rule is now simply: **on a dated build, an unpinned source
contributes nothing that reaches a number** — no nodes, no edges, no
provenance strings, no node fields. A dated graph is the pinned release.

The cost is stated rather than hidden: a dated graph carries no pathway,
structure or interaction enrichment, so `n_pathways`, `n_interactions`,
druggability and evidence-stream counts fall for every dated target.
Identifiability — the quantity the prediction actually tests — is
untouched, because it comes from OpenTargets and OmniPath alone. Pinning
STRING, KEGG, Reactome, UniProt and PDB remains unnecessary; excluding
them from dated builds achieves the same thing at no download cost.

What this section got wrong three times running is worth naming, because
it is the failure mode of the whole design rather than of one sentence:
"identification filters those edges out" is a true statement about
*identification* that was repeatedly assumed to be a statement about *the
score*. Identification reads edge types and evidence classes. The score
reads node counts, source strings, structure counts and pathway
memberships — none of which identification touches, and all of which an
unpinned source was free to supply.

**The candidate frame is pinned with them.** This is the subtle half. Features
computed from 2018 data are useless if the *population* was chosen with
hindsight: a gene that entered the candidate pool because a 2026 source
connected it to something is a gene 2018 would never have evaluated. That is
leakage in the sampling frame rather than in the predictor, and it is just as
fatal. `_get_candidate_nodes` must therefore draw candidates from the pinned
sources only, for any dated run.

### The frame gate

`_get_candidate_nodes` currently admits every node whose type is `gene`,
`protein`, or `pathogen_gene`, so any source that contributes a gene node
contributes a candidate. Monarch, ChEMBL, STRING, KEGG and Reactome all do. The
leak surface is the whole node set, and closing it takes three layers of which
only the third is enforcement.

**One — derive the frame from the pinned source alone.**
`candidate_frame(as_of, disease) -> frozenset[str]` reads only the dated
OpenTargets extract and returns the genes it associated with that disease at
that date. That set is the population the date would have evaluated.

**Two — restrict fetches to it.** STRING, OmniPath, KEGG, Reactome and UniProt
all accept a gene list; a dated run passes the frame.

**Corrected 2026-09-06.** This layer was originally described as "efficiency,
not safety", on the reasoning that Monarch and ChEMBL add gene nodes for the
*disease* rather than from a gene list and so bypass it. The first half of
that is false and the second is true. Passing the frame to the gene-list
sources is what stops an unpinned gene from reaching the pinned OmniPath
reader and drawing causal arrows, and what stops unpinned genes from
displacing frame genes under `max_genes`. It is safety, and it is the layer
that closes the identifiability leak.

The Monarch/ChEMBL observation stands and needs its own answer, which layer
two does not provide: because they are queried by disease rather than by gene
list, a dated build must drop the non-frame gene nodes they return, and must
not let the frame genes they return overwrite a pinned score. That is the
population rule stated under *Scope of pinning*.

**Three — gate candidate selection, and record every exclusion.**
`_get_candidate_nodes` takes the frame on a dated run and admits only nodes
within it. Each excluded node is written to the run record with the source that
contributed it.

Exclusion is deliberately not an error. Monarch legitimately returns genes
OpenTargets did not, so refusing would fail every dated run for a condition that
is normal and expected. But a silent filter is precisely how frame leakage
creeps back after someone refactors, so the exclusions are counted and named.
A run that suddenly excludes four hundred genes where it previously excluded
twelve has had something change upstream, and that is visible in the record
rather than absorbed.

The invariant is held by a test that injects an off-frame gene into a dated
run's assembled graph and asserts it never reaches the candidate list. That test
is the thing a reviewer would ask for.

### Which diseases

Not every disease OpenTargets carries. The corpus is defined by what makes the
downstream prediction scoreable at all, and the criteria are applied **to the
pinned release**, not to current data — selecting diseases because they look
well-studied today, then analysing them as of 2018, is the same hindsight leak
one level up.

A disease enters the corpus for a given time point when, in that release:

1. it carries at least one gene–disease association on genetic or somatic
   evidence — without one the causal subgraph is empty and identifiability is
   trivially zero, which adds noise rather than signal; and
2. at least one of its targets had reached Phase II or beyond — without that
   there is no clinical outcome to predict against.

The second criterion is what binds. Most of OpenTargets' disease space has no
drug that got far enough to have an outcome.

The resulting count is a property of each release, not something to assert in
advance, so measuring it is the first task of the implementation plan rather
than a number written here. The expectation is hundreds of diseases and low
thousands of target–disease pairs — enough for the prediction to have power,
which seven diseases plainly are not.

The seven diseases from the original benchmark are retained wherever they meet
the criteria, so the new results can be set beside the old ones. That
comparison carries every caveat in the corrigendum audit and is context, not
evidence.

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

| Time point | OpenTargets | OmniPath archive | OmniPath data as of | OT layout |
|---|---|---|---|---|
| 2018-06 | `18.06` | `20180614-20181114` | 2018-06-14 | flat `18.06_association_data.json.gz` (0.18 GB) |
| 2021-11 | `21.11` | `20211113-20220114` | 2021-11-13 | ETL era — `output/` Parquet |
| 2025-06 | `25.06` | `20230728-20250813` | **2023-07-28** | current — `output/association_by_datasource_direct` |

**Verified 2026-09-06** against the live archive index, which lists 22
interaction dumps. The first two pairings are as close as the table claims.
The third is not, and the original entry — "latest ≤ 2025-06" — concealed it:
the archive has no refresh between 2023-07-28 and 2025-08-13, so the dump
covering 2025-06 carries data roughly 23 months older than the OpenTargets
release it is paired with. The time point is still usable, because what it
compares against is the other two time points read the same way, but the gap
belongs in the paper's limitations rather than behind a "latest ≤" that reads
as contemporaneous.

Archived dumps are `.tsv.xz`, not the `.tsv.gz` shown in the storage layout
above. The snapshot builder detects compression from the file's magic bytes
rather than its extension, because an archive URL's extension is a claim and
this one was wrong in the spec for the life of the sub-project.

Three readers on the OpenTargets side. OmniPath's archived TSV schema is
uniform across the range, so one reader covers all three — confirmed by
parsing the real 2018 dump, which yields 644,845 rows in the canonical schema
and carries no `consensus_direction` column, exactly as the synthesised-
consensus decision assumed.

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
| What a dated build contains | Only the pinned release. An unpinned source contributes nothing that reaches a number: no nodes, no edges, no provenance strings, no node fields | Corrected twice on 2026-09-06. The first correction stopped unpinned *nodes*; the second stopped unpinned *provenance, fields and edges*, which reach the score without ever reaching identification |
| Storage | Derived Parquet extracts, raw discarded | A release does not fit; the subgraph needs four columns |
| Time points | Three: 2018-06, 2021-11, 2025-06 | One per OT format era, each with a matching OmniPath dump |
| Snapshot vs live | `SourceResolver` on an optional `as_of` | Keeps the interactive path unchanged |
| Missing snapshot | Refuse, naming the source | A silent fallback to live rebuilds the anachronism invisibly |
| Reader | polars | Project standard |
| Frame gate | Filter at candidate selection, record exclusions | Refusing fails every dated run; a silent filter is how leakage returns |
| Disease corpus | Genetic evidence + a target past Phase II, judged on the pinned release | Selecting on today's data and analysing as of 2018 is the same leak one level up |

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
6. A gene injected into a dated run's graph from outside its frame never reaches
   the candidate list, and the exclusion is recorded with its source.
7. `snapshots/manifest.toml` records a release ID, digest, and extractor
   version for every extract, and a run's `env.json` cites them.
8. Building one disease from snapshots is faster than the live path by at
   least an order of magnitude, measured and recorded.

Added 2026-09-06, after a dated build was found to satisfy 4 and 6 while
still reading current data through its node set:

9. No gene or protein node in a dated graph comes from an unpinned source,
   and no unpinned source changes a pinned node's score or provenance.
10. An identifiability verdict computed on a dated graph is unchanged by
    whether the unpinned sources returned anything. This is the criterion
    that fails loudest when the population rule is broken, and the one to
    write first when touching this path.

Added 2026-09-06, after criteria 9 and 10 both held while three further
channels moved `causal_confidence`:

11. A dated graph is **identical** — nodes, edges, node fields, provenance
    strings and metadata alike — whether the unpinned sources return
    everything they can or nothing at all. Criteria 4, 6, 9 and 10 are all
    consequences of this one; each of them passed while a leak was live,
    because each names a mechanism and this names the property.
12. The fixture proving 11 is not vacuous: every dimension it compares is
    exercised by a contribution that actually survives a dated build.
    Removing any one guard must make it fail. A test whose fixture offers
    nothing has been mistaken for a passing test three times on this path,
    twice in tests written specifically to prevent that.

Criterion 12 is not a testing nicety. Of the seven leaks found here, three
were invisible because the test that should have caught them stubbed the
offending source to return nothing — establishing the condition under which
the defect cannot occur, and then verifying it did not occur.

Criteria 1-8 were originally numbered with two 6s. Renumbered here; no
criterion was added to or removed from that range.
