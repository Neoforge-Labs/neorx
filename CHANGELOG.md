# Changelog

All notable changes to NeoRx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] — 2026-09-05

This release exists because 0.2.0 reported numbers that no data supported.
Anyone who installed 0.2.0 should upgrade; that version is yanked.

### Removed

- **Breaking:** `NeoRxResult.causal_effect` and `NeoRxResult.confidence_interval`,
  and `CounterfactualResult.confidence_interval`. Neither confidence interval was
  a bootstrap. Both resampled nothing: they added `rng.normal(0, 0.05)` and
  `rng.normal(0, 0.03)` to deterministic scores and took percentiles, so every
  interval NeoRx ever reported was a readout of those two constants. The effect
  size was a product of heuristic factors, and the p-value derived from it had
  no null distribution behind it. All are deleted rather than deprecated, and a
  CI gate fails if any of the names return.
- `robustness_score` and the leave-one-source-out sensitivity analysis are
  **not** affected — that measurement is real and unchanged.
- **Breaking:** the `genmol`, `dockbot`, `causalbiorl` and `mirrorfold` console
  scripts, deprecated in 0.2.0 and removed here as that release promised. Every
  one remains reachable as a subcommand — `neorx genmol`, `neorx dockbot`, and
  so on — or as `python -m neorx.<package>`.

### Fixed

- **The backdoor criterion is now actually applied.** `identifier.py` called
  `nx.d_separated`, removed from NetworkX, so it raised on every invocation and
  an `except Exception: pass` swallowed it. Identifiability was really
  `len(causal_pathway) > 0` — "some path exists" — and the adjustment set
  computed alongside it was never consulted. Identification now runs on a typed
  causal subgraph and reports one of seven explicit verdicts with the adjustment
  set that justifies it.
- **The RL pipeline had never produced a candidate.** Three faults compounded:
  the planner's reward function was `None` at construction, so its CEM returned
  random latents; the candidate-extraction loop read an `env._target_states`
  attribute that never existed, raising on every run behind a broad `except`
  that logged "collecting partial results" and continued with an empty list; and
  any surviving candidate carried hardcoded `qed_score=0.5` and `sa_score=5.0`.
- **Edge collisions no longer depend on insertion order.** STRING and OmniPath
  both describe protein relationships, and a `DiGraph` holds one edge per node
  pair, so a causal edge could be silently overwritten by an associational one.
  Resolution is now explicit: causal-admissible beats associational, heavier
  weight breaks ties, and provenance is unioned across colliding edges.
- Multi-source corroboration counted aggregators as independent sources, so
  OmniPath re-reporting a STRING interaction counted twice. It now counts
  distinct primary sources.

### Added

- OmniPath as a data source — the only directed, signed gene-to-gene edges in
  the system. Edges are admitted only when the interaction is directed with a
  consensus direction.
- Gene-to-disease edges carry an evidence class, and only genetic association or
  somatic mutation is admitted as causal, on the Mendelian randomisation warrant.
- Per-disease identifiability metrics in the run record: the non-trivial rate,
  the trivial rate reported separately, the cyclic fraction, and a
  confounding-sensitivity count. Trivial and non-trivial are kept apart because
  a knowledge graph is open-world — an absent confounder is absent knowledge,
  not absent confounding.

### Note on prior results

Figures and tables produced with 0.2.0 or earlier cannot be reproduced by this
release, and should not be. They came from an engine that never executed the
criterion it described. The `v0.1.0-paper` tag preserves the state those
manuscripts describe.


## [0.2.0] — 2026-09-02

### Changed
- **Breaking:** `modules.*` imports become `neorx.*`; `modules.neorx` becomes
  `neorx.core`. The `modules/` compatibility shim was removed outright rather
  than deprecated -- there is no shim in 0.2.x, and old `modules.*` imports
  simply no longer exist. (The plan that shipped with this release assumed a
  shim deprecated through 0.2.x and removed in 0.3.0; that step was dropped
  during implementation.)
- Package moves to a `src/` layout, so tests exercise the installed artifact.
- Python floor lowered from 3.13 to 3.12; dependency floors correspondingly
  reassessed (see Fixed).
- `pandas` is replaced with `polars` throughout (was not part of the original
  plan for this release; `benchmark.py`'s reporting is the only affected
  call site).
- Five console scripts unified under `neorx <module> <command>`. The old
  script names remain as deprecated aliases through 0.2.x.

### Added
- All six modules exposed through the public API. `molscreen` previously
  exported nothing.
- Trained GenMol weights ship as package data; `load_pretrained()` raises
  `GenMolAssetError` rather than returning an untrained model.
- **CausalBioRL integration** — RL agent now drives the full drug
  discovery pipeline through `DrugDiscoveryEnv` (Gymnasium).
- **`run_rl_pipeline()`** — RL-driven alternative to the linear
  `run_pipeline()`. Agent iteratively selects targets and generates
  molecules via latent-space navigation.
- **R-GCN graph encoder** — `DiseaseGraphEncoder` maps disease knowledge
  graphs to fixed 128-D embeddings using relational graph convolution.
- **Surrogate docking model** — `SurrogateDockingModel` (MLP) provides
  ~1ms binding affinity predictions, trained on DockBot observations.
- **Adaptive reward learner** — `AdaptiveRewardLearner` with 6 per-objective
  critics and difficulty-adaptive weighting (hindsight shaping).
- **Hierarchical planner** — `HierarchicalPlanner` with UCB1 target
  selection (Level 1) and CEM molecule generation (Level 2).
- **Typed edges in SCM** — Edges tagged with provenance (`api` vs
  `learned`), `augment_graph()` for merging discovered edges,
  `from_disease_graph()` classmethod.
- **CounterfactualValidator bridge** — `validate_with_biorl_scm()` method
  delegates to the shared SCM when CausalBioRL is available.
- **DrugDiscovery-v0** environment registered in Gymnasium.
- **GitHub Actions CI** — lint (Ruff), test (pytest on Ubuntu + macOS),
  type check (mypy), coverage upload (Codecov).
- **CONTRIBUTING.md** — contributor guide with style, testing, and PR
  conventions.
- **CODE_OF_CONDUCT.md** — Contributor Covenant v2.1.
- **SECURITY.md** — vulnerability reporting policy.
- **Ruff configuration** — formatter + linter in `pyproject.toml`.
- **pytest-cov integration** — coverage config with source filtering.
- **Expanded .gitignore** — reports/, results/, *.pdb, *.pdbqt, model
  weights, IDE files, OS files.

### Fixed
- `DrugDiscovery-v0` now generates through the trained VAE. It previously
  built an unvocabularised tokenizer and a randomly initialised model, called
  a `decode_from_latent` method that does not exist, swallowed the resulting
  `AttributeError`, and returned one of twelve hardcoded scaffolds.
- `DrugDiscoveryEnv` now rejects a `latent_dim` that does not match the
  shipped VAE's, instead of silently constructing a mismatched network (not
  part of the original plan; found while wiring the previous fix).
- Four dependency floors (`numpy`, `rdkit`, `pyyaml`, `psycopg2-binary`) had
  been raised past their true minimum on the reasoning that no cp313 wheel
  existed for the lower version on macOS arm64 -- but a floor is a lower
  bound, not a pin, so a wheel gap on a newer interpreter never justifies
  raising it. Re-derived from declared dependency constraints and actual
  binary/API compatibility, verified on Python 3.12.
- **SCM self-loop bug** — autoregressive dependencies (`s0→s0`) were
  being stripped, preventing linear mechanisms from seeing state input.
- **Matplotlib `tostring_rgb` deprecation** — all 3 toy env renderers
  now use `buffer_rgba()` (works on macOS and headless Linux).
- **Pydantic v2 deprecation** — `class Config` replaced with
  `model_config = ConfigDict(...)` in `EpisodeResult` and
  `BenchmarkResult`.
- **Docstring escape sequence** — invalid `\s` in planner docstring.
- **SCM test flakiness** — increased learning rate for reliable
  convergence in unit tests.

## [0.1.0] — 2026-03-29

### Added

- **Causal inference engine** — multi-source evidence triangulation via
  path strength × d-separation quality × centrality × source corroboration.
- **Backdoor criterion** — proper d-separation analysis using
  `networkx.d_separated()` for identifiability testing.
- **Bootstrap confidence intervals** — 95% CIs on causal confidence
  (200 resamples).
- **Leave-one-source-out sensitivity analysis** — robustness validation
  by systematically removing each data source.
- **7 biomedical data source clients** — DisGeNET, Open Targets (GraphQL),
  KEGG, Reactome, STRING, UniProt, RCSB PDB — all with curated mock
  fallbacks for offline use.
- **Parallel API queries** — `ThreadPoolExecutor` for concurrent database
  queries (~2× speedup on graph building).
- **File/Redis caching layer** — 24h TTL for API responses, 7d for graphs.
  Configurable via `NEORX_CACHE_BACKEND` env var.
- **Multi-rule ADMET predictor** — 8 rule systems: Lipinski RO5, Veber,
  Ghose, Egan egg, PAINS (RDKit FilterCatalog), BBB permeability, hERG
  liability, reactive group alerts.  Weighted composite score.
- **Graph persistence & export** — JSON, GraphML, GEXF, Cytoscape formats.
  PostgreSQL persistence for Docker deployments.
- **Disease ontology resolution** — `resolve_disease_id()` maps free-text
  disease names to EFO/MONDO identifiers via Open Targets search.
- **Configurable scorer weights** — override via Python parameter, env var
  (`NEORX_WEIGHTS`), or defaults.  Auto-normalised to sum=1.0.
- **SMILES canonicalization** — duplicate elimination via RDKit canonical
  SMILES before screening.
- **Interactive HTML reports** — vis.js causal knowledge graph, 95% CI
  column, UTC timestamps, updated methodology section.
- **Non-blocking FastAPI endpoints** — `asyncio.to_thread()` on all
  CPU/IO-bound handlers.
- **Rich CLI** — progress bar, `--seed` for reproducibility, `--export`
  for graph formats, `--log-file`, `--no-cache`.
- **Docker Compose infrastructure** — Redis 7 (AOF + LRU), PostgreSQL 16
  (5-table schema with indexes), API container with healthchecks.
- **PEP 561 `py.typed` marker** — full static typing support.
- **Clean public API** — `from neorx import run_pipeline` works
  after `pip install neorx`.
- **108 tests** covering data sources, graph builder, identifier, pipeline,
  scorer, cache, persistence, ADMET, configurable weights, SMILES
  canonicalization, confidence intervals, and disease ID resolution.

[Unreleased]: https://github.com/NeoForge/NeoRx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/NeoForge/NeoRx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NeoForge/NeoRx/releases/tag/v0.1.0
