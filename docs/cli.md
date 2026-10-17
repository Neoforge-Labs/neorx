# Command line

One console script, `neorx`, with the core pipeline commands at the top
level and each engine as a subcommand.

```python
>>> from typer.testing import CliRunner
>>> from neorx.cli import app
>>> result = CliRunner().invoke(app, ["--help"])
>>> result.exit_code
0
>>> for name in ["run", "graph", "identify", "serve", "genmol", "dockbot",
...              "causalbiorl", "mirrorfold", "molscreen", "exp", "snapshot"]:
...     assert name in result.output, name

```

```
neorx run          Run the full NeoRx pipeline.
neorx graph        Build a disease causal knowledge graph.
neorx identify     Identify causal drug targets for a disease.
neorx serve        Start the NeoRx FastAPI server.

neorx genmol       Molecular generation (VAE).
neorx dockbot      Molecular docking.
neorx causalbiorl  Causal RL environments.
neorx mirrorfold   Mirror-image protein analysis.
neorx molscreen    Drug-likeness screening.
neorx exp          Recorded experiments.
neorx snapshot     Build and inspect versioned source snapshots.
```

Every module is also reachable as `python -m neorx.<package>`. The
per-module console scripts (`genmol`, `dockbot`, …) were deprecated in
0.2.0 and removed in 0.3.0, as that release promised.

## The pipeline commands

### `neorx run`

```bash
neorx run HIV --top-n 5
```

| Option | Default | Effect |
|---|---|---|
| `--top-n`, `-n` | 5 | Number of top causal targets to pursue |
| `--candidates`, `-c` | 100 | Molecules generated per target |
| `--no-generation` | off | Skip GenMol |
| `--no-docking` | off | Skip DockBot |
| `--no-report` | off | Skip the HTML report |
| `--no-cache` | off | Bypass the API response cache |
| `--allow-mocks` | off | Permit curated mock data when a live API fails |
| `--seed` | — | Sets `NEORX_SEED` for the run |
| `--log-file` | — | Also write logs to a file |
| `--verbose`, `-v` | off | DEBUG-level logging |

Progress is shown with a Rich progress bar when Rich is importable, and
falls back to plain output otherwise.

!!! warning "`--allow-mocks` is off by default, and should stay off"
    With mocks disabled, a failed API call produces an empty result. With
    them enabled, it produces curated hand-written data that is
    indistinguishable downstream from a measurement. Use it for offline
    development; do not use it for anything you intend to report.

### `neorx graph`

```bash
neorx graph "Type 2 Diabetes"
neorx graph HIV --export cytoscape
```

`--export` takes `json`, `graphml`, `gexf` or `cytoscape`, writing to
`~/.neorx/graphs/`. Prints node counts by type, edge count, and which
sources answered.

### `neorx identify`

```bash
neorx identify "Alzheimer Disease" --top-n 10
```

Builds the graph and identifies targets, printing one line per target
with the confidence, the identification reason, the robustness and
druggability scores, and the classification — which is the quickest way
to see the [reason distribution](identification.md#it-tells-you-why-it-failed)
for a disease.

### `neorx serve`

```bash
neorx serve --port 8000
```

Starts the [REST API](rest-api.md) with uvicorn. Defaults to
`0.0.0.0:8000`.

## Engine subcommands

### `neorx genmol`

```bash
neorx genmol download            # drug-like molecules from ChEMBL
neorx genmol train
neorx genmol generate --n 100
neorx genmol evaluate
neorx genmol interpolate
neorx genmol screen
```

See [GenMol](genmol.md).

### `neorx dockbot`

```bash
neorx dockbot prepare-protein 1BNA
neorx dockbot binding-site 1BNA
neorx dockbot dock-cmd 1BNA --ligand "CC(=O)Oc1ccccc1C(O)=O"
neorx dockbot screen 1BNA --library compounds.smi
```

See [DockBot](dockbot.md).

### `neorx molscreen`

```bash
neorx molscreen screen "CC(=O)Oc1ccccc1C(O)=O"
```

See [MolScreen](molscreen.md).

### `neorx mirrorfold`

```bash
neorx mirrorfold predict YGGFL
neorx mirrorfold compare YGGFL
neorx mirrorfold assess YGGFL
neorx mirrorfold info
```

See [MirrorFold](mirrorfold.md).

### `neorx causalbiorl`

```bash
neorx causalbiorl train --env GeneticToggle-v0
neorx causalbiorl benchmark
neorx causalbiorl play --env CellGrowth-v0
neorx causalbiorl visualise
```

See [CausalBioRL](causalbiorl.md).

### `neorx exp`

```bash
neorx exp list
neorx exp run neorx-7disease
neorx exp show <run-id>
neorx exp figure --from <run-id>
neorx exp replay <run-id>
neorx exp prune <run-id>
```

`replay` exits non-zero when the rows differ, so it can gate CI.

!!! note "`neorx exp` needs the repository"
    The experiment *definitions* are not in the wheel. On a clean
    install these commands say so by name and point at the repository,
    rather than reporting an empty list. See
    [Experiments and run records](experiments.md).

### `neorx snapshot`

```bash
neorx snapshot build opentargets 21.11
neorx snapshot list
```

`list` prints source, release, row count, extractor version and digest
for every built extract. See [Versioned snapshots](snapshots.md).

## Logging

`--verbose` and `--log-file` are on all four pipeline commands. Logs go
to stderr, so stdout stays usable in a pipe.
