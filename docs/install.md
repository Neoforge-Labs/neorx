# Install

NeoRx requires **Python 3.12 or newer** and is managed with
[uv](https://docs.astral.sh/uv/).

```bash
uv pip install neorx
```

From a clone:

```bash
git clone https://github.com/Neoforge-Labs/neorx.git
cd neorx
uv sync
```

`uv sync` creates `.venv/` and installs the project editable with its
runtime dependencies. Add `--group dev` for the test and lint tooling,
`--group docs` to build this site.

!!! warning "`uv`, not bare `pip`"
    This project's dependency floors are deliberate and documented
    in-line in `pyproject.toml` — several of them exist because a lower
    version is genuinely uninstallable or crashes on the declared minimum
    interpreter. `scripts/check_floors.sh` verifies them in CI by
    installing each floor. Use `uv pip install`, never `pip` directly.

## What comes with it

The base install is not small: NeoRx depends on PyTorch, RDKit,
torch-geometric, DoWhy, scikit-learn, Gymnasium and
stable-baselines3, because the generation, screening and
reinforcement-learning engines are part of the package rather than
optional add-ons.

The trained GenMol checkpoint (~16 MB) ships inside the wheel, so
molecule generation works on a fresh install with no download step. See
[GenMol](genmol.md).

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `docking` | `vina` | Real AutoDock Vina docking rather than the CLI path |
| `mirror` | `fair-esm` | Local ESM-2 structure prediction in [MirrorFold](mirrorfold.md) |
| `rl` | `stable-baselines3` | Already a base dependency; the extra is retained for compatibility |

```bash
uv sync --extra docking
uv sync --extra mirror
```

## Verify the install

```bash
neorx --help
```

In Python:

```python
>>> import neorx
>>> neorx.__version__
'0.3.1'

```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `NEORX_CACHE_BACKEND` | `file` | `file` or `redis` for the API response cache |
| `NEORX_SEED` | `42` | Seeds the sensitivity-analysis RNG; `--seed` on the CLI sets it |
| `NEORX_WEIGHTS` | — | JSON object overriding the [composite score weights](scoring.md) |
| `REDIS_URL` | — | Redis connection URL when the cache backend is `redis` |
| `DATABASE_URL` | — | PostgreSQL URL for graph and job persistence |

## Docker

The repository ships a `docker-compose.yml` bringing up Redis,
PostgreSQL and the [REST API](rest-api.md):

```bash
cp .env.example .env
docker compose up -d
```

The API then serves on `http://localhost:8000`, with interactive docs at
`/docs`.

## Running the tests

```bash
.venv/bin/python -m pytest -m "not network and not slow"
```

The `network` marker covers tests that perform real HTTP I/O and `slow`
covers tests that execute a real experiment run; CI deselects `network`
so that a third-party outage cannot turn the gates red.
