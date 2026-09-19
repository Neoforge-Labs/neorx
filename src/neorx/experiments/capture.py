"""Freeze an experiment's HTTP inputs so a run can be replayed exactly.

The seven HTTP data sources call module-level ``requests.get``/``post`` at
13 sites and construct a Session per call, so a transport adapter mounted
on a Session would intercept nothing. vcrpy patches at the connection
level and therefore captures all of them without editing a single source
module.

This module is the one sanctioned use of monkeypatching in the project:
it is imported only by the experiment runner, is active only inside the
``capture`` context, and never touches a library code path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from neorx.experiments.record import RunRecord


class RecorderUnavailableError(RuntimeError):
    """vcrpy is not installed, so HTTP cannot be frozen or replayed."""


def _vcr():
    """Import vcrpy at the point of use, not at import time.

    vcrpy is a development dependency: it exists to freeze an experiment's
    HTTP inputs, which only a recording or replaying run does. Importing
    it at module level made it a runtime dependency of the whole CLI --
    `neorx.cli` imports the experiments app, which imports the registry,
    which imports this module -- so every command failed on a clean
    install with `ModuleNotFoundError: No module named 'vcr'`. The nightly
    end-to-end job caught it and reported it as "likely an upstream API
    change", which sent the diagnosis in the wrong direction.

    Raising here rather than degrading: a run that asked for its inputs to
    be frozen and silently did not freeze them is unreplayable and does
    not say so.
    """
    try:
        import vcr
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by test
        raise RecorderUnavailableError(
            "recording or replaying an experiment's HTTP needs vcrpy, which "
            "is a development dependency. Install it with "
            "`uv sync --group dev`. Experiments that do not capture HTTP "
            "run without it."
        ) from exc
    return vcr


class NoInteractionsRecordedError(RuntimeError):
    """A recording session captured no HTTP traffic at all.

    This almost always means the code under test bypassed the recorder --
    for example a source that switched to httpx or aiohttp -- rather than
    that the experiment genuinely made no calls.
    """


def cassette_path(record: RunRecord) -> Path:
    return record.path / "inputs" / "http.yaml"


def interaction_count(record: RunRecord) -> int:
    path = cassette_path(record)
    if not path.exists():
        return 0
    import yaml  # type: ignore[import-untyped]

    data = yaml.safe_load(path.read_text()) or {}
    return len(data.get("interactions", []))


@contextmanager
def capture(record: RunRecord, *, mode: str = "record") -> Iterator[None]:
    """Record or replay the HTTP traffic of the enclosed block.

    Parameters
    ----------
    mode : {"record", "replay"}
        ``record`` performs live calls and writes them to the cassette.
        ``replay`` serves them back and refuses any request the cassette
        does not contain.
    """
    if mode not in ("record", "replay"):
        raise ValueError(f"mode must be 'record' or 'replay', got {mode!r}")

    path = cassette_path(record)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "replay" and not path.exists():
        raise FileNotFoundError(
            f"cannot replay {record.run_id}: no cassette at {path}. "
            f"Its inputs may have been pruned."
        )

    config = _vcr().VCR(
        record_mode="all" if mode == "record" else "none",
        match_on=["method", "scheme", "host", "port", "path", "query"],
        decode_compressed_response=True,
    )

    with config.use_cassette(str(path)):
        yield

    if mode == "record" and interaction_count(record) == 0:
        raise NoInteractionsRecordedError(
            f"recording {record.run_id} captured no HTTP interactions. "
            f"The code under test did not route through the recorder -- check "
            f"whether a data source now uses a client other than requests."
        )
