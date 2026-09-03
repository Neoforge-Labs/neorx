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

import vcr

from neorx.experiments.record import RunRecord


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

    config = vcr.VCR(
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
