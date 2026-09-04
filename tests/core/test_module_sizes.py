"""
Module size ceilings.

A file that has grown past a few hundred lines is usually doing more
than one thing, and both humans and agents edit focused files more
reliably. identifier.py reached 1291 lines while holding graph
semantics, identification, scoring, and evidence counting at once.
"""

from pathlib import Path

import pytest

import neorx

MAX_LINES = 600

# Task 12 appends "causalbiorl/envs" to this tuple. It is not listed here
# because drug_discovery.py is 884 lines until Task 12 splits it, and a task
# must never commit a red suite.
_PACKAGES = ("core/causal",)


def _modules() -> list[Path]:
    root = Path(neorx.__file__).parent
    return sorted(
        path
        for package in _PACKAGES
        for path in (root / package).rglob("*.py")
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_module_is_under_the_line_ceiling(path):
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= MAX_LINES, (
        f"{path.name} is {n_lines} lines, over the {MAX_LINES}-line ceiling. "
        f"Split it by responsibility rather than raising the ceiling."
    )
