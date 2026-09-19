"""One version number, declared in one place, agreeing everywhere."""

import tomllib
from pathlib import Path

import neorx


def test_version_matches_pyproject():
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert neorx.__version__ == declared


def test_version_is_0_3_1():
    """Pinned to the literal, so a bump cannot happen without review.

    `test_version_matches_pyproject` only proves the two files agree; both
    could drift together. This one has to be edited deliberately, which is
    the point -- a release is a decision, not a side effect.
    """
    assert neorx.__version__ == "0.3.1"
