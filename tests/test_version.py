"""One version number, declared in one place, agreeing everywhere."""

import tomllib
from pathlib import Path

import neorx


def test_version_matches_pyproject():
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert neorx.__version__ == declared


def test_version_is_0_2_0():
    assert neorx.__version__ == "0.2.0"
