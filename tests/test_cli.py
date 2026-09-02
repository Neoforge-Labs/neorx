"""One CLI, with a subcommand per module."""

import pytest
from typer.testing import CliRunner

from neorx.cli import app

runner = CliRunner()

SUBCOMMANDS = ["genmol", "dockbot", "causalbiorl", "mirrorfold"]


def test_root_help_lists_every_module():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in SUBCOMMANDS:
        assert name in result.stdout


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_each_subcommand_has_its_own_help(name):
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0


def test_core_commands_are_top_level():
    result = runner.invoke(app, ["--help"])
    assert "identify" in result.stdout or "run" in result.stdout
