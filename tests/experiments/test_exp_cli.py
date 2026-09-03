"""`neorx exp` is the only supported way to run an experiment."""

import pytest
from typer.testing import CliRunner

from neorx.cli import app

runner = CliRunner()


def test_exp_is_a_neorx_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "exp" in result.stdout


@pytest.mark.xfail(
    reason="neorx-7disease is registered in Task 8; remove this marker then",
    strict=True,
)
def test_exp_list_names_the_registered_experiments():
    result = runner.invoke(app, ["exp", "list"])
    assert result.exit_code == 0
    assert "neorx-7disease" in result.stdout


def test_exp_run_rejects_an_unknown_name():
    result = runner.invoke(app, ["exp", "run", "no-such-experiment"])
    assert result.exit_code != 0
