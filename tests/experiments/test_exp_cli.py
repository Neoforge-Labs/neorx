"""`neorx exp` is the only supported way to run an experiment."""

from typer.testing import CliRunner

from neorx.cli import app
from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

runner = CliRunner()


def test_exp_is_a_neorx_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "exp" in result.stdout


def test_exp_list_names_the_registered_experiments():
    result = runner.invoke(app, ["exp", "list"])
    assert result.exit_code == 0
    assert "neorx-7disease" in result.stdout


def test_exp_run_rejects_an_unknown_name():
    result = runner.invoke(app, ["exp", "run", "no-such-experiment"])
    assert result.exit_code != 0


@experiment(name="cli-replay-timing-fixture", help="", volatile_fields=("wall_clock_s",))
def _cli_timing_fixture(record: RunRecord) -> None:
    """Like the three real experiments, writes a wall-clock field on every

    row, so an honest replay must report a volatile diff even while
    reporting IDENTICAL overall.
    """
    import time

    record.append_row({"disease": "HIV", "F1": 0.5, "wall_clock_s": time.perf_counter()})


def test_exp_replay_reports_identical_but_still_prints_volatile_diffs():
    """IDENTICAL must not swallow the timing deltas it excludes from the

    verdict -- they still have to reach stdout, or the CLI silently
    discards exactly the differences volatile_fields was built to keep
    visible.
    """
    run_result = runner.invoke(app, ["exp", "run", "cli-replay-timing-fixture"])
    assert run_result.exit_code == 0, run_result.stdout
    run_id = run_result.stdout.split()[0]

    replay_result = runner.invoke(app, ["exp", "replay", run_id])

    assert replay_result.exit_code == 0, replay_result.stdout
    assert "IDENTICAL" in replay_result.stdout
    assert "volatile field(s) differed" in replay_result.stdout
    assert "wall_clock_s" in replay_result.stdout
