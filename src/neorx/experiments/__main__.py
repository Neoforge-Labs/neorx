"""`neorx exp` -- run, list and inspect experiment records."""

from __future__ import annotations

import typer

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import list_experiments, run_experiment

app = typer.Typer(
    name="exp",
    help="Run recorded experiments and inspect their run records.",
    no_args_is_help=True,
)


@app.command("list")
def list_cmd() -> None:
    """List every registered experiment."""
    import experiments  # noqa: F401  -- import registers the definitions

    for defn in list_experiments():
        typer.echo(f"{defn.name:22} {defn.help}")


@app.command("run")
def run_cmd(
    name: str = typer.Argument(..., help="Experiment name, from `neorx exp list`."),
    allow_large: bool = typer.Option(
        False, "--allow-large", help="Permit a record larger than 250 MB."
    ),
) -> None:
    """Run an experiment and write its record."""
    import experiments  # noqa: F401

    record = run_experiment(name, allow_large=allow_large)
    typer.echo(f"{record.run_id}  status={record.status}  citable={record.citable}")


@app.command("show")
def show_cmd(run_id: str = typer.Argument(..., help="Run ID to inspect.")) -> None:
    """Print a run record's status and rows."""
    record = RunRecord.load(run_id)
    typer.echo(f"{record.run_id}  status={record.status}  citable={record.citable}")
    for row in record.rows():
        typer.echo(f"  {row}")


@app.command("figure")
def figure_cmd(
    from_run: str = typer.Option(..., "--from", help="Run ID to render from."),
) -> None:
    """Render manuscript figures from a run record."""
    import os

    import experiments  # noqa: F401

    os.environ["NEORX_FIGURE_RUN"] = from_run
    record = run_experiment("figures")
    typer.echo(f"rendered from {from_run}; figure run {record.run_id}")


@app.command("replay")
def replay_cmd(run_id: str = typer.Argument(..., help="Run ID to replay.")) -> None:
    """Re-execute a recorded run against its frozen inputs and diff the rows."""
    import experiments  # noqa: F401
    from neorx.experiments.replay import replay_experiment

    result = replay_experiment(run_id)
    if result.identical:
        typer.echo(f"IDENTICAL  ({run_id} reproduced by {result.replay_run_id})")
        return
    typer.echo(f"DIFFERS  ({len(result.diffs)} field(s))")
    for d in result.diffs:
        typer.echo(f"  row {d.index}  {d.key}: recorded={d.recorded!r} replayed={d.replayed!r}")
    raise typer.Exit(code=1)


@app.command("prune")
def prune_cmd(run_id: str = typer.Argument(..., help="Run ID to prune.")) -> None:
    """Drop a run's frozen inputs, keeping its recorded results."""
    from neorx.experiments.replay import prune_record

    freed = prune_record(run_id)
    typer.echo(f"pruned {run_id}: freed {freed / 1e6:.1f} MB; results retained")


def main() -> None:
    app()
