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
        False, "--allow-large", help="Permit a record larger than 50 MB."
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


def main() -> None:
    app()
