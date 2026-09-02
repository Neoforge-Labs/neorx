"""Unified NeoRx command-line interface.

The core pipeline commands sit at the top level; each module is a
subcommand::

    neorx run HIV --top-n 5
    neorx genmol sample --n 100
    neorx dockbot dock-cmd 1BNA --ligand "CC(=O)O"
"""

from __future__ import annotations

import typer

from neorx.causalbiorl.__main__ import app as causalbiorl_app
from neorx.core.__main__ import app as core_app
from neorx.dockbot.__main__ import app as dockbot_app
from neorx.genmol.__main__ import app as genmol_app
from neorx.mirrorfold.__main__ import app as mirrorfold_app

app = typer.Typer(
    name="neorx",
    help="Causal drug target discovery via Pearl's do-calculus.",
    no_args_is_help=True,
)

# Core pipeline commands stay top-level: `neorx run`, `neorx identify`.
for _command in core_app.registered_commands:
    app.registered_commands.append(_command)

app.add_typer(genmol_app, name="genmol", help="Molecular generation (VAE).")
app.add_typer(dockbot_app, name="dockbot", help="Molecular docking.")
app.add_typer(causalbiorl_app, name="causalbiorl", help="Causal RL environments.")
app.add_typer(mirrorfold_app, name="mirrorfold", help="Mirror-image protein analysis.")

__all__ = ["app", "main"]


def main() -> None:
    """Console-script entry point."""
    app()
