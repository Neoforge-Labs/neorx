"""
MolScreen CLI
==============

Command-line interface for drug-likeness screening.

Commands
--------
* ``molscreen screen``  — Screen a single SMILES and print the verdict

Examples
--------
.. code-block:: bash

    # Screen aspirin
    molscreen screen "CC(=O)Oc1ccccc1C(O)=O"

    # Screen with a name and JSON output
    molscreen screen "CC(=O)Oc1ccccc1C(O)=O" --name aspirin --json
"""

from __future__ import annotations

import json as json_module

import typer

app = typer.Typer(
    name="molscreen",
    help="MolScreen — drug-likeness screening and molecular property calculation.",
    no_args_is_help=True,
)


@app.command()
def screen(
    smiles: str = typer.Argument(..., help="SMILES string of the molecule to screen."),
    name: str = typer.Option("", "--name", "-n", help="Optional molecule name for the report."),
    json: bool = typer.Option(False, "--json", help="Print the result as JSON."),
) -> None:
    """Screen a molecule and print its drug-likeness verdict."""
    from . import (
        calculate_properties,
        classify_drug_likeness,
        parse_smiles,
        qed_score,
        run_all_filters,
    )

    mol = parse_smiles(smiles)
    if mol is None:
        typer.echo(f"Invalid SMILES: {smiles!r}", err=True)
        raise typer.Exit(code=1)

    props = calculate_properties(mol)
    filters = run_all_filters(mol)
    verdict = classify_drug_likeness(filters)
    qed = qed_score(mol)

    if json:
        result = {
            "smiles": smiles,
            "name": name or None,
            "verdict": verdict,
            "qed_score": qed,
            "properties": props.model_dump() if props is not None else None,
            "filters": {f.name: f.passed for f in filters},
        }
        typer.echo(json_module.dumps(result, indent=2))
        return

    label = f" ({name})" if name else ""
    typer.echo(f"SMILES{label}: {smiles}")
    typer.echo(f"Verdict: {verdict}")
    typer.echo(f"QED: {qed:.3f}" if qed is not None else "QED: n/a")
    for f in filters:
        status = "PASS" if f.passed else "FAIL"
        typer.echo(f"  [{status}] {f.name}")


def main() -> None:
    """Entry point for the ``molscreen`` CLI."""
    app()


if __name__ == "__main__":
    main()
