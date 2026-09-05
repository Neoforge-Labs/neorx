"""The snapshot CLI.

Downloading lives here and only here, so every reader stays a pure
function of already-fetched data and unit tests need no network. These
tests exercise the wiring with a local file rather than a URL.
"""

from typer.testing import CliRunner

from neorx.snapshots.__main__ import app

runner = CliRunner()


def test_list_on_an_empty_root_reports_no_snapshots(tmp_path):
    result = runner.invoke(app, ["list", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "no snapshots" in result.stdout.lower()


def test_build_writes_an_extract_and_a_manifest_entry(tmp_path):
    src = tmp_path / "18.06.jsonl"
    src.write_text(
        '{"target": {"id": "ENSG1", "gene_info": {"symbol": "PIK3CA"}},'
        ' "disease": {"id": "EFO_1"},'
        ' "association_score": {"datatypes": {"genetic_association": 0.6}}}\n'
    )
    root = tmp_path / "snapshots"
    result = runner.invoke(
        app,
        [
            "build",
            "opentargets",
            "18.06",
            "--from-file",
            str(src),
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (root / "opentargets" / "18.06" / "associations.parquet").exists()
    assert (root / "manifest.toml").exists()


def test_list_reports_a_built_snapshot_with_its_row_count(tmp_path):
    src = tmp_path / "18.06.jsonl"
    src.write_text(
        '{"target": {"id": "ENSG1", "gene_info": {"symbol": "PIK3CA"}},'
        ' "disease": {"id": "EFO_1"},'
        ' "association_score": {"datatypes": {"genetic_association": 0.6}}}\n'
    )
    root = tmp_path / "snapshots"
    runner.invoke(
        app, ["build", "opentargets", "18.06", "--from-file", str(src), "--root", str(root)]
    )
    result = runner.invoke(app, ["list", "--root", str(root)])
    assert "18.06" in result.stdout
    assert "1" in result.stdout


def test_building_an_unknown_source_fails_with_a_named_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "build",
            "nosuchsource",
            "1.0",
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "nosuchsource" in result.stdout
