"""The snapshot CLI.

Downloading lives here and only here, so every reader stays a pure
function of already-fetched data and unit tests need no network. These
tests exercise the wiring with a local file rather than a URL.
"""

import re

import polars as pl
from typer.testing import CliRunner

from neorx.snapshots.__main__ import app

runner = CliRunner()

# The exact subpaths `_build_opentargets` joins onto its `local_path`
# argument for each era -- see the module docstring of
# `neorx.snapshots.opentargets` and `_OPENTARGETS_SUBPATHS` /
# `_OPENTARGETS_MODERN_SUBPATHS` in `neorx.snapshots.__main__`. Duplicated
# here deliberately: the tests below exist precisely to catch the CLI
# module using the wrong one of these for a given release.
_SUBPATHS_2111 = ("output/etl/parquet/associationByDatasourceDirect", "output/etl/parquet/targets")
_SUBPATHS_MODERN = ("output/association_by_datasource_direct", "output/target")


def _write_parquet_release_tree(root, assoc_subpath: str, target_subpath: str) -> None:
    """Build a tiny two-table Parquet release tree at the given subpaths.

    One association row scores positive and one target row supplies its
    symbol, so a correctly-wired build yields exactly one output row --
    the same shape `read_2111`/`read_modern`'s own unit tests use.
    """
    assoc_dir = root / assoc_subpath
    target_dir = root / target_subpath
    assoc_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "datatypeId": "genetic_association",
                "datasourceId": "eva",
                "diseaseId": "EFO_0000616",
                "targetId": "ENSG00000121879",
                "score": 0.61,
                "evidenceCount": 3,
            }
        ]
    ).write_parquet(assoc_dir / "part-0.parquet")
    pl.DataFrame(
        [{"id": "ENSG00000121879", "approvedSymbol": "PIK3CA", "biotype": "protein_coding"}]
    ).write_parquet(target_dir / "part-0.parquet")


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
    # Three distinct target/disease pairs, each with one positive-score
    # datatype, so the built extract has a distinctive row count (3) that
    # could not be mistaken for the extractor_version (always 1) that also
    # appears on the line -- `assert "1" in result.stdout` would pass even
    # if the row count printed were wrong, since extractor_version=1 is
    # always present.
    lines = [
        f'{{"target": {{"id": "ENSG{i}", "gene_info": {{"symbol": "GENE{i}"}}}},'
        f' "disease": {{"id": "EFO_{i}"}},'
        f' "association_score": {{"datatypes": {{"genetic_association": 0.6}}}}}}'
        for i in range(3)
    ]
    src = tmp_path / "18.06.jsonl"
    src.write_text("\n".join(lines) + "\n")
    root = tmp_path / "snapshots"
    runner.invoke(
        app, ["build", "opentargets", "18.06", "--from-file", str(src), "--root", str(root)]
    )
    result = runner.invoke(app, ["list", "--root", str(root)])
    assert "18.06" in result.stdout
    match = re.search(r"rows=(\d+)", result.stdout)
    assert match is not None, result.stdout
    assert match.group(1) == "3"


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


def test_build_writes_an_extract_from_a_2111_parquet_tree(tmp_path):
    release_root = tmp_path / "release-2111"
    _write_parquet_release_tree(release_root, *_SUBPATHS_2111)
    root = tmp_path / "snapshots"
    result = runner.invoke(
        app,
        ["build", "opentargets", "21.11", "--from-file", str(release_root), "--root", str(root)],
    )
    assert result.exit_code == 0, result.stdout
    extract_path = root / "opentargets" / "21.11" / "associations.parquet"
    assert extract_path.exists()
    extract = pl.read_parquet(extract_path)
    assert extract.height == 1
    assert extract["target_symbol"].to_list() == ["PIK3CA"]


def test_build_writes_an_extract_from_a_modern_parquet_tree(tmp_path):
    release_root = tmp_path / "release-2506"
    _write_parquet_release_tree(release_root, *_SUBPATHS_MODERN)
    root = tmp_path / "snapshots"
    result = runner.invoke(
        app,
        ["build", "opentargets", "25.06", "--from-file", str(release_root), "--root", str(root)],
    )
    assert result.exit_code == 0, result.stdout
    extract_path = root / "opentargets" / "25.06" / "associations.parquet"
    assert extract_path.exists()
    extract = pl.read_parquet(extract_path)
    assert extract.height == 1
    assert extract["target_symbol"].to_list() == ["PIK3CA"]


def test_a_2111_build_does_not_silently_read_a_modern_shaped_tree(tmp_path):
    # 21.11 and modern use different subpaths (output/etl/parquet/... with
    # camelCase names vs. the flatter output/<snake_case>/ layout). A tree
    # laid out only in the modern shape has nothing at the 21.11 subpaths,
    # so a correctly-wired build for release "21.11" must fail rather than
    # find data by accident. If the two eras' subpath tables (or the
    # read_2111/read_modern reader selection) were ever swapped, this build
    # would find the modern-shaped tree's files and silently succeed --
    # which is exactly what this test exists to catch.
    release_root = tmp_path / "release-2506"
    _write_parquet_release_tree(release_root, *_SUBPATHS_MODERN)
    root = tmp_path / "snapshots"
    result = runner.invoke(
        app,
        ["build", "opentargets", "21.11", "--from-file", str(release_root), "--root", str(root)],
    )
    assert result.exit_code != 0
    assert not (root / "opentargets" / "21.11" / "associations.parquet").exists()


def test_list_marks_a_synthesised_consensus_row(tmp_path):
    # The 2018 OmniPath archive has no consensus_direction column at all;
    # read_archive_tsv fills it from is_directed and reports that it did
    # so via the second element of its return tuple, which build_cmd
    # threads into SnapshotEntry.synthesised_consensus. `list` is the only
    # place a user inspects what was built, so this fact must be visible
    # there, not just recorded silently in the manifest.
    tsv = (
        "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\t"
        "is_stimulation\tis_inhibition\tsources\treferences\tdip_url\tomnipath\t"
        "ncbi_tax_id_source\tncbi_tax_id_target\n"
        "P04626\tP01133\tERBB2\tEGF\t1\t1\t0\tSIGNOR\tSIGNOR:12345678\t\t1\t"
        "9606\t9606\n"
    )
    src = tmp_path / "omnipath_2018.tsv"
    src.write_text(tsv)
    root = tmp_path / "snapshots"
    build_result = runner.invoke(
        app, ["build", "omnipath", "2018", "--from-file", str(src), "--root", str(root)]
    )
    assert build_result.exit_code == 0, build_result.stdout

    result = runner.invoke(app, ["list", "--root", str(root)])
    assert "synthesised-consensus" in result.stdout
    # A row without the caveat must not carry the same tag.
    src_18_06 = tmp_path / "18.06.jsonl"
    src_18_06.write_text(
        '{"target": {"id": "ENSG1", "gene_info": {"symbol": "PIK3CA"}},'
        ' "disease": {"id": "EFO_1"},'
        ' "association_score": {"datatypes": {"genetic_association": 0.6}}}\n'
    )
    runner.invoke(
        app, ["build", "opentargets", "18.06", "--from-file", str(src_18_06), "--root", str(root)]
    )
    result = runner.invoke(app, ["list", "--root", str(root)])
    lines = {line.split()[0]: line for line in result.stdout.splitlines() if line.strip()}
    assert "synthesised-consensus" in lines["omnipath"]
    assert "synthesised-consensus" not in lines["opentargets"]
