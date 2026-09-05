"""`neorx snapshot` -- build and inspect versioned source snapshots.

This is where downloading happens, and the only place it happens. Every
reader in ``neorx.snapshots.opentargets`` and ``neorx.snapshots.omnipath``
is a pure function of already-fetched data precisely so that unit tests
for those readers need no network; ``build`` is what actually streams a
release, reduces it through the matching reader, writes the canonical
Parquet extract, and records it in the manifest.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import polars as pl
import requests
import typer

from neorx.snapshots.manifest import SnapshotEntry, digest_file, read_manifest, write_entry
from neorx.snapshots.omnipath import read_archive_tsv
from neorx.snapshots.opentargets import read_1806, read_2111, read_modern
from neorx.snapshots.reader import SnapshotStore
from neorx.snapshots.schema import EXTRACTOR_VERSION

app = typer.Typer(
    name="snapshot",
    help="Build and inspect versioned source snapshots.",
    no_args_is_help=True,
)

# Relative paths within a downloaded OpenTargets release tree. Verified
# against the real releases -- see the module docstring of
# neorx.snapshots.opentargets for the source of these paths. 21.11 nests
# its tables under output/etl/parquet/ with camelCase names; every later
# era (the "modern" bucket read_modern covers) uses the flatter
# output/<snake_case>/ layout.
_OPENTARGETS_SUBPATHS: dict[str, tuple[str, str]] = {
    "21.11": ("output/etl/parquet/associationByDatasourceDirect", "output/etl/parquet/targets"),
}
_OPENTARGETS_MODERN_SUBPATHS = ("output/association_by_datasource_direct", "output/target")


def _read_text(local_path: Path) -> str:
    """Read a local file as text, transparently decompressing gzip.

    Upstream dumps of the line-delimited eras are typically gzipped; a
    file already unpacked by hand (as the tests do) is plain text. Both
    must work without a separate flag telling the CLI which one it is.
    """
    with open(local_path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(local_path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return local_path.read_text(encoding="utf-8")


def _build_opentargets(release: str, local_path: Path) -> tuple[pl.DataFrame, bool]:
    """Dispatch on release era and return (canonical rows, False).

    OpenTargets has no synthesised-column caveat like OmniPath's 2018
    archive, so the second element is always False; it exists so this
    function has the same return shape as ``_build_omnipath`` and the
    caller does not need to know which source it is dealing with.
    """
    if release == "18.06":
        lines = _read_text(local_path).splitlines()
        return read_1806(lines), False

    assoc_rel, target_rel = _OPENTARGETS_SUBPATHS.get(release, _OPENTARGETS_MODERN_SUBPATHS)
    associations = pl.read_parquet(local_path / assoc_rel)
    targets = pl.read_parquet(local_path / target_rel)
    reader = read_2111 if release == "21.11" else read_modern
    return reader(associations, targets), False


def _build_omnipath(release: str, local_path: Path) -> tuple[pl.DataFrame, bool]:
    """Read an archived OmniPath TSV.

    ``release`` is accepted but not used to choose a reader -- there is
    exactly one, unlike OpenTargets' three eras -- so that the dispatch
    table below can call every builder the same way.
    """
    _ = release
    return read_archive_tsv(_read_text(local_path))


# One entry per known source. A source not in this table is refused by
# name rather than attempted and failing deeper in the stack.
_BUILDERS = {
    "opentargets": _build_opentargets,
    "omnipath": _build_omnipath,
}


def _download(url: str) -> Path:
    """Stream `url` to a private temp file and return its path.

    Streamed rather than loaded whole so a multi-gigabyte release does
    not have to fit in memory before it is even reduced.
    """
    fd, tmp_name = tempfile.mkstemp(prefix="neorx-snapshot-")
    tmp_path = Path(tmp_name)
    # `open(fd, ...)` is entered before the request so the descriptor is
    # always owned by a context manager -- a failure at any point (the
    # connection, `raise_for_status`, or a chunk write) closes it on the
    # way out. `wrote` only flips once the whole body has been written;
    # if it never does, the partial file is removed before re-raising, so
    # a failed fetch leaves nothing on disk for the caller to clean up.
    wrote = False
    try:
        with open(fd, "wb") as fh, requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
        wrote = True
        return tmp_path
    finally:
        if not wrote:
            tmp_path.unlink(missing_ok=True)


@app.command("build")
def build_cmd(
    source: str = typer.Argument(..., help="Source name: opentargets, omnipath."),
    release: str = typer.Argument(..., help="Release identifier, e.g. 18.06, 21.11, 25.06."),
    root: Path = typer.Option(Path("snapshots"), "--root", help="Snapshot store root."),
    from_file: Path | None = typer.Option(
        None, "--from-file", help="A local path already downloaded, in place of --url."
    ),
    url: str | None = typer.Option(
        None, "--url", help="Source URL, streamed to a temp file and discarded after reduction."
    ),
) -> None:
    """Fetch, reduce, and record one source release as a versioned extract."""
    if source not in _BUILDERS:
        typer.echo(f"unknown source {source!r}; known sources: {', '.join(sorted(_BUILDERS))}")
        raise typer.Exit(code=1)

    if from_file is None and url is None:
        typer.echo("build needs one of --from-file or --url")
        raise typer.Exit(code=1)

    fetched_path: Path | None = None
    try:
        if from_file is not None:
            local_path = from_file
        else:
            # The guard above guarantees url is set when from_file is not.
            local_path = fetched_path = _download(url)  # type: ignore[arg-type]

        frame, synthesised_consensus = _BUILDERS[source](release, local_path)

        store = SnapshotStore(root)
        # Reuses SnapshotStore's own path convention (source ->
        # filename) so the extract's write location and its read
        # location can never drift apart.
        out_path = store._path(source, release)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(out_path)

        sha256 = digest_file(out_path)
        entry = SnapshotEntry(
            source=source,
            release=release,
            url=url or "",
            sha256=sha256,
            extractor_version=EXTRACTOR_VERSION,
            rows=frame.height,
            synthesised_consensus=synthesised_consensus,
        )
        write_entry(root / "manifest.toml", entry)
    finally:
        # Only a fetched download is raw and disposable; a caller-supplied
        # --from-file is theirs to keep.
        if fetched_path is not None:
            fetched_path.unlink(missing_ok=True)

    typer.echo(f"{source} {release}: wrote {frame.height} rows to {out_path}")


@app.command("list")
def list_cmd(
    root: Path = typer.Option(Path("snapshots"), "--root", help="Snapshot store root."),
) -> None:
    """List every built snapshot: source, release, rows, extractor version, digest.

    A row built from an archive lacking a real consensus_direction column
    (OmniPath's 2018 dump) is marked with a trailing `[synthesised-consensus]`
    tag, and a one-line legend is printed after the table when any row
    carries it -- recording the flag in the manifest and then hiding it
    here would defeat the reason it is recorded at all.
    """
    entries = read_manifest(root / "manifest.toml")
    if not entries:
        typer.echo("no snapshots built yet")
        return
    any_synthesised = False
    for (source, release), entry in sorted(entries.items()):
        tag = ""
        if entry.synthesised_consensus:
            any_synthesised = True
            tag = " [synthesised-consensus]"
        typer.echo(
            f"{source:12} {release:10} rows={entry.rows:<8} "
            f"extractor_version={entry.extractor_version} sha256={entry.sha256[:12]}{tag}"
        )
    if any_synthesised:
        typer.echo(
            "[synthesised-consensus]: consensus_direction was not present in the "
            "source and was derived rather than read."
        )


def main() -> None:
    app()
