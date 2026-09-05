"""The builder must read what upstream actually ships.

Every OmniPath archive dump is ``.tsv.xz`` -- all 22 of them, back to
20180614 -- and this function read xz as UTF-8 until 2026-09-06, so
``neorx snapshot build omnipath`` failed on every real input with
``UnicodeDecodeError: invalid start byte``. The suite did not catch it
because it only ever handed the function text unpacked by hand, which is
the one case that was never broken.

These tests compress in the test rather than committing binary fixtures,
so what is exercised is real gzip and real xz rather than a stand-in.
"""

import gzip
import lzma

from neorx.snapshots.__main__ import _read_text

CONTENT = "source\ttarget\nHOMER1\tTRPC3\n"


def test_plain_text_is_read_as_is(tmp_path):
    path = tmp_path / "interactions.tsv"
    path.write_text(CONTENT, encoding="utf-8")
    assert _read_text(path) == CONTENT


def test_gzip_is_decompressed(tmp_path):
    path = tmp_path / "associations.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(CONTENT)
    assert _read_text(path) == CONTENT


def test_xz_is_decompressed(tmp_path):
    # The case that was broken. Named as the archive names it.
    path = tmp_path / "omnipath_webservice_interactions__20180614-20181114.tsv.xz"
    with lzma.open(path, "wt", encoding="utf-8") as fh:
        fh.write(CONTENT)
    assert _read_text(path) == CONTENT


def test_the_format_is_read_from_the_bytes_not_the_name(tmp_path):
    # An archive URL's extension is a claim about the file. This one is a
    # lie, and the bytes are what the reader must believe.
    path = tmp_path / "definitely_plain.tsv"
    with lzma.open(path, "wt", encoding="utf-8") as fh:
        fh.write(CONTENT)
    assert _read_text(path) == CONTENT
