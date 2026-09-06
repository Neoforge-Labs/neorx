"""Which release a date resolves to, for each pinned source.

``SourceResolver`` takes ``release_for`` as an injected callable so the
date-to-release table lives in one place instead of at every call site.
Until now the only implementations were lambdas inside tests, which meant
no production path could run a dated build at all: the machinery existed
and nothing could reach it.

Every entry here was verified against the live archive on 2026-09-06 by
listing it, not by reading the spec's table. Two of the three pairings are
as close as that table claimed; the third is not, and the discrepancy is
recorded rather than smoothed over.
"""

from __future__ import annotations

__all__ = [
    "TIME_POINTS",
    "UnknownTimePointError",
    "omnipath_archive_url",
    "release_for",
]


class UnknownTimePointError(KeyError):
    """A date with no pinned release table entry."""


# OmniPath's archive names a dump for the window it was current in, so the
# dump covering a date carries data as of that window's START. That is why
# `data_as_of` is recorded separately from the archive name: for 2018 and
# 2021 they are within days of the OpenTargets release, and for 2025 they
# are 23 months apart.
TIME_POINTS: dict[str, dict[str, str]] = {
    "2018-06": {
        "opentargets": "18.06",
        "omnipath": "20180614-20181114",
        "omnipath_data_as_of": "2018-06-14",
    },
    "2021-11": {
        "opentargets": "21.11",
        "omnipath": "20211113-20220114",
        "omnipath_data_as_of": "2021-11-13",
    },
    "2025-06": {
        "opentargets": "25.06",
        # The archive's last interaction dump. Its window CONTAINS 2025-06,
        # so it is the correct choice -- but OmniPath published no refresh
        # between 2023-07-28 and 2025-08-13, so this time point pairs a
        # 2025 OpenTargets release with regulatory data roughly 23 months
        # older. The spec's original "latest <= 2025-06" concealed that.
        # It belongs in the paper's limitations.
        "omnipath": "20230728-20250813",
        "omnipath_data_as_of": "2023-07-28",
    },
}

_ARCHIVE_ROOT = "https://archive.omnipathdb.org"


def release_for(source: str, as_of: str) -> str:
    """The release identifier ``source`` is pinned to at ``as_of``.

    Raises rather than guessing. A dated run that silently fell back to
    some nearby release would produce exactly the anachronistic graph this
    sub-project exists to prevent, and would do it invisibly -- the same
    reasoning that makes ``SourceResolver`` refuse an unbuilt snapshot
    instead of using the live client.
    """
    try:
        point = TIME_POINTS[as_of]
    except KeyError:
        raise UnknownTimePointError(
            f"No pinned releases for {as_of!r}. Known time points: "
            f"{', '.join(sorted(TIME_POINTS))}. Add one to "
            f"neorx.snapshots.releases.TIME_POINTS only after verifying "
            f"against the upstream archive that both releases exist."
        ) from None

    try:
        return point[source]
    except KeyError:
        raise UnknownTimePointError(
            f"{as_of!r} pins no release for source {source!r}. Pinned "
            f"sources at that date: "
            f"{', '.join(s for s in sorted(point) if not s.endswith('_data_as_of'))}."
        ) from None


def omnipath_archive_url(as_of: str) -> str:
    """The archive URL for the OmniPath dump pinned to ``as_of``.

    Every dump is ``.tsv.xz``; the builder detects compression from the
    file's magic bytes rather than this extension, because an archive
    URL's extension is a claim and this one was wrong in the spec for the
    life of the sub-project.
    """
    release = release_for("omnipath", as_of)
    return f"{_ARCHIVE_ROOT}/omnipath_webservice_interactions__{release}.tsv.xz"
