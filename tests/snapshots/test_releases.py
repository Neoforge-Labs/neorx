"""The date-to-release table, and why it refuses rather than guesses.

Until this module existed, every implementation of `release_for` was a
lambda inside a test, so no production path could run a dated build. These
tests pin the table's values because they are measurements of the upstream
archive, not conventions: they were verified by listing the archive on
2026-09-06, and the 2025 entry in particular contradicts what the spec
originally asserted.
"""

import pytest

from neorx.snapshots.releases import (
    TIME_POINTS,
    UnknownTimePointError,
    omnipath_archive_url,
    release_for,
)


def test_the_three_time_points_resolve_to_their_verified_releases():
    assert release_for("opentargets", "2018-06") == "18.06"
    assert release_for("omnipath", "2018-06") == "20180614-20181114"
    assert release_for("opentargets", "2021-11") == "21.11"
    assert release_for("omnipath", "2021-11") == "20211113-20220114"
    assert release_for("opentargets", "2025-06") == "25.06"
    assert release_for("omnipath", "2025-06") == "20230728-20250813"


def test_the_2025_time_point_records_its_own_staleness():
    """The one entry that is not what the spec first claimed.

    OmniPath published no interaction dump between 2023-07-28 and
    2025-08-13, so the dump whose window contains 2025-06 carries data
    roughly 23 months older than the OpenTargets release beside it. The
    spec said "latest <= 2025-06", which reads as contemporaneous. This is
    a limitation to state in the paper, and it cannot be stated if the
    table does not carry it.
    """
    assert TIME_POINTS["2025-06"]["omnipath_data_as_of"] == "2023-07-28"
    # The other two are within days of their OpenTargets release.
    assert TIME_POINTS["2018-06"]["omnipath_data_as_of"] == "2018-06-14"
    assert TIME_POINTS["2021-11"]["omnipath_data_as_of"] == "2021-11-13"


def test_an_unknown_date_raises_and_names_what_is_known():
    with pytest.raises(UnknownTimePointError) as excinfo:
        release_for("opentargets", "2019-03")
    message = str(excinfo.value)
    assert "2019-03" in message
    # It must not silently pick a nearby release: that is the anachronism
    # the whole sub-project exists to prevent.
    for known in ("2018-06", "2021-11", "2025-06"):
        assert known in message


def test_an_unpinned_source_raises_rather_than_returning_a_default():
    with pytest.raises(UnknownTimePointError) as excinfo:
        release_for("string", "2018-06")
    assert "string" in str(excinfo.value)


def test_the_archive_url_is_built_from_the_pinned_release():
    url = omnipath_archive_url("2018-06")
    assert url == (
        "https://archive.omnipathdb.org/"
        "omnipath_webservice_interactions__20180614-20181114.tsv.xz"
    )


def test_every_time_point_pins_both_sources():
    # A time point that pins only one source would fail halfway through a
    # dated build, after the other source's snapshot had already been
    # resolved.
    for as_of in TIME_POINTS:
        assert release_for("opentargets", as_of)
        assert release_for("omnipath", as_of)
