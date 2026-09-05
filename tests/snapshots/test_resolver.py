"""Choosing between live and snapshot, and refusing in between.

A dated run that quietly mixes in current data produces exactly the
anachronistic graph this sub-project exists to prevent, and does so
invisibly. So a source with no snapshot at a requested date is a named
refusal, never a fallback.
"""

import pytest

from neorx.snapshots.resolver import SourceResolver, UnpinnedSourceError


class _FakeStore:
    def __init__(self, present):
        self._present = present

    def has(self, source, release):
        return (source, release) in self._present


RELEASES = {
    "2018-06": {"opentargets": "18.06", "omnipath": "20180614"},
    "2025-06": {"opentargets": "25.06", "omnipath": "20250813"},
}


def _resolver(present=(("opentargets", "18.06"), ("omnipath", "20180614"))):
    return SourceResolver(
        store=_FakeStore(set(present)),
        release_for=lambda source, as_of: RELEASES[as_of][source],
        live={"opentargets": lambda: "LIVE_OT", "omnipath": lambda: "LIVE_OMNI",
              "string": lambda: "LIVE_STRING"},
    )


def test_no_date_returns_the_live_client():
    assert _resolver().resolve("opentargets", as_of=None)() == "LIVE_OT"


def test_an_unpinned_source_stays_live_even_on_a_dated_run():
    # STRING contributes only associational edges, which identification
    # filters out, so pinning it would buy rigour against a leak that does
    # not exist.
    assert _resolver().resolve("string", as_of="2018-06")() == "LIVE_STRING"


def test_a_dated_run_uses_the_snapshot_for_a_pinned_source():
    reader = _resolver().resolve("opentargets", as_of="2018-06")
    assert reader() != "LIVE_OT"


def test_a_missing_snapshot_refuses_rather_than_falling_back():
    with pytest.raises(UnpinnedSourceError):
        _resolver().resolve("opentargets", as_of="2025-06")


def test_the_refusal_names_the_source_and_the_release():
    with pytest.raises(UnpinnedSourceError, match="opentargets"):
        _resolver().resolve("opentargets", as_of="2025-06")
    with pytest.raises(UnpinnedSourceError, match="25.06"):
        _resolver().resolve("opentargets", as_of="2025-06")


def test_the_refusal_says_it_will_not_fall_back():
    # The message is the mechanism: someone hitting this must not conclude
    # that omitting the date is the fix.
    with pytest.raises(UnpinnedSourceError, match="not fall back"):
        _resolver().resolve("omnipath", as_of="2025-06")
