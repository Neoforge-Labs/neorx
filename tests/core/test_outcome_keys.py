"""Today's clinical phase must not ride on a dated node.

Sub-project 6 asks whether causal structure predicts trial outcomes.
ChEMBL's `clinical_phase` -- and the score it derives 60% from -- IS that
outcome. A live source may enrich a pinned node with what the release is
silent about; it may not hand it the answer.

Only `evaluate_pathogen_target` reads `clinical_phase` today, and dated
builds no longer reach it because pathogen nodes are excluded by type. So
this is currently inert. It is inert by the coincidence of a different
rule, which is why it is pinned down here rather than left to hold.
"""

import pytest

from neorx.core.graph.dated_frame import (
    OUTCOME_KEYS,
    enrich,
    live_metadata_for_pinned,
)
from neorx.core.graph.graph_builder import _merge_nodes
from neorx.core.graph.models import GraphNode, NodeType


def _pinned(**metadata):
    md = {"snapshot_release": "18.06"}
    md.update(metadata)
    return GraphNode(node_id="gene:CCR5", name="CCR5", node_type=NodeType.GENE,
                     source="Open Targets", score=0.3, metadata=md)


def _live(**metadata):
    return GraphNode(node_id="gene:CCR5", name="CCR5", node_type=NodeType.GENE,
                     source="ChEMBL", score=0.8, metadata=dict(metadata))


def _live_named(source, **metadata):
    return GraphNode(node_id="gene:CCR5", name="CCR5", node_type=NodeType.GENE,
                     source=source, score=0.7, metadata=dict(metadata))


@pytest.mark.parametrize("key", sorted(OUTCOME_KEYS))
def test_no_outcome_key_can_be_enriched_onto_a_pinned_node(key):
    node = _pinned()
    enrich(node, key, 4)
    assert key not in node.metadata


def test_ordinary_enrichment_still_reaches_a_pinned_node(key="subcellular_location"):
    # The rule refuses the outcome, not enrichment in general -- otherwise
    # it would be a reason not to query the unpinned sources at all.
    node = _pinned()
    enrich(node, key, "membrane")
    assert node.metadata[key] == "membrane"


def test_enrichment_still_cannot_restate_what_the_release_said():
    node = _pinned(has_known_drug=False)
    enrich(node, "has_known_drug", True)
    assert node.metadata["has_known_drug"] is False


def test_an_unpinned_node_is_unaffected():
    # Undated builds must behave exactly as they did.
    node = _live()
    enrich(node, "clinical_phase", 4)
    assert node.metadata["clinical_phase"] == 4


def test_the_filter_keeps_everything_that_is_not_an_outcome():
    kept = live_metadata_for_pinned(
        {"clinical_phase": 4, "go_terms": ["GO:1"], "uniprot_id": "P0DTC2"}
    )
    assert kept == {"go_terms": ["GO:1"], "uniprot_id": "P0DTC2"}


# ── The merge path, both orderings ──────────────────────────────────
#
# Monarch and ChEMBL are queried before Open Targets, so on a dated build
# the LIVE node routinely arrives first and the pinned node second. That
# ordering takes a different branch of `_merge_nodes` from the one every
# other test here exercises, and `update()` only overwrites keys the
# release HAS -- so a live outcome key would survive onto a node that has
# by then become pinned. Removing that branch's filter left the whole
# suite green until this test existed.


def _merge(first, second):
    nodes, _edges = _merge_nodes([first, second], [])
    assert len(nodes) == 1, [n.node_id for n in nodes]
    return nodes[0]


def test_a_live_outcome_key_does_not_survive_a_pinned_node_arriving_second():
    merged = _merge(_live(clinical_phase=4, n_drugs=3), _pinned())
    assert merged.score == 0.3
    assert "clinical_phase" not in merged.metadata
    assert "n_drugs" not in merged.metadata


def test_a_live_outcome_key_does_not_survive_a_pinned_node_arriving_first():
    merged = _merge(_pinned(), _live(clinical_phase=4, n_drugs=3))
    assert merged.score == 0.3
    assert "clinical_phase" not in merged.metadata
    assert "n_drugs" not in merged.metadata


def test_live_enrichment_survives_both_orderings():
    # The rule must not cost the enrichment that justifies querying the
    # unpinned sources on a dated build at all.
    for first, second in (
        (_live(go_terms=["GO:1"]), _pinned()),
        (_pinned(), _live(go_terms=["GO:1"])),
    ):
        assert _merge(first, second).metadata["go_terms"] == ["GO:1"]


def test_two_unpinned_nodes_merge_exactly_as_before():
    # Undated builds are untouched: highest score wins, metadata merges,
    # outcome keys included.
    merged = _merge(_live(clinical_phase=4), _live())
    assert merged.score == 0.8
    assert merged.metadata["clinical_phase"] == 4


# ── Guards that are unreachable through a build, and still must hold ──
#
# `admit` now withholds every unpinned contribution on a dated build, so
# nothing unpinned reaches `_merge_nodes` or `enrich` by that route. That
# makes the guards below defence in depth -- and it makes them exactly the
# kind of guard this session has repeatedly found untested, because a
# build-level test cannot reach them. Removing either one leaves the whole
# invariance suite green. They are tested directly instead.


def test_a_live_source_does_not_extend_a_pinned_nodes_source_string():
    """Provenance is an input, not a label.

    evidence.py SPLITS `source` into collect_source_scores, so a name in
    it becomes an evidence stream and a unit of n_active_sources -- the
    denominator of the consensus term. Measured before this guard: the one
    target ChEMBL had a drug for was the only one whose confidence did not
    move, and the others fell by up to 0.0375.
    """
    merged = _merge(_pinned(), _live())
    assert merged.source == "Open Targets"


def test_two_unpinned_nodes_still_record_both_sources():
    # The guard is about pinned provenance, not about abolishing the
    # corroboration record. An undated build is unchanged.
    merged = _merge(_live(), _live_named("Monarch"))
    assert "ChEMBL" in merged.source and "Monarch" in merged.source


def test_tractability_is_refused_on_a_pinned_node():
    """Named explicitly, because the parametrised test cannot catch this.

    `test_no_outcome_key_can_be_enriched_onto_a_pinned_node` parametrises
    over OUTCOME_KEYS itself, so deleting a key from that set deletes its
    test case along with it and the suite stays green. scoring.py reads
    `tractability` for +0.15 of assess_druggability.
    """
    assert "tractability" in OUTCOME_KEYS
    node = _pinned()
    enrich(node, "tractability", [{"value": True}])
    assert "tractability" not in node.metadata


def test_is_druggable_is_refused_on_a_pinned_node():
    # Same reasoning as tractability: named so that removing it from the
    # set is a failure rather than a silently smaller parametrisation.
    assert "is_druggable" in OUTCOME_KEYS
    node = _pinned()
    enrich(node, "is_druggable", True)
    assert "is_druggable" not in node.metadata
