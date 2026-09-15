"""The tissue gate is an unpinned source, and a dated build must not use it.

The eighth leak channel found on this path, and the first that reached a
reported number *after* the graph was built -- so no fingerprint of a
`DiseaseGraph` could see it.

`TissueFilter` queries the Human Protein Atlas live
(`proteinatlas.org/search/<gene>?format=json`). `scoring.classify_target`
treats its answer as a boolean gate: a gene that fails it is demoted to
CORRELATIONAL whatever its confidence. So on a graph built from a 2018
release, today's expression data decided `classification`,
`is_causal_target` and `tissue_relevant` -- which is exactly what "a dated
build takes nothing from an unpinned source" forbids.

Skipping the gate is not the same as passing it, and a dated run is
therefore more permissive here than a live one. That difference is
recorded on every result rather than left to be inferred from a
`tissue_relevant=True` that looks indistinguishable from a pass.
"""

from unittest.mock import patch

import pytest

from neorx.core.causal.identifier import evaluate_all_targets
from neorx.core.graph.models import (
    DiseaseGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)


def _graph(as_of=None):
    return DiseaseGraph(
        disease_name="HIV infection",
        disease_id="EFO_0000764",
        as_of=as_of,
        nodes=[
            GraphNode(
                node_id="disease:hiv_infection",
                name="HIV infection",
                node_type=NodeType.DISEASE,
                source="NeoRx",
                score=1.0,
            ),
            GraphNode(
                node_id="gene:CCR5",
                name="CCR5",
                node_type=NodeType.GENE,
                source="Open Targets",
                score=0.75,
                metadata={"datatype_scores": {"genetic_association": 0.75}},
            ),
        ],
        edges=[
            GraphEdge(
                source_id="gene:CCR5",
                target_id="disease:hiv_infection",
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=0.75,
                source_db="Open Targets",
                evidence="OT genetic",
                evidence_class="genetic_association",
                primary_sources=["Open Targets"],
            ),
        ],
    )


# A tissue answer hostile enough to flip the gate: the gene is expressed
# only somewhere unrelated to the disease.
_FAILS_GATE = (False, 0.0, "expressed only in testis")


def test_a_dated_build_does_not_query_the_tissue_atlas_at_all():
    """The strongest form: the unpinned source is never reached.

    Asserted on the call rather than on the result, because a result that
    happens to match could still have been produced by a live call whose
    answer agreed.
    """
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant"
    ) as probe:
        probe.return_value = _FAILS_GATE
        evaluate_all_targets(_graph(as_of="2018-06"))
    probe.assert_not_called()


def test_an_undated_build_still_applies_the_gate():
    # The live path is unchanged: this is about dated provenance, not
    # about abandoning the tissue gate.
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant"
    ) as probe:
        probe.return_value = _FAILS_GATE
        results = evaluate_all_targets(_graph())
    probe.assert_called()
    assert results[0].tissue_relevant is False


def test_todays_tissue_data_cannot_change_a_dated_classification():
    """Criterion 11, for a field computed after the graph is built.

    Same snapshot, same graph; the only difference is what the Human
    Protein Atlas says today. Before this fix the two runs disagreed on
    `classification`, `is_causal_target` and `tissue_relevant`.
    """
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant",
        return_value=_FAILS_GATE,
    ):
        hostile = evaluate_all_targets(_graph(as_of="2018-06"))
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant",
        return_value=(True, 0.9, "expressed in blood"),
    ):
        friendly = evaluate_all_targets(_graph(as_of="2018-06"))

    def _fingerprint(results):
        return [
            (
                r.protein_id,
                r.classification,
                r.is_causal_target,
                r.tissue_relevant,
                round(r.causal_confidence, 9),
            )
            for r in results
        ]

    assert _fingerprint(hostile) == _fingerprint(friendly)


def test_the_skipped_gate_is_recorded_not_silent():
    """`tissue_relevant=True` alone would read as a pass.

    A reader comparing a dated classification with a live one needs to
    know the gate never ran, and needs it on the result rather than in a
    log line nobody reads back.
    """
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant",
        return_value=_FAILS_GATE,
    ):
        results = evaluate_all_targets(_graph(as_of="2018-06"))

    explanation = results[0].tissue_explanation
    assert "2018-06" in explanation
    assert "not applied" in explanation.lower()
    # And it must say why, so the asymmetry is not mistaken for a pass.
    assert "not the same as passed" in explanation.lower()


@pytest.mark.parametrize("as_of", ["2018-06", "2021-11", "2025-06"])
def test_every_time_point_skips_it(as_of):
    with patch(
        "neorx.core.bio.tissue_filter.TissueFilter.is_tissue_relevant"
    ) as probe:
        probe.return_value = _FAILS_GATE
        evaluate_all_targets(_graph(as_of=as_of))
    probe.assert_not_called()
