"""
Tests for the OmniPath regulatory source.

OmniPath aggregates signed, directed causal interactions from SIGNOR,
TRRUST, and others. Only interactions the aggregator marks as directed
with a consensus direction may become arrows -- an undirected or
direction-disputed interaction carries no causal claim.
"""

import pytest

from neorx.core.graph.models import EdgeType
from neorx.core.sources.omnipath import _interactions_to_edges


def _row(**overrides) -> dict:
    row = {
        "source_genesymbol": "A",
        "target_genesymbol": "B",
        "is_directed": True,
        "is_stimulation": True,
        "is_inhibition": False,
        "consensus_direction": True,
        "sources": ["SIGNOR", "TRRUST"],
        "references": "SIGNOR:16331690;TRRUST:14983059",
    }
    row.update(overrides)
    return row


def test_stimulation_becomes_a_positively_signed_activates_edge():
    edges = _interactions_to_edges([_row()], {"A", "B"})
    assert len(edges) == 1
    edge = edges[0]
    assert edge.source_id == "gene:A"
    assert edge.target_id == "gene:B"
    assert edge.edge_type is EdgeType.ACTIVATES
    assert edge.sign == 1
    assert edge.evidence_class == "regulatory"


def test_inhibition_becomes_a_negatively_signed_inhibits_edge():
    edges = _interactions_to_edges(
        [_row(is_stimulation=False, is_inhibition=True)], {"A", "B"},
    )
    assert edges[0].edge_type is EdgeType.INHIBITS
    assert edges[0].sign == -1


def test_directed_but_unsigned_interaction_becomes_an_unsigned_regulates_edge():
    edges = _interactions_to_edges(
        [_row(is_stimulation=False, is_inhibition=False)], {"A", "B"},
    )
    assert edges[0].edge_type is EdgeType.REGULATES
    assert edges[0].sign == 0


def test_undirected_interaction_is_not_an_arrow():
    assert _interactions_to_edges([_row(is_directed=False)], {"A", "B"}) == []


def test_direction_without_consensus_is_not_an_arrow():
    assert _interactions_to_edges(
        [_row(consensus_direction=False)], {"A", "B"},
    ) == []


def test_contradictory_sign_is_recorded_as_unsigned_regulation():
    # Curators disagree on the sign; the direction still holds, so the
    # arrow stands but makes no claim about which way it pushes.
    edges = _interactions_to_edges(
        [_row(is_stimulation=True, is_inhibition=True)], {"A", "B"},
    )
    assert edges[0].edge_type is EdgeType.REGULATES
    assert edges[0].sign == 0


def test_primary_sources_are_preserved_for_corroboration_counting():
    edges = _interactions_to_edges([_row()], {"A", "B"})
    assert edges[0].primary_sources == ["SIGNOR", "TRRUST"]
    assert edges[0].source_db == "OmniPath"


def test_pubmed_ids_are_extracted_from_the_references_field():
    edges = _interactions_to_edges([_row()], {"A", "B"})
    assert sorted(edges[0].pmids) == ["14983059", "16331690"]


def test_interactions_outside_the_requested_gene_set_are_dropped():
    # OmniPath returns a gene's whole neighbourhood; only edges between
    # genes already in the disease graph are relevant.
    assert _interactions_to_edges([_row(target_genesymbol="ZZZ")], {"A", "B"}) == []


def test_self_loops_are_dropped():
    assert _interactions_to_edges([_row(target_genesymbol="A")], {"A"}) == []


def test_missing_gene_symbols_are_dropped_rather_than_guessed():
    assert _interactions_to_edges([_row(source_genesymbol="")], {"A", "B"}) == []


def test_a_malformed_row_that_is_a_list_raises_rather_than_being_absorbed():
    # Pins the failure mode of the ``genes=`` query parameter bug (R7):
    # that parameter returns HTTP 200 with a JSON body that is a list of
    # lists rather than a list of interaction dicts. ``_interactions_to_edges``
    # must not silently swallow a malformed row -- a response shaped
    # nothing like an interaction record should fail loudly, not be
    # coerced into an empty result.
    with pytest.raises(AttributeError):
        _interactions_to_edges([["A", "B", "not", "a", "dict"]], {"A", "B"})


# ── A mixed-case gene keeps its arrows on a live build ──────────────
#
# The dated path was fixed months of review ago; this is the same fault
# on the live one. `graph_builder` upper-cases the gene list it queries
# with, OmniPath returns real HGNC spellings, and the membership test was
# exact -- so C9orf72 and the 330 other C#orf# genes lost their entire
# regulatory layer from every live graph, with no log line.


def test_a_mixed_case_gene_keeps_its_regulatory_edges():
    from neorx.core.sources.omnipath import _interactions_to_edges

    rows = [
        {
            "source_genesymbol": "CCR5",
            "target_genesymbol": "C9orf72",
            "is_directed": True,
            "consensus_direction": True,
            "is_stimulation": False,
            "is_inhibition": True,
            "sources": ["SIGNOR"],
            "references": "",
        }
    ]
    # What graph_builder actually passes: upper-cased symbols.
    edges = _interactions_to_edges(rows, {"CCR5", "C9ORF72"})

    assert len(edges) == 1
    # The endpoints keep the row's real spelling, which is what the
    # graph's own nodes are named, so the edge connects something.
    assert edges[0].source_id == "gene:CCR5"
    assert edges[0].target_id == "gene:C9orf72"


def test_a_gene_outside_the_requested_set_is_still_excluded():
    # Relaxing case must not relax membership.
    from neorx.core.sources.omnipath import _interactions_to_edges

    rows = [
        {
            "source_genesymbol": "CCR5",
            "target_genesymbol": "UNRELATED",
            "is_directed": True,
            "consensus_direction": True,
            "is_stimulation": True,
            "is_inhibition": False,
            "sources": ["SIGNOR"],
            "references": "",
        }
    ]
    assert _interactions_to_edges(rows, {"CCR5", "C9ORF72"}) == []
