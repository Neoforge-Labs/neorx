"""A dated build end to end.

Two invariants, and they are different.

The first is about the *frame*: a gene injected into a dated run's
assembled graph, from a source that bypasses gene-list restriction, never
reaches the candidate list.

The second is about the *data*, and is the one this file exists for. A
dated build must be made of the pinned release. Restricting which genes
may be evaluated while every score attached to them is fetched live
produces a graph that looks correctly dated and is not: for temporal
holdout the current association score encodes the outcome being
predicted. So the tests below stub the live Open Targets and OmniPath
clients to emit sentinel genes and assert those sentinels are absent,
and build the snapshot as real parquet on disk rather than mocking
``SnapshotStore`` -- the previous version of this defect survived a
green suite precisely because the snapshot layer was never executed.
"""

import polars as pl
import pytest

from neorx.core.causal.identifier import identify_causal_targets
from neorx.core.graph.models import (
    DiseaseGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)
from neorx.snapshots.schema import ASSOCIATION_COLUMNS, INTERACTION_COLUMNS

DISEASE = "HIV infection"
DISEASE_ID = "EFO_0000764"
RELEASE = "18.06"

# Genes that exist only in the snapshot, and genes that exist only in the
# live clients. A dated build must contain the first and none of the
# second.
SNAPSHOT_GENES = ("CCR5", "CXCR4", "CONFND", "C9orf72")
LIVE_ONLY_GENE = "TODAYONLY"

# A real HGNC symbol carrying lowercase, and a real ALS/FTD gene. The
# release writes it the way HGNC does; ChEMBL upper-cases every symbol it
# resolves (``chembl.py`` ``gs.upper()``). Node identity is the node id,
# so ``gene:C9orf72`` and ``gene:C9ORF72`` are two nodes for one gene
# unless every comparison between a frame and a node normalises the same
# way. 15,603 of the 2018 OmniPath extract's 34,211 symbols carry
# lowercase, and 872 of the human ones do.
MIXED_CASE_GENE = "C9orf72"

# The gene the release names as a regulator of CCR5. It is in the frame,
# so it is the confounder the backdoor search has to adjust for, and the
# undisturbed verdict on CCR5 is `identifiable_by_adjustment` rather than
# the trivial one.
CONFOUNDER_GENE = "CONFND"

# Genes no unpinned source may put into a dated graph. LIVE_BRIDGE_GENE is
# the dangerous one: the pinned OmniPath extract *does* know it -- OmniPath
# is not disease-scoped -- so admitting the live node hands the snapshot
# reader a symbol that closes a regulatory cycle through CCR5 that the
# release, read on its own terms for this disease, does not contain.
LIVE_BRIDGE_GENE = "LIVEBRIDGE"
LIVE_EXTRA_GENES = ("LIVEX1", "LIVEX2")

# One off-frame gene-like node per unpinned source, so that deleting any
# single source's restriction is visible. Before this file carried them,
# ChEMBL -- the source the design notes single out as the worst offender,
# because its score is 60% of today's ``max_phase`` -- had no off-frame
# gene in any fixture, and STRING, KEGG and Reactome returned nothing at
# all. Deleting their restrictions changed no test.
CHEMBL_ONLY_GENE = "CHEMBLONLY"
STRING_ONLY_PROTEIN = "STRINGONLY"
KEGG_ONLY_GENE = "KEGGONLY"
REACTOME_ONLY_GENE = "REACTONLY"

# A pathogen target whose symbol *is* a frame gene's symbol. ChEMBL
# namespaces the pathogen node's id (``pathogen:<organism>:<SYMBOL>``) but
# leaves its ``name`` the bare symbol, so a frame check that reads the
# name admits it whenever the human gene of the same name is in the
# frame. P. falciparum DHFR and human DHFR are the real instance; CCR5 is
# used here because it is already this fixture's frame gene.
PATHOGEN_ORGANISM = "plasmodium_falciparum"
PATHOGEN_SYMBOL = "CCR5"

DISEASE_NODE_ID = f"disease:{DISEASE.lower().replace(' ', '_')}"


def _graph_with_an_off_frame_gene():
    nodes = [
        GraphNode(node_id="gene:PIK3CA", name="PIK3CA", node_type=NodeType.GENE,
                  source="Open Targets", score=0.9),
        GraphNode(node_id="gene:LATER", name="LATER", node_type=NodeType.GENE,
                  source="Monarch", score=0.9),
        GraphNode(node_id="disease:d", name="d", node_type=NodeType.DISEASE,
                  source="NeoRx", score=1.0),
    ]
    edges = [
        GraphEdge(source_id=n.node_id, target_id="disease:d",
                  edge_type=EdgeType.ASSOCIATED_WITH, weight=0.9,
                  source_db="Open Targets", evidence_class="genetic_association")
        for n in nodes[:2]
    ]
    return DiseaseGraph(disease_name="d", disease_id="disease:d",
                        nodes=nodes, edges=edges)


def test_an_off_frame_gene_never_reaches_the_results():
    results = identify_causal_targets(
        _graph_with_an_off_frame_gene(), top_n=10,
        frame=frozenset({"PIK3CA"}),
    )
    assert all(r.gene_name != "LATER" for r in results)


def test_without_a_frame_the_same_gene_is_evaluated():
    results = identify_causal_targets(_graph_with_an_off_frame_gene(), top_n=10)
    assert any(r.gene_name == "LATER" for r in results)


def test_a_dated_build_refuses_when_the_snapshot_is_missing(tmp_path):
    from neorx.core.graph.graph_builder import build_disease_graph
    from neorx.snapshots.reader import SnapshotStore
    from neorx.snapshots.resolver import SourceResolver, UnpinnedSourceError

    resolver = SourceResolver(
        store=SnapshotStore(tmp_path),
        release_for=lambda source, as_of: RELEASE,
        live={},
    )
    with pytest.raises(UnpinnedSourceError):
        build_disease_graph(
            DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
        )


# ── A real snapshot on disk ─────────────────────────────────────────


def _write_snapshot(root):
    """Write real parquet extracts, in the canonical schema.

    Not a mocked ``SnapshotStore``: the point of these tests is that the
    reading actually happens, and a mock would assert only that this test
    file agrees with itself.
    """
    associations = root / "opentargets" / RELEASE
    associations.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "target_id": ["ENSG0000001", "ENSG0000001", "ENSG0000002",
                          "ENSG0000003", "ENSG0000001", "ENSG0000004",
                          "ENSG0000005"],
            # The release writes C9orf72 the way HGNC does. Nothing in the
            # pipeline may quietly re-case it: the symbol *is* half the
            # node's identity.
            "target_symbol": ["CCR5", "CCR5", "CXCR4", "LITONLY", "CCR5",
                              CONFOUNDER_GENE, MIXED_CASE_GENE],
            # The fifth row is a DIFFERENT disease. It must not reach the
            # graph -- the build filters to DISEASE_ID -- but it must
            # reach CCR5's n_associated_diseases, which is a statement
            # about the release rather than about this disease.
            "disease_id": [DISEASE_ID] * 4 + ["EFO_0000000", DISEASE_ID,
                                              DISEASE_ID],
            # CCR5 carries both a genetic and a non-genetic datatype, so
            # the breakdown has something to preserve and the node score
            # has something to pick out of it.
            "datatype": ["genetic_association", "known_drug",
                         "somatic_mutation", "literature",
                         "genetic_association", "genetic_association",
                         "genetic_association"],
            "score": [0.75, 0.90, 0.40, 0.99, 0.60, 0.55, 0.30],
        },
        schema=ASSOCIATION_COLUMNS,
    ).write_parquet(associations / "associations.parquet")

    interactions = root / "omnipath" / RELEASE
    interactions.mkdir(parents=True, exist_ok=True)
    # OmniPath is not disease-scoped: the extract carries regulatory
    # interactions for every gene in the release, LIVE_BRIDGE_GENE
    # included. Which of them reach a dated graph is decided entirely by
    # the symbol list the builder hands the reader, which is why that list
    # is the thing under test.
    pl.DataFrame(
        {
            # The last row is the mixed-case gene's only arrow, and it is
            # in the *source* column -- both columns are filtered, so both
            # have to survive the comparison. The extract spells it the
            # way the archive does; the 2018 dump holds `C9orf72` and does
            # not hold `C9ORF72`.
            "source_symbol": ["CCR5", CONFOUNDER_GENE, "CXCR4",
                              LIVE_BRIDGE_GENE, MIXED_CASE_GENE],
            "target_symbol": ["CXCR4", "CCR5", LIVE_BRIDGE_GENE, "CCR5",
                              "CXCR4"],
            "is_directed": [True] * 5,
            "consensus_direction": [True] * 5,
            "is_stimulation": [False, True, True, True, False],
            "is_inhibition": [True, False, False, False, True],
            "primary_sources": ["SIGNOR;TRRUST", "SIGNOR", "SIGNOR",
                                "SIGNOR", "SIGNOR"],
            "references": ["SIGNOR:11111;TRRUST:22222", "SIGNOR:33333",
                           "SIGNOR:44444", "SIGNOR:55555", "SIGNOR:66666"],
        },
        schema=INTERACTION_COLUMNS,
    ).write_parquet(interactions / "interactions.parquet")


def _make_resolver(root):
    """A resolver over a real snapshot store."""
    from neorx.snapshots.reader import SnapshotStore
    from neorx.snapshots.resolver import SourceResolver

    _write_snapshot(root)
    return SourceResolver(
        store=SnapshotStore(root),
        release_for=lambda source, as_of: RELEASE,
        live={},
    )


def _live_monarch_pair():
    """What live Monarch returns when it is allowed to contribute.

    One frame gene at Monarch's flat causal score of 0.85 -- above CCR5's
    pinned 0.75, so an unprotected merge takes it -- and three genes the
    release does not associate with this disease at all. The edges carry
    no ``evidence_class``, exactly as ``neorx.core.sources.monarch`` emits
    them: the *edges* were never the leak, the nodes were.
    """
    nodes, edges = [], []
    for gene in ("CCR5", LIVE_BRIDGE_GENE, *LIVE_EXTRA_GENES):
        node_id = f"gene:{gene}"
        nodes.append(GraphNode(
            node_id=node_id, name=gene, node_type=NodeType.GENE,
            source="Monarch", score=0.85,
            metadata={
                "mondo_id": "MONDO:0005109",
                "hgnc_id": f"HGNC:{gene}",
                "association_type": "causal",
                "provenance": "Monarch",
            },
        ))
        edges.append(GraphEdge(
            source_id=node_id, target_id=DISEASE_NODE_ID,
            edge_type=EdgeType.ASSOCIATED_WITH, weight=0.85,
            source_db="Monarch", evidence="Causal association",
        ))
    return nodes, edges


def _live_chembl_pair():
    """What live ChEMBL returns when it is allowed to contribute.

    CXCR4 is the frame gene here, and it is the one that shows the
    provenance leak: the release gives it a somatic-mutation score of 0.40
    and no known drug, while ChEMBL -- whose score is 60% of *today's*
    ``max_phase`` -- reports 0.88 and ``has_known_drug`` True. Both the
    score and the flag are statements about today's clinic, which is the
    outcome a dated build exists to predict.
    """
    def _target(node_id, name, node_type, score, **extra):
        return GraphNode(
            node_id=node_id, name=name, node_type=node_type,
            source="ChEMBL", score=score,
            metadata={
                "chembl_target_id": "CHEMBL2107",
                "chembl_drug_evidence_score": score,
                "clinical_phase": 4,
                "has_known_drug": True,
                "is_druggable": True,
                "is_pathogen_target": False,
                **extra,
            },
        )

    nodes = [
        _target("gene:CXCR4", "CXCR4", NodeType.GENE, 0.88),
        # The mixed-case frame gene, upper-cased the way chembl.py
        # upper-cases every symbol it resolves. Its *symbol* matches the
        # frame; its *node id* does not, and node identity is the node id.
        _target(f"gene:{MIXED_CASE_GENE.upper()}", MIXED_CASE_GENE.upper(),
                NodeType.GENE, 0.80),
        # A gene the release never associated with this disease. ChEMBL
        # queries by disease rather than from a gene list, so it can and
        # does introduce genes no gene-list restriction ever saw.
        _target(f"gene:{CHEMBL_ONLY_GENE}", CHEMBL_ONLY_GENE,
                NodeType.GENE, 0.92),
        # A pathogen target whose bare symbol collides with a frame gene.
        GraphNode(
            node_id=f"pathogen:{PATHOGEN_ORGANISM}:{PATHOGEN_SYMBOL}",
            name=PATHOGEN_SYMBOL, node_type=NodeType.PATHOGEN_GENE,
            source="ChEMBL", score=0.80,
            metadata={
                "chembl_target_id": "CHEMBL1234",
                "clinical_phase": 4,
                "has_known_drug": True,
                "is_pathogen_target": True,
                "pathogen_organism": "Plasmodium falciparum",
            },
        ),
    ]
    edges = [
        GraphEdge(
            source_id=node.node_id, target_id=DISEASE_NODE_ID,
            edge_type=EdgeType.ASSOCIATED_WITH, weight=node.score,
            source_db="ChEMBL", evidence="ChEMBL max_phase 4",
        )
        for node in nodes
    ]
    return nodes, edges


def _live_string_pair():
    """What live STRING returns when it is allowed to contribute.

    STRING nodes are ``NodeType.PROTEIN``, and STRING names them with its
    own ``preferredName``, not with the symbol it was queried on -- so a
    protein the release never associated with the disease reaches the
    builder even though the gene list handed to STRING was the frame.
    """
    node = GraphNode(
        node_id=f"gene:{STRING_ONLY_PROTEIN}", name=STRING_ONLY_PROTEIN,
        node_type=NodeType.PROTEIN, source="STRING", score=0.91,
    )
    edge = GraphEdge(
        source_id=node.node_id, target_id="gene:CCR5",
        edge_type=EdgeType.INTERACTS_WITH, weight=0.91,
        source_db="STRING", evidence="STRING combined score: 0.910",
    )
    return [node], [edge]


def _live_pathway_pair(source_db, gene):
    """A pathway source's contribution: a pathway, and an off-frame gene.

    The restriction is a contract about what an *unpinned source* may add,
    checked per source, not a statement about which node types today's
    KEGG and Reactome clients happen to construct. A client that starts
    returning gene nodes -- or a mock path, or a schema change upstream --
    must not be able to widen a dated build's candidate population, and
    the only way a test can say so is to hand the gate a gene node.
    """
    nodes = [
        GraphNode(
            node_id=f"pathway:{source_db.lower()}:1", name=f"{source_db} pathway",
            node_type=NodeType.PATHWAY, source=source_db, score=0.8,
        ),
        GraphNode(
            node_id=f"gene:{gene}", name=gene, node_type=NodeType.GENE,
            source=source_db, score=0.8,
        ),
    ]
    edges = [
        GraphEdge(
            source_id=f"gene:{gene}", target_id=f"pathway:{source_db.lower()}:1",
            edge_type=EdgeType.PARTICIPATES_IN, weight=0.8, source_db=source_db,
        ),
        GraphEdge(
            source_id="gene:CCR5", target_id=f"pathway:{source_db.lower()}:1",
            edge_type=EdgeType.PARTICIPATES_IN, weight=0.8, source_db=source_db,
        ),
    ]
    return nodes, edges


def _stub_sources(monkeypatch, *, live_nodes=False):
    """Replace every network-backed source call with an instant stub.

    Open Targets and OmniPath emit a sentinel gene that exists nowhere in
    the snapshot, so a dated build that reached them is detectable in the
    graph itself and not only in a call counter. Returns the counters.

    ``live_nodes`` switches on the configuration this file previously
    could not express: Monarch and ChEMBL *contributing nodes*, which is
    what they do on every real dated run. With it off they return
    ``[], []`` and the leak the pinning rule closes cannot occur, which is
    the condition the earlier version of this file established before
    checking that the leak had not occurred.
    """
    import neorx.core.graph.graph_builder as graph_builder
    import neorx.core.sources.open_targets as open_targets_module

    calls = {"n": 0, "open_targets": 0, "omnipath": 0,
             "omnipath_gene_lists": []}

    def _empty_pair(*_args, **_kwargs):
        calls["n"] += 1
        return [], []

    def _empty_dict(*_args, **_kwargs):
        calls["n"] += 1
        return {}

    def _live_open_targets(*_args, **_kwargs):
        calls["n"] += 1
        calls["open_targets"] += 1
        node = GraphNode(
            node_id=f"gene:{LIVE_ONLY_GENE}", name=LIVE_ONLY_GENE,
            node_type=NodeType.GENE, source="Open Targets", score=1.0,
        )
        edge = GraphEdge(
            source_id=node.node_id,
            target_id=f"disease:{DISEASE.lower().replace(' ', '_')}",
            edge_type=EdgeType.ASSOCIATED_WITH, weight=1.0,
            source_db="Open Targets", evidence_class="genetic_association",
        )
        return [node], [edge]

    def _live_omnipath(*_args, **_kwargs):
        calls["n"] += 1
        calls["omnipath"] += 1
        return [], [GraphEdge(
            source_id=f"gene:{LIVE_ONLY_GENE}", target_id="gene:CCR5",
            edge_type=EdgeType.ACTIVATES, weight=0.9, source_db="OmniPath",
            evidence_class="regulatory", sign=1,
        )]

    def _live_monarch(*_args, **_kwargs):
        calls["n"] += 1
        return _live_monarch_pair()

    def _live_chembl(*_args, **_kwargs):
        calls["n"] += 1
        return _live_chembl_pair()

    monkeypatch.setattr(
        graph_builder, "query_monarch",
        _live_monarch if live_nodes else _empty_pair,
    )
    monkeypatch.setattr(graph_builder, "query_open_targets", _live_open_targets)
    monkeypatch.setattr(
        graph_builder, "query_chembl",
        _live_chembl if live_nodes else _empty_pair,
    )
    def _live_kegg(*_args, **_kwargs):
        calls["n"] += 1
        return _live_pathway_pair("KEGG", KEGG_ONLY_GENE)

    def _live_reactome(*_args, **_kwargs):
        calls["n"] += 1
        return _live_pathway_pair("Reactome", REACTOME_ONLY_GENE)

    def _live_string(*_args, **_kwargs):
        calls["n"] += 1
        return _live_string_pair()

    monkeypatch.setattr(
        graph_builder, "query_kegg_pathways",
        _live_kegg if live_nodes else _empty_pair,
    )
    monkeypatch.setattr(
        graph_builder, "query_reactome_pathways",
        _live_reactome if live_nodes else _empty_pair,
    )
    monkeypatch.setattr(
        graph_builder, "query_string_interactions",
        _live_string if live_nodes else _empty_pair,
    )
    monkeypatch.setattr(graph_builder, "query_omnipath", _live_omnipath)
    monkeypatch.setattr(graph_builder, "query_uniprot", _empty_dict)
    monkeypatch.setattr(graph_builder, "query_pdb_structures", _empty_dict)
    monkeypatch.setattr(
        open_targets_module, "resolve_disease_id", lambda *_a, **_kw: None
    )
    return calls


def _fresh_cache(tmp_path, monkeypatch):
    import neorx.core.graph.graph_builder as graph_builder
    from neorx.core.cache import FileCache

    cache = FileCache(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(graph_builder, "get_cache", lambda: cache)
    return cache


class _RecordingResolver:
    """A resolver that records the symbol list the OmniPath reader is given.

    Wrapping the resolved reader rather than mocking it keeps the real
    snapshot read in the path -- what is recorded is exactly what the
    parquet was queried with.
    """

    def __init__(self, inner, gene_lists):
        self._inner = inner
        self._gene_lists = gene_lists

    def resolve(self, source, as_of):
        reader = self._inner.resolve(source, as_of)
        if source != "omnipath":
            return reader

        def recording(gene_symbols):
            self._gene_lists.append(list(gene_symbols))
            return reader(gene_symbols)

        return recording


def _dated_build(tmp_path, monkeypatch, *, live_nodes=False, max_genes=20):
    from neorx.core.graph.graph_builder import build_disease_graph

    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = _stub_sources(monkeypatch, live_nodes=live_nodes)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _RecordingResolver(
        _make_resolver(tmp_path / "snapshots"), calls["omnipath_gene_lists"],
    )
    graph = build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
        max_genes=max_genes,
    )
    return graph, calls


# ── The data a dated build is made of ───────────────────────────────


def test_a_dated_build_calls_neither_live_opentargets_nor_live_omnipath(
    tmp_path, monkeypatch
):
    _graph, calls = _dated_build(tmp_path, monkeypatch)
    assert calls["open_targets"] == 0
    assert calls["omnipath"] == 0
    # The unpinned sources are still live -- that is the design, not an
    # oversight -- so the build did do work.
    assert calls["n"] > 0


def test_a_gene_present_only_in_the_snapshot_reaches_the_graph(
    tmp_path, monkeypatch
):
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    names = {n.name for n in graph.nodes}
    for gene in SNAPSHOT_GENES:
        assert gene in names


def test_a_gene_present_only_in_the_live_client_does_not(tmp_path, monkeypatch):
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    assert LIVE_ONLY_GENE not in {n.name for n in graph.nodes}
    assert all(
        LIVE_ONLY_GENE not in edge.source_id and LIVE_ONLY_GENE not in edge.target_id
        for edge in graph.edges
    )


def test_the_node_score_is_the_releases_genetic_score_not_an_overall_score(
    tmp_path, monkeypatch
):
    # CCR5's rows are genetic_association 0.75 and known_drug 0.90. The
    # score must be 0.75: the genetic evidence, which is a value in the
    # extract -- not 0.90, not a sum, not a harmonic mean of the two.
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    ccr5 = next(n for n in graph.nodes if n.name == "CCR5")
    assert ccr5.score == pytest.approx(0.75)
    assert ccr5.metadata["datatype_scores"] == {
        "genetic_association": 0.75, "known_drug": 0.90,
    }
    assert ccr5.metadata["has_known_drug"] is True


def test_n_associated_diseases_counts_the_release_not_this_disease(
    tmp_path, monkeypatch
):
    # CCR5 is associated with two diseases in the extract, CXCR4 with one.
    # The build is for one of them, so a count taken from the filtered
    # rows would report 1 for both and the specificity penalty would stop
    # discriminating. It is counted release-wide, which is what
    # specificity means.
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    counts = {
        n.name: n.metadata["n_associated_diseases"]
        for n in graph.nodes
        if n.node_type == NodeType.GENE
    }
    assert counts["CCR5"] == 2
    assert counts["CXCR4"] == 1


def test_the_other_diseases_rows_do_not_reach_the_graph(tmp_path, monkeypatch):
    # The same row that lifts CCR5's count to 2 carries a 0.60 genetic
    # score for another disease. The node score must stay 0.75 -- the
    # count reads the release, the evidence reads this disease.
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    ccr5 = next(n for n in graph.nodes if n.name == "CCR5")
    assert ccr5.score == pytest.approx(0.75)
    assert ccr5.metadata["datatype_scores"] == {
        "genetic_association": 0.75, "known_drug": 0.90,
    }


def test_a_target_without_genetic_evidence_is_not_given_an_invented_score(
    tmp_path, monkeypatch
):
    # LITONLY has a literature score of 0.99 and nothing genetic. It gets
    # no node rather than a stand-in score, and so is not a candidate.
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    assert "LITONLY" not in {n.name for n in graph.nodes}


def test_the_dated_omnipath_edges_come_from_the_snapshot(tmp_path, monkeypatch):
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    regulatory = [e for e in graph.edges if e.source_db == "OmniPath"]
    # Exactly the interactions among the release's own genes for this
    # disease. The extract also holds CXCR4 -> LIVEBRIDGE and
    # LIVEBRIDGE -> CCR5; neither may appear, because LIVEBRIDGE is not
    # one of those genes.
    assert sorted((e.source_id, e.target_id) for e in regulatory) == [
        (f"gene:{MIXED_CASE_GENE}", "gene:CXCR4"),
        ("gene:CCR5", "gene:CXCR4"),
        (f"gene:{CONFOUNDER_GENE}", "gene:CCR5"),
    ]
    inhibition = next(
        e for e in regulatory
        if e.target_id == "gene:CXCR4" and e.source_id == "gene:CCR5"
    )
    assert inhibition.edge_type == EdgeType.INHIBITS
    assert inhibition.sign == -1
    assert inhibition.primary_sources == ["SIGNOR", "TRRUST"]
    assert inhibition.pmids == ["11111", "22222"]


def test_the_graph_records_the_date_it_was_built_as_of(tmp_path, monkeypatch):
    graph, _calls = _dated_build(tmp_path, monkeypatch)
    assert graph.as_of == "2018-06"
    assert graph.disease_id == DISEASE_ID


def test_an_undated_build_records_no_as_of(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    graph = build_disease_graph(DISEASE)
    assert graph.as_of is None
    # And the live client's gene is present, because nothing was pinned.
    assert LIVE_ONLY_GENE in {n.name for n in graph.nodes}


def test_a_dated_build_refuses_without_a_disease_id(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")
    with pytest.raises(ValueError, match="disease_id"):
        build_disease_graph(DISEASE, as_of="2018-06", resolver=resolver)


# ── Cache key must include `as_of` ──────────────────────────────────
#
# The resolver check runs before the cache lookup and correctly fails
# fast when a snapshot is missing. But when the snapshot exists,
# resolution passes -- and before dd52a7b, the cache key was built from
# disease/max_genes/string_min_score alone, so a dated call could hit a
# cache entry written by a live (undated) call of the same disease and
# silently return the live graph. That is the exact anachronism this
# sub-project exists to prevent, so it must be exercised directly rather
# than trusted to the resolver check alone.


def test_a_dated_build_does_not_return_a_cached_live_graph(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")

    live = build_disease_graph(DISEASE)
    after_live = calls["n"]
    assert after_live > 0
    assert LIVE_ONLY_GENE in {n.name for n in live.nodes}

    dated = build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    # A cache hit on the live entry would leave the call count unchanged
    # and hand back the live graph.
    assert calls["n"] > after_live
    assert dated.as_of == "2018-06"
    assert LIVE_ONLY_GENE not in {n.name for n in dated.nodes}


def test_two_dated_builds_with_different_as_of_do_not_share_a_cache_entry(
    tmp_path, monkeypatch
):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")

    build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    after_first = calls["n"]
    assert after_first > 0

    build_disease_graph(
        DISEASE, as_of="2020-01", resolver=resolver, disease_id=DISEASE_ID,
    )
    assert calls["n"] > after_first


def test_an_undated_build_still_hits_its_own_cache(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)

    build_disease_graph(DISEASE)
    after_first = calls["n"]
    assert after_first > 0

    build_disease_graph(DISEASE)
    # Same disease, same params, no as_of -- must be a cache hit.
    assert calls["n"] == after_first


def test_a_dated_build_hits_its_own_cache(tmp_path, monkeypatch):
    from neorx.core.graph.graph_builder import build_disease_graph

    calls = _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")

    first = build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    after_first = calls["n"]

    second = build_disease_graph(
        DISEASE, as_of="2018-06", resolver=resolver, disease_id=DISEASE_ID,
    )
    assert calls["n"] == after_first
    # A cached dated graph must still say it is dated.
    assert second.as_of == first.as_of == "2018-06"


# ── With the unpinned sources actually contributing nodes ───────────
#
# Everything above this line runs with Monarch and ChEMBL stubbed to
# ``[], []``. That is the configuration in which the pinning rule cannot
# be violated, so it cannot detect a violation either. Below, they
# contribute what they contribute on a real dated run: frame genes at
# scores drawn from today, and genes the release never associated with
# this disease.


def _identification(graph, gene):
    """The identification verdict on ``gene`` in an assembled graph."""
    from neorx.core.causal.backdoor import find_adjustment_set
    from neorx.core.graph.graph_builder import disease_graph_to_networkx

    return find_adjustment_set(
        disease_graph_to_networkx(graph), f"gene:{gene}", DISEASE_NODE_ID,
    )


def test_live_nodes_do_not_change_an_identifiability_verdict(
    tmp_path, monkeypatch
):
    """The verdict is the predictor the research programme tests.

    Two builds of the same disease, at the same date, against byte-identical
    snapshots. The only difference is whether the unpinned sources are
    allowed to contribute nodes. If that changes the verdict, then what a
    dated run reports about 2018 depends on what Monarch says today, and
    the temporal holdout is not one.
    """
    pinned_only, _ = _dated_build(tmp_path / "pinned-only", monkeypatch)
    with_live, _ = _dated_build(
        tmp_path / "with-live", monkeypatch, live_nodes=True,
    )

    quiet = _identification(pinned_only, "CCR5")
    loud = _identification(with_live, "CCR5")

    # Stated absolutely as well as relatively: an equality assertion alone
    # would pass if both builds broke in the same way.
    assert quiet.identifiable is True
    assert quiet.reason.value == "identifiable_by_adjustment"
    assert quiet.adjustment_set == (f"gene:{CONFOUNDER_GENE}",)

    assert loud.identifiable == quiet.identifiable
    assert loud.reason == quiet.reason
    assert loud.adjustment_set == quiet.adjustment_set


def test_a_live_gene_outside_the_frame_does_not_become_a_node(
    tmp_path, monkeypatch
):
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    names = {n.name for n in graph.nodes}
    for gene in (LIVE_BRIDGE_GENE, *LIVE_EXTRA_GENES):
        assert gene not in names
    # And no edge is left pointing at a node that is not there.
    node_ids = {n.node_id for n in graph.nodes}
    for edge in graph.edges:
        assert edge.source_id in node_ids
        assert edge.target_id in node_ids


def test_the_frame_genes_still_arrive_when_live_sources_are_active(
    tmp_path, monkeypatch
):
    """The rule restricts the population; it does not empty it."""
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    names = {n.name for n in graph.nodes}
    for gene in SNAPSHOT_GENES:
        assert gene in names


def test_a_live_source_cannot_raise_a_pinned_nodes_score(tmp_path, monkeypatch):
    # Monarch reports CCR5 at its flat 0.85, above the release's 0.75.
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    ccr5 = next(n for n in graph.nodes if n.name == "CCR5")
    assert ccr5.score == pytest.approx(0.75)
    assert ccr5.metadata["score_is"] == (
        "max genetic-evidence score in this release"
    )
    assert ccr5.metadata["snapshot_release"] == RELEASE
    assert ccr5.metadata["datatype_scores"] == {
        "genetic_association": 0.75, "known_drug": 0.90,
    }


def test_a_live_source_cannot_overwrite_pinned_provenance(tmp_path, monkeypatch):
    # ChEMBL reports CXCR4 at 0.88 with has_known_drug True, both derived
    # from today's max_phase. The release says 0.40, somatic mutation
    # only, no known drug.
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    cxcr4 = next(n for n in graph.nodes if n.name == "CXCR4")
    assert cxcr4.score == pytest.approx(0.40)
    assert cxcr4.metadata["has_known_drug"] is False
    assert cxcr4.metadata["datatype_scores"] == {"somatic_mutation": 0.40}
    assert cxcr4.metadata["score_is"] == (
        "max genetic-evidence score in this release"
    )


def test_the_omnipath_gene_list_holds_no_live_only_gene(tmp_path, monkeypatch):
    _graph, calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    assert calls["omnipath_gene_lists"], "the OmniPath reader was never called"
    for gene_list in calls["omnipath_gene_lists"]:
        assert sorted(gene_list) == sorted(SNAPSHOT_GENES)


def test_live_genes_cannot_displace_frame_genes_from_the_cap(
    tmp_path, monkeypatch
):
    """With four live genes and a cap of four, the frame must still win.

    Monarch scores its genes 0.85 and ChEMBL scores CXCR4 0.88, both above
    every genetic score in the release, so under a score-ranked cap the
    live genes take the whole budget and the regulatory layer the release
    actually contains -- the arrow the backdoor search adjusts on -- is
    never fetched.
    """
    graph, _calls = _dated_build(
        tmp_path, monkeypatch, live_nodes=True, max_genes=4,
    )
    regulatory = sorted(
        (e.source_id, e.target_id)
        for e in graph.edges if e.source_db == "OmniPath"
    )
    assert regulatory == [
        (f"gene:{MIXED_CASE_GENE}", "gene:CXCR4"),
        ("gene:CCR5", "gene:CXCR4"),
        (f"gene:{CONFOUNDER_GENE}", "gene:CCR5"),
    ]


def test_the_dropped_nodes_are_counted_and_named(tmp_path, monkeypatch, caplog):
    """Filtering silently is the failure mode; the count must be logged."""
    import logging

    with caplog.at_level(logging.INFO, logger="neorx.core.graph.graph_builder"):
        _dated_build(tmp_path, monkeypatch, live_nodes=True)

    dropped = [
        r.getMessage() for r in caplog.records
        if "off-frame" in r.getMessage()
    ]
    assert dropped, "the build dropped nodes without saying so"
    message = " ".join(dropped)
    assert "Monarch" in message
    assert "3" in message


# ── One identity rule, checked at every source ──────────────────────
#
# Everything in this block is the same defect seen from a different
# side: the frame gate was enforced on a bare symbol string while node
# identity is a node id, and the two normalise differently.


def test_an_off_frame_chembl_gene_does_not_become_a_node(tmp_path, monkeypatch):
    """ChEMBL is the source with the worst leak and had no fixture for it.

    Its score is 60% of today's ``max_phase`` -- the clinical outcome a
    dated build exists to predict -- and it queries by disease, so no
    gene-list restriction constrains which genes it returns.
    """
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    assert CHEMBL_ONLY_GENE not in {n.name for n in graph.nodes}
    assert f"gene:{CHEMBL_ONLY_GENE}" not in {n.node_id for n in graph.nodes}


def test_an_off_frame_string_protein_does_not_become_a_node(
    tmp_path, monkeypatch
):
    """A protein node is a candidate too, not enrichment hanging off one.

    ``_get_candidate_nodes`` admits ``protein`` alongside ``gene``, so a
    node type narrower than that in the restriction is a hole rather than
    a simplification.
    """
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    assert STRING_ONLY_PROTEIN not in {n.name for n in graph.nodes}
    node_ids = {n.node_id for n in graph.nodes}
    for edge in graph.edges:
        assert edge.source_id in node_ids
        assert edge.target_id in node_ids


def test_an_off_frame_pathway_source_gene_does_not_become_a_node(
    tmp_path, monkeypatch
):
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    names = {n.name for n in graph.nodes}
    assert KEGG_ONLY_GENE not in names
    assert REACTOME_ONLY_GENE not in names
    # The pathways themselves are enrichment and stay: the rule restricts
    # the candidate population, it does not switch the sources off.
    assert any(n.node_type == NodeType.PATHWAY for n in graph.nodes)


def test_a_mixed_case_frame_gene_is_one_node_at_the_pinned_score(
    tmp_path, monkeypatch
):
    """The gene is in the frame, so ChEMBL may enrich it -- not duplicate it.

    ChEMBL reports ``gene:C9ORF72`` at 0.80 with ``clinical_phase`` 4.
    The frame gate reads the *symbol*, which matches; ``_merge_nodes``
    merges on the *node id*, which does not. The result is two nodes for
    one gene, the second carrying today's clinical phase as its score in a
    2018 graph.
    """
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    for_gene = [
        n for n in graph.nodes
        if n.node_id.lower() == f"gene:{MIXED_CASE_GENE}".lower()
    ]
    assert len(for_gene) == 1, [n.node_id for n in for_gene]
    node = for_gene[0]
    # The release's spelling and the release's score survive.
    assert node.node_id == f"gene:{MIXED_CASE_GENE}"
    assert node.name == MIXED_CASE_GENE
    assert node.score == pytest.approx(0.30)
    assert node.metadata["snapshot_release"] == RELEASE
    # And ChEMBL's contribution arrived as enrichment on that one node,
    # rather than being dropped: a comparison that refuses the match is
    # not a fix, it is symptom 3.
    assert node.metadata["chembl_target_id"] == "CHEMBL2107"


def test_a_mixed_case_frame_gene_keeps_its_pinned_arrows(tmp_path, monkeypatch):
    """The regulatory layer identifiability is computed from.

    ``_extract_gene_symbols`` upper-cases every symbol and the pinned
    OmniPath filter is an exact ``is_in``, so a symbol the release spells
    with lowercase is asked for in a form the extract does not contain and
    loses every arrow it has -- silently. The 2018 dump holds 321 C#orf#
    symbols and 872 mixed-case human symbols in total.
    """
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    regulatory = [
        (e.source_id, e.target_id)
        for e in graph.edges if e.source_db == "OmniPath"
    ]
    assert (f"gene:{MIXED_CASE_GENE}", "gene:CXCR4") in regulatory


def test_a_pathogen_target_is_excluded_even_when_its_symbol_collides(
    tmp_path, monkeypatch
):
    """"A dated build therefore has no pathogen targets" -- module docstring.

    ChEMBL namespaces the pathogen node's id but leaves ``name`` the bare
    symbol, and the frame check reads ``name``. So a pathogen target whose
    symbol happens to be a frame gene's symbol is admitted, and is never
    counted as dropped. Excluding by node type is what actually holds.
    """
    graph, _calls = _dated_build(tmp_path, monkeypatch, live_nodes=True)
    assert all(
        n.node_type != NodeType.PATHOGEN_GENE for n in graph.nodes
    ), [n.node_id for n in graph.nodes if n.node_type == NodeType.PATHOGEN_GENE]
    assert f"pathogen:{PATHOGEN_ORGANISM}:{PATHOGEN_SYMBOL}" not in {
        n.node_id for n in graph.nodes
    }
    # And the human gene of the same name kept the release's score rather
    # than the pathogen target's 0.80.
    ccr5 = next(n for n in graph.nodes if n.node_id == "gene:CCR5")
    assert ccr5.score == pytest.approx(0.75)


def test_the_pathogen_exclusion_is_counted_like_any_other(
    tmp_path, monkeypatch, caplog
):
    import logging

    with caplog.at_level(logging.INFO, logger="neorx.core.graph.graph_builder"):
        _dated_build(tmp_path, monkeypatch, live_nodes=True)

    # Not merely "pathogen appears somewhere in the log": the per-source
    # line already says how many pathogen nodes ChEMBL returned. What must
    # be visible is that they were *excluded*, with a count.
    excluded = [
        r.getMessage() for r in caplog.records
        if "excluded" in r.getMessage() and "pathogen" in r.getMessage()
    ]
    assert excluded, "pathogen nodes were dropped without saying so"
    message = " ".join(excluded)
    assert "ChEMBL" in message
    assert f"pathogen:{PATHOGEN_ORGANISM}:{PATHOGEN_SYMBOL}" in message


def test_a_dated_build_refuses_allow_mocks(tmp_path, monkeypatch):
    """Curated mock data must not be able to become a 2018 node score."""
    from neorx.core.graph.graph_builder import build_disease_graph

    _stub_sources(monkeypatch)
    _fresh_cache(tmp_path, monkeypatch)
    resolver = _make_resolver(tmp_path / "snapshots")
    with pytest.raises(ValueError, match="allow_mocks"):
        build_disease_graph(
            DISEASE, as_of="2018-06", resolver=resolver,
            disease_id=DISEASE_ID, allow_mocks=True,
        )


def test_the_persisted_row_is_keyed_on_the_date_and_the_disease(
    tmp_path, monkeypatch
):
    """A dated and a live build of one disease are different rows.

    ``save_graph_to_db`` upserts on ``(disease_name, parameters)``, so a
    parameters blob that does not name the date lets a live build
    overwrite a dated one under the same key.
    """
    import neorx.core.graph.persistence as persistence

    captured = []
    monkeypatch.setattr(
        persistence, "save_graph_to_db",
        lambda graph, params=None: captured.append(dict(params or {})),
    )
    _dated_build(tmp_path, monkeypatch, live_nodes=True)

    assert captured, "the graph was never handed to persistence"
    assert captured[-1]["as_of"] == "2018-06"
    assert captured[-1]["disease_id"] == DISEASE_ID
    assert captured[-1]["max_genes"] == 20


# ── The same identity rule, at the identifier's own frame ───────────


def test_the_identifier_frame_matches_a_mixed_case_symbol():
    """``candidate_frame`` is a different frame, compared in the same form.

    It is drawn from the extract's ``target_symbol`` column while the node
    carries whatever spelling the source that built it used, and the
    comparison here was case-*sensitive* while the builder's was
    case-blind. A gene could therefore pass one gate and fail the other.
    """
    nodes = [
        GraphNode(node_id=f"gene:{MIXED_CASE_GENE.upper()}",
                  name=MIXED_CASE_GENE.upper(), node_type=NodeType.GENE,
                  source="ChEMBL", score=0.9),
        GraphNode(node_id="disease:d", name="d", node_type=NodeType.DISEASE,
                  source="NeoRx", score=1.0),
    ]
    edges = [
        GraphEdge(source_id=nodes[0].node_id, target_id="disease:d",
                  edge_type=EdgeType.ASSOCIATED_WITH, weight=0.9,
                  source_db="Open Targets",
                  evidence_class="genetic_association"),
    ]
    graph = DiseaseGraph(disease_name="d", disease_id="disease:d",
                         nodes=nodes, edges=edges)
    results = identify_causal_targets(
        graph, top_n=10, frame=frozenset({MIXED_CASE_GENE}),
    )
    assert [r.gene_name for r in results] == [MIXED_CASE_GENE.upper()]


def test_the_identifier_frame_still_excludes_a_gene_that_is_not_in_it():
    """Case-insensitive is not the same as permissive."""
    results = identify_causal_targets(
        _graph_with_an_off_frame_gene(), top_n=10,
        frame=frozenset({"pik3ca"}),
    )
    assert [r.gene_name for r in results] == ["PIK3CA"]


# ── The two claims the previous round made and did not test ─────────


def test_merging_does_not_reach_back_into_the_source_nodes_metadata():
    """``_merge_nodes`` copies deeply, and something has to say why.

    With a shallow copy the merged node shares its ``metadata`` dict with
    the node it was copied from, so "a pinned node keeps its metadata" is
    true only by accident: the live source's keys land in the pinned
    node's own dict as well, and any later reader of that node -- the
    source list it came from is still live in the builder -- sees them.
    """
    from neorx.core.graph.graph_builder import _merge_nodes

    pinned = GraphNode(
        node_id="gene:CCR5", name="CCR5", node_type=NodeType.GENE,
        source="Open Targets", score=0.75,
        metadata={"snapshot_release": RELEASE, "datatype_scores": {"a": 0.75}},
    )
    live = GraphNode(
        node_id="gene:CCR5", name="CCR5", node_type=NodeType.GENE,
        source="ChEMBL", score=0.88,
        metadata={"clinical_phase": 4},
    )

    merged, _edges = _merge_nodes([pinned, live], [])

    assert "clinical_phase" not in pinned.metadata
    assert pinned.metadata["datatype_scores"] == {"a": 0.75}
    assert merged[0].metadata is not pinned.metadata
    assert merged[0].metadata["datatype_scores"] is not pinned.metadata[
        "datatype_scores"
    ]


def test_uniprot_enrichment_cannot_overwrite_what_a_release_said():
    """This runs before ``_merge_nodes``, i.e. outside its protection.

    Today UniProt only adds keys a snapshot node does not carry, so
    nothing changes -- but it assigns with ``=``, and a key name
    coinciding is all it would take for a live answer to become the
    release's.
    """
    from neorx.core.graph.graph_builder import _enrich_nodes_with_uniprot

    pinned = GraphNode(
        node_id="gene:CCR5", name="CCR5", node_type=NodeType.GENE,
        source="Open Targets", score=0.75,
        metadata={"snapshot_release": RELEASE, "is_druggable": False},
    )
    live = GraphNode(
        node_id="gene:CXCR4", name="CXCR4", node_type=NodeType.GENE,
        source="Monarch", score=0.85,
        metadata={"is_druggable": False},
    )
    info = {"is_druggable": True, "subcellular_location": "membrane",
            "go_terms": ["GO:1"]}

    _enrich_nodes_with_uniprot([pinned, live], {"ccr5": info, "CXCR4": info})

    # Keyed through the same normalisation as everything else, so the
    # lower-cased key still found the node.
    assert pinned.metadata["subcellular_location"] == "membrane"
    # ... but the release's own answer stands.
    assert pinned.metadata["is_druggable"] is False
    # An unpinned node is enriched exactly as before.
    assert live.metadata["is_druggable"] is True
