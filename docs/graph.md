# The graph and its sources

The graph builder is the first stage of a run. It queries the biomedical
sources for one disease, merges duplicate entities, records where every
edge came from, and returns a `DiseaseGraph`.

What matters here is not the number of sources but **which attributes
each one is allowed to set.** The causal engine reads only `edge_type`
and `evidence_class`, so an edge's source decides whether it ever becomes
an arrow — see [Which edges earn an arrow](admissibility.md).

## The sources

| Source | Supplies | Access | Causal-admissible edges? |
|---|---|---|---|
| [Monarch Initiative](https://monarchinitiative.org/) | Gene–disease associations from OMIM, ClinGen, Orphanet, HPO | REST v3 | Only where the evidence class is genetic |
| [Open Targets](https://platform.opentargets.org/) | Target–disease evidence, per-datatype scores, tractability | GraphQL v4 | Only `genetic_association` / `somatic_mutation` |
| [KEGG](https://rest.kegg.jp/) | Pathway membership | REST | No — `participates_in` is not directed |
| [Reactome](https://reactome.org/) | Hierarchical pathway membership | REST | No |
| [STRING](https://string-db.org/) | Protein–protein interactions, 0–1000 confidence | REST, batched | No — PPIs are undirected |
| [OmniPath](https://omnipathdb.org/) | Signed, directed regulatory interactions | REST | **Yes** — the only source of gene→gene arrows |
| [UniProt](https://www.uniprot.org/) | Protein function, PDB cross-references, location | REST | Enrichment only |
| [RCSB PDB](https://www.rcsb.org/) | 3D structures for docking | Search v2 + Data API | Enrichment only |
| ChEMBL | Drug–target relationships, incl. pathogen targets | Local SQLite (`chembl_36.db`, ~28 GB) | No |

OmniPath is the load-bearing one for causal inference, and it is worth
being clear why. STRING has far more interactions, but they are
undirected: an arrow drawn from a STRING row records which protein
appeared in the first column, not which acts on which. OmniPath
aggregates SIGNOR, TRRUST, PhosphoSite and around a hundred other
resources and resolves direction and sign *by consensus across them*.

Aggregation does not cost provenance: each OmniPath interaction carries
the primary databases that assert it and the PubMed IDs behind them, both
recorded on the edge. So multi-source corroboration counts distinct
primary evidence rather than counting OmniPath twice for an interaction
STRING also reported.

ChEMBL is the only source that can surface a **non-human** target —
*Plasmodium* DHFR-TS, HIV-1 protease — which become `PATHOGEN_GENE`
nodes rather than merging into the human gene population.

## Building one

!!! warning "Requires network"
    This block queries every live source, so it is not executed as a
    doctest.

```python
from neorx import build_disease_graph

graph = build_disease_graph(
    "HIV",
    max_genes=20,          # cap on genes carried forward
    string_min_score=400,  # STRING combined-score threshold, 0-1000
    use_cache=True,
    allow_mocks=False,
)
print(graph.n_genes, graph.n_proteins, graph.n_pathways)
print(graph.sources_queried)
```

`allow_mocks=False` is the default and it matters: with mocks disabled, a
failed API call produces an empty result rather than curated stand-in
data. Mock fallbacks exist for offline development and tests, and turning
them on in anger would mean a reported number resting on hand-written
data.

## Node merging

Monarch calls it `CCR5`; Open Targets calls it `ENSG00000160791`. Nodes
are normalised to gene symbols and merged, keeping the highest confidence
score and the union of metadata, with every contributing source recorded
in the node's `source` field — which is what the multi-source consensus
term in [causal confidence](targets.md#causal-confidence) counts.

## The data model

A `DiseaseGraph` is Pydantic, so it serialises losslessly and validates
on construction. Edges carry more than a weight:

```python
>>> from neorx import GraphEdge, EdgeType
>>> edge = GraphEdge(
...     source_id="CCL5", target_id="CCR5",
...     edge_type=EdgeType.REGULATES,
...     weight=0.7,
...     source_db="OmniPath",
...     evidence_class="regulatory",
...     sign=1,
...     primary_sources=["SIGNOR", "TRRUST"],
...     pmids=["12345678"],
... )
>>> edge.edge_type.value
'regulates'
>>> edge.primary_sources
['SIGNOR', 'TRRUST']

```

`sign` records activation versus inhibition, `primary_sources` the
databases behind an aggregated edge, `evidence_class` the attribute
admissibility turns on.

The twelve edge types and eight node types:

```python
>>> from neorx import EdgeType, NodeType
>>> [e.value for e in EdgeType]
['activates', 'inhibits', 'phosphorylates', 'binds', 'regulates',
 'participates_in', 'associated_with', 'causes', 'treats',
 'interacts_with', 'upregulates', 'downregulates']
>>> [n.value for n in NodeType]
['gene', 'protein', 'pathway', 'phenotype', 'disease', 'drug',
 'metabolite', 'pathogen_gene']

```

## To NetworkX and back

The causal engine works on a `nx.DiGraph`. The conversion is explicit:

```python
>>> from neorx import (DiseaseGraph, GraphNode, NodeType,
...                    disease_graph_to_networkx)
>>> graph = DiseaseGraph(
...     disease_name="HIV",
...     nodes=[
...         GraphNode(node_id="CCR5", name="CCR5", node_type=NodeType.GENE),
...         GraphNode(node_id="MONDO:0005109", name="HIV",
...                   node_type=NodeType.DISEASE),
...     ],
...     edges=[edge.model_copy(update={"source_id": "CCR5",
...                                    "target_id": "MONDO:0005109"})],
... )
>>> G = disease_graph_to_networkx(graph)
>>> sorted(G.nodes)
['CCR5', 'MONDO:0005109']
>>> G.edges["CCR5", "MONDO:0005109"]["evidence_class"]
'regulatory'

```

Edge attributes survive the conversion, which is the whole point —
`is_causal_admissible` reads them off the NetworkX graph.

## Persistence and export

```python
>>> import tempfile
>>> from neorx import save_graph, load_graph
>>> with tempfile.TemporaryDirectory() as tmp:
...     path = save_graph(graph, output_dir=tmp, fmt="json")
...     reloaded = load_graph(path)
...     for fmt in ("graphml", "gexf", "cytoscape"):
...         _ = save_graph(graph, output_dir=tmp, fmt=fmt)
>>> reloaded.disease_name
'HIV'
>>> len(reloaded.nodes)
2

```

Four formats: `json` (lossless, round-trips through `load_graph`),
`graphml`, `gexf` for Gephi, and `cytoscape` for Cytoscape. The default
directory is `~/.neorx/graphs/`, and filenames carry a UTC timestamp.

With `DATABASE_URL` set, graphs and jobs are also written to PostgreSQL —
`save_graph_to_db`, `load_graph_from_db`.

## Caching

Every HTTP client goes through a cache layer, backed by JSON files under
`~/.neorx/cache/` by default or by Redis when
`NEORX_CACHE_BACKEND=redis`. API responses are held for 24 hours and
assembled graphs for 7 days. Pass `use_cache=False`, or `--no-cache` on
the CLI, to bypass it.

All HTTP traffic in `neorx.core.sources` uses `requests`, which is what
lets the [experiment runner](experiments.md) freeze a run's traffic into
a cassette for replay. **A source added with a different HTTP client
would silently escape that recording.**

## Dated builds

A build can be pinned to a past release, so that a run reads what that
date would have read rather than what the sources say today:

```python
from neorx import build_disease_graph
from neorx.snapshots.reader import SNAPSHOTS_DIR, SnapshotStore
from neorx.snapshots.releases import release_for
from neorx.snapshots.resolver import SourceResolver

store = SnapshotStore(SNAPSHOTS_DIR, on_read=None)
# An empty `live` mapping on purpose: a dated build must not be able to
# reach a live client for a pinned source even by accident.
resolver = SourceResolver(store=store, release_for=release_for, live={})

graph = build_disease_graph(
    "HIV infection",
    as_of="2021-11",
    resolver=resolver,
    disease_id="EFO_0000764",
)
```

`as_of` is a key in the pinned release table — `2018-06`, `2021-11`,
`2025-06` — not a free-form date, and an unknown one raises rather than
being rounded to something nearby:

```python
>>> from neorx.snapshots.releases import TIME_POINTS, release_for
>>> sorted(TIME_POINTS)
['2018-06', '2021-11', '2025-06']
>>> release_for("opentargets", "2021-11")
'21.11'
>>> release_for("omnipath", "2021-11")
'20211113-20220114'

```

`as_of` requires both `resolver` and `disease_id`, and refuses
`allow_mocks` outright — curated mock data is a measurement of no
release, and a dated build would carry it as a node score *for that
date*.

!!! danger "A dated score and a live score are different quantities"
    On a live build, a gene node's `score` is Open Targets' overall
    association score — a harmonic sum over datasources that the platform
    publishes. The derived extract does not carry that number and it is
    not recoverable from the per-datatype rows; reconstructing it would
    mean inventing a quantity and presenting it as a measurement.

    So a dated node carries the **maximum genetic-evidence score in the
    pinned release** instead. Both are real Open Targets numbers. They are
    not the same number, and they must not be placed in the same column,
    ranked against each other, or pooled. `DiseaseGraph.as_of` is how a
    consumer tells which it is holding. The full per-datatype breakdown
    is on both, in `metadata["datatype_scores"]`.

Unpinned sources stay live on a dated build, because their edges are not
causal-admissible and identification filters them out. Their *nodes* are
restricted to the pinned release's own gene population: they may enrich a
node the release put in the graph, never introduce one. What they offered
and did not get to contribute is recorded in
`DiseaseGraph.frame_exclusions`, with a reason per entry, because a
silent filter is how frame leakage returns after a refactor.

See [Versioned snapshots](snapshots.md) for how the extracts are built.
