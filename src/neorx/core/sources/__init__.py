"""
Data Sources
==============

API clients for biomedical databases used to build disease
causal graphs.  Each client follows the same pattern:

1. Query the database for disease-associated entities
2. Parse the response into ``GraphNode`` and ``GraphEdge`` objects
3. Handle rate limits, timeouts, and API errors gracefully
4. Return empty lists (never crash) on failure — the graph
   builder aggregates results from multiple sources

All clients use ``requests`` and are synchronous.  For
production, consider ``httpx`` with async support.

All HTTP traffic in this package is subject to recording. When these
modules run inside ``neorx.experiments.capture``, every request and
response is frozen into the run's cassette so the run can be replayed
exactly. Adding a source that uses a client other than ``requests`` will
silently escape that recording -- see ``neorx/experiments/capture.py``.
"""

from .monarch import query_monarch
from .open_targets import query_open_targets
from .kegg import query_kegg_pathways
from .reactome import query_reactome_pathways
from .string_db import query_string_interactions
from .omnipath import query_omnipath
from .uniprot import query_uniprot
from .pdb import query_pdb_structures
from .chembl import query_chembl

__all__ = [
    "query_monarch",
    "query_open_targets",
    "query_kegg_pathways",
    "query_reactome_pathways",
    "query_string_interactions",
    "query_omnipath",
    "query_uniprot",
    "query_pdb_structures",
    "query_chembl",
]
