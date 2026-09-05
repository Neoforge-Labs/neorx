"""
Versioned snapshots of the sources the causal subgraph is built from.

Upstream releases are streamed, reduced to a compact Parquet extract, and
discarded. What persists is four columns of gene-disease evidence and a
table of directed regulatory interactions -- enough to rebuild the causal
subgraph as it would have looked at a past date, and small enough that a
decade of releases costs less disk than one of them.
"""
