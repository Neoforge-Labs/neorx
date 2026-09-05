"""Experiment definitions for this project.

Importing this package registers every experiment. It is deliberately not
part of the installed wheel: these are this repository's experiments, not
a library feature.
"""

from experiments import (  # noqa: F401
    causalbiorl_bench,
    corpus_census,
    figures,
    genmol_eval,
    neorx_7disease,
)
