"""MolScreen — drug-likeness screening and molecular property calculation.

>>> from neorx.molscreen import qed_score, lipinski_filter, parse_smiles
>>> mol = parse_smiles("CC(=O)Oc1ccccc1C(O)=O")
>>> round(qed_score(mol), 3)
0.55
"""

from .accessibility import qed_score, sa_score
from .filters import (
    brenk_filter,
    classify_drug_likeness,
    egan_filter,
    ghose_filter,
    lipinski_filter,
    pains_filter,
    run_all_filters,
    veber_filter,
)
from .models import (
    ComparisonReport,
    DrugLikelihoodCategory,
    FilterResult,
    MolecularProperties,
    ScreeningReport,
    SimilarDrug,
)
from .parser import canonicalise, name_to_smiles, parse_smiles, smart_parse, validate_smiles
from .properties import calculate_properties
from .similarity import build_cache, clear_cache, find_similar_drugs, tanimoto_similarity

__all__ = [
    # Parsing
    "parse_smiles",
    "validate_smiles",
    "canonicalise",
    "name_to_smiles",
    "smart_parse",
    # Properties
    "calculate_properties",
    "qed_score",
    "sa_score",
    # Filters
    "lipinski_filter",
    "veber_filter",
    "ghose_filter",
    "egan_filter",
    "pains_filter",
    "brenk_filter",
    "run_all_filters",
    "classify_drug_likeness",
    # Similarity
    "build_cache",
    "clear_cache",
    "find_similar_drugs",
    "tanimoto_similarity",
    # Models
    "MolecularProperties",
    "FilterResult",
    "ScreeningReport",
    "ComparisonReport",
    "SimilarDrug",
    "DrugLikelihoodCategory",
]
