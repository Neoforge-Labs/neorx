"""The seven-disease causal target benchmark behind Tables 2-5.

Ported from benchmark_paper.py, which computed these numbers and never
persisted any of them -- no json.dump, no write_text. Every reported value
existed only in terminal scrollback.

The evaluation logic is carried across unchanged. Two known defects are
recorded faithfully rather than fixed here, because fixing them is
sub-project 3's job and conflating the two would make neither reviewable:

* ``t_graph`` used to read ``graph._build_time``, an attribute nothing
  set, so it was always 0.0. This migration measures real elapsed time
  instead -- a recording fix, not a methodology change -- but the value
  it replaces was never meaningful.
* the correlation-only baseline filters to node types ``gene`` and
  ``protein``, so it cannot see pathogen targets that NeoRx can. That
  filter is carried across unchanged.

Online: eight external APIs, so the run is recorded to a cassette and can
be replayed exactly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from neorx.experiments.chembl import chembl_provenance
from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

DISEASES = [
    "HIV",
    "malaria",
    "type 2 diabetes",
    "alzheimer disease",
    "lung cancer",
    "breast cancer",
    "ebola",
]

TOP_N = 20
CHEMBL_DB = Path("chembl_36.db")


class UnvalidatableDiseaseError(RuntimeError):
    """Raised when a configured disease has no ground truth to validate against.

    ``KnownTargetValidator.validate()`` returns ``validated=False`` in this
    case. Recording precision/recall/F1 as 0.0 when validation could not
    even run would be a fabricated metric, not a measurement -- exactly
    what this subsystem exists to make impossible. A disease that cannot
    be validated must fail the run, not degrade into an invented number.
    """


@experiment(
    name="neorx-7disease",
    help="Causal target benchmark across 7 diseases.",
    captures_http=True,
    volatile_fields=("wall_clock_s", "t_graph", "t_identify"),
)
def neorx_7disease(record: RunRecord) -> None:
    if CHEMBL_DB.exists():
        prov = chembl_provenance(CHEMBL_DB)
        (record.path / "inputs" / "chembl.json").write_text(json.dumps(prov, indent=2))

    _run_all(record)


def _run_all(record: RunRecord) -> None:
    for disease in DISEASES:
        started = time.perf_counter()
        result = _evaluate_disease(disease)
        record.append_row(
            {
                "disease": disease,
                "wall_clock_s": round(time.perf_counter() - started, 1),
                **result,
            }
        )


def _evaluate_disease(disease: str) -> dict:
    """Evaluate one disease. Logic ported from benchmark_paper.py.

    The validator API used here differs from benchmark_paper.py's
    unreachable helpers (``validator.ground_truth``, ``report.grade``):
    it uses the real ``KnownTargetValidator`` surface -- ``validate()``
    returning a ``ValidationResult`` with ``.quality_grade``, plus
    ``get_ground_truth_genes()`` / ``get_known_false_targets()`` for the
    correlation-only baseline.
    """
    from neorx.core.causal.identifier import identify_causal_targets
    from neorx.core.graph.graph_builder import build_disease_graph
    from neorx.core.validator import KnownTargetValidator

    validator = KnownTargetValidator()

    t0 = time.perf_counter()
    graph = build_disease_graph(disease, use_cache=False)
    t_graph = round(time.perf_counter() - t0, 1)

    t1 = time.perf_counter()
    targets = identify_causal_targets(graph, top_n=TOP_N)
    t_identify = round(time.perf_counter() - t1, 1)

    target_dicts = [{"gene_symbol": t.gene_name, "is_causal": t.is_causal_target} for t in targets]
    report = validator.validate(disease, target_dicts)

    if not report.validated:
        raise UnvalidatableDiseaseError(
            f"{disease!r} has no ground truth to validate against "
            f"({getattr(report, 'reason', 'validation did not run')}); "
            f"refusing to record a fabricated P/R/F1 of 0.0"
        )

    tp_names = sorted(report.true_positives)
    fp_names = sorted(report.false_positives_known)
    missed = sorted(report.missed_targets)
    P, R, F1 = report.precision, report.recall, report.f1
    grade = report.quality_grade

    causal = [t for t in targets if t.is_causal_target]

    corr = _correlation_only(graph, disease, validator)

    neorx_fp_set = set(fp_names)
    corr_fp_set = set(corr["fp"])
    demoted = sorted(corr_fp_set - neorx_fp_set)

    return {
        "found": len(targets),
        "causal": len(causal),
        "tp": tp_names,
        "fp": fp_names,
        "missed": missed,
        "P": P,
        "R": R,
        "F1": F1,
        "grade": grade,
        "corr_P": corr["P"],
        "corr_R": corr["R"],
        "corr_F1": corr["F1"],
        "corr_fp": corr["fp"],
        "demoted": demoted,
        "fp_ranks": corr["fp_ranks"],
        "t_graph": t_graph,
        "t_identify": t_identify,
    }


def _correlation_only(graph, disease: str, validator) -> dict:
    """Rank by raw association score, no causal analysis.

    NOTE: filters to node types ``gene`` and ``protein``, so pathogen
    targets are invisible to this baseline. That asymmetry is a known
    defect carried across unchanged; sub-project 3 owns it.
    """
    nodes = []
    for node in graph.nodes:
        if node.node_type.value in ("gene", "protein"):
            name = node.node_id.split(":", 1)[1] if ":" in node.node_id else node.node_id
            nodes.append((name, node.score or 0.0))
    nodes.sort(key=lambda x: x[1], reverse=True)
    top = nodes[:TOP_N]

    known_tp = validator.get_ground_truth_genes(disease)
    known_fp = validator.get_known_false_targets(disease)
    found = {n.upper() for n, _ in top}

    tp = found & known_tp
    fp = found & known_fp
    P = len(tp) / len(top) if top else 0.0
    R = len(tp) / len(known_tp) if known_tp else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    fp_ranks = {}
    for fp_gene in sorted(fp):
        for rank, (gname, gscore) in enumerate(top, 1):
            if gname.upper() == fp_gene:
                fp_ranks[fp_gene] = {"rank": rank, "score": gscore}
                break

    return {"P": P, "R": R, "F1": F1, "fp": sorted(fp), "fp_ranks": fp_ranks}
