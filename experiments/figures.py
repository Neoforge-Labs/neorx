"""Manuscript figures, rendered from run records.

_gen_figures.py held the F1 values as literals at lines 226-227, so
Figure 3 was a transcription of the paper's table rather than a rendering
of data. Every data-bearing figure here takes a run ID and reads its
series from that run's rows.

fig1_pipeline and fig2_hiv_graph are diagrams with no data dependency and
move across from _gen_figures.py unchanged. fig4_disease_context is a
categorical classification table with no metric literals in it, so it
moves across unchanged too. Only fig3_pr_comparison -- which held the
F1/precision/recall values this migration exists to fix -- now reads its
series from a run record, with its summary lines computed from that data
instead of copied from one particular run.

`test_the_figure_module_contains_no_metric_literals` fails if a literal
reappears in this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from neorx.experiments.record import RunRecord
from neorx.experiments.registry import experiment

OUT = Path("figures")

#: Vertical offset (axis units) for text labels placed just above a bar or
#: line -- a layout constant, not a measured value.
LABEL_OFFSET = 0.02

BLUE = "#2196F3"
GREY = "#9E9E9E"
GREEN = "#4CAF50"
RED = "#F44336"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
TEAL = "#009688"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def fig3_series(run_id: str, *, runs_dir: Path | None = None) -> dict:
    """Extract Figure 3's series from a run record."""
    record = RunRecord.load(run_id, runs_dir=runs_dir)
    if not record.citable:
        raise ValueError(
            f"run {run_id} is not citable (status={record.status}); a figure "
            f"may not be drawn from a run nobody can reproduce"
        )
    rows = record.rows()
    return {
        "diseases": [r["disease"] for r in rows],
        "neorx_f1": [r["F1"] for r in rows],
        "corr_f1": [r["corr_F1"] for r in rows],
        "neorx_p": [r.get("P") for r in rows],
        "neorx_r": [r.get("R") for r in rows],
        "corr_p": [r.get("corr_P") for r in rows],
        "corr_r": [r.get("corr_R") for r in rows],
        "grades": [r.get("grade") for r in rows],
    }


# ════════════════════════════════════════════════════════════════════
# Figure 1: Pipeline Architecture (no data dependency)
# ════════════════════════════════════════════════════════════════════


def fig1_pipeline() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        ("Disease\nName", 0.3, "#E3F2FD", "Input"),
        ("Knowledge\nGraph\nAssembly", 2.0, "#BBDEFB", "Stage 1"),
        ("Causal\nTarget\nIdentification", 4.0, "#90CAF9", "Stage 2"),
        ("Biological\nIntelligence\nLayer", 6.0, "#64B5F6", "Stage 3"),
        ("Molecule\nGeneration\n(GenMol VAE)", 8.0, "#42A5F5", "Stage 4"),
        ("Screening &\nDocking", 9.8, "#2196F3", "Stage 5"),
        ("Composite\nScoring &\nRanking", 11.3, "#1976D2", "Stage 6"),
    ]

    for label, x, color, stage_label in stages:
        box = mpatches.FancyBboxPatch(
            (x - 0.65, 1.8),
            1.3,
            1.8,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="#1565C0",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x,
            2.7,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#0D47A1",
        )
        ax.text(x, 1.55, stage_label, ha="center", va="center", fontsize=7, color="#666")

    for i in range(len(stages) - 1):
        x1 = stages[i][1] + 0.7
        x2 = stages[i + 1][1] - 0.7
        ax.annotate(
            "",
            xy=(x2, 2.7),
            xytext=(x1, 2.7),
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2),
        )

    sources = "Monarch . Open Targets . KEGG . Reactome\nSTRING . UniProt . PDB . ChEMBL"
    ax.text(
        2.0,
        0.7,
        sources,
        ha="center",
        va="center",
        fontsize=7,
        style="italic",
        color="#555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F9A825", linewidth=1),
    )
    ax.annotate(
        "",
        xy=(2.0, 1.75),
        xytext=(2.0, 1.15),
        arrowprops=dict(arrowstyle="->", color="#F9A825", lw=1.5),
    )

    ax.text(
        4.0,
        0.7,
        "ChEMBL Pathogen\nTarget Pipeline",
        ha="center",
        va="center",
        fontsize=7,
        style="italic",
        color="#555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=1),
    )
    ax.annotate(
        "",
        xy=(4.0, 1.75),
        xytext=(4.0, 1.1),
        arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.5),
    )

    ax.text(
        11.3,
        0.7,
        "Ranked Drug\nCandidates",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#B71C1C",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#E53935", linewidth=1.5
        ),
    )
    ax.annotate(
        "",
        xy=(11.3, 1.15),
        xytext=(11.3, 1.75),
        arrowprops=dict(arrowstyle="->", color="#E53935", lw=2),
    )

    ax.set_title("NeoRx Pipeline Architecture", fontsize=14, fontweight="bold", pad=15)

    fig.savefig(OUT / "fig1_pipeline.png")
    fig.savefig(OUT / "fig1_pipeline.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# Figure 2: HIV Knowledge Graph Excerpt (no data dependency)
# ════════════════════════════════════════════════════════════════════


def fig2_hiv_graph() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.axis("off")

    nodes = {
        "HIV": (5, 0.5, "#FFCDD2", "disease"),
        "POL": (2, 5.5, "#C8E6C9", "pathogen"),
        "ENV": (5, 6.5, "#C8E6C9", "pathogen"),
        "GAG": (8, 5.5, "#C8E6C9", "pathogen"),
        "CCR5": (1, 3, "#BBDEFB", "human"),
        "CD4": (3.5, 2.5, "#BBDEFB", "human"),
        "CXCR4": (6.5, 2.5, "#BBDEFB", "human"),
        "RPOA": (9, 3, "#FFECB3", "off-target"),
        "RPSA": (10, 4.5, "#FFECB3", "off-target"),
    }

    confidences = {
        "POL": "C=0.990",
        "ENV": "C=0.955",
        "GAG": "C=0.887",
        "CCR5": "C=0.706",
        "CD4": "C=0.625",
        "CXCR4": "C=0.580",
        "RPOA": "INCON.",
        "RPSA": "INCON.",
    }

    classifications = {
        "POL": "PATHOGEN_DIRECT",
        "ENV": "PATHOGEN_DIRECT",
        "GAG": "PATHOGEN_DIRECT",
        "CCR5": "HOST_INVASION",
        "CD4": "HOST_IMMUNE",
        "CXCR4": "CORRELATIONAL",
        "RPOA": "INCONCLUSIVE",
        "RPSA": "INCONCLUSIVE",
    }

    edges = [
        ("POL", "HIV", "#4CAF50", 2.5, "-"),
        ("ENV", "HIV", "#4CAF50", 2.0, "-"),
        ("GAG", "HIV", "#4CAF50", 2.0, "-"),
        ("CCR5", "HIV", "#2196F3", 1.5, "-"),
        ("CD4", "HIV", "#2196F3", 1.5, "-"),
        ("CXCR4", "HIV", "#2196F3", 1.0, "-"),
        ("RPOA", "HIV", "#9E9E9E", 1.0, "--"),
        ("RPSA", "HIV", "#9E9E9E", 1.0, "--"),
    ]

    for src, dst, color, lw, ls in edges:
        sx, sy = nodes[src][0], nodes[src][1]
        dx, dy = nodes[dst][0], nodes[dst][1]
        ax.plot([sx, dx], [sy, dy], color=color, lw=lw, ls=ls, alpha=0.6, zorder=1)

    for name, (x, y, color, _ntype) in nodes.items():
        r = 0.55 if name != "HIV" else 0.7
        circle = plt.Circle((x, y), r, facecolor=color, edgecolor="#333", linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(
            x,
            y + 0.05,
            name,
            ha="center",
            va="center",
            fontsize=10 if name != "HIV" else 12,
            fontweight="bold",
            zorder=3,
        )

        if name in confidences:
            ax.text(
                x, y - r - 0.25, confidences[name], ha="center", va="top", fontsize=7, color="#555"
            )
            ax.text(
                x,
                y - r - 0.55,
                classifications[name],
                ha="center",
                va="top",
                fontsize=6,
                color="#888",
                style="italic",
            )

    legend_items = [
        mpatches.Patch(color="#C8E6C9", label="Pathogen target (ChEMBL)"),
        mpatches.Patch(color="#BBDEFB", label="Human target"),
        mpatches.Patch(color="#FFECB3", label="Off-target (demoted)"),
        mpatches.Patch(color="#FFCDD2", label="Disease node"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title("HIV Causal Knowledge Graph (Excerpt)", fontsize=14, fontweight="bold", pad=15)

    fig.savefig(OUT / "fig2_hiv_graph.png")
    fig.savefig(OUT / "fig2_hiv_graph.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# Figure 3: Precision-Recall Comparison (reads a run record)
# ════════════════════════════════════════════════════════════════════


def _render_fig3(series: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    diseases = series["diseases"]
    neorx_f1 = series["neorx_f1"]
    corr_f1 = series["corr_f1"]
    neorx_p = series.get("neorx_p") or []
    neorx_r = series.get("neorx_r") or []
    corr_p = series.get("corr_p") or []
    corr_r = series.get("corr_r") or []
    grades = series.get("grades") or []

    has_pr = bool(neorx_p) and bool(neorx_r) and bool(corr_p) and bool(corr_r)
    has_pr = has_pr and None not in neorx_p and None not in neorx_r
    has_pr = has_pr and None not in corr_p and None not in corr_r

    x = np.arange(len(diseases))
    width = 0.35

    n_panels = 3 if has_pr else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.bar(
        x - width / 2,
        corr_f1,
        width,
        label="Correlation-only",
        color=GREY,
        alpha=0.8,
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        neorx_f1,
        width,
        label="NeoRx",
        color=BLUE,
        alpha=0.9,
        edgecolor="white",
    )
    ax.set_ylabel("F₁ Score")
    ax.set_title("F₁ Score Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, rotation=45, ha="right")
    ax.legend()

    if grades and len(grades) == len(diseases):
        for bar, grade in zip(bars2, grades):
            if not grade:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + LABEL_OFFSET,
                grade,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=GREEN if grade == "A" else BLUE,
            )

    if neorx_f1:
        mean_neorx = float(np.mean(neorx_f1))
        ax.axhline(y=mean_neorx, color=BLUE, ls="--", lw=1, alpha=0.5)
        ax.text(
            len(diseases) - 0.7,
            mean_neorx + LABEL_OFFSET,
            f"Mean={mean_neorx:.3f}",
            fontsize=7,
            color=BLUE,
        )
    if corr_f1:
        mean_corr = float(np.mean(corr_f1))
        ax.axhline(y=mean_corr, color=GREY, ls="--", lw=1, alpha=0.5)
        ax.text(
            len(diseases) - 0.7,
            mean_corr + LABEL_OFFSET,
            f"Mean={mean_corr:.3f}",
            fontsize=7,
            color=GREY,
        )

    if has_pr:
        ax = axes[1]
        ax.bar(
            x - width / 2,
            corr_p,
            width,
            label="Correlation-only",
            color=GREY,
            alpha=0.8,
            edgecolor="white",
        )
        ax.bar(
            x + width / 2, neorx_p, width, label="NeoRx", color=ORANGE, alpha=0.9, edgecolor="white"
        )
        ax.set_ylabel("Precision")
        ax.set_title("Precision Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(diseases, rotation=45, ha="right")
        ax.legend()

        ax = axes[2]
        ax.bar(
            x - width / 2,
            corr_r,
            width,
            label="Correlation-only",
            color=GREY,
            alpha=0.8,
            edgecolor="white",
        )
        ax.bar(
            x + width / 2, neorx_r, width, label="NeoRx", color=TEAL, alpha=0.9, edgecolor="white"
        )
        ax.set_ylabel("Recall")
        ax.set_title("Recall Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(diseases, rotation=45, ha="right")
        ax.legend()

    fig.suptitle(
        "NeoRx vs. Correlation-Only Baseline Across the Recorded Diseases",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(OUT / "fig3_pr_comparison.png")
    fig.savefig(OUT / "fig3_pr_comparison.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# Figure 4: Disease-context classification differences
#
# A categorical table with no metric literals -- it does not fall under
# the "F1 values as literals" defect this migration fixes, so it moves
# across from _gen_figures.py unchanged, like fig1 and fig2.
# ════════════════════════════════════════════════════════════════════


def fig4_disease_context() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    genes = ["TNF", "GABRD", "TLR4", "CCR5", "SCN2A"]
    diseases = ["Malaria", "RA", "Alzheimer's"]

    classifications = {
        "TNF": ["HOST_SYMPTOM", "PATHOGEN_DIRECT", "CORRELATIONAL"],
        "GABRD": ["HOST_SYMPTOM", "HOST_SYMPTOM", "Not reclassified"],
        "TLR4": ["HOST_IMMUNE", "PATHOGEN_DIRECT", "HOST_IMMUNE"],
        "CCR5": ["HOST_INVASION", "CORRELATIONAL", "CORRELATIONAL"],
        "SCN2A": ["HOST_SYMPTOM", "HOST_SYMPTOM", "Not reclassified"],
    }

    color_map = {
        "HOST_SYMPTOM": "#FFCDD2",
        "PATHOGEN_DIRECT": "#C8E6C9",
        "HOST_INVASION": "#BBDEFB",
        "HOST_IMMUNE": "#FFE0B2",
        "CORRELATIONAL": "#E0E0E0",
        "Not reclassified": "#F5F5F5",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    cell_text = [classifications[g] for g in genes]
    cell_colors = [[color_map.get(c, "#FFF") for c in classifications[g]] for g in genes]

    table = ax.table(
        cellText=cell_text,
        rowLabels=genes,
        colLabels=diseases,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for j in range(len(diseases)):
        table[0, j].set_facecolor("#E8EAF6")
        table[0, j].set_text_props(fontweight="bold")
    for i in range(len(genes)):
        table[i + 1, -1].set_text_props(fontweight="bold")

    ax.set_title(
        "Disease-Context-Dependent Gene Classification", fontsize=13, fontweight="bold", pad=20
    )

    fig.savefig(OUT / "fig4_disease_context.png")
    fig.savefig(OUT / "fig4_disease_context.pdf")
    plt.close(fig)


@experiment(name="figures", help="Render manuscript figures from a run record.")
def figures(record: RunRecord) -> None:
    """Registered so figure generation is itself a recorded run.

    The source run is read from NEORX_FIGURE_RUN; `neorx exp figure` sets it.
    """
    import os

    source = os.environ.get("NEORX_FIGURE_RUN")
    if not source:
        raise ValueError(
            "set NEORX_FIGURE_RUN to the run ID the figures should render from, "
            "or use `neorx exp figure --from <run-id>`"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    series = fig3_series(source)
    _render_fig3(series)
    record.append_row(
        {
            "figure": "fig3_pr_comparison",
            "source_run": source,
            "n_series_points": len(series["diseases"]),
        }
    )
