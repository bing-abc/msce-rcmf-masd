from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables_md_revision"
FIGURES = ROOT / "figures_md_revision"
MANUSCRIPT_DIR = ROOT / "manuscript_md_revision_files"
MANUSCRIPT_FIGURES = MANUSCRIPT_DIR / "figures"
READY_DIR = MANUSCRIPT_DIR / "submission_package_md_ready"
READY_FIGURES = READY_DIR / "figures"
SUBMISSION_DIR = ROOT / "submission_final"
SUBMISSION_FIGURES = SUBMISSION_DIR / "figures"
SUBMISSION_SOURCE_DATA = SUBMISSION_DIR / "source_data"
SUBMISSION_SOURCE_FIGURES = SUBMISSION_DIR / "source" / "figures"
POLY_JSON = ROOT / "outputs" / "exp" / "diagnostics" / "polybert_baseline_results.json"
STATS_JSON = ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run" / "stats.json"
HARD_IMPROVEMENTS = FIGURES / "final_fig3_source_data.csv"

MANIFEST = pd.read_csv(
    TABLES / "final_metric_manifest.csv",
    engine="python",
    on_bad_lines="skip",
)
EXTERNAL = pd.read_csv(TABLES / "external_stratified_performance.csv")
CLUSTER_DIAG = pd.read_csv(TABLES / "cluster_failure_diagnosis.csv")
CLUSTER_DESIGN = pd.read_csv(TABLES / "cluster_design_relevance.csv")
RANKING = pd.read_csv(TABLES / "materials_design_candidate_ranking.csv")

for path in (FIGURES, MANUSCRIPT_FIGURES, READY_FIGURES, SUBMISSION_FIGURES, SUBMISSION_SOURCE_DATA, SUBMISSION_SOURCE_FIGURES):
    path.mkdir(parents=True, exist_ok=True)

P = {
    "navy": "#204E7A",
    "blue": "#2F6DAA",
    "sky": "#85A9D0",
    "teal": "#157A6E",
    "teal_soft": "#D7EFEA",
    "orange": "#C65A1E",
    "orange_soft": "#F8E1D3",
    "green": "#2F855A",
    "green_soft": "#D9EEDD",
    "red": "#B54742",
    "red_soft": "#F7D8D6",
    "gold": "#B88A12",
    "slate": "#56616B",
    "gray": "#7B8794",
    "light": "#EEF2F6",
    "grid": "#CDD6E1",
    "text": "#1F2933",
}


def manifest_row(metric: str, split: str, model: str) -> pd.Series:
    rows = MANIFEST[
        (MANIFEST["metric"] == metric)
        & (MANIFEST["split"] == split)
        & (MANIFEST["model"] == model)
    ]
    if rows.empty:
        raise KeyError(f"Missing manifest row for {metric=} {split=} {model=}")
    return rows.iloc[0]


def t_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size <= 1:
        return mean, mean
    sem = float(values.std(ddof=1) / math.sqrt(values.size))
    delta = 1.96 * sem
    return mean - delta, mean + delta


def poly_payload() -> dict[str, object]:
    return json.loads(POLY_JSON.read_text(encoding="utf-8"))


def poly_metric_array(payload: dict[str, object], split: str, metric: str) -> np.ndarray:
    values: list[float] = []
    for row in payload.get("rows", []):
        block = row.get(f"polybert_{split}")
        if isinstance(block, dict) and block.get(metric) is not None:
            values.append(float(block[metric]))
    return np.asarray(values, dtype=float)


def poly_summary(payload: dict[str, object], split: str) -> tuple[float, float, float]:
    arr = poly_metric_array(payload, split, "mae_k")
    low, high = t_ci(arr)
    return float(arr.mean()), low, high


def hard_stats_payload() -> dict[str, object]:
    return json.loads(STATS_JSON.read_text(encoding="utf-8"))


def add_round_box(ax, xy, width, height, title, lines, facecolor, edgecolor, title_size=17, text_size=11):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    x0, y0 = xy
    ax.text(
        x0 + 0.04 * width,
        y0 + height - 0.14 * height,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=P["navy"],
    )
    ax.text(
        x0 + 0.04 * width,
        y0 + height - 0.34 * height,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=text_size,
        color=P["slate"],
        linespacing=1.35,
    )


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.0,
            color=P["orange"],
            transform=ax.transAxes,
        )
    )


def build_figure1() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_axis_off()
    ax.text(
        0.03,
        0.95,
        "Polymer Tg prediction at the computer-materials interface",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        color=P["orange"],
        va="top",
    )

    add_round_box(
        ax,
        (0.035, 0.58),
        0.20,
        0.22,
        "Polymer Tg data",
        [
            "Train/validation/test +",
            "external holdout",
            "Overlap removed; splits fixed",
            "Primary + supplementary sources",
        ],
        P["light"],
        P["navy"],
    )
    add_round_box(
        ax,
        (0.285, 0.58),
        0.23,
        0.22,
        "Multimodal views",
        [
            "Graph backbone",
            "Descriptor branch",
            "Chain-context view",
        ],
        P["light"],
        P["navy"],
    )
    add_round_box(
        ax,
        (0.565, 0.58),
        0.20,
        0.22,
        "MSCE + MASD prediction",
        [
            "Validation-selected checkpoints",
            "Mechanism-aware correction",
            "Frozen test/external audit",
        ],
        P["light"],
        P["navy"],
    )
    add_round_box(
        ax,
        (0.81, 0.58),
        0.15,
        0.22,
        "Evaluation",
        [
            "Primary average",
            "Baseline-defined difficult subset",
            "External chemistry-space shift",
        ],
        P["light"],
        P["navy"],
        title_size=15,
        text_size=10.5,
    )

    chip = FancyBboxPatch(
        (0.60, 0.83),
        0.13,
        0.055,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.6,
        edgecolor=P["teal"],
        facecolor=P["teal_soft"],
        transform=ax.transAxes,
    )
    ax.add_patch(chip)
    ax.text(
        0.665,
        0.857,
        "Auxiliary diagnostic\nconditioning",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.5,
        color=P["teal"],
        fontweight="bold",
    )
    add_arrow(ax, (0.14, 0.69), (0.285, 0.69))
    add_arrow(ax, (0.515, 0.69), (0.565, 0.69))
    add_arrow(ax, (0.765, 0.69), (0.81, 0.69))
    add_arrow(ax, (0.665, 0.83), (0.665, 0.80))

    add_round_box(
        ax,
        (0.29, 0.18),
        0.40,
        0.24,
        "Structure-property and design implications",
        [
            "Aromatic-dense and ether-oxygen families improve most",
            "Amide and imide-like families remain challenging",
            "Bounded retrospective screening, not de novo discovery",
        ],
        "#F7F9FC",
        P["navy"],
        title_size=17,
        text_size=11.5,
    )
    add_arrow(ax, (0.40, 0.58), (0.45, 0.42))
    add_arrow(ax, (0.67, 0.58), (0.58, 0.42))

    src = pd.DataFrame(
        [
            {"stage": "Data", "content": "Train/validation/test + external holdout; overlap removed"},
            {"stage": "Multimodal views", "content": "Graph, descriptors, chain context"},
            {"stage": "Prediction", "content": "MSCE + MASD with validation-only selection"},
            {"stage": "Auxiliary diagnostic conditioning", "content": "Reliability used as bounded diagnostic path"},
            {"stage": "Evaluation", "content": "Primary, baseline-defined difficult subset, external shift"},
            {"stage": "Design implication", "content": "Bounded retrospective Tg screening and failure disclosure"},
        ]
    )
    src.to_csv(FIGURES / "final_fig1_source_data.csv", index=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.03)
    fig.savefig(FIGURES / "final_fig1_workflow.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "final_fig1_workflow.pdf", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 1. Study workflow at the computer-materials interface. Curated polymer Tg data "
        "are converted into graph, descriptor, and chain-context views, processed by MSCE + MASD "
        "with validation-only model selection, and evaluated on the primary test set, a "
        "baseline-defined difficult subset, and an external chemistry-space-shift holdout. The "
        "auxiliary diagnostic conditioning path is included as a bounded support component rather "
        "than as a standalone accuracy claim. The bottom panel summarizes retrospective "
        "structure-property and design implications."
    )
    (TABLES / "final_fig1_caption.md").write_text(caption + "\n", encoding="utf-8")


def build_figure2(payload: dict[str, object]) -> None:
    baseline_primary = manifest_row("MAE", "primary", "strongest_baseline")
    proposed_primary = manifest_row("MAE", "primary", "final_model")
    baseline_hard = manifest_row("MAE", "hard_subgroup", "strongest_baseline")
    proposed_hard = manifest_row("MAE", "hard_subgroup", "final_model")
    baseline_external = manifest_row("MAE", "external_holdout", "strongest_baseline")
    proposed_external = manifest_row("MAE", "external_holdout", "final_model")
    poly_primary_mean, poly_primary_low, poly_primary_high = poly_summary(payload, "primary")
    poly_external_mean, _, _ = poly_summary(payload, "external")

    src = pd.DataFrame(
        [
            {
                "panel": "primary_mae",
                "method": "Strongest baseline",
                "mean": float(baseline_primary["mean_K"]),
                "ci_low": float(baseline_primary["ci95_low_K"]),
                "ci_high": float(baseline_primary["ci95_high_K"]),
                "n_runs": int(baseline_primary["n_runs"]),
            },
            {
                "panel": "primary_mae",
                "method": "polyBERT audit",
                "mean": poly_primary_mean,
                "ci_low": poly_primary_low,
                "ci_high": poly_primary_high,
                "n_runs": int(poly_metric_array(payload, "primary", "mae_k").size),
            },
            {
                "panel": "primary_mae",
                "method": "Proposed",
                "mean": float(proposed_primary["mean_K"]),
                "ci_low": float(proposed_primary["ci95_low_K"]),
                "ci_high": float(proposed_primary["ci95_high_K"]),
                "n_runs": int(proposed_primary["n_runs"]),
            },
            {
                "panel": "hard_mae",
                "method": "Strongest baseline",
                "mean": float(baseline_hard["mean_K"]),
                "ci_low": float(baseline_hard["ci95_low_K"]),
                "ci_high": float(baseline_hard["ci95_high_K"]),
                "n_runs": int(baseline_hard["n_runs"]),
            },
            {
                "panel": "hard_mae",
                "method": "Proposed",
                "mean": float(proposed_hard["mean_K"]),
                "ci_low": float(proposed_hard["ci95_low_K"]),
                "ci_high": float(proposed_hard["ci95_high_K"]),
                "n_runs": int(proposed_hard["n_runs"]),
            },
            {
                "panel": "delta_mae",
                "subset": "Primary test",
                "method": "polyBERT audit",
                "delta_vs_baseline_k": float(baseline_primary["mean_K"]) - poly_primary_mean,
            },
            {
                "panel": "delta_mae",
                "subset": "Primary test",
                "method": "Proposed",
                "delta_vs_baseline_k": float(baseline_primary["mean_K"]) - float(proposed_primary["mean_K"]),
            },
            {
                "panel": "delta_mae",
                "subset": "External holdout",
                "method": "polyBERT audit",
                "delta_vs_baseline_k": float(baseline_external["mean_K"]) - poly_external_mean,
            },
            {
                "panel": "delta_mae",
                "subset": "External holdout",
                "method": "Proposed",
                "delta_vs_baseline_k": float(baseline_external["mean_K"]) - float(proposed_external["mean_K"]),
            },
        ]
    )
    src.to_csv(FIGURES / "final_fig2_source_data.csv", index=False)

    fig = plt.figure(figsize=(14.2, 5.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.15], wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    primary_order = ["Strongest baseline", "polyBERT audit", "Proposed"]
    hard_order = ["Strongest baseline", "Proposed"]
    colors = {
        "Strongest baseline": P["blue"],
        "polyBERT audit": P["teal"],
        "Proposed": P["orange"],
    }

    primary_panel = src[src["panel"] == "primary_mae"].copy()
    primary_panel["method"] = pd.Categorical(primary_panel["method"], primary_order, ordered=True)
    primary_panel = primary_panel.sort_values("method")
    y_primary = np.arange(len(primary_panel))[::-1]
    for yi, (_, row) in zip(y_primary, primary_panel.iterrows()):
        ax1.hlines(yi, float(row["ci_low"]), float(row["ci_high"]), color=colors[row["method"]], linewidth=3)
        ax1.scatter(float(row["mean"]), yi, s=85, color=colors[row["method"]], zorder=3)
        ax1.text(
            float(row["ci_high"]) + 0.12,
            yi,
            f"{float(row['mean']):.2f} K",
            va="center",
            fontsize=10.5,
            color=P["text"],
        )
    ax1.set_yticks(y_primary)
    ax1.set_yticklabels(primary_order, fontsize=11)
    ax1.set_xlabel("MAE (K)", fontsize=11)
    ax1.set_title("Primary-set MAE", fontsize=12, fontweight="bold", color=P["text"])
    ax1.grid(axis="x", linestyle="--", linewidth=0.9, color=P["grid"])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    hard_panel = src[src["panel"] == "hard_mae"].copy()
    hard_panel["method"] = pd.Categorical(hard_panel["method"], hard_order, ordered=True)
    hard_panel = hard_panel.sort_values("method")
    y_hard = np.arange(len(hard_panel))[::-1]
    for yi, (_, row) in zip(y_hard, hard_panel.iterrows()):
        ax2.hlines(yi, float(row["ci_low"]), float(row["ci_high"]), color=colors[row["method"]], linewidth=3)
        ax2.scatter(float(row["mean"]), yi, s=85, color=colors[row["method"]], zorder=3)
        ax2.text(
            float(row["ci_high"]) + 0.45,
            yi,
            f"{float(row['mean']):.2f} K",
            va="center",
            fontsize=10.5,
            color=P["text"],
        )
    ax2.set_yticks(y_hard)
    ax2.set_yticklabels(hard_order, fontsize=11)
    ax2.set_xlabel("MAE (K)", fontsize=11)
    ax2.set_title("Baseline-defined difficult subset MAE", fontsize=12, fontweight="bold", color=P["text"])
    ax2.grid(axis="x", linestyle="--", linewidth=0.9, color=P["grid"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax1.set_xlim(23.5, 25.0)
    ax2.set_xlim(24.0, 31.0)
    ax2.text(
        0.0,
        -0.19,
        "polyBERT difficult-subset audit is reported separately in Supplementary Table S3",
        transform=ax2.transAxes,
        fontsize=9.6,
        color=P["slate"],
    )

    delta = src[src["panel"] == "delta_mae"].copy()
    subset_order = ["Primary test", "External holdout"]
    delta["subset"] = pd.Categorical(delta["subset"], subset_order, ordered=True)
    delta["method"] = pd.Categorical(delta["method"], ["polyBERT audit", "Proposed"], ordered=True)
    delta = delta.sort_values(["subset", "method"])
    ypos = np.arange(len(subset_order))
    width = 0.26
    offsets = {"polyBERT audit": -0.16, "Proposed": 0.16}
    for method in ["polyBERT audit", "Proposed"]:
        rows = delta[delta["method"] == method]
        vals = [float(rows[rows["subset"] == subset]["delta_vs_baseline_k"].iloc[0]) for subset in subset_order]
        ax3.barh(
            ypos + offsets[method],
            vals,
            height=width,
            color=colors[method],
            alpha=0.95,
            label=method,
        )
        for yy, val in zip(ypos + offsets[method], vals):
            ax3.text(val + 0.03, yy, f"{val:.3f}", va="center", fontsize=10.5, color=P["text"])
    ax3.axvline(0, color=P["gray"], linewidth=1.2, linestyle="--")
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(subset_order, fontsize=11)
    ax3.set_xlabel("MAE reduction vs strongest baseline (K)", fontsize=11)
    ax3.set_title("Average-case deltas", fontsize=12.5, fontweight="bold", color=P["text"])
    ax3.grid(axis="x", linestyle="--", linewidth=0.9, color=P["grid"])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(frameon=False, fontsize=10, loc="lower right")

    fig.suptitle("Overall performance with difficult-subset context", fontsize=15, fontweight="bold", color=P["text"], y=0.98)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.16, wspace=0.34)
    fig.savefig(FIGURES / "final_fig2_performance.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "final_fig2_performance.pdf", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 2. Overall performance with difficult-subset context. Left, primary-set MAE for "
        "the strongest multimodal baseline, the matched five-seed polyBERT audit, and the "
        "proposed model. Center, hard-subgroup MAE on the fixed baseline-defined difficult subset "
        "for the 100-run strongest-baseline and proposed-model package only; the matched five-seed "
        "polyBERT difficult-subset audit is reported separately in Supplementary Table S3. "
        "Right, MAE reductions relative to the strongest baseline on the primary and external "
        "splits, showing a moderate primary gain for the proposed model, bounded external gain, "
        "and a slightly stronger external average for polyBERT. Error bars denote 95% confidence "
        "intervals across frozen runs or matched audit seeds where available."
    )
    (TABLES / "final_fig2_caption.md").write_text(caption + "\n", encoding="utf-8")


def build_figure3() -> None:
    stats_payload = hard_stats_payload()
    records = pd.DataFrame(stats_payload["per_seed_records"])
    records = records.rename(
        columns={
            "seed": "seed",
            "baseline_hard_mae_k": "baseline_hard_mae_k",
            "final_hard_mae_k": "proposed_hard_mae_k",
        }
    )
    records["hard_improvement_k"] = records["baseline_hard_mae_k"] - records["proposed_hard_mae_k"]
    records = records[["seed", "baseline_hard_mae_k", "proposed_hard_mae_k", "hard_improvement_k"]].copy()
    records.to_csv(FIGURES / "final_fig3_source_data.csv", index=False)

    manifest_baseline = manifest_row("MAE", "hard_subgroup", "strongest_baseline")
    manifest_proposed = manifest_row("MAE", "hard_subgroup", "final_model")
    stat_row = pd.read_csv(TABLES / "statistical_tests.csv")
    stat_row = stat_row[stat_row["label"] == "hard_subgroup_MAE"].iloc[0]

    improvements = records["hard_improvement_k"].to_numpy(dtype=float)
    positive = int((improvements > 0).sum())
    negative = int((improvements <= 0).sum())

    fig = plt.figure(figsize=(14.8, 5.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 0.95, 0.95], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    colors = np.where(records["hard_improvement_k"] > 0, P["green"], P["red"])
    ax1.scatter(
        records["baseline_hard_mae_k"],
        records["proposed_hard_mae_k"],
        c=colors,
        s=42,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    low = min(records["baseline_hard_mae_k"].min(), records["proposed_hard_mae_k"].min()) - 1.0
    high = max(records["baseline_hard_mae_k"].max(), records["proposed_hard_mae_k"].max()) + 1.0
    ax1.plot([low, high], [low, high], linestyle="--", linewidth=1.2, color=P["gray"])
    ax1.set_xlim(low, high)
    ax1.set_ylim(low, high)
    ax1.set_xlabel("Baseline hard-subgroup MAE (K)", fontsize=11)
    ax1.set_ylabel("Proposed hard-subgroup MAE (K)", fontsize=11)
    ax1.set_title("Run-wise paired comparison", fontsize=12.5, fontweight="bold", color=P["text"])
    ax1.text(
        0.03,
        0.97,
        f"{positive}/100 runs below the diagonal",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        color=P["text"],
    )
    ax1.grid(linestyle="--", color=P["grid"], linewidth=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bins = np.linspace(improvements.min() - 0.5, improvements.max() + 0.5, 18)
    ax2.hist(improvements, bins=bins, color=P["orange"], edgecolor="white", alpha=0.9)
    ax2.axvline(0.0, color=P["gray"], linestyle="--", linewidth=1.2)
    ax2.axvline(float(stat_row["mean_improvement_K"]), color=P["blue"], linewidth=2.0)
    ax2.set_xlabel("Hard-subgroup MAE reduction (K)", fontsize=11)
    ax2.set_ylabel("Run count", fontsize=11)
    ax2.set_title("Improvement distribution", fontsize=12.5, fontweight="bold", color=P["text"])
    ax2.text(
        0.03,
        0.97,
        f"Sign rate: {positive}/100 positive, {negative}/100 non-positive",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        color=P["text"],
    )
    ax2.grid(axis="y", linestyle="--", color=P["grid"], linewidth=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    summary = pd.DataFrame(
        [
            {
                "method": "Strongest baseline",
                "mean": float(manifest_baseline["mean_K"]),
                "ci_low": float(manifest_baseline["ci95_low_K"]),
                "ci_high": float(manifest_baseline["ci95_high_K"]),
                "color": P["blue"],
            },
            {
                "method": "Proposed",
                "mean": float(manifest_proposed["mean_K"]),
                "ci_low": float(manifest_proposed["ci95_low_K"]),
                "ci_high": float(manifest_proposed["ci95_high_K"]),
                "color": P["orange"],
            },
        ]
    )
    y_positions = np.array([1, 0])
    for yy, row in zip(y_positions, summary.itertuples(index=False)):
        ax3.hlines(yy, row.ci_low, row.ci_high, color=row.color, linewidth=3.0)
        ax3.scatter(row.mean, yy, s=90, color=row.color, zorder=3)
        ax3.text(row.ci_high + 0.35, yy, f"{row.mean:.2f} K", va="center", fontsize=10.5, color=P["text"])
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(summary["method"], fontsize=11)
    ax3.set_xlabel("MAE (K)", fontsize=11)
    ax3.set_title("Aggregate hard-subgroup MAE", fontsize=12.5, fontweight="bold", color=P["text"])
    ax3.grid(axis="x", linestyle="--", color=P["grid"], linewidth=0.9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.tick_params(axis="y", length=0)
    ax3.set_xlim(22.0, 31.5)
    ax3.text(
        0.03,
        0.16,
        (
            f"$\\Delta$MAE = {float(stat_row['mean_improvement_K']):.3f} K  "
            f"(95% bootstrap CI [{float(stat_row['bootstrap_ci95_low']):.3f}, {float(stat_row['bootstrap_ci95_high']):.3f}])\n"
            f"Paired $t$ p = {stat_row['paired_t_pvalue']}; Wilcoxon p = {stat_row['wilcoxon_pvalue']}"
        ),
        transform=ax3.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.8,
        color=P["text"],
    )

    fig.suptitle("Fixed baseline-defined hard-subgroup evidence", fontsize=15, fontweight="bold", color=P["text"], y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.16, wspace=0.28)
    fig.savefig(FIGURES / "final_fig3_hard_subgroup.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "final_fig3_hard_subgroup.pdf", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 3. Performance on the fixed baseline-defined hard subgroup. Left, paired run-wise "
        "hard-subgroup MAE comparison between the strongest baseline and the proposed model, with "
        "points below the diagonal indicating improvement. Center, distribution of run-wise hard-"
        "subgroup MAE reductions across the frozen 100-run package. Right, aggregate hard-subgroup "
        "MAE with 95% confidence intervals, showing a 4.224 K reduction and positive gain in 96 of "
        "100 frozen runs."
    )
    (TABLES / "final_fig3_caption.md").write_text(caption + "\n", encoding="utf-8")


def build_figure4() -> None:
    label_map = {
        "near_train": "Near-train",
        "within_domain": "Within-domain",
        "high_tg": "High Tg",
        "high_aromatic": "High aromaticity",
        "shifted_domain": "Shifted domain",
        "far_train": "Far-train",
    }
    order = [
        "near_train",
        "within_domain",
        "high_tg",
        "high_aromatic",
        "shifted_domain",
        "far_train",
    ]
    df = EXTERNAL.copy()
    df = df[df["subset"].isin(order)].copy()
    df["label"] = df["subset"].map(label_map)
    df["order"] = df["subset"].map({name: idx for idx, name in enumerate(order)})
    df = df.sort_values("order")
    df.to_csv(FIGURES / "final_fig4_source_data.csv", index=False)

    fig = plt.figure(figsize=(14.4, 6.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0], wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    y = np.arange(len(df))[::-1]
    colors = []
    for _, row in df.iterrows():
        if float(row["delta_ci95_low_K"]) > 0:
            colors.append(P["green"])
        elif float(row["delta_K"]) > 0:
            colors.append(P["gold"])
        else:
            colors.append(P["gray"])
    ax1.barh(y, df["delta_K"], color=colors, alpha=0.9)
    xerr = np.vstack(
        [
            np.maximum(0.0, df["delta_K"] - df["delta_ci95_low_K"]),
            np.maximum(0.0, df["delta_ci95_high_K"] - df["delta_K"]),
        ]
    )
    ax1.errorbar(df["delta_K"], y, xerr=xerr, fmt="none", color=P["text"], capsize=3, linewidth=1.2)
    ax1.axvline(0, color=P["gray"], linestyle="--", linewidth=1.2)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{label}\n(n={n})" for label, n in zip(df["label"], df["n_samples"])], fontsize=10.2)
    ax1.set_xlabel("MAE reduction vs strongest baseline (K)", fontsize=11)
    ax1.set_title("External subset deltas", fontsize=13, fontweight="bold", color=P["text"])
    ax1.grid(axis="x", linestyle="--", color=P["grid"], linewidth=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for yy, val in zip(y, df["delta_K"]):
        ax1.text(float(val) + 0.02, yy, f"{float(val):.3f}", va="center", fontsize=9.8, color=P["text"])

    bubble = ax2.scatter(
        df["mean_max_train_tanimoto"],
        df["delta_K"],
        s=df["n_samples"] * 3.2,
        c=df["mean_aromatic_rings"],
        cmap="YlOrBr",
        edgecolor="white",
        linewidth=0.9,
    )
    ax2.axhline(0, color=P["gray"], linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Mean maximum train Tanimoto", fontsize=11)
    ax2.set_ylabel("MAE reduction (K)", fontsize=11)
    ax2.set_title("Improvement concentrates in closer chemistry", fontsize=13, fontweight="bold", color=P["text"])
    ax2.grid(linestyle="--", color=P["grid"], linewidth=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    labels_to_show = {"near_train", "high_tg", "shifted_domain", "far_train", "high_aromatic"}
    label_offsets = {
        "near_train": (0.006, 0.012),
        "within_domain": (0.016, -0.010),
        "high_tg": (0.004, 0.01),
        "shifted_domain": (0.006, 0.01),
        "far_train": (0.006, 0.01),
        "high_aromatic": (0.004, 0.01),
    }
    for _, row in df.iterrows():
        if row["subset"] in labels_to_show:
            dx, dy = label_offsets[row["subset"]]
            ax2.text(
                float(row["mean_max_train_tanimoto"]) + dx,
                float(row["delta_K"]) + dy,
                row["label"],
                fontsize=10.2,
                color=P["text"],
            )
    cbar = fig.colorbar(bubble, ax=ax2, fraction=0.048, pad=0.03)
    cbar.set_label("Mean aromatic rings", fontsize=10.5)

    fig.suptitle("External holdout under moderate chemistry-space shift", fontsize=15, fontweight="bold", color=P["text"], y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12, wspace=0.18)
    fig.savefig(FIGURES / "final_fig4_external.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "final_fig4_external.pdf", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 4. External-holdout behavior under moderate chemistry-space shift. Left, stratified "
        "MAE reductions relative to the strongest baseline, with subgroup sizes shown beside each "
        "label and 95% confidence intervals from the frozen-model audit. Right, the same subsets "
        "projected against mean maximum train-set Tanimoto similarity, showing that larger gains are "
        "concentrated in chemically closer or within-domain subsets, whereas far-train and shifted "
        "subsets show only bounded benefit."
    )
    (TABLES / "final_fig4_caption.md").write_text(caption + "\n", encoding="utf-8")


def build_figure5() -> None:
    merged = CLUSTER_DESIGN.merge(CLUSTER_DIAG, left_on="Cluster", right_on="cluster", how="left")
    merged["display"] = merged["Cluster"].str.replace("_", " ").str.title()
    merged = merged.sort_values("MAE_delta_K", ascending=True)

    top10 = RANKING[RANKING["screening_rank"] <= 10].copy()
    counts: dict[str, int] = {}
    for family_text in top10["chemistry_family"]:
        for token in str(family_text).split(";"):
            token = token.strip()
            if token:
                counts[token] = counts.get(token, 0) + 1
    top_family_text = ", ".join(
        f"{name.replace('_', ' ')} ({count})"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if name in {"aromatic_dense", "ether_oxygen", "fluorinated"}
    )

    src = merged[
        [
            "Cluster",
            "N_test",
            "MAE_delta_K",
            "Interpretation",
            "Design_verdict",
            "mean_aromatic_rings",
            "mean_max_train_tanimoto",
            "n_samples",
            "delta_K",
        ]
    ].copy()
    src.to_csv(FIGURES / "final_fig6_source_data.csv", index=False)

    fig = plt.figure(figsize=(14.6, 8.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.95], hspace=0.28, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    colors = []
    for _, row in merged.iterrows():
        verdict = str(row["Design_verdict"])
        if verdict.startswith("RELIABLE"):
            colors.append(P["green"])
        elif verdict.startswith("CAUTION"):
            colors.append(P["red"])
        else:
            colors.append(P["gold"])
    y = np.arange(len(merged))
    ax1.barh(y, merged["MAE_delta_K"], color=colors, alpha=0.92)
    ax1.axvline(0, color=P["gray"], linestyle="--", linewidth=1.2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(
        [
            f"{d}{'*' if int(n) < 10 else ''}\n(n={int(n)})"
            for d, n in zip(merged["display"], merged["N_test"])
        ],
        fontsize=10.2,
    )
    ax1.set_xlabel("Family-level MAE reduction (K)", fontsize=11)
    ax1.set_title("Family-level model behavior", fontsize=13, fontweight="bold", color=P["text"])
    ax1.grid(axis="x", linestyle="--", color=P["grid"], linewidth=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for yy, val in zip(y, merged["MAE_delta_K"]):
        shift = 0.02 if float(val) >= 0 else -0.02
        ax1.text(float(val) + shift, yy, f"{float(val):.3f}", va="center", ha="left" if float(val) >= 0 else "right", fontsize=9.8)

    bubble = ax2.scatter(
        merged["mean_aromatic_rings"],
        merged["delta_K"],
        s=merged["n_samples"] * 3.0,
        c=merged["mean_max_train_tanimoto"],
        cmap="Blues",
        edgecolor="white",
        linewidth=0.9,
    )
    ax2.axhline(0, color=P["gray"], linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Mean aromatic rings per polymer", fontsize=11)
    ax2.set_ylabel("MAE reduction (K)", fontsize=11)
    ax2.set_title("Structural proxies vs family-level improvement", fontsize=13, fontweight="bold", color=P["text"])
    ax2.grid(linestyle="--", color=P["grid"], linewidth=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    label_offsets = {
        "Aromatic Dense": (0.05, 0.03),
        "Ether Oxygen": (0.05, 0.035),
        "Fluorinated": (0.05, 0.03),
        "Other": (0.08, -0.05),
        "Imide Like": (0.05, -0.03),
        "Amide": (0.05, 0.03),
        "Sulfone": (0.05, 0.03),
        "Ester Or Carbonate": (0.05, 0.03),
    }
    for _, row in merged.iterrows():
        dx, dy = label_offsets.get(row["display"], (0.05, 0.03))
        family_label = f"{row['display']}{'*' if int(row['n_samples']) < 10 else ''}"
        ax2.text(
            float(row["mean_aromatic_rings"]) + dx,
            float(row["delta_K"]) + dy,
            family_label,
            fontsize=9.6,
            color=P["gray"] if int(row["n_samples"]) < 10 else P["text"],
        )
    cbar = fig.colorbar(bubble, ax=ax2, fraction=0.048, pad=0.03)
    cbar.set_label("Mean maximum train Tanimoto", fontsize=10.5)
    ax2.text(
        0.99,
        0.02,
        "* n < 10, exploratory only",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        color=P["gray"],
    )

    ax3.set_axis_off()
    ax3.set_title("Bounded design implications from frozen-model analyses", fontsize=13, fontweight="bold", color=P["text"], loc="left", pad=8)
    cards = [
        (
            "Rigid aromatic backbones",
            "Aromatic-dense / fluorinated clusters:\n+0.52 to +0.61 K stable gain.",
            "Use for cautious retrospective\nhigh-Tg screening in covered chemistry.",
            P["green_soft"],
        ),
        (
            "Ether-containing segments",
            "Ether-oxygen family:\n+0.664 K across 175 samples.",
            "Multimodal context helps separate\nflexibility from local rigidity.",
            "#E8F4F2",
        ),
        (
            "Hydrogen-bonding families",
            "Amide and imide-like families:\nno stable improvement.",
            "Add H-bonding, crystallinity,\nor processing descriptors.",
            P["orange_soft"],
        ),
        (
            "Shifted chemistry space",
            "Far-train / shifted-domain subsets:\nonly 0.011 to 0.095 K gain.",
            "Keep deployment bounded outside\nwell-covered chemistry domains.",
            "#F4F6F8",
        ),
    ]
    x_positions = [0.01, 0.26, 0.51, 0.76]
    for (title, evidence, implication, facecolor), x0 in zip(cards, x_positions):
        box = FancyBboxPatch(
            (x0, 0.14),
            0.22,
            0.70,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor=P["navy"],
            facecolor=facecolor,
            transform=ax3.transAxes,
        )
        ax3.add_patch(box)
        ax3.text(x0 + 0.012, 0.80, title, transform=ax3.transAxes, fontsize=11.1, fontweight="bold", color=P["navy"], va="top")
        ax3.text(x0 + 0.012, 0.61, evidence, transform=ax3.transAxes, fontsize=9.7, color=P["text"], va="top", linespacing=1.3)
        ax3.text(x0 + 0.012, 0.35, implication, transform=ax3.transAxes, fontsize=9.6, color=P["slate"], va="top", linespacing=1.3)
    ax3.text(
        0.01,
        0.03,
        f"Top-ranked retrospective external candidates were enriched in {top_family_text}; this membership reflects predicted high Tg rather than improved prediction accuracy. This figure is retrospective and hypothesis-generating rather than evidence of material discovery.",
        transform=ax3.transAxes,
        fontsize=10.2,
        color=P["slate"],
    )

    fig.suptitle("Structure-property links and bounded design relevance", fontsize=15, fontweight="bold", color=P["text"], y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, hspace=0.32, wspace=0.22)
    fig.savefig(FIGURES / "final_fig6_design_insight.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "final_fig6_design_insight.pdf", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 5. Structure-property and bounded design relevance analysis. Top left, family-level "
        "MAE deltas relative to the strongest baseline, showing stable gains for aromatic-dense, "
        "fluorinated, and ether-oxygen families but not for amide, imide-like, or heterogeneous "
        "other polymers. Top right, family-level behavior projected against structural proxies, "
        "with bubble size proportional to family size and color indicating train-set proximity. "
        "Bottom, design-oriented implications derived from the frozen-model analyses and retrospective "
        "external ranking patterns. Families marked with an asterisk have n < 10 and are shown for "
        "transparency only as exploratory diagnostics. The figure is hypothesis-generating and does "
        "not claim de novo material discovery."
    )
    (TABLES / "final_fig6_caption.md").write_text(caption + "\n", encoding="utf-8")


def sync_outputs() -> None:
    figure_map = {
        "Figure_1": ("final_fig1_workflow.pdf", "final_fig1_workflow.png"),
        "Figure_2": ("final_fig2_performance.pdf", "final_fig2_performance.png"),
        "Figure_3": ("final_fig3_hard_subgroup.pdf", "final_fig3_hard_subgroup.png"),
        "Figure_4": ("final_fig4_external.pdf", "final_fig4_external.png"),
        "Figure_5": ("final_fig6_design_insight.pdf", "final_fig6_design_insight.png"),
    }
    for stem, (pdf_name, png_name) in figure_map.items():
        pdf_src = FIGURES / pdf_name
        png_src = FIGURES / png_name
        if pdf_src.exists():
            pdf_target = f"{stem}.pdf"
            shutil.copy2(pdf_src, MANUSCRIPT_FIGURES / pdf_target)
            shutil.copy2(pdf_src, SUBMISSION_FIGURES / pdf_target)
            shutil.copy2(pdf_src, SUBMISSION_SOURCE_FIGURES / pdf_target)
            if READY_FIGURES.exists():
                shutil.copy2(pdf_src, READY_FIGURES / pdf_target)
        if png_src.exists():
            png_target = f"{stem}.png"
            shutil.copy2(png_src, SUBMISSION_FIGURES / png_target)
            shutil.copy2(png_src, SUBMISSION_SOURCE_FIGURES / png_target)

    source_map = {
        FIGURES / "final_fig1_source_data.csv": "Figure_1_source.csv",
        FIGURES / "final_fig2_source_data.csv": "Figure_2_source.csv",
        FIGURES / "final_fig3_source_data.csv": "Figure_3_source.csv",
        FIGURES / "final_fig4_source_data.csv": "Figure_4_source.csv",
        FIGURES / "final_fig6_source_data.csv": "Figure_5_source.csv",
    }
    for src, name in source_map.items():
        if src.exists():
            shutil.copy2(src, SUBMISSION_SOURCE_DATA / name)
            shutil.copy2(src, SUBMISSION_SOURCE_DATA / name.replace("_source.csv", "_source_data.csv"))


def main() -> None:
    payload = poly_payload()
    build_figure1()
    build_figure2(payload)
    build_figure3()
    build_figure4()
    build_figure5()
    sync_outputs()
    print("Rebuilt main manuscript figures 1-5 and synced submission assets.")


if __name__ == "__main__":
    main()
