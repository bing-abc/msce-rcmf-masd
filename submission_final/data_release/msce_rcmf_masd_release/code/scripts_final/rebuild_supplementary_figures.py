from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SUPP_DIR = ROOT / "submission_final" / "supplementary"
SUPP_FIG_DIR = SUPP_DIR / "supplementary_figures"
SOURCE_DATA_DIR = ROOT / "submission_final" / "source_data"
DATASET = ROOT / "data" / "dataset.csv"
BUNDLE = ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run_merged_raw" / "mainline_bundle.pt"
CLUSTER_DIAG = ROOT / "tables_md_revision" / "cluster_failure_diagnosis.csv"
CLUSTER_DESIGN = ROOT / "tables_md_revision" / "cluster_design_relevance.csv"
ABLATION = ROOT / "results_md_revision" / "rcmf_masd_ablation_results.csv"

for path in (SUPP_DIR, SUPP_FIG_DIR, SOURCE_DATA_DIR):
    path.mkdir(parents=True, exist_ok=True)


P = {
    "blue": "#2F6DAA",
    "green": "#2F855A",
    "orange": "#C65A1E",
    "red": "#B54742",
    "gold": "#B88A12",
    "gray": "#6B7280",
    "grid": "#D5DCE5",
    "text": "#1F2933",
    "light": "#F6F8FB",
}


def mol(smiles: str):
    return Chem.MolFromSmiles(str(smiles))


def fp(molecule):
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)


def save_outputs(fig: plt.Figure, stem: str, *, pdf: bool = True) -> None:
    png_path = SUPP_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    shutil.copy2(png_path, SUPP_FIG_DIR / png_path.name)
    if pdf:
        pdf_path = SUPP_DIR / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        shutil.copy2(pdf_path, SUPP_FIG_DIR / pdf_path.name)
    plt.close(fig)


def rebuild_s1() -> None:
    df = pd.read_csv(DATASET)
    external_df = df[df["role"] == "external_holdout"].copy()
    train_df = df[df["role"] == "primary_pool"].copy()

    train_fps = []
    for smiles in train_df["canonical_smiles"]:
        molecule = mol(smiles)
        if molecule is not None:
            train_fps.append(fp(molecule))

    rows = []
    for row in external_df.itertuples():
        molecule = mol(row.canonical_smiles)
        if molecule is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp(molecule), train_fps)
        rows.append(
            {
                "sample_index": int(row.Index),
                "tg_k": float(row.tg_k),
                "max_train_tanimoto": float(np.max(sims)),
                "mean_train_tanimoto": float(np.mean(sims)),
                "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(molecule)),
            }
        )
    source = pd.DataFrame(rows).sort_values("sample_index")
    source.to_csv(SOURCE_DATA_DIR / "Supplementary_Figure_S1_source.csv", index=False)

    mean_val = float(source["max_train_tanimoto"].mean())
    median_val = float(source["max_train_tanimoto"].median())

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.hist(source["max_train_tanimoto"], bins=30, color=P["green"], edgecolor="white", alpha=0.88)
    ax.axvline(mean_val, color=P["orange"], linewidth=2.0, label=f"Mean = {mean_val:.3f}")
    ax.axvline(median_val, color=P["gray"], linewidth=1.6, linestyle="--", label=f"Median = {median_val:.3f}")
    ax.set_xlabel("Maximum train-set Tanimoto similarity", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("External holdout proximity to the training chemistry space", fontsize=13, fontweight="bold", color=P["text"])
    ax.grid(axis="y", linestyle="--", color=P["grid"], linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    save_outputs(fig, "Supp_Figure_S1")


def rebuild_s2() -> None:
    diag = pd.read_csv(CLUSTER_DIAG)
    design = pd.read_csv(CLUSTER_DESIGN)
    merged = design.merge(diag, left_on="Cluster", right_on="cluster", how="left")
    merged["display"] = merged["Cluster"].str.replace("_", " ").str.title()
    merged["display_star"] = merged.apply(
        lambda row: f"{row['display']}{'*' if int(row['n_samples']) < 10 else ''}",
        axis=1,
    )
    merged = merged.sort_values("MAE_delta_K", ascending=True)
    merged.to_csv(SOURCE_DATA_DIR / "Supplementary_Figure_S2_source.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.8), gridspec_kw={"width_ratios": [1.05, 1.0]})

    bar_colors = []
    for _, row in merged.iterrows():
        if float(row["MAE_delta_K"]) > 0.15:
            bar_colors.append(P["green"])
        elif float(row["MAE_delta_K"]) >= 0:
            bar_colors.append(P["gold"])
        else:
            bar_colors.append(P["red"])
    y = np.arange(len(merged))
    ax1.barh(y, merged["MAE_delta_K"], color=bar_colors, alpha=0.9)
    ax1.axvline(0, color=P["gray"], linestyle="--", linewidth=1.2)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{label}\n(n={int(n)})" for label, n in zip(merged["display_star"], merged["n_samples"])], fontsize=10)
    ax1.set_xlabel("MAE reduction vs strongest baseline (K)", fontsize=11)
    ax1.set_title("Family-level delta", fontsize=12.5, fontweight="bold", color=P["text"])
    ax1.grid(axis="x", linestyle="--", color=P["grid"], linewidth=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bubble = ax2.scatter(
        merged["mean_max_train_tanimoto"],
        merged["mean_hetero_atoms"],
        s=merged["n_samples"] * 5.0,
        c=merged["delta_K"],
        cmap="RdYlGn",
        edgecolor="white",
        linewidth=0.9,
        vmin=min(-0.8, float(merged["delta_K"].min())),
        vmax=max(0.8, float(merged["delta_K"].max())),
    )
    ax2.set_xlabel("Mean maximum train Tanimoto", fontsize=11)
    ax2.set_ylabel("Mean hetero atom count", fontsize=11)
    ax2.set_title("Shift and structural complexity", fontsize=12.5, fontweight="bold", color=P["text"])
    ax2.grid(linestyle="--", color=P["grid"], linewidth=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for _, row in merged.iterrows():
        dx = 0.006
        dy = 0.10 if row["display"] in {"Other", "Fluorinated"} else 0.04
        ax2.text(
            float(row["mean_max_train_tanimoto"]) + dx,
            float(row["mean_hetero_atoms"]) + dy,
            row["display_star"],
            fontsize=9.6,
            color=P["gray"] if int(row["n_samples"]) < 10 else P["text"],
        )

    cbar = fig.colorbar(bubble, ax=ax2, fraction=0.048, pad=0.03)
    cbar.set_label("MAE reduction (K)", fontsize=10.5)
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

    fig.tight_layout()
    save_outputs(fig, "Supp_Figure_S2")


def rebuild_s3() -> None:
    df = pd.read_csv(ABLATION)
    df.to_csv(SOURCE_DATA_DIR / "Supplementary_Figure_S3_source.csv", index=False)

    order = [
        "MSCE only",
        "MSCE + RCMF",
        "MSCE + MASD (no RCMF)",
        "MSCE + RCMF + MASD",
    ]
    colors = [P["green"], P["gray"], P["orange"], "#8F2045"]
    panels = [
        ("primary_mae_mean_k", "primary_ci95_low_k", "primary_ci95_high_k", "Primary"),
        ("hard_mae_mean_k", "hard_ci95_low_k", "hard_ci95_high_k", "Hard subgroup"),
        ("external_mae_mean_k", "external_ci95_low_k", "external_ci95_high_k", "External"),
    ]

    plot_df = df[df["display_name"].isin(order)].copy()
    plot_df["display_name"] = pd.Categorical(plot_df["display_name"], order, ordered=True)
    plot_df = plot_df.sort_values("display_name")

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.4), sharey=False)
    for ax, (mean_col, low_col, high_col, title) in zip(axes, panels):
        means = plot_df[mean_col].to_numpy(dtype=float)
        lows = plot_df[low_col].to_numpy(dtype=float)
        highs = plot_df[high_col].to_numpy(dtype=float)
        xpos = np.arange(len(plot_df))
        ax.bar(xpos, means, color=colors, alpha=0.92, width=0.65)
        ax.errorbar(
            xpos,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            color=P["text"],
            capsize=4,
            linewidth=1.2,
        )
        ax.set_xticks(xpos)
        ax.set_xticklabels(order, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("MAE (K)", fontsize=11)
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=P["text"])
        ax.grid(axis="y", linestyle="--", color=P["grid"], linewidth=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Limited five-seed ablation audit: MASD carries the measurable correction",
        fontsize=14.5,
        fontweight="bold",
        color=P["text"],
        y=0.99,
    )
    fig.text(
        0.5,
        0.02,
        "MSCE + MASD (no RCMF) remains better than the full MSCE + RCMF + MASD variant on the same five-seed slice.",
        ha="center",
        fontsize=10.2,
        color=P["text"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_outputs(fig, "Supp_Figure_S3")


def main() -> None:
    rebuild_s1()
    rebuild_s2()
    rebuild_s3()
    print("Rebuilt supplementary figures S1-S3 and exported source data.")


if __name__ == "__main__":
    main()
