from __future__ import annotations

"""Fasttrack scan over minimal-risk tweaks around the locked full model.

These runs are decision support for the paper package and are not alternative
headline results unless a candidate survives the multi-seed confirmation stage.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polyuatg_clean.scripts.masd_v3_run as masd_run
from polyuatg_clean.scripts.masd_v3_eval import paired_stats, reduction_interpretation, summary_stats, summarize_payload_metrics
from train.experiment_overrides import temporary_experiment_overrides
from train.full_train import diagnostic_config, load_artifacts, make_loader, prepare_seed_tensors
from train.mspce_repair import ensure_multiscale_features, train_repair_student


RUN_NAME = "pr_fasttrack_20260407"
RUN_DIR = ROOT / "outputs" / "exp" / "diagnostics" / RUN_NAME
CACHE_PATH = RUN_DIR / "cache.pt"
SCREEN_SEEDS = (0, 1)
FINAL_SEEDS = (0, 1, 2, 3, 4)

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "label": "current_full",
        "family": "current_full",
        "overrides": {},
    },
    {
        "label": "full_hard_reweight_a03",
        "family": "hard_reweight",
        "overrides": {
            "hard_reweight_alpha": 0.30,
        },
    },
    {
        "label": "full_thresholded_masd_b2_t07",
        "family": "thresholded_masd",
        "overrides": {
            "thresholded_masd_enabled": True,
            "thresholded_masd_bound_k": 2.0,
            "thresholded_masd_tau": 0.70,
            "thresholded_masd_gamma": 8.0,
        },
    },
    {
        "label": "full_combo_a03_b2_t07_T15_e1e4",
        "family": "combo",
        "overrides": {
            "hard_reweight_alpha": 0.30,
            "thresholded_masd_enabled": True,
            "thresholded_masd_bound_k": 2.0,
            "thresholded_masd_tau": 0.70,
            "thresholded_masd_gamma": 8.0,
            "rcmf_q_temperature": 1.5,
            "rcmf_entropy_lambda": 1.0e-4,
        },
    },
)


def fmt_ci(low: float, high: float, *, unit: str = "K") -> str:
    suffix = f" {unit}" if unit else ""
    return f"[{low:.4f}, {high:.4f}]{suffix}"


def fmt_mean_std(mean: float, std: float, *, unit: str = "K") -> str:
    suffix = f" {unit}" if unit else ""
    return f"{mean:.4f} +- {std:.4f}{suffix}"


def load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"mainline": {}, "masd_only": {}}
    return torch.load(CACHE_PATH, map_location="cpu")


def save_cache(cache: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)


def row_by_name(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    for row in rows:
        if str(row["model_name"]) == model_name:
            return row
    raise KeyError(model_name)


def sign_consistency(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    pos = int(np.sum(arr > 1e-8))
    neg = int(np.sum(arr < -1e-8))
    zero = int(arr.size - pos - neg)
    return f"{pos}/{arr.size} positive, {neg}/{arr.size} negative, {zero}/{arr.size} zero"


def model_payload_key(stage_key: str, split_key: str) -> str:
    mapping = {
        ("baseline", "primary"): "baseline_primary_clean",
        ("baseline", "external"): "baseline_external",
        ("mspce", "primary"): "mspce_primary_clean",
        ("mspce", "external"): "mspce_external",
        ("rcmf", "primary"): "rcmf_primary_clean",
        ("rcmf", "external"): "rcmf_external",
        ("full", "primary"): "masd_primary_clean",
        ("full", "external"): "masd_external",
    }
    return mapping[(stage_key, split_key)]


def cluster_delta(
    *,
    baseline_external_payload: dict[str, np.ndarray],
    candidate_external_payload: dict[str, np.ndarray],
    cluster_masks: dict[str, np.ndarray],
    cluster_name: str,
) -> float:
    row = masd_run.external_cluster_reduction_row(
        baseline_external_payload=baseline_external_payload,
        candidate_external_payload=candidate_external_payload,
        cluster_masks=cluster_masks,
    )
    return float(row.get(f"cluster_{cluster_name}_mae_reduction_k", float("nan")))


def safe_load_mainline(
    *,
    cache: dict[str, Any],
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    candidate: dict[str, Any],
    seed: int,
    epoch_log: list[float],
) -> dict[str, Any]:
    key = f"{candidate['label']}::seed{seed}"
    if key in cache["mainline"]:
        return cache["mainline"][key]
    with temporary_experiment_overrides(label=str(candidate["label"]), **candidate["overrides"]):
        rows, bundle = masd_run.run_mainline_seed(
            seed=seed,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            selection_policy="tailfix",
            epoch_log=epoch_log,
        )
    payload = {
        "label": candidate["label"],
        "family": candidate["family"],
        "seed": int(seed),
        "overrides": dict(candidate["overrides"]),
        "rows": rows,
        "bundle": bundle,
    }
    cache["mainline"][key] = payload
    save_cache(cache)
    return payload


def safe_load_masd_only(
    *,
    cache: dict[str, Any],
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    best_candidate: dict[str, Any],
    seed: int,
    epoch_log: list[float],
) -> dict[str, Any]:
    key = f"{best_candidate['label']}::masd_only::seed{seed}"
    if key in cache["masd_only"]:
        return cache["masd_only"][key]
    split = masd_run.ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    _baseline_model, mspce_model = train_repair_student(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=seed,
    )
    masd_overrides = {
        "hard_reweight_alpha": float(best_candidate["overrides"].get("hard_reweight_alpha", 0.0)),
        "thresholded_masd_enabled": bool(best_candidate["overrides"].get("thresholded_masd_enabled", False)),
        "thresholded_masd_bound_k": float(best_candidate["overrides"].get("thresholded_masd_bound_k", 0.0)),
        "thresholded_masd_tau": float(best_candidate["overrides"].get("thresholded_masd_tau", 0.70)),
        "thresholded_masd_gamma": float(best_candidate["overrides"].get("thresholded_masd_gamma", 8.0)),
        "disable_rcmf_anchor": True,
    }
    with temporary_experiment_overrides(label=f"{best_candidate['label']}_mspce_masd", **masd_overrides):
        model = masd_run.train_masd_current_student(
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=seed,
            current_rcmf=mspce_model,
            mode=masd_run.CURRENT_MODE,
            selection_policy="tailfix",
            epoch_log=epoch_log,
        )
    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)
    clean_metrics, clean_payload = masd_run.evaluate_stage(
        model,
        primary_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 809 + 1,
        return_payload=True,
    )
    external_metrics, external_payload = masd_run.evaluate_stage(
        model,
        external_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 809 + 3,
        return_payload=True,
    )
    payload = {
        "label": f"{best_candidate['label']}_mspce_masd",
        "seed": int(seed),
        "overrides": masd_overrides,
        "metrics": {
            "primary_clean": float(clean_metrics["mae_k"]),
            "primary_hard_subgroup": float(clean_metrics["hard_subgroup_mae_k"]),
            "external_holdout": float(external_metrics["mae_k"]),
            "external_hard_subgroup": float(external_metrics["hard_subgroup_mae_k"]),
        },
        "primary_payload": clean_payload,
        "external_payload": external_payload,
    }
    cache["masd_only"][key] = payload
    save_cache(cache)
    return payload


def screen_record(
    result: dict[str, Any],
    *,
    cluster_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows = result["rows"]
    bundle = result["bundle"]
    baseline_row = row_by_name(rows, "strongest_baseline")
    final_row = masd_run.current_stage_row(rows)
    other_delta = cluster_delta(
        baseline_external_payload=bundle["baseline_external"],
        candidate_external_payload=bundle["masd_external"],
        cluster_masks=cluster_masks,
        cluster_name="other",
    )
    return {
        "label": str(result["label"]),
        "family": str(result["family"]),
        "seed": int(result["seed"]),
        "main_delta": float(baseline_row["primary_clean"] - final_row["primary_clean"]),
        "hard_delta": float(baseline_row["primary_hard_subgroup"] - final_row["primary_hard_subgroup"]),
        "external_delta": float(baseline_row["external_holdout"] - final_row["external_holdout"]),
        "other_cluster_delta": float(other_delta),
    }


def summarize_screen(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    grouped = []
    for label, group in df.groupby("label", sort=False):
        grouped.append(
            {
                "label": str(label),
                "family": str(group["family"].iloc[0]),
                "num_seeds": int(group["seed"].nunique()),
                "main_delta_mean": float(group["main_delta"].mean()),
                "hard_delta_mean": float(group["hard_delta"].mean()),
                "external_delta_mean": float(group["external_delta"].mean()),
                "other_cluster_delta_mean": float(group["other_cluster_delta"].mean()),
                "main_sign_consistency": sign_consistency(group["main_delta"].tolist()),
                "hard_sign_consistency": sign_consistency(group["hard_delta"].tolist()),
                "external_sign_consistency": sign_consistency(group["external_delta"].tolist()),
                "other_sign_consistency": sign_consistency(group["other_cluster_delta"].tolist()),
            }
        )
    return pd.DataFrame(grouped)


def choose_best_candidate(screen_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    current_row = screen_df.loc[screen_df["label"] == "current_full"].iloc[0]
    ranked_rows: list[dict[str, Any]] = []
    for item in CANDIDATES:
        row = screen_df.loc[screen_df["label"] == item["label"]].iloc[0].to_dict()
        pass_all = bool(
            float(row["main_delta_mean"]) >= float(current_row["main_delta_mean"])
            and float(row["external_delta_mean"]) >= float(current_row["external_delta_mean"])
            and float(row["hard_delta_mean"]) >= float(current_row["hard_delta_mean"])
            and float(row["other_cluster_delta_mean"]) >= float(current_row["other_cluster_delta_mean"])
            and "negative, 0/" not in str(row["main_sign_consistency"])
            and "negative, 0/" not in str(row["external_sign_consistency"])
            and "negative, 0/" not in str(row["hard_sign_consistency"])
        )
        row["pass_all"] = pass_all
        ranked_rows.append(row)
    ranked = pd.DataFrame(ranked_rows).sort_values(
        by=["pass_all", "other_cluster_delta_mean", "external_delta_mean", "main_delta_mean", "hard_delta_mean"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    if bool(ranked["pass_all"].any()):
        best_label = str(ranked.iloc[0]["label"])
    else:
        best_label = "current_full"
    selected = next(item for item in CANDIDATES if item["label"] == best_label)
    return selected, ranked


def collect_stage_payloads(results: dict[int, dict[str, Any]], stage_key: str, split_key: str) -> list[dict[str, Any]]:
    return [results[seed]["bundle"][model_payload_key(stage_key, split_key)] for seed in sorted(results)]


def collect_stage_metric_values(results: dict[int, dict[str, Any]], stage_name: str, metric_key: str) -> list[float]:
    values: list[float] = []
    for seed in sorted(results):
        rows = results[seed]["rows"]
        if stage_name == "full":
            row = masd_run.current_stage_row(rows)
        elif stage_name == "baseline":
            row = row_by_name(rows, "strongest_baseline")
        elif stage_name == "mspce":
            row = row_by_name(rows, "strongest_baseline_plus_mspce")
        elif stage_name == "rcmf":
            row = row_by_name(rows, "strongest_baseline_plus_mspce_rcmf")
        else:
            raise KeyError(stage_name)
        values.append(float(row[metric_key]))
    return values


def paired_comparison_rows(
    *,
    current_results: dict[int, dict[str, Any]],
    best_results: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_payloads = collect_stage_payloads(current_results, "baseline", "primary")
    current_payloads = collect_stage_payloads(current_results, "full", "primary")
    best_payloads = collect_stage_payloads(best_results, "full", "primary")

    baseline_main = summarize_payload_metrics(baseline_payloads)
    current_main = summarize_payload_metrics(current_payloads)
    best_main = summarize_payload_metrics(best_payloads)

    current_hard = summary_stats(collect_stage_metric_values(current_results, "full", "primary_hard_subgroup"))
    best_hard = summary_stats(collect_stage_metric_values(best_results, "full", "primary_hard_subgroup"))
    baseline_hard = summary_stats(collect_stage_metric_values(current_results, "baseline", "primary_hard_subgroup"))

    current_external = summary_stats(collect_stage_metric_values(current_results, "full", "external_holdout"))
    best_external = summary_stats(collect_stage_metric_values(best_results, "full", "external_holdout"))
    baseline_external = summary_stats(collect_stage_metric_values(current_results, "baseline", "external_holdout"))

    current_vs_best_main_diff = np.asarray(current_main["mae_values"], dtype=np.float64) - np.asarray(best_main["mae_values"], dtype=np.float64)
    current_vs_best_hard_diff = np.asarray(collect_stage_metric_values(current_results, "full", "primary_hard_subgroup"), dtype=np.float64) - np.asarray(
        collect_stage_metric_values(best_results, "full", "primary_hard_subgroup"),
        dtype=np.float64,
    )
    current_vs_best_external_diff = np.asarray(collect_stage_metric_values(current_results, "full", "external_holdout"), dtype=np.float64) - np.asarray(
        collect_stage_metric_values(best_results, "full", "external_holdout"),
        dtype=np.float64,
    )

    current_vs_best_main = paired_stats(current_vs_best_main_diff)
    current_vs_best_hard = paired_stats(current_vs_best_hard_diff)
    current_vs_best_external = paired_stats(current_vs_best_external_diff)

    main_rows = pd.DataFrame(
        [
            {
                "row_type": "model",
                "name": "strongest_baseline",
                "n": int(baseline_main["mae"]["n"]),
                "mae_mean_k": float(baseline_main["mae"]["mean"]),
                "mae_std_k": float(baseline_main["mae"]["std"]),
                "mae_ci95_low_k": float(baseline_main["mae"]["ci95_low"]),
                "mae_ci95_high_k": float(baseline_main["mae"]["ci95_high"]),
                "rmse_mean_k": float(baseline_main["rmse"]["mean"]),
                "rmse_ci95_low_k": float(baseline_main["rmse"]["ci95_low"]),
                "rmse_ci95_high_k": float(baseline_main["rmse"]["ci95_high"]),
                "pearson_mean": float(baseline_main["pearson"]["mean"]),
                "pearson_ci95_low": float(baseline_main["pearson"]["ci95_low"]),
                "pearson_ci95_high": float(baseline_main["pearson"]["ci95_high"]),
            },
            {
                "row_type": "model",
                "name": "current_full",
                "n": int(current_main["mae"]["n"]),
                "mae_mean_k": float(current_main["mae"]["mean"]),
                "mae_std_k": float(current_main["mae"]["std"]),
                "mae_ci95_low_k": float(current_main["mae"]["ci95_low"]),
                "mae_ci95_high_k": float(current_main["mae"]["ci95_high"]),
                "rmse_mean_k": float(current_main["rmse"]["mean"]),
                "rmse_ci95_low_k": float(current_main["rmse"]["ci95_low"]),
                "rmse_ci95_high_k": float(current_main["rmse"]["ci95_high"]),
                "pearson_mean": float(current_main["pearson"]["mean"]),
                "pearson_ci95_low": float(current_main["pearson"]["ci95_low"]),
                "pearson_ci95_high": float(current_main["pearson"]["ci95_high"]),
            },
            {
                "row_type": "model",
                "name": "best_config",
                "n": int(best_main["mae"]["n"]),
                "mae_mean_k": float(best_main["mae"]["mean"]),
                "mae_std_k": float(best_main["mae"]["std"]),
                "mae_ci95_low_k": float(best_main["mae"]["ci95_low"]),
                "mae_ci95_high_k": float(best_main["mae"]["ci95_high"]),
                "rmse_mean_k": float(best_main["rmse"]["mean"]),
                "rmse_ci95_low_k": float(best_main["rmse"]["ci95_low"]),
                "rmse_ci95_high_k": float(best_main["rmse"]["ci95_high"]),
                "pearson_mean": float(best_main["pearson"]["mean"]),
                "pearson_ci95_low": float(best_main["pearson"]["ci95_low"]),
                "pearson_ci95_high": float(best_main["pearson"]["ci95_high"]),
            },
            {
                "row_type": "comparison",
                "name": "current_full_vs_best_config",
                "n": int(current_vs_best_main["n"]),
                "paired_delta_mean_k": float(current_vs_best_main["mean"]),
                "paired_delta_ci95_low_k": float(current_vs_best_main["ci95_low"]),
                "paired_delta_ci95_high_k": float(current_vs_best_main["ci95_high"]),
                "paired_t_pvalue": float(current_vs_best_main["t_pvalue"]),
                "permutation_pvalue": float(current_vs_best_main["perm_pvalue"]),
                "sign_consistency": sign_consistency(current_vs_best_main_diff.tolist()),
            },
        ]
    )

    hard_rows = pd.DataFrame(
        [
            {
                "row_type": "model",
                "name": "strongest_baseline",
                "n": int(baseline_hard["n"]),
                "mae_mean_k": float(baseline_hard["mean"]),
                "mae_std_k": float(baseline_hard["std"]),
                "mae_ci95_low_k": float(baseline_hard["ci95_low"]),
                "mae_ci95_high_k": float(baseline_hard["ci95_high"]),
            },
            {
                "row_type": "model",
                "name": "current_full",
                "n": int(current_hard["n"]),
                "mae_mean_k": float(current_hard["mean"]),
                "mae_std_k": float(current_hard["std"]),
                "mae_ci95_low_k": float(current_hard["ci95_low"]),
                "mae_ci95_high_k": float(current_hard["ci95_high"]),
            },
            {
                "row_type": "model",
                "name": "best_config",
                "n": int(best_hard["n"]),
                "mae_mean_k": float(best_hard["mean"]),
                "mae_std_k": float(best_hard["std"]),
                "mae_ci95_low_k": float(best_hard["ci95_low"]),
                "mae_ci95_high_k": float(best_hard["ci95_high"]),
            },
            {
                "row_type": "comparison",
                "name": "current_full_vs_best_config",
                "n": int(current_vs_best_hard["n"]),
                "paired_delta_mean_k": float(current_vs_best_hard["mean"]),
                "paired_delta_ci95_low_k": float(current_vs_best_hard["ci95_low"]),
                "paired_delta_ci95_high_k": float(current_vs_best_hard["ci95_high"]),
                "paired_t_pvalue": float(current_vs_best_hard["t_pvalue"]),
                "permutation_pvalue": float(current_vs_best_hard["perm_pvalue"]),
                "sign_consistency": sign_consistency(current_vs_best_hard_diff.tolist()),
            },
        ]
    )

    external_rows = pd.DataFrame(
        [
            {
                "row_type": "model",
                "name": "strongest_baseline",
                "n": int(baseline_external["n"]),
                "mae_mean_k": float(baseline_external["mean"]),
                "mae_std_k": float(baseline_external["std"]),
                "mae_ci95_low_k": float(baseline_external["ci95_low"]),
                "mae_ci95_high_k": float(baseline_external["ci95_high"]),
            },
            {
                "row_type": "model",
                "name": "current_full",
                "n": int(current_external["n"]),
                "mae_mean_k": float(current_external["mean"]),
                "mae_std_k": float(current_external["std"]),
                "mae_ci95_low_k": float(current_external["ci95_low"]),
                "mae_ci95_high_k": float(current_external["ci95_high"]),
            },
            {
                "row_type": "model",
                "name": "best_config",
                "n": int(best_external["n"]),
                "mae_mean_k": float(best_external["mean"]),
                "mae_std_k": float(best_external["std"]),
                "mae_ci95_low_k": float(best_external["ci95_low"]),
                "mae_ci95_high_k": float(best_external["ci95_high"]),
            },
            {
                "row_type": "comparison",
                "name": "current_full_vs_best_config",
                "n": int(current_vs_best_external["n"]),
                "paired_delta_mean_k": float(current_vs_best_external["mean"]),
                "paired_delta_ci95_low_k": float(current_vs_best_external["ci95_low"]),
                "paired_delta_ci95_high_k": float(current_vs_best_external["ci95_high"]),
                "paired_t_pvalue": float(current_vs_best_external["t_pvalue"]),
                "permutation_pvalue": float(current_vs_best_external["perm_pvalue"]),
                "sign_consistency": sign_consistency(current_vs_best_external_diff.tolist()),
            },
        ]
    )
    return main_rows, hard_rows, external_rows


def cluster_summary_rows(
    *,
    current_results: dict[int, dict[str, Any]],
    best_results: dict[int, dict[str, Any]],
    cluster_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_name in masd_run.CHEMISTRY_CLUSTER_ORDER:
        mask = cluster_masks.get(cluster_name)
        if mask is None or not bool(mask.any()):
            continue
        baseline_values: list[float] = []
        current_values: list[float] = []
        best_values: list[float] = []
        for seed in sorted(current_results):
            baseline_err = np.asarray(current_results[seed]["bundle"]["baseline_external"]["error"], dtype=np.float64).reshape(-1)
            current_err = np.asarray(current_results[seed]["bundle"]["masd_external"]["error"], dtype=np.float64).reshape(-1)
            best_err = np.asarray(best_results[seed]["bundle"]["masd_external"]["error"], dtype=np.float64).reshape(-1)
            baseline_values.append(float(baseline_err[mask].mean()))
            current_values.append(float(current_err[mask].mean()))
            best_values.append(float(best_err[mask].mean()))
        baseline_stats = summary_stats(baseline_values)
        current_stats = summary_stats(current_values)
        best_stats = summary_stats(best_values)
        baseline_to_best = paired_stats(np.asarray(baseline_values, dtype=np.float64) - np.asarray(best_values, dtype=np.float64))
        current_to_best = paired_stats(np.asarray(current_values, dtype=np.float64) - np.asarray(best_values, dtype=np.float64))
        rows.append(
            {
                "cluster_name": cluster_name,
                "sample_count": int(mask.sum()),
                "baseline_mae_mean_k": float(baseline_stats["mean"]),
                "current_full_mae_mean_k": float(current_stats["mean"]),
                "best_config_mae_mean_k": float(best_stats["mean"]),
                "baseline_to_best_delta_k": float(baseline_to_best["mean"]),
                "baseline_to_best_ci95_low_k": float(baseline_to_best["ci95_low"]),
                "baseline_to_best_ci95_high_k": float(baseline_to_best["ci95_high"]),
                "baseline_to_best_t_pvalue": float(baseline_to_best["t_pvalue"]),
                "baseline_to_best_perm_pvalue": float(baseline_to_best["perm_pvalue"]),
                "baseline_to_best_interpretation": reduction_interpretation(baseline_to_best),
                "current_to_best_delta_k": float(current_to_best["mean"]),
                "current_to_best_ci95_low_k": float(current_to_best["ci95_low"]),
                "current_to_best_ci95_high_k": float(current_to_best["ci95_high"]),
                "current_to_best_t_pvalue": float(current_to_best["t_pvalue"]),
                "current_to_best_perm_pvalue": float(current_to_best["perm_pvalue"]),
                "current_to_best_sign_consistency": sign_consistency((np.asarray(current_values) - np.asarray(best_values)).tolist()),
            }
        )
    return pd.DataFrame(rows)


def ablation_summary_rows(
    *,
    current_results: dict[int, dict[str, Any]],
    best_results: dict[int, dict[str, Any]],
    masd_only_results: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_primary = summarize_payload_metrics(collect_stage_payloads(current_results, "baseline", "primary"))
    baseline_hard = summary_stats(collect_stage_metric_values(current_results, "baseline", "primary_hard_subgroup"))
    baseline_external = summary_stats(collect_stage_metric_values(current_results, "baseline", "external_holdout"))

    mspce_primary = summarize_payload_metrics(collect_stage_payloads(current_results, "mspce", "primary"))
    mspce_hard = summary_stats(collect_stage_metric_values(current_results, "mspce", "primary_hard_subgroup"))
    mspce_external = summary_stats(collect_stage_metric_values(current_results, "mspce", "external_holdout"))

    rcmf_primary = summarize_payload_metrics(collect_stage_payloads(best_results, "rcmf", "primary"))
    rcmf_hard = summary_stats(collect_stage_metric_values(best_results, "rcmf", "primary_hard_subgroup"))
    rcmf_external = summary_stats(collect_stage_metric_values(best_results, "rcmf", "external_holdout"))

    masd_primary_payloads = [masd_only_results[seed]["primary_payload"] for seed in sorted(masd_only_results)]
    masd_primary = summarize_payload_metrics(masd_primary_payloads)
    masd_hard = summary_stats([masd_only_results[seed]["metrics"]["primary_hard_subgroup"] for seed in sorted(masd_only_results)])
    masd_external = summary_stats([masd_only_results[seed]["metrics"]["external_holdout"] for seed in sorted(masd_only_results)])

    full_primary = summarize_payload_metrics(collect_stage_payloads(best_results, "full", "primary"))
    full_hard = summary_stats(collect_stage_metric_values(best_results, "full", "primary_hard_subgroup"))
    full_external = summary_stats(collect_stage_metric_values(best_results, "full", "external_holdout"))

    stage_specs = [
        ("Baseline", baseline_primary, baseline_hard, baseline_external),
        ("+MSPCE", mspce_primary, mspce_hard, mspce_external),
        ("+MSPCE+RCMF", rcmf_primary, rcmf_hard, rcmf_external),
        ("+MSPCE+MASD", masd_primary, masd_hard, masd_external),
        ("Full", full_primary, full_hard, full_external),
    ]
    for name, primary_stats, hard_stats, external_stats in stage_specs:
        rows.append(
            {
                "model_name": name,
                "n": int(primary_stats["mae"]["n"]),
                "primary_mae_mean_k": float(primary_stats["mae"]["mean"]),
                "primary_ci95_low_k": float(primary_stats["mae"]["ci95_low"]),
                "primary_ci95_high_k": float(primary_stats["mae"]["ci95_high"]),
                "hard_mae_mean_k": float(hard_stats["mean"]),
                "hard_ci95_low_k": float(hard_stats["ci95_low"]),
                "hard_ci95_high_k": float(hard_stats["ci95_high"]),
                "external_mae_mean_k": float(external_stats["mean"]),
                "external_ci95_low_k": float(external_stats["ci95_low"]),
                "external_ci95_high_k": float(external_stats["ci95_high"]),
                "delta_vs_baseline_main_k": float(baseline_primary["mae"]["mean"] - primary_stats["mae"]["mean"]),
                "delta_vs_baseline_hard_k": float(baseline_hard["mean"] - hard_stats["mean"]),
                "delta_vs_baseline_external_k": float(baseline_external["mean"] - external_stats["mean"]),
                "delta_vs_mspce_main_k": float(mspce_primary["mae"]["mean"] - primary_stats["mae"]["mean"]),
                "delta_vs_mspce_hard_k": float(mspce_hard["mean"] - hard_stats["mean"]),
                "delta_vs_mspce_external_k": float(mspce_external["mean"] - external_stats["mean"]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    *,
    main_df: pd.DataFrame,
    hard_df: pd.DataFrame,
    external_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    best_candidate: dict[str, Any],
    screen_ranked: pd.DataFrame,
) -> None:
    main_cmp = main_df.loc[main_df["row_type"] == "comparison"].iloc[0]
    hard_cmp = hard_df.loc[hard_df["row_type"] == "comparison"].iloc[0]
    external_cmp = external_df.loc[external_df["row_type"] == "comparison"].iloc[0]
    other_row = cluster_df.loc[cluster_df["cluster_name"] == "other"].iloc[0]

    summary_main_lines = [
        "# Main test summary",
        "",
        f"- Current full vs best config paired delta: {float(main_cmp['paired_delta_mean_k']):+.4f} K with 95% CI {fmt_ci(float(main_cmp['paired_delta_ci95_low_k']), float(main_cmp['paired_delta_ci95_high_k']))}.",
        f"- Paired t-test p-value: {float(main_cmp['paired_t_pvalue']):.4g}; permutation p-value: {float(main_cmp['permutation_pvalue']):.4g}.",
        f"- Sign consistency across seeds: {str(main_cmp['sign_consistency'])}.",
    ]
    (RUN_DIR / "summary_main.md").write_text("\n".join(summary_main_lines) + "\n", encoding="utf-8")

    summary_hard_lines = [
        "# Hard subgroup summary",
        "",
        f"- Current full vs best config paired delta: {float(hard_cmp['paired_delta_mean_k']):+.4f} K with 95% CI {fmt_ci(float(hard_cmp['paired_delta_ci95_low_k']), float(hard_cmp['paired_delta_ci95_high_k']))}.",
        f"- Paired t-test p-value: {float(hard_cmp['paired_t_pvalue']):.4g}; permutation p-value: {float(hard_cmp['permutation_pvalue']):.4g}.",
        f"- Sign consistency across seeds: {str(hard_cmp['sign_consistency'])}.",
    ]
    (RUN_DIR / "summary_hard.md").write_text("\n".join(summary_hard_lines) + "\n", encoding="utf-8")

    summary_external_lines = [
        "# External holdout summary",
        "",
        f"- Current full vs best config paired delta: {float(external_cmp['paired_delta_mean_k']):+.4f} K with 95% CI {fmt_ci(float(external_cmp['paired_delta_ci95_low_k']), float(external_cmp['paired_delta_ci95_high_k']))}.",
        f"- Paired t-test p-value: {float(external_cmp['paired_t_pvalue']):.4g}; permutation p-value: {float(external_cmp['permutation_pvalue']):.4g}.",
        f"- Sign consistency across seeds: {str(external_cmp['sign_consistency'])}.",
    ]
    (RUN_DIR / "summary_external.md").write_text("\n".join(summary_external_lines) + "\n", encoding="utf-8")

    cluster_lines = ["# Cluster summary", ""]
    for cluster_name in masd_run.CHEMISTRY_CLUSTER_ORDER:
        row = cluster_df.loc[cluster_df["cluster_name"] == cluster_name]
        if row.empty:
            cluster_lines.append(f"- `{cluster_name}`: not sampled in the external holdout.")
            continue
        item = row.iloc[0]
        low = float(item["baseline_to_best_ci95_low_k"])
        high = float(item["baseline_to_best_ci95_high_k"])
        if low > 0.0:
            status = "stable positive improvement"
        elif float(item["baseline_to_best_delta_k"]) > 0.0:
            status = "mean positive but CI crosses zero"
        else:
            status = "negative or unsupported"
        cluster_lines.append(
            f"- `{cluster_name}`: baseline to best delta {float(item['baseline_to_best_delta_k']):+.4f} K, 95% CI {fmt_ci(low, high)}, status = {status}; current full to best delta {float(item['current_to_best_delta_k']):+.4f} K."
        )
    cluster_lines.append("")
    cluster_lines.append(
        f"- `other`: repair status = {'repaired or non-negative' if float(other_row['baseline_to_best_delta_k']) >= 0.0 else 'still negative vs baseline'}, current full to best delta {float(other_row['current_to_best_delta_k']):+.4f} K."
    )
    (RUN_DIR / "summary_cluster.md").write_text("\n".join(cluster_lines) + "\n", encoding="utf-8")

    rcmf_row = ablation_df.loc[ablation_df["model_name"] == "+MSPCE+RCMF"].iloc[0]
    masd_row = ablation_df.loc[ablation_df["model_name"] == "+MSPCE+MASD"].iloc[0]
    full_row = ablation_df.loc[ablation_df["model_name"] == "Full"].iloc[0]
    ablation_lines = [
        "# Ablation summary",
        "",
        f"- `+MSPCE+RCMF` vs `+MSPCE`: main {float(rcmf_row['delta_vs_mspce_main_k']):+.4f} K, hard {float(rcmf_row['delta_vs_mspce_hard_k']):+.4f} K, external {float(rcmf_row['delta_vs_mspce_external_k']):+.4f} K.",
        f"- `+MSPCE+MASD` vs `+MSPCE`: main {float(masd_row['delta_vs_mspce_main_k']):+.4f} K, hard {float(masd_row['delta_vs_mspce_hard_k']):+.4f} K, external {float(masd_row['delta_vs_mspce_external_k']):+.4f} K.",
        f"- `Full` vs `+MSPCE`: main {float(full_row['delta_vs_mspce_main_k']):+.4f} K, hard {float(full_row['delta_vs_mspce_hard_k']):+.4f} K, external {float(full_row['delta_vs_mspce_external_k']):+.4f} K.",
    ]
    (RUN_DIR / "summary_ablation.md").write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")

    best_lines = [
        "# Best configuration",
        "",
        f"- Selected label: `{best_candidate['label']}`.",
        f"- Candidate family: `{best_candidate['family']}`.",
        f"- Overrides: `{json.dumps(best_candidate['overrides'], ensure_ascii=True, sort_keys=True)}`.",
        "",
        "## Two-seed screen ranking",
        "",
    ]
    for _, row in screen_ranked.iterrows():
        best_lines.append(
            f"- `{row['label']}`: pass_all={bool(row['pass_all'])}, main={float(row['main_delta_mean']):+.4f} K, hard={float(row['hard_delta_mean']):+.4f} K, external={float(row['external_delta_mean']):+.4f} K, other={float(row['other_cluster_delta_mean']):+.4f} K."
        )
    (RUN_DIR / "summary_best_config.md").write_text("\n".join(best_lines) + "\n", encoding="utf-8")

    rcmf_support = "mixed or weak"
    if float(rcmf_row["delta_vs_mspce_main_k"]) > 0.0 and float(rcmf_row["delta_vs_mspce_external_k"]) > 0.0:
        rcmf_support = "supported on main and external"
    elif float(rcmf_row["delta_vs_mspce_hard_k"]) > 0.0:
        rcmf_support = "main signal weak but hard-case signal present"
    advisor_lines = [
        "# Advisor summary",
        "",
        f"- Most effective modification: `{best_candidate['label']}` with overrides `{json.dumps(best_candidate['overrides'], ensure_ascii=True, sort_keys=True)}`.",
        f"- External holdout vs current full: {float(external_cmp['paired_delta_mean_k']):+.4f} K, 95% CI {fmt_ci(float(external_cmp['paired_delta_ci95_low_k']), float(external_cmp['paired_delta_ci95_high_k']))}.",
        f"- Hard subgroup vs current full: {float(hard_cmp['paired_delta_mean_k']):+.4f} K, 95% CI {fmt_ci(float(hard_cmp['paired_delta_ci95_low_k']), float(hard_cmp['paired_delta_ci95_high_k']))}.",
        f"- Other cluster vs strongest baseline: {float(other_row['baseline_to_best_delta_k']):+.4f} K; vs current full: {float(other_row['current_to_best_delta_k']):+.4f} K.",
        f"- RCMF ablation support: {rcmf_support}.",
        f"- This version is {'more suitable' if float(external_cmp['paired_delta_mean_k']) >= 0.0 and float(other_row['current_to_best_delta_k']) >= 0.0 else 'not clearly more suitable'} to start writing as a PR-facing result package than the original current full control.",
        "- Remaining risks that cannot be packaged as solved: average-gain headroom is still limited, hard subgroup remains the primary sales point, and `other` may still fail to become a stable positive cluster even if it narrows.",
    ]
    (RUN_DIR / "summary_for_advisor.md").write_text("\n".join(advisor_lines) + "\n", encoding="utf-8")


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    ensure_multiscale_features()
    gpu_payload = masd_run.ensure_gpu()
    dataset, features, splits = load_artifacts()
    masd_run.CHEMISTRY_TAG_LOOKUP = masd_run.build_chemistry_tag_lookup(dataset)
    cluster_masks = masd_run.external_cluster_masks(dataset)
    config = diagnostic_config()
    epoch_log: list[float] = []

    screen_records: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for seed in SCREEN_SEEDS:
            result = safe_load_mainline(
                cache=cache,
                dataset=dataset,
                features=features,
                splits=splits,
                config=config,
                candidate=candidate,
                seed=seed,
                epoch_log=epoch_log,
            )
            screen_records.append(screen_record(result, cluster_masks=cluster_masks))
    screen_seed_df = pd.DataFrame(screen_records)
    screen_seed_df.to_csv(RUN_DIR / "screen_seed_results.csv", index=False)
    screen_df = summarize_screen(screen_records)
    screen_df.to_csv(RUN_DIR / "screen_summary.csv", index=False)
    best_candidate, screen_ranked = choose_best_candidate(screen_df)
    screen_ranked.to_csv(RUN_DIR / "screen_ranking.csv", index=False)

    current_candidate = next(item for item in CANDIDATES if item["label"] == "current_full")
    current_results = {
        seed: safe_load_mainline(
            cache=cache,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            candidate=current_candidate,
            seed=seed,
            epoch_log=epoch_log,
        )
        for seed in FINAL_SEEDS
    }
    best_results = {
        seed: safe_load_mainline(
            cache=cache,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            candidate=best_candidate,
            seed=seed,
            epoch_log=epoch_log,
        )
        for seed in FINAL_SEEDS
    }
    masd_only_results = {
        seed: safe_load_masd_only(
            cache=cache,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            best_candidate=best_candidate,
            seed=seed,
            epoch_log=epoch_log,
        )
        for seed in FINAL_SEEDS
    }

    main_df, hard_df, external_df = paired_comparison_rows(
        current_results=current_results,
        best_results=best_results,
    )
    cluster_df = cluster_summary_rows(
        current_results=current_results,
        best_results=best_results,
        cluster_masks=cluster_masks,
    )
    ablation_df = ablation_summary_rows(
        current_results=current_results,
        best_results=best_results,
        masd_only_results=masd_only_results,
    )

    main_df.to_csv(RUN_DIR / "summary_main.csv", index=False)
    hard_df.to_csv(RUN_DIR / "summary_hard.csv", index=False)
    external_df.to_csv(RUN_DIR / "summary_external.csv", index=False)
    cluster_df.to_csv(RUN_DIR / "summary_cluster.csv", index=False)
    ablation_df.to_csv(RUN_DIR / "summary_ablation.csv", index=False)

    write_markdown(
        main_df=main_df,
        hard_df=hard_df,
        external_df=external_df,
        cluster_df=cluster_df,
        ablation_df=ablation_df,
        best_candidate=best_candidate,
        screen_ranked=screen_ranked,
    )

    run_summary = {
        "run_name": RUN_NAME,
        "gpu_payload": gpu_payload,
        "best_candidate": best_candidate,
        "screen_ranked": screen_ranked.to_dict(orient="records"),
        "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
    }
    (RUN_DIR / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

