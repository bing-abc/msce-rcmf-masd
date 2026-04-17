from __future__ import annotations

"""Follow-up thresholded-MASD scan after the main fasttrack screen."""

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
import polyuatg_clean.scripts.pr_fasttrack_run as base_runner
from train.experiment_overrides import temporary_experiment_overrides
from train.full_train import diagnostic_config, load_artifacts
from train.mspce_repair import ensure_multiscale_features


RUN_NAME = "pr_threshold_scan_20260407"
RUN_DIR = ROOT / "outputs" / "exp" / "diagnostics" / RUN_NAME
CACHE_PATH = RUN_DIR / "cache.pt"
BASE_CACHE_PATH = base_runner.RUN_DIR / "cache.pt"
SCREEN_SEEDS = (0, 1)
FINAL_SEEDS = (0, 1, 2, 3, 4)

THRESHOLD_CANDIDATES: tuple[dict[str, Any], ...] = (
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
        "label": "full_thresholded_masd_b2_t06",
        "family": "thresholded_masd",
        "overrides": {
            "thresholded_masd_enabled": True,
            "thresholded_masd_bound_k": 2.0,
            "thresholded_masd_tau": 0.60,
            "thresholded_masd_gamma": 8.0,
        },
    },
    {
        "label": "full_thresholded_masd_b4_t07",
        "family": "thresholded_masd",
        "overrides": {
            "thresholded_masd_enabled": True,
            "thresholded_masd_bound_k": 4.0,
            "thresholded_masd_tau": 0.70,
            "thresholded_masd_gamma": 8.0,
        },
    },
    {
        "label": "full_thresholded_masd_b4_t06",
        "family": "thresholded_masd",
        "overrides": {
            "thresholded_masd_enabled": True,
            "thresholded_masd_bound_k": 4.0,
            "thresholded_masd_tau": 0.60,
            "thresholded_masd_gamma": 8.0,
        },
    },
)


def load_local_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"mainline": {}}
    return torch.load(CACHE_PATH, map_location="cpu")


def save_local_cache(cache: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)


def load_base_cache() -> dict[str, Any]:
    if not BASE_CACHE_PATH.exists():
        return {"mainline": {}, "masd_only": {}}
    return torch.load(BASE_CACHE_PATH, map_location="cpu")


def get_current_full_results(base_cache: dict[str, Any]) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for seed in FINAL_SEEDS:
        key = f"current_full::seed{seed}"
        if key not in base_cache["mainline"]:
            raise RuntimeError(f"missing cached current_full result for seed {seed} in {BASE_CACHE_PATH}")
        results[int(seed)] = base_cache["mainline"][key]
    return results


def get_or_train_threshold_candidate(
    *,
    local_cache: dict[str, Any],
    base_cache: dict[str, Any],
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    candidate: dict[str, Any],
    seed: int,
    epoch_log: list[float],
) -> dict[str, Any]:
    key = f"{candidate['label']}::seed{seed}"
    if key in local_cache["mainline"]:
        return local_cache["mainline"][key]
    if key in base_cache.get("mainline", {}):
        payload = base_cache["mainline"][key]
        local_cache["mainline"][key] = payload
        save_local_cache(local_cache)
        return payload
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
    local_cache["mainline"][key] = payload
    save_local_cache(local_cache)
    return payload


def summarize_screen(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    grouped = []
    for label, group in df.groupby("label", sort=False):
        grouped.append(
            {
                "label": str(label),
                "num_seeds": int(group["seed"].nunique()),
                "main_delta_mean": float(group["main_delta"].mean()),
                "hard_delta_mean": float(group["hard_delta"].mean()),
                "external_delta_mean": float(group["external_delta"].mean()),
                "other_cluster_delta_mean": float(group["other_cluster_delta"].mean()),
                "main_sign_consistency": base_runner.sign_consistency(group["main_delta"].tolist()),
                "hard_sign_consistency": base_runner.sign_consistency(group["hard_delta"].tolist()),
                "external_sign_consistency": base_runner.sign_consistency(group["external_delta"].tolist()),
                "other_sign_consistency": base_runner.sign_consistency(group["other_cluster_delta"].tolist()),
            }
        )
    return pd.DataFrame(grouped)


def choose_threshold_candidate(screen_df: pd.DataFrame) -> dict[str, Any]:
    ranked = screen_df.sort_values(
        by=["external_delta_mean", "other_cluster_delta_mean", "main_delta_mean", "hard_delta_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    label = str(ranked.iloc[0]["label"])
    return next(item for item in THRESHOLD_CANDIDATES if item["label"] == label)


def write_summary(
    *,
    screen_df: pd.DataFrame,
    selected_candidate: dict[str, Any],
    main_df: pd.DataFrame,
    hard_df: pd.DataFrame,
    external_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    main_cmp = main_df.loc[main_df["row_type"] == "comparison"].iloc[0]
    hard_cmp = hard_df.loc[hard_df["row_type"] == "comparison"].iloc[0]
    external_cmp = external_df.loc[external_df["row_type"] == "comparison"].iloc[0]
    other_row = cluster_df.loc[cluster_df["cluster_name"] == "other"].iloc[0]
    lines = [
        "# Thresholded MASD follow-up",
        "",
        "This round only extends the `full + thresholded_MASD` family because it was the only family with positive two-seed external signal in the first fast track.",
        "",
        "## Two-seed scan",
        "",
    ]
    for _, row in screen_df.sort_values(by="external_delta_mean", ascending=False).iterrows():
        lines.append(
            f"- `{row['label']}`: main={float(row['main_delta_mean']):+.4f} K, hard={float(row['hard_delta_mean']):+.4f} K, external={float(row['external_delta_mean']):+.4f} K, other={float(row['other_cluster_delta_mean']):+.4f} K."
        )
    lines.extend(
        [
            "",
            f"## Five-seed confirmation: `{selected_candidate['label']}`",
            "",
            f"- Main vs `current_full`: {float(main_cmp['paired_delta_mean_k']):+.4f} K, 95% CI {base_runner.fmt_ci(float(main_cmp['paired_delta_ci95_low_k']), float(main_cmp['paired_delta_ci95_high_k']))}.",
            f"- Hard vs `current_full`: {float(hard_cmp['paired_delta_mean_k']):+.4f} K, 95% CI {base_runner.fmt_ci(float(hard_cmp['paired_delta_ci95_low_k']), float(hard_cmp['paired_delta_ci95_high_k']))}.",
            f"- External vs `current_full`: {float(external_cmp['paired_delta_mean_k']):+.4f} K, 95% CI {base_runner.fmt_ci(float(external_cmp['paired_delta_ci95_low_k']), float(external_cmp['paired_delta_ci95_high_k']))}.",
            f"- `other` vs strongest baseline: {float(other_row['baseline_to_best_delta_k']):+.4f} K; vs `current_full`: {float(other_row['current_to_best_delta_k']):+.4f} K.",
        ]
    )
    (RUN_DIR / "threshold_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ensure_multiscale_features()
    masd_run.ensure_gpu()
    dataset, features, splits = load_artifacts()
    masd_run.CHEMISTRY_TAG_LOOKUP = masd_run.build_chemistry_tag_lookup(dataset)
    cluster_masks = masd_run.external_cluster_masks(dataset)
    config = diagnostic_config()
    local_cache = load_local_cache()
    base_cache = load_base_cache()
    epoch_log: list[float] = []

    current_full_results = get_current_full_results(base_cache)

    screen_records: list[dict[str, Any]] = []
    for candidate in THRESHOLD_CANDIDATES:
        for seed in SCREEN_SEEDS:
            result = get_or_train_threshold_candidate(
                local_cache=local_cache,
                base_cache=base_cache,
                dataset=dataset,
                features=features,
                splits=splits,
                config=config,
                candidate=candidate,
                seed=seed,
                epoch_log=epoch_log,
            )
            screen_records.append(base_runner.screen_record(result, cluster_masks=cluster_masks))
    screen_df = summarize_screen(screen_records)
    screen_df.to_csv(RUN_DIR / "threshold_screen_summary.csv", index=False)
    selected_candidate = choose_threshold_candidate(screen_df)

    selected_results = {
        seed: get_or_train_threshold_candidate(
            local_cache=local_cache,
            base_cache=base_cache,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            candidate=selected_candidate,
            seed=seed,
            epoch_log=epoch_log,
        )
        for seed in FINAL_SEEDS
    }

    main_df, hard_df, external_df = base_runner.paired_comparison_rows(
        current_results=current_full_results,
        best_results=selected_results,
    )
    cluster_df = base_runner.cluster_summary_rows(
        current_results=current_full_results,
        best_results=selected_results,
        cluster_masks=cluster_masks,
    )

    main_df.to_csv(RUN_DIR / "threshold_summary_main.csv", index=False)
    hard_df.to_csv(RUN_DIR / "threshold_summary_hard.csv", index=False)
    external_df.to_csv(RUN_DIR / "threshold_summary_external.csv", index=False)
    cluster_df.to_csv(RUN_DIR / "threshold_summary_cluster.csv", index=False)

    write_summary(
        screen_df=screen_df,
        selected_candidate=selected_candidate,
        main_df=main_df,
        hard_df=hard_df,
        external_df=external_df,
        cluster_df=cluster_df,
    )
    (RUN_DIR / "threshold_run_summary.json").write_text(
        json.dumps(
            {
                "selected_candidate": selected_candidate,
                "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

