from __future__ import annotations

"""RCMF anchor contribution ablation.

Compares the full MSCE-RCMF-MASD chain against the same chain with RCMF's
anchor contribution disabled (MASD anchors on the MSCE prediction instead of
the RCMF-conditioned prediction).  This isolates the value of reliability-
conditioned fusion as a bridge for the MASD correction step.

Run on the fixed five-seed tranche (15-19).
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.experiment_overrides import clear_experiment_overrides, set_experiment_overrides
from train.full_train import diagnostic_config, load_artifacts, make_loader, prepare_seed_tensors
from train.msce_stage import ensure_msce_features, train_msce_stage
from train.rcmf_stage import train_rcmf_external_focus_stage, train_rcmf_stage

from polymer_tg.scripts.mainline_run import (
    CURRENT_MODE,
    DIAG_ROOT,
    build_chemistry_tag_lookup,
    enable_determinism,
    ensure_gpu,
    ensure_protocol_split,
    evaluate_stage,
    lock_snapshot,
    parse_seed_list,
    save_bundle,
    save_results_csv,
    train_masd_current_student,
)

CONDITIONS = [
    {"label": "rcmf_anchor_enabled",  "disable_rcmf_anchor": False},
    {"label": "rcmf_anchor_disabled", "disable_rcmf_anchor": True},
]


def run_anchor_ablation_seed(
    *,
    seed: int,
    conditions: list[dict[str, Any]],
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    base_config: Any,
    epoch_log: list[float],
) -> list[dict[str, Any]]:
    split = ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    primary_loader = make_loader(seed_tensors, split["test"], base_config.batch_size, shuffle=False)

    rows: list[dict[str, Any]] = []
    for cond in conditions:
        label = cond["label"]
        disable = cond["disable_rcmf_anchor"]
        # Set override BEFORE all build_model calls inside training stages.
        set_experiment_overrides(disable_rcmf_anchor=disable)

        _baseline_model, msce_model = train_msce_stage(
            split=split, seed_tensors=seed_tensors, config=base_config, repeat_id=seed
        )
        minimal_rcmf = train_rcmf_stage(
            split=split, seed_tensors=seed_tensors, config=base_config, repeat_id=seed,
            repair_model=msce_model,
        )
        current_rcmf = train_rcmf_external_focus_stage(
            split=split, seed_tensors=seed_tensors, config=base_config, repeat_id=seed,
            minimal_rcmf=minimal_rcmf,
        )
        model = train_masd_current_student(
            split=split, seed_tensors=seed_tensors, config=base_config, seed=seed,
            current_rcmf=current_rcmf, mode=CURRENT_MODE,
            selection_policy="tailfix", epoch_log=epoch_log,
        )
        clear_experiment_overrides()

        clean_metrics, _ = evaluate_stage(
            model, primary_loader, seed_tensors,
            variant="clean", noise_seed=seed * 1709 + (0 if not disable else 1),
            return_payload=False,
        )
        rows.append({
            "seed": int(seed),
            "result_group": "rcmf_anchor_ablation",
            "condition": label,
            "disable_rcmf_anchor": bool(disable),
            "primary_clean": float(clean_metrics["mae_k"]),
            "primary_hard_subgroup": float(clean_metrics["hard_subgroup_mae_k"]),
            "hard_mask_rate": float(clean_metrics["hard_mask_rate"]),
        })
    return rows


def write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    summary = (
        df.groupby(["condition", "disable_rcmf_anchor"], as_index=False)[
            ["primary_clean", "primary_hard_subgroup"]
        ]
        .mean()
    )
    summary_path = run_dir / "rcmf_anchor_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    lines = [
        "# RCMF Anchor Contribution Ablation",
        "",
        "Fixed five-seed tranche: 15--19.",
        "",
        "| Condition | Disable RCMF anchor | Primary MAE (K) | Hard MAE (K) |",
        "|:---|:---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.disable_rcmf_anchor} |"
            f" {float(row.primary_clean):.4f} | {float(row.primary_hard_subgroup):.4f} |"
        )
    lines += [
        "",
        "RCMF anchor enabled = MASD anchors on rcmf_min_pred (full chain).",
        "RCMF anchor disabled = MASD anchors on msce_stage_pred (RCMF conditioning bypassed for MASD).",
    ]
    (run_dir / "rcmf_anchor_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="RCMF anchor contribution ablation.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="rcmf_anchor_ablation")
    parser.add_argument("--seeds", type=str, default="15,16,17,18,19")
    args = parser.parse_args()

    enable_determinism(strict=False)
    ensure_msce_features()
    ensure_gpu()
    dataset, features, splits = load_artifacts()
    _ = build_chemistry_tag_lookup(dataset)
    base_config = diagnostic_config()
    seeds = parse_seed_list(args.seeds)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_snapshot()
    epoch_log: list[float] = []
    all_rows: list[dict[str, Any]] = []

    for seed in seeds:
        rows = run_anchor_ablation_seed(
            seed=seed, conditions=CONDITIONS, dataset=dataset,
            features=features, splits=splits, base_config=base_config,
            epoch_log=epoch_log,
        )
        all_rows.extend(rows)
        save_results_csv(run_dir, args.output_prefix, all_rows)

    write_summary(run_dir, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
