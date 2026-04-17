from __future__ import annotations

"""MASD internal design choices ablation.

Compares the full MSCE-RCMF-MASD chain against variants that ablate one design
element of MASD at a time.  Stages 1-3 (baseline, MSCE repair, RCMF repair) are
shared across all modes within a seed; only MASD training (stage 4) is rerun.

Modes tested:
  full          main_core_sci2_masd_final          (current paper model)
  no_competition  main_core_sci2_masd_no_competition  (uniform alpha, no sparsemax competition)
  no_sparse_alpha main_core_sci2_masd_v3_no_sparse_alpha (softmax instead of sparsemax)
  no_risk_alpha   main_core_sci2_masd_current_no_risk_adaptive_alpha
  no_mono_gate    main_core_sci2_masd_current_no_monotone_risk_gate

Run on the fixed five-seed tranche (15-19).
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.experiment_overrides import clear_experiment_overrides
from train.full_train import diagnostic_config, load_artifacts, make_loader, prepare_seed_tensors
from train.mspce_repair import ensure_multiscale_features, train_repair_student
from train.rcmf_min_repair import train_rcmf_external_focus_student, train_rcmf_student

from polyuatg_clean.scripts.masd_v3_run import (
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

DESIGN_MODES = [
    ("full",          CURRENT_MODE,                                          "risk-adaptive α + monotone gate [paper default]"),
    ("no_risk_alpha", "main_core_sci2_masd_current_no_risk_adaptive_alpha",  "no risk-adaptive α temperature"),
    ("no_mono_gate",  "main_core_sci2_masd_current_no_monotone_risk_gate",   "no monotone risk gate"),
]


def run_design_ablation_seed(
    *,
    seed: int,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    base_config: Any,
    epoch_log: list[float],
) -> list[dict[str, Any]]:
    clear_experiment_overrides()
    split = ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    primary_loader = make_loader(seed_tensors, split["test"], base_config.batch_size, shuffle=False)

    # Shared stages 1-3 (trained once per seed).
    _baseline_model, mspce_model = train_repair_student(
        split=split, seed_tensors=seed_tensors, config=base_config, seed=seed
    )
    minimal_rcmf = train_rcmf_student(
        split=split, seed_tensors=seed_tensors, config=base_config, seed=seed,
        repair_model=mspce_model,
    )
    current_rcmf = train_rcmf_external_focus_student(
        split=split, seed_tensors=seed_tensors, config=base_config, seed=seed,
        minimal_rcmf=minimal_rcmf,
    )

    rows: list[dict[str, Any]] = []
    for short_label, mode_str, description in DESIGN_MODES:
        model = train_masd_current_student(
            split=split, seed_tensors=seed_tensors, config=base_config, seed=seed,
            current_rcmf=current_rcmf, mode=mode_str,
            selection_policy="tailfix", epoch_log=epoch_log,
        )
        clean_metrics, _ = evaluate_stage(
            model, primary_loader, seed_tensors,
            variant="clean", noise_seed=seed * 1709 + hash(short_label) % 997,
            return_payload=False,
        )
        rows.append({
            "seed": int(seed),
            "result_group": "masd_design_ablation",
            "label": short_label,
            "mode": mode_str,
            "description": description,
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
        df.groupby(["label", "description"], as_index=False)[["primary_clean", "primary_hard_subgroup"]]
        .mean()
    )
    # Sort so "full" appears first.
    label_order = [m[0] for m in DESIGN_MODES]
    summary["_order"] = summary["label"].map({l: i for i, l in enumerate(label_order)})
    summary = summary.sort_values("_order").drop(columns="_order")
    summary.to_csv(run_dir / "masd_design_ablation_summary.csv", index=False)

    lines = [
        "# MASD Internal Design Ablation",
        "",
        "Fixed five-seed tranche: 15--19. Stages 1-3 shared per seed.",
        "",
        "| Label | Primary MAE (K) | Hard MAE (K) | Description |",
        "|:---|---:|---:|:---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.label} | {float(row.primary_clean):.4f} |"
            f" {float(row.primary_hard_subgroup):.4f} | {row.description} |"
        )
    (run_dir / "masd_design_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="MASD design choices ablation.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="masd_design_ablation")
    parser.add_argument("--seeds", type=str, default="15,16,17,18,19")
    args = parser.parse_args()

    enable_determinism(strict=False)
    ensure_multiscale_features()
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
        rows = run_design_ablation_seed(
            seed=seed, dataset=dataset, features=features, splits=splits,
            base_config=base_config, epoch_log=epoch_log,
        )
        all_rows.extend(rows)
        save_results_csv(run_dir, args.output_prefix, all_rows)

    write_summary(run_dir, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
