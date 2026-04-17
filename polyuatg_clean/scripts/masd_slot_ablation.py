from __future__ import annotations

"""Run the MASD slot-count sensitivity study on the fixed ablation tranche."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


SLOT_PRIOR_LABELS = {
    2: "[+,-]",
    3: "[+,+,-]",
    4: "[+,+,-,-]",
    6: "[+,+,+,-,-,-]",
}


def parse_slot_count_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    for value in values:
        if value not in SLOT_PRIOR_LABELS:
            raise ValueError(f"unsupported slot count {value}; expected one of {sorted(SLOT_PRIOR_LABELS)}")
    return values


def run_slot_ablation_seed(
    *,
    seed: int,
    slot_counts: list[int],
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    base_config: Any,
    epoch_log: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    _baseline_model, mspce_model = train_repair_student(split=split, seed_tensors=seed_tensors, config=base_config, seed=seed)
    minimal_rcmf = train_rcmf_student(split=split, seed_tensors=seed_tensors, config=base_config, seed=seed, repair_model=mspce_model)
    current_rcmf = train_rcmf_external_focus_student(
        split=split,
        seed_tensors=seed_tensors,
        config=base_config,
        seed=seed,
        minimal_rcmf=minimal_rcmf,
    )
    primary_loader = make_loader(seed_tensors, split["test"], base_config.batch_size, shuffle=False)

    rows: list[dict[str, Any]] = []
    slot_bundles: list[dict[str, Any]] = []
    for slot_count in slot_counts:
        slot_config = replace(base_config, masd_slot_count=int(slot_count))
        model = train_masd_current_student(
            split=split,
            seed_tensors=seed_tensors,
            config=slot_config,
            seed=seed,
            current_rcmf=current_rcmf,
            mode=CURRENT_MODE,
            selection_policy="tailfix",
            epoch_log=epoch_log,
        )
        clean_metrics, _ = evaluate_stage(
            model,
            primary_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 1709 + slot_count,
            return_payload=False,
        )
        checkpoint_meta = getattr(model, "_masd_checkpoint_meta", {})
        row = {
            "seed": int(seed),
            "result_group": "slot_ablation",
            "slot_count": int(slot_count),
            "sign_prior": SLOT_PRIOR_LABELS[int(slot_count)],
            "primary_clean": float(clean_metrics["mae_k"]),
            "primary_hard_subgroup": float(clean_metrics["hard_subgroup_mae_k"]),
            "hard_mask_rate": float(clean_metrics["hard_mask_rate"]),
            "selection_policy": "tailfix",
            "selected_stage": str(checkpoint_meta.get("selected_stage", checkpoint_meta.get("stage", ""))),
            "selected_epoch": int(checkpoint_meta.get("selected_epoch", checkpoint_meta.get("epoch", -1))),
            "pass_flag": True,
        }
        rows.append(row)
        slot_bundles.append(
            {
                "slot_count": int(slot_count),
                "sign_prior": SLOT_PRIOR_LABELS[int(slot_count)],
                "checkpoint_meta": checkpoint_meta,
                "checkpoint_candidates": getattr(model, "_masd_checkpoint_candidates", []),
            }
        )
    return rows, {"seed": int(seed), "slot_bundles": slot_bundles}


def write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    summary = (
        df.groupby(["slot_count", "sign_prior"], as_index=False)[["primary_clean", "primary_hard_subgroup"]]
        .mean()
        .sort_values("slot_count")
    )
    summary_path = run_dir / "slot_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    lines = [
        "# MASD Slot Count Sensitivity",
        "",
        "Fixed five-seed tranche: 15--19.",
        "",
        "| Slots | Sign prior | Primary MAE (K) | Hard MAE (K) |",
        "|---:|:---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.slot_count)} | `{row.sign_prior}` | {float(row.primary_clean):.4f} | {float(row.primary_hard_subgroup):.4f} |"
        )
    lines.extend(
        [
            "",
            "For 6 slots, the additional positive and negative slots are initialized from the mean of their sibling proxy families, so correction capacity increases without adding new polymer-semantic labels.",
        ]
    )
    (run_dir / "slot_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MASD slot-count sensitivity on the fixed five-seed tranche.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="masd_slot_ablation")
    parser.add_argument("--seeds", type=str, default="15,16,17,18,19")
    parser.add_argument("--slot-counts", type=str, default="2,3,4,6")
    args = parser.parse_args()

    enable_determinism(strict=False)
    ensure_multiscale_features()
    gpu_payload = ensure_gpu()
    dataset, features, splits = load_artifacts()
    _ = build_chemistry_tag_lookup(dataset)
    base_config = diagnostic_config()
    seeds = parse_seed_list(args.seeds)
    slot_counts = parse_slot_count_list(args.slot_counts)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.output_prefix)
    locked = lock_snapshot()
    epoch_log: list[float] = []
    all_rows: list[dict[str, Any]] = []
    seed_bundles: list[dict[str, Any]] = []

    for seed in seeds:
        rows, seed_bundle = run_slot_ablation_seed(
            seed=seed,
            slot_counts=slot_counts,
            dataset=dataset,
            features=features,
            splits=splits,
            base_config=base_config,
            epoch_log=epoch_log,
        )
        all_rows.extend(rows)
        seed_bundles.append(seed_bundle)
        save_results_csv(run_dir, output_prefix, all_rows)
        save_bundle(
            run_dir,
            "slot_ablation_bundle",
            {
                "gpu_payload": gpu_payload,
                "rows": all_rows,
                "seed_bundles": seed_bundles,
                "seeds": seeds,
                "slot_counts": slot_counts,
                "locked_snapshot": locked,
                "output_prefix": output_prefix,
                "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
            },
        )

    write_summary(run_dir, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
