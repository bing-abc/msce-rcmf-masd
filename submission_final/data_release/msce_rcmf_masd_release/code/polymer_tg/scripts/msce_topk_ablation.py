from __future__ import annotations

"""MSCE top-k sensitivity ablation.

Varies the number of active context scales selected by the MSCE top-k gate
(k = 1, 2, 3, 4) on the fixed five-seed tranche (15-19).

k=1: only one scale active per sample (extreme selection).
k=2: two scales active.
k=3: three scales active (paper default).
k=4: all four scales active (no gated selection, full pooling).

The full pipeline (stages 1-4) is rerun for each k value because MSCE is trained
in Stage 2 and the gate behavior affects downstream RCMF and MASD training.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

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
    save_results_csv,
    train_masd_current_student,
)

K_VALUES = [1, 2, 3, 4]


def run_k_ablation_seed(
    *,
    seed: int,
    k_values: list[int],
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
    for k in k_values:
        # Legacy override key remains `mspce_top_k`; it controls the paper-facing MSCE top-k.
        set_experiment_overrides(mspce_top_k=k)

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
            variant="clean", noise_seed=seed * 1709 + k,
            return_payload=False,
        )
        rows.append({
            "seed": int(seed),
            "result_group": "msce_topk_ablation",
            "top_k": int(k),
            "description": "all scales (no selection)" if k == 4 else (
                "paper default" if k == 3 else f"top-{k} active"
            ),
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
        df.groupby(["top_k", "description"], as_index=False)[["primary_clean", "primary_hard_subgroup"]]
        .mean()
        .sort_values("top_k")
    )
    summary.to_csv(run_dir / "msce_topk_ablation_summary.csv", index=False)

    lines = [
        "# MSCE Top-k Sensitivity",
        "",
        "Fixed five-seed tranche: 15--19. Full pipeline rerun for each k.",
        "",
        "| k | Primary MAE (K) | Hard MAE (K) | Note |",
        "|:---:|---:|---:|:---|",
    ]
    for row in summary.itertuples(index=False):
        marker = " *" if int(row.top_k) == 3 else ""
        lines.append(
            f"| {int(row.top_k)} | {float(row.primary_clean):.4f} |"
            f" {float(row.primary_hard_subgroup):.4f} | {row.description}{marker} |"
        )
    lines += ["", "* paper default"]
    (run_dir / "msce_topk_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="MSCE top-k sensitivity ablation.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="msce_topk_ablation")
    parser.add_argument("--seeds", type=str, default="15,16,17,18,19")
    parser.add_argument("--k-values", type=str, default="1,2,3,4")
    args = parser.parse_args()

    enable_determinism(strict=False)
    ensure_msce_features()
    ensure_gpu()
    dataset, features, splits = load_artifacts()
    _ = build_chemistry_tag_lookup(dataset)
    base_config = diagnostic_config()
    seeds = parse_seed_list(args.seeds)
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_snapshot()
    epoch_log: list[float] = []
    all_rows: list[dict[str, Any]] = []

    for seed in seeds:
        rows = run_k_ablation_seed(
            seed=seed, k_values=k_values, dataset=dataset,
            features=features, splits=splits, base_config=base_config,
            epoch_log=epoch_log,
        )
        all_rows.extend(rows)
        save_results_csv(run_dir, args.output_prefix, all_rows)

    write_summary(run_dir, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
