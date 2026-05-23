"""Curriculum-controlled Simple Concat baseline.

This experiment tests the strongest procedural alternative explanation for the
paper's hard-subgroup gains: can hard-sample-focused curriculum alone, without
MSCE/RCMF/MASD, reproduce the improvement?

Protocol:
1. Train a Stage-1 Simple Concat baseline.
2. Score training samples by absolute prediction error.
3. Upweight the top-20% hardest samples 3x with WeightedRandomSampler.
4. Fine-tune the same Simple Concat model for a short second phase.
5. Compare main MAE and hard-subgroup MAE before and after the curriculum step.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.full_train import (  # noqa: E402
    _to_device,
    diagnostic_config,
    full_config,
    load_artifacts,
    make_loader,
    prepare_seed_tensors,
    set_seed,
    stable_seed,
    standard_loss,
    train_standard_model,
)

DEFAULT_AUDIT_SEEDS = (15, 16, 17, 18, 19)
HARD_UPSAMPLE_WEIGHT = 3.0
CURRICULUM_LR = 5e-4
CURRICULUM_EPOCHS = 8
CURRICULUM_PATIENCE = 3
HARD_PERCENTILE = 0.80
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "exp" / "diagnostics" / "curriculum_controlled_baseline_results.json"


def parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _collect_errors(model: nn.Module, loader: DataLoader, seed_tensors: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    errors: list[np.ndarray] = []
    indices_out: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch)
            out = model(batch["graph"], batch["desc"], batch["ctx"])
            pred = (out["pred"].cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]).numpy().ravel()
            y = (batch["y"].cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]).numpy().ravel()
            errors.append(np.abs(pred - y))
            indices_out.append(batch["sample_index"].cpu().numpy().ravel())
    return np.concatenate(errors), np.concatenate(indices_out)


def _eval_hard_subgroup(model: nn.Module, loader: DataLoader, seed_tensors: dict) -> dict[str, float | int]:
    model.eval()
    preds, ys, conflicts, uncertainties = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch)
            out = model(batch["graph"], batch["desc"], batch["ctx"])
            pred = (out["pred"].cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]).numpy().ravel()
            y = (batch["y"].cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]).numpy().ravel()
            conflict = out["conflict_level"].cpu().numpy().ravel()
            uncertainty = out["uncertainty_level"].cpu().numpy().ravel()
            preds.append(pred)
            ys.append(y)
            conflicts.append(conflict)
            uncertainties.append(uncertainty)
    pred_a = np.concatenate(preds)
    y_a = np.concatenate(ys)
    conflict_a = np.concatenate(conflicts)
    uncertainty_a = np.concatenate(uncertainties)
    hard_score = conflict_a + 1.20 * uncertainty_a
    threshold = np.quantile(hard_score, HARD_PERCENTILE)
    hard_mask = hard_score >= threshold
    return {
        "main_mae": float(np.mean(np.abs(pred_a - y_a))),
        "hard_mae": float(np.mean(np.abs(pred_a[hard_mask] - y_a[hard_mask]))),
        "hard_n": int(hard_mask.sum()),
    }


def _build_hard_sample_weights(
    train_errors: np.ndarray,
    train_indices: np.ndarray,
    all_train_indices: Sequence[int],
) -> np.ndarray:
    idx_to_error = {int(idx): float(err) for idx, err in zip(train_indices, train_errors)}
    errors_ordered = np.array([idx_to_error.get(int(idx), 0.0) for idx in all_train_indices], dtype=np.float64)
    threshold = np.quantile(errors_ordered, HARD_PERCENTILE)
    return np.where(errors_ordered >= threshold, HARD_UPSAMPLE_WEIGHT, 1.0)


def _curriculum_finetune(
    model: nn.Module,
    train_loader_weighted: DataLoader,
    val_loader: DataLoader,
    seed_tensors: dict,
    config,
) -> nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=CURRICULUM_LR, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch_idx in range(CURRICULUM_EPOCHS):
        model.train()
        for batch in train_loader_weighted:
            batch = _to_device(batch)
            out = model(batch["graph"], batch["desc"], batch["ctx"])
            loss = standard_loss(
                model,
                out,
                batch,
                loss_fn,
                config,
                "simple_concat",
                epoch_idx,
                CURRICULUM_EPOCHS,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        val_preds, val_ys = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch)
                out = model(batch["graph"], batch["desc"], batch["ctx"])
                val_preds.append(
                    (out["pred"].cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]).numpy().ravel()
                )
                val_ys.append(
                    (batch["y"].cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]).numpy().ravel()
                )
        val_mae_k = float(np.mean(np.abs(np.concatenate(val_preds) - np.concatenate(val_ys))))
        if val_mae_k < best_val:
            best_val = val_mae_k
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= CURRICULUM_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_curriculum_controlled_baseline(
    *,
    audit_seeds: Sequence[int] = DEFAULT_AUDIT_SEEDS,
    use_full_config: bool = False,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    dataset, features, splits_raw = load_artifacts()
    splits = splits_raw.get("seeds", splits_raw)
    config = full_config() if use_full_config else diagnostic_config()
    audit_seeds = tuple(int(seed) for seed in audit_seeds)

    print(f"\n{'=' * 72}")
    print("Curriculum-Controlled Baseline Experiment")
    print(f"Seeds: {audit_seeds}  |  Config: {'full' if use_full_config else 'diagnostic'}")
    print(f"Hard-sample upsample weight: {HARD_UPSAMPLE_WEIGHT}x  |  Curriculum epochs: {CURRICULUM_EPOCHS}")
    print(f"{'=' * 72}\n")

    rows: list[dict[str, float | int]] = []
    for seed in audit_seeds:
        print(f"--- Seed {seed} ---")
        split_key = str(seed)
        if split_key not in splits:
            print(f"  [skip] no split for seed {seed}")
            continue
        split = splits[split_key]

        set_seed(stable_seed(seed, "simple_concat", "main"))
        seed_tensors = prepare_seed_tensors(features, split["train"], dataset)

        print("  Stage 1: training simple_concat ...")
        stage1_model = train_standard_model(
            mode="simple_concat",
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=stable_seed(seed, "simple_concat", "main"),
        )

        test_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
        val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)
        stage1_metrics = _eval_hard_subgroup(stage1_model, test_loader, seed_tensors)
        print(
            f"  Stage-1 baseline  -> main: {stage1_metrics['main_mae']:.4f} K,"
            f" hard: {stage1_metrics['hard_mae']:.4f} K"
        )

        train_loader_plain = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=False)
        train_errors, train_indices = _collect_errors(stage1_model, train_loader_plain, seed_tensors)
        hard_weights = _build_hard_sample_weights(train_errors, train_indices, split["train"])

        train_loader_weighted = make_loader(
            seed_tensors,
            split["train"],
            config.batch_size,
            shuffle=False,
            sample_weights=hard_weights,
            loader_seed=seed * 7 + 3,
        )
        print(f"  Curriculum fine-tune (hard-upsample 3x, {CURRICULUM_EPOCHS} epochs) ...")
        set_seed(seed * 31 + 7)
        curriculum_model = copy.deepcopy(stage1_model)
        curriculum_model = _curriculum_finetune(
            curriculum_model,
            train_loader_weighted,
            val_loader,
            seed_tensors,
            config,
        )

        curriculum_metrics = _eval_hard_subgroup(curriculum_model, test_loader, seed_tensors)
        print(
            f"  Curriculum-only   -> main: {curriculum_metrics['main_mae']:.4f} K,"
            f" hard: {curriculum_metrics['hard_mae']:.4f} K"
        )
        print(
            "  Delta hard MAE (curriculum - stage1):"
            f" {curriculum_metrics['hard_mae'] - stage1_metrics['hard_mae']:+.4f} K"
        )

        rows.append(
            {
                "seed": seed,
                "stage1_main_mae": float(stage1_metrics["main_mae"]),
                "stage1_hard_mae": float(stage1_metrics["hard_mae"]),
                "curriculum_main_mae": float(curriculum_metrics["main_mae"]),
                "curriculum_hard_mae": float(curriculum_metrics["hard_mae"]),
                "curriculum_hard_delta": float(curriculum_metrics["hard_mae"] - stage1_metrics["hard_mae"]),
            }
        )

    if not rows:
        print("No seeds completed.")
        return

    stage1_mains = [row["stage1_main_mae"] for row in rows]
    stage1_hards = [row["stage1_hard_mae"] for row in rows]
    cur_mains = [row["curriculum_main_mae"] for row in rows]
    cur_hards = [row["curriculum_hard_mae"] for row in rows]

    print(f"\n{'=' * 72}")
    print("SUMMARY (mean over completed seeds)")
    print(f"{'=' * 72}")
    print(
        f"  Simple Concat Stage-1 baseline   -> main: {np.mean(stage1_mains):.4f} K,"
        f" hard: {np.mean(stage1_hards):.4f} K"
    )
    print(
        f"  Curriculum-controlled (hard 3x)  -> main: {np.mean(cur_mains):.4f} K,"
        f" hard: {np.mean(cur_hards):.4f} K"
    )
    print(f"  Hard-subgroup delta (curriculum) : {np.mean(cur_hards) - np.mean(stage1_hards):+.4f} K")
    print("\n  Reference: full MSCE+RCMF+MASD model (current paper tranche) -> hard: ~27.90 K")
    print("  If curriculum hard MAE >> 27.90 K: architecture (not curriculum) drives the gain.")
    print("  If curriculum hard MAE ~= 27.90 K: curriculum alone may suffice; revise the claim.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage1_main_mae_mean": float(np.mean(stage1_mains)),
        "stage1_hard_mae_mean": float(np.mean(stage1_hards)),
        "curriculum_main_mae_mean": float(np.mean(cur_mains)),
        "curriculum_hard_mae_mean": float(np.mean(cur_hards)),
        "curriculum_hard_delta_mean": float(np.mean(cur_hards) - np.mean(stage1_hards)),
        "n_seeds": len(rows),
        "audit_seeds": list(audit_seeds),
    }
    output_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    md_path = output_path.with_suffix(".md")
    md_lines = [
        "# Curriculum-Controlled Baseline Summary",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in audit_seeds)}",
        f"- Stage-1 baseline main MAE: {summary['stage1_main_mae_mean']:.4f} K",
        f"- Stage-1 baseline hard MAE: {summary['stage1_hard_mae_mean']:.4f} K",
        f"- Curriculum-only main MAE: {summary['curriculum_main_mae_mean']:.4f} K",
        f"- Curriculum-only hard MAE: {summary['curriculum_hard_mae_mean']:.4f} K",
        f"- Hard MAE delta (curriculum - stage1): {summary['curriculum_hard_delta_mean']:+.4f} K",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nResults saved to: {output_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum-controlled Simple Concat baseline.")
    parser.add_argument("--seeds", type=str, default="15,16,17,18,19", help="Comma-separated audit seeds.")
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="Use the full training configuration instead of the diagnostic configuration.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="JSON path for detailed results. CSV and markdown are written beside it.",
    )
    args = parser.parse_args()
    run_curriculum_controlled_baseline(
        audit_seeds=parse_seed_list(args.seeds),
        use_full_config=args.full_config,
        output_path=Path(args.output_json),
    )
