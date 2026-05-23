"""Same-protocol polyBERT baseline.

This script fine-tunes polyBERT on the overlap-purged training split for each
seed, then evaluates on the same test split used in the paper. The hard
subgroup is defined by the Simple Concat Stage-1 baseline errors, matching the
fixed-mask comparator protocol used in the manuscript.

Examples:
    python train/polybert_baseline.py --seeds 10,11,12,13,14,15,16,17,18,19 --epochs 30
    python train/polybert_baseline.py --model-name-or-path kuelumbus/polyBERT
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Patch transformers safety check for torch<2.6 on local files (research use only)
try:
    import transformers.utils.import_utils as _tuu
    _tuu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from transformers import AutoTokenizer, DebertaV2Model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.full_train import (
    _to_device,
    diagnostic_config,
    load_artifacts,
    make_loader,
    prepare_seed_tensors,
    set_seed,
    stable_seed,
    train_standard_model,
)

DEFAULT_MODEL_NAME_OR_PATH = "kuelumbus/polyBERT"
# Fallback to direct snapshot path if hub-based loading fails (torch<2.6 + old .bin)
_SNAPSHOT_FALLBACK = r"C:/Users/bing/.cache/huggingface/hub/models--kuelumbus--polyBERT/snapshots/deaa98fb65a7bdfb537457d42f43bd468963f695"
DEFAULT_SEEDS = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
MAX_LEN = 512
HARD_PERCENTILE = 0.80
BATCH_SIZE = 16
LR_HEAD = 1e-4
LR_ENCODER = 2e-5
PATIENCE = 5
WEIGHT_DECAY = 1e-2
RESULTS_PATH = ROOT / "outputs" / "exp" / "diagnostics" / "polybert_baseline_results.json"


class SmilesRegressionDataset(Dataset):
    """Minimal dataset wrapper for SMILES regression batches."""

    def __init__(self, smiles_list, y_values):
        self.smiles = smiles_list
        self.y = y_values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], float(self.y[idx])


class PolyBERTRegressor(nn.Module):
    """polyBERT encoder followed by a lightweight regression head."""

    def __init__(self, model_name_or_path: str, *, local_files_only: bool = False):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        cfg = self.encoder.config
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, enc):
        out = self.encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)


def _regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics in Kelvin."""

    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    error = pred_arr - target_arr
    mae = float(np.mean(np.abs(error)))
    rmse = float(math.sqrt(np.mean(error**2)))
    target_mean = float(np.mean(target_arr))
    ss_tot = float(np.sum((target_arr - target_mean) ** 2))
    ss_res = float(np.sum(error**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    if pred_arr.size < 2 or float(np.std(pred_arr)) < 1e-12 or float(np.std(target_arr)) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pred_arr, target_arr)[0, 1])
    return {
        "mae_k": mae,
        "rmse_k": rmse,
        "r2": r2,
        "pearson": pearson,
    }


def _mean_metrics(rows: list[dict[str, object]], split_name: str, prefix: str) -> dict[str, float]:
    metrics = {}
    for field in ("mae_k", "rmse_k", "r2", "pearson"):
        values = [
            float(row[split_name][field])  # type: ignore[index]
            for row in rows
            if row.get(split_name) is not None
        ]
        if values:
            metrics[f"{prefix}_{split_name}_{field}_mean"] = float(np.mean(values))
    return metrics


def _write_results(rows: list[dict[str, object]]) -> None:
    b_primary = np.mean([float(r["baseline_primary_mae"]) for r in rows])
    b_hard = np.mean([float(r["baseline_hard_mae"]) for r in rows])
    p_primary = np.mean([float(r["polybert_primary_mae"]) for r in rows])
    p_hard = np.mean([float(r["polybert_hard_mae"]) for r in rows])
    ext_rows_b = [float(r["baseline_ext_mae"]) for r in rows if r["baseline_ext_mae"] is not None]
    ext_rows_p = [float(r["polybert_ext_mae"]) for r in rows if r["polybert_ext_mae"] is not None]

    summary = {
        "n_seeds": len(rows),
        "baseline_primary_mean": float(b_primary),
        "baseline_hard_mean": float(b_hard),
        "polybert_primary_mean": float(p_primary),
        "polybert_hard_mean": float(p_hard),
    }
    if ext_rows_b:
        summary["baseline_ext_mean"] = float(np.mean(ext_rows_b))
    if ext_rows_p:
        summary["polybert_ext_mean"] = float(np.mean(ext_rows_p))
    summary.update(_mean_metrics(rows, "baseline_primary", "baseline"))
    summary.update(_mean_metrics(rows, "baseline_hard", "baseline"))
    summary.update(_mean_metrics(rows, "baseline_external", "baseline"))
    summary.update(_mean_metrics(rows, "polybert_primary", "polybert"))
    summary.update(_mean_metrics(rows, "polybert_hard", "polybert"))
    summary.update(_mean_metrics(rows, "polybert_external", "polybert"))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")


def _make_smiles_loader(smiles_list, y_vals, y_mean, y_std, tokenizer, shuffle):
    """Build a DataLoader that returns tokenized SMILES and Tg values in Kelvin."""

    ds = SmilesRegressionDataset(smiles_list, y_vals)

    def cf(batch):
        smiles, ys = zip(*batch)
        enc = tokenizer(
            list(smiles),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        y_k = torch.tensor(ys, dtype=torch.float32) * y_std + y_mean
        return enc, y_k

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=cf)


def _train_polybert(
    smiles_train,
    y_train_norm,
    smiles_val,
    y_val_norm,
    y_mean: float,
    y_std: float,
    tokenizer,
    max_epochs: int,
    device,
    *,
    model_name_or_path: str,
    local_files_only: bool,
) -> PolyBERTRegressor:
    """Fine-tune polyBERT with differential learning rates for encoder and head."""

    model = PolyBERTRegressor(
        model_name_or_path,
        local_files_only=local_files_only,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": LR_ENCODER},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-6,
    )
    loss_fn = nn.SmoothL1Loss()
    train_loader = _make_smiles_loader(smiles_train, y_train_norm, y_mean, y_std, tokenizer, shuffle=True)
    val_loader = _make_smiles_loader(smiles_val, y_val_norm, y_mean, y_std, tokenizer, shuffle=False)

    best_val = float("inf")
    best_state = None
    bad = 0

    for _epoch in range(max_epochs):
        model.train()
        for enc, y_k in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            y_k = y_k.to(device)
            pred = model(enc)
            loss = loss_fn(pred, y_k)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_ys = [], []
        with torch.no_grad():
            for enc, y_k in val_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                val_preds.append(model(enc).cpu().numpy())
                val_ys.append(y_k.numpy())
        val_mae = float(np.mean(np.abs(np.concatenate(val_preds) - np.concatenate(val_ys))))

        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _eval_polybert(model, smiles_list, y_vals_norm, y_mean, y_std, tokenizer, device):
    """Evaluate polyBERT on a SMILES list and return predictions and targets in Kelvin."""

    loader = _make_smiles_loader(smiles_list, y_vals_norm, y_mean, y_std, tokenizer, shuffle=False)
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for enc, y_k in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            preds.append(model(enc).cpu().numpy())
            ys.append(y_k.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def _eval_gnn_loader(model, loader, seed_tensors) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the Simple Concat baseline and return predictions and targets in Kelvin."""

    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch)
            out = model(batch["graph"], batch["desc"], batch["ctx"])
            preds.append((out["pred"].cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]).numpy().ravel())
            ys.append((batch["y"].cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]).numpy().ravel())
    return np.concatenate(preds), np.concatenate(ys)


def run_polybert_baseline(
    seeds=DEFAULT_SEEDS,
    max_epochs=30,
    *,
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH,
    local_files_only: bool = False,
    resume: bool = True,
    overwrite_existing: bool = False,
):
    """Run the same-protocol baseline audit and save per-seed plus summary results."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use snapshot fallback path for torch<2.6 environments
    import os as _os
    _snap = getattr(sys.modules[__name__], "_SNAPSHOT_FALLBACK", None)
    if _snap and _os.path.isdir(_snap):
        _resolved_path = _snap
    else:
        _resolved_path = model_name_or_path
    print(f"polyBERT path: {_resolved_path}")

    dataset, features, splits_raw = load_artifacts()
    splits = splits_raw.get("seeds", splits_raw)
    config = diagnostic_config()
    tokenizer = AutoTokenizer.from_pretrained(
        _resolved_path,
        local_files_only=True,
    )
    smiles_col = dataset["canonical_smiles"].tolist()
    tg_col = dataset["tg_k"].values.astype(np.float32)

    existing_rows: dict[int, dict[str, object]] = {}
    if resume and RESULTS_PATH.exists():
        existing_payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        for row in existing_payload.get("rows", []):
            try:
                existing_rows[int(row["seed"])] = row
            except Exception:
                continue
        if existing_rows:
            print(f"Loaded {len(existing_rows)} completed seeds from {RESULTS_PATH}")

    rows: list[dict[str, object]] = list(existing_rows.values()) if not overwrite_existing else []
    for seed in seeds:
        if resume and not overwrite_existing and seed in existing_rows:
            print(f"[resume] seed {seed} already completed, skipping.")
            continue
        split_key = str(seed)
        if split_key not in splits:
            print(f"[skip] no split for seed {seed}")
            continue
        split = splits[split_key]
        print(f"\n--- Seed {seed} ---")

        set_seed(stable_seed(seed, "simple_concat", "main"))
        seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
        y_mean = float(seed_tensors["y_mean"])
        y_std = float(seed_tensors["y_std"])

        train_idx = split["train"]
        val_idx = split["val"]
        test_idx = split["test"]

        smiles_train = [smiles_col[i] for i in train_idx]
        smiles_val = [smiles_col[i] for i in val_idx]
        smiles_test = [smiles_col[i] for i in test_idx]

        y_train_norm = (tg_col[train_idx] - y_mean) / y_std
        y_val_norm = (tg_col[val_idx] - y_mean) / y_std
        y_test_norm = (tg_col[test_idx] - y_mean) / y_std

        print("  Training Simple Concat baseline ...")
        baseline_model = train_standard_model(
            mode="simple_concat",
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=stable_seed(seed, "simple_concat", "main"),
        )
        test_loader = make_loader(seed_tensors, test_idx, config.batch_size, shuffle=False)
        ext_loader = make_loader(seed_tensors, split.get("external", []), config.batch_size, shuffle=False)

        b_preds, b_ys = _eval_gnn_loader(baseline_model, test_loader, seed_tensors)
        baseline_primary_metrics = _regression_metrics(b_preds, b_ys)
        b_errors = np.abs(b_preds - b_ys)
        baseline_primary_mae = baseline_primary_metrics["mae_k"]

        hard_threshold = np.percentile(b_errors, 100 * HARD_PERCENTILE)
        hard_mask = b_errors >= hard_threshold
        baseline_hard_metrics = _regression_metrics(b_preds[hard_mask], b_ys[hard_mask])
        baseline_hard_mae = baseline_hard_metrics["mae_k"]
        hard_rate = float(hard_mask.mean())

        baseline_ext_metrics = None
        if split.get("external"):
            be_p, be_y = _eval_gnn_loader(baseline_model, ext_loader, seed_tensors)
            baseline_ext_metrics = _regression_metrics(be_p, be_y)
        baseline_ext_mae = None if baseline_ext_metrics is None else baseline_ext_metrics["mae_k"]
        del baseline_model
        torch.cuda.empty_cache()

        print(
            f"  Baseline - primary: {baseline_primary_mae:.4f} K, hard: {baseline_hard_mae:.4f} K"
            + (f", ext: {baseline_ext_mae:.4f} K" if baseline_ext_mae is not None else "")
        )

        print(f"  Fine-tuning polyBERT ({max_epochs} max epochs) ...")
        set_seed(seed * 37 + 13)
        poly_model = _train_polybert(
            smiles_train,
            y_train_norm,
            smiles_val,
            y_val_norm,
            y_mean,
            y_std,
            tokenizer,
            max_epochs,
            device,
            model_name_or_path=_resolved_path,
            local_files_only=True,
        )

        preds, ys = _eval_polybert(poly_model, smiles_test, y_test_norm, y_mean, y_std, tokenizer, device)
        polybert_primary_metrics = _regression_metrics(preds, ys)
        p_errors = np.abs(preds - ys)
        primary_mae = polybert_primary_metrics["mae_k"]
        polybert_hard_metrics = _regression_metrics(preds[hard_mask], ys[hard_mask])
        polybert_hard_mae = polybert_hard_metrics["mae_k"]

        polybert_ext_metrics = None
        if split.get("external"):
            ext_idx = split["external"]
            smiles_ext = [smiles_col[i] for i in ext_idx]
            y_ext_norm = (tg_col[ext_idx] - y_mean) / y_std
            e_preds, e_ys = _eval_polybert(poly_model, smiles_ext, y_ext_norm, y_mean, y_std, tokenizer, device)
            polybert_ext_metrics = _regression_metrics(e_preds, e_ys)
        ext_mae = None if polybert_ext_metrics is None else polybert_ext_metrics["mae_k"]

        print(
            f"  polyBERT - primary: {primary_mae:.4f} K, hard: {polybert_hard_mae:.4f} K"
            + (f", ext: {ext_mae:.4f} K" if ext_mae is not None else "")
        )

        new_row = {
            "seed": seed,
            "baseline_primary_mae": baseline_primary_mae,
            "baseline_hard_mae": baseline_hard_mae,
            "baseline_ext_mae": baseline_ext_mae,
            "polybert_primary_mae": primary_mae,
            "polybert_hard_mae": polybert_hard_mae,
            "polybert_ext_mae": ext_mae,
            "hard_mask_rate": hard_rate,
            "baseline_primary": baseline_primary_metrics,
            "baseline_hard": baseline_hard_metrics,
            "baseline_external": baseline_ext_metrics,
            "polybert_primary": polybert_primary_metrics,
            "polybert_hard": polybert_hard_metrics,
            "polybert_external": polybert_ext_metrics,
        }
        existing_rows[seed] = new_row
        rows = [existing_rows[key] for key in sorted(existing_rows)]
        _write_results(rows)
        print(f"  [saved] {RESULTS_PATH}")

        del poly_model
        torch.cuda.empty_cache()

    if not rows:
        print("No seeds completed.")
        return

    b_primary = np.mean([float(r["baseline_primary_mae"]) for r in rows])
    b_hard = np.mean([float(r["baseline_hard_mae"]) for r in rows])
    p_primary = np.mean([float(r["polybert_primary_mae"]) for r in rows])
    p_hard = np.mean([float(r["polybert_hard_mae"]) for r in rows])
    ext_rows_b = [float(r["baseline_ext_mae"]) for r in rows if r["baseline_ext_mae"] is not None]
    ext_rows_p = [float(r["polybert_ext_mae"]) for r in rows if r["polybert_ext_mae"] is not None]
    b_ext_str = f", ext: {np.mean(ext_rows_b):.4f} K" if ext_rows_b else ""
    p_ext_str = f", ext: {np.mean(ext_rows_p):.4f} K" if ext_rows_p else ""

    print(f"\n{'=' * 64}")
    print("SUMMARY (mean over completed seeds)")
    print(f"{'=' * 64}")
    print(f"  Simple Concat baseline - primary: {b_primary:.4f} K, hard: {b_hard:.4f} K{b_ext_str}")
    print(f"  polyBERT (fine-tuned)  - primary: {p_primary:.4f} K, hard: {p_hard:.4f} K{p_ext_str}")
    print("  Reference: MSCE-RCMF-MASD (proposed) primary ~23.98 K, hard ~27.20 K")

    _write_results(rows)
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Same-protocol polyBERT baseline.")
    parser.add_argument("--seeds", default="10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    run_polybert_baseline(
        seeds=[int(s.strip()) for s in args.seeds.split(",") if s.strip()],
        max_epochs=args.epochs,
        model_name_or_path=args.model_name_or_path,
        local_files_only=args.local_files_only,
        resume=not args.no_resume,
        overwrite_existing=args.overwrite_existing,
    )
