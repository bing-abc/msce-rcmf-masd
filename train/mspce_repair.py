from __future__ import annotations

"""MSPCE/MSPCE repair-stage utilities used before the full-chain experiments.

These helpers build the multiscale context features and train the first repair
student that later stages inherit from.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = ROOT / "outputs" / "exp" / "diagnostics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.featurize import build_feature_cache, build_mspce_multiscale_context
from train.calibration import copy_shared_weights
from train.full_train import (
    DEVICE,
    _to_device,
    build_model,
    diagnostic_config,
    load_artifacts,
    make_loader,
    prepare_seed_tensors,
    set_seed,
    stable_seed,
    train_standard_model,
)

PRIMARY_CLEAN_PASS_DELTA = 0.10
PRIMARY_NOISY_PASS_DELTA = 0.00
EXTERNAL_PASS_DELTA = 0.30


def ensure_multiscale_features() -> dict[str, Any]:
    """Materialize the cached multiscale context features expected by MSPCE."""
    dataset = pd.read_csv(ROOT / "data/dataset.csv")
    expected_vec, expected_layout = build_mspce_multiscale_context(str(dataset["canonical_smiles"].iloc[0]))
    expected_dim = int(expected_vec.shape[0])
    features_path = ROOT / "data/features.pt"
    rebuilt = False
    reason = "already_multiscale"
    features: dict[str, Any] | None = None
    if features_path.exists():
        features = torch.load(features_path, map_location="cpu", weights_only=False)
        current_layout = features.get("context_layout") or {}
        current_scales = list(current_layout.get("scales", []))
        current_dim = int(features["contexts"].shape[1])
        if len(current_scales) < 2:
            rebuilt = True
            reason = "missing_context_layout"
        elif current_dim != expected_dim:
            rebuilt = True
            reason = f"stale_context_dim_{current_dim}_vs_{expected_dim}"
    else:
        rebuilt = True
        reason = "features_missing"

    if rebuilt:
        if features_path.exists():
            base_features = torch.load(features_path, map_location="cpu", weights_only=False)
            contexts = []
            context_layout: dict[str, Any] | None = None
            for smiles in dataset["canonical_smiles"].astype(str):
                context_vec, meta = build_mspce_multiscale_context(smiles)
                contexts.append(context_vec.astype(np.float32))
                if context_layout is None:
                    context_layout = meta
            if context_layout is None:
                raise RuntimeError("failed to rebuild MSPCE context layout")
            base_features["contexts"] = torch.tensor(np.stack(contexts), dtype=torch.float32)
            base_features["context_layout"] = context_layout
            features = base_features
        else:
            features = build_feature_cache(dataset)
        torch.save(features, features_path)
    elif features is None:
        features = torch.load(features_path, map_location="cpu", weights_only=False)

    return {
        "features": features,
        "rebuilt": rebuilt,
        "reason": reason,
        "expected_context_dim": expected_dim,
        "expected_scale_count": len(expected_layout.get("scales", [])),
        "actual_context_dim": int(features["contexts"].shape[1]),
        "actual_scale_count": len((features.get("context_layout") or {}).get("scales", [])),
    }


def set_repair_trainable(model: nn.Module, *, allow_backbone_tail: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module_name in ("ctx_encoder", "mspce_context_injector", "mspce_repair_gate"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
    if allow_backbone_tail:
        desc_tail = getattr(getattr(model.backbone, "descriptor_branch", None), "net", None)
        if isinstance(desc_tail, nn.Sequential) and len(desc_tail) >= 2 and isinstance(desc_tail[1], nn.Sequential):
            for param in desc_tail[1][-1].parameters():
                param.requires_grad = True


def repair_loss(out: dict[str, torch.Tensor], y_true: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    loss_fn = nn.SmoothL1Loss()
    pred_loss = loss_fn(out["pred"], y_true)
    concat_err = torch.abs(out["concat_pred"] - y_true)
    pred_err = torch.abs(out["pred"] - y_true)
    anchor_margin = torch.relu(pred_err - concat_err + 0.0015).mean()
    hidden_consistency = out["mspce_repair_hidden_consistency"].mean()
    scale_consistency = out["mspce_repair_scale_consistency"].mean()
    gate_mean = out["mspce_repair_gate"].mean()
    stability_penalty = (out["mspce_repair_gate"] * out["mspce_repair_stability_proxy"]).mean()
    delta_penalty = torch.abs(out["pred"] - out["concat_pred"]).mean()
    total = (
        pred_loss
        + 0.90 * anchor_margin
        + 0.22 * hidden_consistency
        + 0.10 * scale_consistency
        + 0.65 * stability_penalty
        + 0.10 * gate_mean
        + 0.08 * delta_penalty
    )
    return total, {
        "pred_loss": float(pred_loss.detach().cpu()),
        "anchor_margin": float(anchor_margin.detach().cpu()),
        "hidden_consistency": float(hidden_consistency.detach().cpu()),
        "scale_consistency": float(scale_consistency.detach().cpu()),
        "gate_mean": float(gate_mean.detach().cpu()),
        "stability_penalty": float(stability_penalty.detach().cpu()),
        "delta_penalty": float(delta_penalty.detach().cpu()),
    }


@torch.no_grad()
def collect_repair_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    seed_tensors: dict[str, Any],
    *,
    variant: str = "clean",
    noise_seed: int = 0,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=DEVICE.type if DEVICE.type != "cpu" else "cpu")
    generator.manual_seed(noise_seed)
    y_scaled = []
    pred_scaled = []
    concat_scaled = []
    gate_vals = []
    stability_vals = []
    hidden_shift_vals = []
    scale_consistency_vals = []
    for batch in loader:
        batch = _to_device(batch)
        desc = batch["desc"]
        ctx = batch["ctx"]
        if variant == "noisy":
            desc = desc + 0.015 * torch.randn(desc.shape, device=desc.device, dtype=desc.dtype, generator=generator)
            ctx = ctx + 0.030 * torch.randn(ctx.shape, device=ctx.device, dtype=ctx.dtype, generator=generator)
        elif variant == "drop":
            ctx = torch.zeros_like(ctx)
        out = model(batch["graph"], desc, ctx, led=batch["led"], led_mask=batch["led_mask"])
        y_scaled.append(batch["y"].detach().cpu())
        pred_scaled.append(out["pred"].detach().cpu())
        concat_scaled.append(out["concat_pred"].detach().cpu())
        gate_vals.append(out.get("mspce_repair_gate", torch.zeros_like(out["pred"])).detach().cpu())
        stability_vals.append(out.get("mspce_repair_stability_proxy", torch.zeros_like(out["pred"])).detach().cpu())
        hidden_cons = out.get("mspce_repair_hidden_consistency", torch.zeros_like(out["pred"]))
        scale_cons = out.get("mspce_repair_scale_consistency", torch.zeros_like(out["pred"]))
        hidden_shift_vals.append(torch.sqrt(hidden_cons.clamp_min(0.0)).detach().cpu())
        scale_consistency_vals.append(scale_cons.detach().cpu())
    y_scaled_t = torch.cat(y_scaled, dim=0)
    pred_scaled_t = torch.cat(pred_scaled, dim=0)
    concat_scaled_t = torch.cat(concat_scaled, dim=0)
    y = y_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    pred = pred_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    concat = concat_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    y_np = y.numpy().squeeze(1)
    pred_np = pred.numpy().squeeze(1)
    concat_np = concat.numpy().squeeze(1)
    return {
        "mae_k": float(np.mean(np.abs(y_np - pred_np))),
        "concat_mae_k": float(np.mean(np.abs(y_np - concat_np))),
        "gate_mean": float(torch.cat(gate_vals, dim=0).mean().item()),
        "stability_mean": float(torch.cat(stability_vals, dim=0).mean().item()),
        "hidden_shift_mean": float(torch.cat(hidden_shift_vals, dim=0).mean().item()),
        "scale_consistency_mean": float(torch.cat(scale_consistency_vals, dim=0).mean().item()),
    }


def train_repair_student(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
) -> tuple[nn.Module, nn.Module]:
    baseline_model = train_standard_model(
        mode="simple_concat",
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=stable_seed(seed, "simple_concat", "main"),
    )
    set_seed(stable_seed(seed, "mspce_context_injection", "main"))
    student = build_model("mspce_context_injection", seed_tensors, config)
    copy_shared_weights(baseline_model, student)
    set_repair_trainable(student, allow_backbone_tail=False)
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=2.5e-4,
        weight_decay=config.weight_decay,
    )
    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)
    best_state: dict[str, Any] | None = None
    best_score = float("inf")
    bad_epochs = 0
    for _epoch in range(10):
        student.train()
        for batch in train_loader:
            batch = _to_device(batch)
            out = student(batch["graph"], batch["desc"], batch["ctx"])
            loss, _ = repair_loss(out, batch["y"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()
        val_metrics = collect_repair_metrics(student, val_loader, seed_tensors, variant="clean", noise_seed=seed * 17 + 3)
        val_score = (
            val_metrics["mae_k"]
            + 0.85 * max(0.0, val_metrics["mae_k"] - val_metrics["concat_mae_k"])
            + 0.25 * val_metrics["stability_mean"]
            + 0.20 * val_metrics["gate_mean"]
            + 0.10 * val_metrics["hidden_shift_mean"]
            + 0.05 * val_metrics["scale_consistency_mean"]
        )
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 3:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint stored for mspce_context_injection")
    student.load_state_dict(best_state)
    return baseline_model, student


def run_seed(
    *,
    seed: int,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    split = splits["seeds"][str(seed)]
    seed_tensors = prepare_seed_tensors(features, split["train"])
    baseline_model, repair_model = train_repair_student(split=split, seed_tensors=seed_tensors, config=config, seed=seed)
    loaders = {
        "primary_clean": make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False),
        "external_holdout": make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False),
    }
    baseline_primary_clean = collect_repair_metrics(baseline_model, loaders["primary_clean"], seed_tensors, variant="clean", noise_seed=seed * 31 + 1)["mae_k"]
    baseline_primary_noisy = collect_repair_metrics(baseline_model, loaders["primary_clean"], seed_tensors, variant="noisy", noise_seed=seed * 31 + 2)["mae_k"]
    baseline_external = collect_repair_metrics(baseline_model, loaders["external_holdout"], seed_tensors, variant="clean", noise_seed=seed * 31 + 3)["mae_k"]

    repair_primary_clean = collect_repair_metrics(repair_model, loaders["primary_clean"], seed_tensors, variant="clean", noise_seed=seed * 31 + 11)["mae_k"]
    repair_primary_noisy = collect_repair_metrics(repair_model, loaders["primary_clean"], seed_tensors, variant="noisy", noise_seed=seed * 31 + 12)["mae_k"]
    repair_external = collect_repair_metrics(repair_model, loaders["external_holdout"], seed_tensors, variant="clean", noise_seed=seed * 31 + 13)["mae_k"]

    repair_row = {
        "model_name": "Baseline + MSPCE",
        "seed": seed,
        "primary_clean": repair_primary_clean,
        "primary_noisy": repair_primary_noisy,
        "external_holdout": repair_external,
        "delta_vs_simple_concat_primary_clean": repair_primary_clean - baseline_primary_clean,
        "delta_vs_simple_concat_primary_noisy": repair_primary_noisy - baseline_primary_noisy,
        "delta_vs_simple_concat_external": repair_external - baseline_external,
    }
    repair_row["pass_flag"] = bool(
        repair_row["delta_vs_simple_concat_primary_clean"] <= PRIMARY_CLEAN_PASS_DELTA
        and repair_row["delta_vs_simple_concat_primary_noisy"] <= PRIMARY_NOISY_PASS_DELTA
        and repair_row["delta_vs_simple_concat_external"] <= EXTERNAL_PASS_DELTA
    )
    baseline_row = {
        "model_name": "Simple Concat",
        "seed": seed,
        "primary_clean": baseline_primary_clean,
        "primary_noisy": baseline_primary_noisy,
        "external_holdout": baseline_external,
        "delta_vs_simple_concat_primary_clean": 0.0,
        "delta_vs_simple_concat_primary_noisy": 0.0,
        "delta_vs_simple_concat_external": 0.0,
        "pass_flag": "reference",
    }
    return [baseline_row, repair_row], repair_row


def write_outputs(
    *,
    feature_info: dict[str, Any],
    rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    previous_repair: dict[str, float] | None,
) -> None:
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_path = DIAG_ROOT / "mspce_repair_results.csv"
    results_df.to_csv(results_path, index=False)

    latest = repair_rows[-1]
    current_primary_clean = float(np.mean([float(row["delta_vs_simple_concat_primary_clean"]) for row in repair_rows]))
    current_primary_noisy = float(np.mean([float(row["delta_vs_simple_concat_primary_noisy"]) for row in repair_rows]))
    current_external = float(np.mean([float(row["delta_vs_simple_concat_external"]) for row in repair_rows]))
    all_seed_pass = all(row["pass_flag"] is True for row in repair_rows)
    rcmf_allowed_next = bool(all_seed_pass and len(repair_rows) >= 2)
    seed_list = ", ".join(str(int(row["seed"])) for row in repair_rows)

    trend_line = "No previous repair result available for comparison."
    if previous_repair is not None:
        delta_improve = previous_repair["primary_clean"] - current_primary_clean
        external_improve = previous_repair["external"] - current_external
        trend_line = (
            f"Compared with the previous repair round, primary clean delta changed from {previous_repair['primary_clean']:+.4f} K "
            f"to {current_primary_clean:+.4f} K and external delta changed from {previous_repair['external']:+.4f} K "
            f"to {current_external:+.4f} K."
        )
        if rcmf_allowed_next:
            trend_line += " This is good enough to reopen the next minimal RCMF gate."
        elif delta_improve > 0.0 or external_improve > 0.0:
            trend_line += " This is directionally better, but still below the gate."
        else:
            trend_line += " This is not better enough to change the gate."

    note_md = "\n".join(
        [
            "# MSPCE Repair Note",
            "",
            "- Previous failure root cause: MSPCE over-perturbed the `Simple Concat` anchor when context entered as a free correction path; this round targeted integration strength rather than protocol or innovation definition changes.",
            "- This round changed code in three places: MSPCE scale selection is now learnable sparse top-k; MSPCE injection is now baseline-anchored low-rank FiLM plus a weak gated residual bottleneck; repair training now loads `Simple Concat`, freezes the core path, and only trains MSPCE/context injection parameters.",
            "- These changes do not alter the innovation definition. MSPCE still carries polymer chain multiscale context plus 2D neighborhood context; only the way that context modulates the strongest baseline was changed.",
            f"- Gate standard for this round: primary clean delta <= {PRIMARY_CLEAN_PASS_DELTA:.2f} K, primary noisy delta <= {PRIMARY_NOISY_PASS_DELTA:.2f} K, external delta <= {EXTERNAL_PASS_DELTA:.2f} K before any RCMF reopening.",
        ]
    )
    (DIAG_ROOT / "mspce_repair_note.md").write_text(note_md, encoding="utf-8")

    gate_md = "\n".join(
        [
            "# MSPCE Gate Decision",
            "",
            f"- Does baseline + MSPCE beat `Simple Concat`: {'yes' if current_primary_clean <= 0.0 else 'no'} ({current_primary_clean:+.4f} K mean delta on primary clean).",
            f"- Does primary noisy avoid regression: {'yes' if current_primary_noisy <= 0.0 else 'no'} ({current_primary_noisy:+.4f} K).",
            f"- Does external holdout stay stable: {'yes' if current_external <= EXTERNAL_PASS_DELTA else 'no'} ({current_external:+.4f} K).",
            f"- Current pass status: {'pass' if all_seed_pass else 'fail'}.",
            f"- RCMF allowed next: {'yes' if rcmf_allowed_next else 'no'}.",
            "- Full allowed now: no.",
            f"- Next correct action: {'open the smallest RCMF validation on top of the repaired MSPCE anchor while keeping Full and 20-seed closed' if rcmf_allowed_next else ('run seed1 confirmation under the same anchored repair' if all_seed_pass else 'keep repairing MSPCE anchoring strength and reduce external perturbation before any RCMF reopening')}.",
        ]
    )
    (DIAG_ROOT / "mspce_gate_decision.md").write_text(gate_md, encoding="utf-8")

    summary_md = "\n".join(
        [
            "# Summary",
            "",
            "Canonical protocol remains ready and unchanged; this turn only targeted baseline + MSPCE repair.",
            f"Seeds executed this turn: {seed_list}.",
            f"Feature status: context cache rebuilt = {str(bool(feature_info['rebuilt'])).lower()}, context dimension = {feature_info['actual_context_dim']}, active scales = top-3 sparse selection over {feature_info['actual_scale_count']} scales.",
            trend_line,
            f"Current repair result: primary clean delta {current_primary_clean:+.4f} K, primary noisy delta {current_primary_noisy:+.4f} K, external delta {current_external:+.4f} K.",
            f"Current status remains `{'RCMF_ALLOWED_NEXT' if rcmf_allowed_next else 'MSPCE_REPAIR_CONTINUES'}`. Full and 20-seed remain closed.",
        ]
    )
    (DIAG_ROOT / "summary.md").write_text(summary_md, encoding="utf-8")


def previous_repair_result() -> dict[str, float] | None:
    path = DIAG_ROOT / "mspce_repair_results.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "model_name" in df.columns:
        hit = df.loc[df["model_name"] == "Baseline + MSPCE"].copy()
        if hit.empty:
            return None
        row = hit.iloc[-1]
        return {
            "primary_clean": float(row["delta_vs_simple_concat_primary_clean"]),
            "primary_noisy": float(row["delta_vs_simple_concat_primary_noisy"]),
            "external": float(row["delta_vs_simple_concat_external"]),
        }
    if "mode" in df.columns and "delta_k_vs_concat" in df.columns and "external_delta_k_vs_concat" in df.columns:
        hit = df.loc[df["mode"].astype(str).str.contains("mspce_context_injection", na=False)].copy()
        if hit.empty:
            return None
        row = hit.iloc[-1]
        return {
            "primary_clean": float(row["delta_k_vs_concat"]),
            "primary_noisy": float("nan"),
            "external": float(row["external_delta_k_vs_concat"]),
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run anchored MSPCE repair on the recovered canonical protocol.")
    parser.add_argument("--seed0_only", action="store_true")
    args = parser.parse_args()

    previous = previous_repair_result()
    feature_info = ensure_multiscale_features()
    _, _, splits = load_artifacts()
    features = feature_info["features"]
    config = diagnostic_config()

    rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    seed0_rows, seed0_repair = run_seed(seed=0, features=features, splits=splits, config=config)
    rows.extend(seed0_rows)
    repair_rows.append(seed0_repair)
    if not args.seed0_only and bool(seed0_repair["pass_flag"]):
        seed1_rows, seed1_repair = run_seed(seed=1, features=features, splits=splits, config=config)
        rows.extend(seed1_rows)
        repair_rows.append(seed1_repair)

    write_outputs(
        feature_info=feature_info,
        rows=rows,
        repair_rows=repair_rows,
        previous_repair=previous,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

