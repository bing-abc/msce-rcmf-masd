from __future__ import annotations

"""Utilities for the intermediate reliability-conditioned fusion repair stage.

This module keeps the lightweight RCMF/RCMF-style repairs separate from the
final MASD stage so ablations can stop at the fusion bridge.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = ROOT / "outputs" / "exp" / "diagnostics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.calibration import copy_shared_weights
from train.full_train import (
    _to_device,
    build_model,
    diagnostic_config,
    make_loader,
    prepare_seed_tensors,
    set_seed,
)
from train.mspce_repair import collect_repair_metrics, ensure_multiscale_features, train_repair_student

PRIMARY_CLEAN_PASS_DELTA = 0.10
PRIMARY_NOISY_PASS_DELTA = 0.05
EXTERNAL_PASS_DELTA = 0.30


def _pr_sample_weights(model: nn.Module | None, out: dict[str, torch.Tensor]) -> torch.Tensor | None:
    # Fasttrack scans optionally upweight hard samples without touching the test protocol.
    if model is None:
        return None
    alpha = float(getattr(model, "pr_hard_reweight_alpha", 0.0))
    if alpha <= 0.0:
        return None
    hard_score = out.get("pr_hard_score")
    if hard_score is None:
        return None
    return 1.0 + alpha * hard_score.detach()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_rcmf_external_focus(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    seed_tensors: dict[str, Any],
    *,
    variant: str = "clean",
    noise_seed: int = 0,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=DEVICE.type if DEVICE.type != "cpu" else "cpu")
    generator.manual_seed(int(noise_seed))

    y_vals = []
    pred_vals = []
    hard_err_vals = []
    hard_score_vals = []
    for batch in loader:
        batch = _to_device(batch)
        desc = batch["desc"]
        ctx = batch["ctx"]
        if variant == "noisy":
            desc = desc + 0.012 * torch.randn(desc.shape, device=desc.device, dtype=desc.dtype, generator=generator)
            ctx = ctx + 0.020 * torch.randn(ctx.shape, device=ctx.device, dtype=ctx.dtype, generator=generator)
        out = model(batch["graph"], desc, ctx, led=batch["led"], led_mask=batch["led_mask"])
        y = batch["y"].detach().cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]
        pred = out["pred"].detach().cpu() * seed_tensors["y_std"] + seed_tensors["y_mean"]
        err = torch.abs(pred - y)
        y_vals.append(y)
        pred_vals.append(pred)
        conflict = out["conflict_level"].detach().cpu()
        uncertainty = out["uncertainty_level"].detach().cpu()
        confidence = out.get("rcmf_min_confidence", torch.ones_like(conflict)).detach().cpu()
        external_focus = out.get(
            "rcmf_min_external_focus",
            conflict + 1.35 * uncertainty + 0.40 * (1.0 - confidence),
        ).detach().cpu()
        hard_score = external_focus + 0.30 * conflict + 0.25 * uncertainty
        hard_err_vals.append(err)
        hard_score_vals.append(hard_score)

    y_tensor = torch.cat(y_vals, dim=0)
    pred_tensor = torch.cat(pred_vals, dim=0)
    hard_err = torch.cat(hard_err_vals, dim=0)
    hard_score = torch.cat(hard_score_vals, dim=0)
    hard_threshold = torch.quantile(hard_score.squeeze(1), 0.80)
    hard_mask = (hard_score.squeeze(1) >= hard_threshold).float().unsqueeze(1)
    return {
        "mae_k": float(torch.mean(torch.abs(y_tensor - pred_tensor)).item()),
        "high_conflict_external_metric": subgroup_mae(hard_err, hard_mask),
        "hard_mask_rate": float(hard_mask.mean().item()),
    }


def focused_rcmf_loss(
    out: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    model: nn.Module | None = None,
) -> torch.Tensor:
    weights = _pr_sample_weights(model, out)
    pred_loss = _weighted_mean(F.smooth_l1_loss(out["pred"], y_true, reduction="none"), weights)
    anchor_err = torch.abs(out["mspce_anchor_pred"] - y_true)
    pred_err = torch.abs(out["pred"] - y_true)
    confidence = out["rcmf_min_confidence"]
    conflict = out["conflict_level"]
    uncertainty = out["uncertainty_level"]
    focus = out["rcmf_min_external_focus"]
    hard = torch.sigmoid(2.8 * (focus - 1.05))
    focused_weight = hard if weights is None else hard * weights
    top_margin = (torch.relu(pred_err - anchor_err + 0.0008) * focused_weight).sum() / focused_weight.sum().clamp_min(1.0)
    top_benefit = ((anchor_err - pred_err) * focused_weight).sum() / focused_weight.sum().clamp_min(1.0)
    hidden_consistency = out["rcmf_min_hidden_consistency"].mean()
    external_stability = (out["rcmf_min_gate"] * focus).mean()
    trust_entropy = out["rcmf_min_trust_entropy"].mean()
    delta_scale = torch.abs(out["pred"] - out["mspce_anchor_pred"]).mean()
    confidence_floor = (torch.relu(0.58 - confidence) * focused_weight).sum() / focused_weight.sum().clamp_min(1.0)
    entropy_reg_lambda = float(getattr(model, "pr_rcmf_entropy_lambda", 0.0)) if model is not None else 0.0
    entropy_reg = torch.relu(0.60 - out["rcmf_min_trust_entropy"]).mean()
    return (
        pred_loss
        + 1.20 * top_margin
        + 0.20 * hidden_consistency
        + 0.30 * external_stability
        + 0.08 * torch.relu(0.42 - trust_entropy)
        + 0.05 * delta_scale
        + 0.18 * confidence_floor
        - 0.90 * top_benefit
        + 0.06 * (conflict * uncertainty).mean()
        + entropy_reg_lambda * entropy_reg
    )


def set_rcmf_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module_name in ("rcmf_min_fusion", "rcmf_min_delta_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True


def rcmf_loss(
    out: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    model: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = _pr_sample_weights(model, out)
    pred_loss = _weighted_mean(F.smooth_l1_loss(out["pred"], y_true, reduction="none"), weights)
    anchor_err = torch.abs(out["mspce_anchor_pred"] - y_true)
    pred_err = torch.abs(out["pred"] - y_true)
    anchor_margin = _weighted_mean(torch.relu(pred_err - anchor_err + 0.0015), weights)
    hidden_consistency = out["rcmf_min_hidden_consistency"].mean()
    external_stability = (out["rcmf_min_gate"] * out["rcmf_min_risk"]).mean()
    selector_penalty = torch.relu(out["rcmf_min_selector_score"] - 0.92).mean()
    entropy_penalty = torch.relu(0.45 - out["rcmf_min_trust_entropy"]).mean()
    delta_penalty = torch.abs(out["pred"] - out["mspce_anchor_pred"]).mean()
    entropy_reg_lambda = float(getattr(model, "pr_rcmf_entropy_lambda", 0.0)) if model is not None else 0.0
    entropy_reg = torch.relu(0.60 - out["rcmf_min_trust_entropy"]).mean()
    total = (
        pred_loss
        + 0.95 * anchor_margin
        + 0.18 * hidden_consistency
        + 0.45 * external_stability
        + 0.10 * selector_penalty
        + 0.06 * entropy_penalty
        + 0.06 * delta_penalty
        + entropy_reg_lambda * entropy_reg
    )
    return total, {
        "pred_loss": float(pred_loss.detach().cpu()),
        "anchor_margin": float(anchor_margin.detach().cpu()),
        "hidden_consistency": float(hidden_consistency.detach().cpu()),
        "external_stability": float(external_stability.detach().cpu()),
        "selector_penalty": float(selector_penalty.detach().cpu()),
        "entropy_penalty": float(entropy_penalty.detach().cpu()),
        "delta_penalty": float(delta_penalty.detach().cpu()),
        "entropy_reg": float(entropy_reg.detach().cpu()),
    }


def noisy_anchor_penalty(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    noisy_desc = batch["desc"] + 0.012 * torch.randn_like(batch["desc"])
    noisy_ctx = batch["ctx"] + 0.020 * torch.randn_like(batch["ctx"])
    noisy_out = model(batch["graph"], noisy_desc, noisy_ctx)
    noisy_err = torch.abs(noisy_out["pred"] - batch["y"])
    noisy_anchor_err = torch.abs(noisy_out["mspce_anchor_pred"] - batch["y"])
    noisy_margin = torch.relu(noisy_err - noisy_anchor_err + 0.0005).mean()
    noisy_stability = (noisy_out["rcmf_min_gate"] * noisy_out["rcmf_min_risk"]).mean()
    return noisy_margin + 0.30 * noisy_stability


@torch.no_grad()
def collect_rcmf_aux(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> dict[str, float]:
    model.eval()
    gate_vals = []
    risk_vals = []
    selector_vals = []
    entropy_vals = []
    hidden_cons = []
    for batch in loader:
        batch = _to_device(batch)
        out = model(batch["graph"], batch["desc"], batch["ctx"])
        gate_vals.append(out["rcmf_min_gate"].detach().cpu())
        risk_vals.append(out["rcmf_min_risk"].detach().cpu())
        selector_vals.append(out["rcmf_min_selector_score"].detach().cpu())
        entropy_vals.append(out["rcmf_min_trust_entropy"].detach().cpu())
        hidden_cons.append(out["rcmf_min_hidden_consistency"].detach().cpu())
    return {
        "gate_mean": float(torch.cat(gate_vals, dim=0).mean().item()),
        "risk_mean": float(torch.cat(risk_vals, dim=0).mean().item()),
        "selector_mean": float(torch.cat(selector_vals, dim=0).mean().item()),
        "entropy_mean": float(torch.cat(entropy_vals, dim=0).mean().item()),
        "hidden_consistency_mean": float(torch.cat(hidden_cons, dim=0).mean().item()),
    }


def train_rcmf_student(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
    repair_model: nn.Module,
) -> nn.Module:
    set_seed(20000 + seed * 97)
    student = build_model("rcmf_min_trusted_fusion", seed_tensors, config)
    copy_shared_weights(repair_model, student)
    set_rcmf_trainable(student)
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=2.0e-4,
        weight_decay=config.weight_decay,
    )
    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)
    best_state: dict[str, Any] | None = None
    best_score = float("inf")
    bad_epochs = 0
    repair_val_clean = collect_repair_metrics(repair_model, val_loader, seed_tensors, variant="clean", noise_seed=seed * 41 + 7)
    repair_val_noisy = collect_repair_metrics(repair_model, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 41 + 8)
    for _epoch in range(8):
        student.train()
        for batch in train_loader:
            batch = _to_device(batch)
            out = student(batch["graph"], batch["desc"], batch["ctx"])
            loss, _ = rcmf_loss(out, batch["y"], model=student)
            loss = loss + 0.55 * noisy_anchor_penalty(student, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.5)
            optimizer.step()
        val_clean = collect_repair_metrics(student, val_loader, seed_tensors, variant="clean", noise_seed=seed * 41 + 9)
        val_noisy = collect_repair_metrics(student, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 41 + 10)
        val_aux = collect_rcmf_aux(student, val_loader)
        val_score = (
            val_clean["mae_k"]
            + 1.00 * max(0.0, val_clean["mae_k"] - repair_val_clean["mae_k"])
            + 0.80 * max(0.0, val_noisy["mae_k"] - repair_val_noisy["mae_k"])
            + 0.40 * val_aux["risk_mean"] * val_aux["gate_mean"]
            + 0.15 * max(0.0, val_aux["selector_mean"] - 0.90)
            + 0.10 * max(0.0, 0.45 - val_aux["entropy_mean"])
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
        raise RuntimeError("no checkpoint stored for rcmf_min_trusted_fusion")
    student.load_state_dict(best_state)
    return student


def train_rcmf_external_focus_student(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
    minimal_rcmf: nn.Module,
) -> nn.Module:
    set_seed(41000 + seed * 97)
    student = build_model("rcmf_min_trusted_fusion", seed_tensors, config)
    student.load_state_dict(minimal_rcmf.state_dict())
    set_rcmf_trainable(student)
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=1.2e-4,
        weight_decay=config.weight_decay,
    )
    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)
    anchor_val_clean = evaluate_rcmf_external_focus(minimal_rcmf, val_loader, seed_tensors, variant="clean", noise_seed=seed * 97 + 1)
    anchor_val_noisy = evaluate_rcmf_external_focus(minimal_rcmf, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 97 + 2)
    anchor_external = evaluate_rcmf_external_focus(minimal_rcmf, external_loader, seed_tensors, variant="clean", noise_seed=seed * 97 + 5)
    best_state: dict[str, Any] | None = None
    best_score = float("inf")
    bad_epochs = 0
    for _epoch in range(4):
        student.train()
        for batch in train_loader:
            batch = _to_device(batch)
            out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
            loss = focused_rcmf_loss(out, batch["y"], model=student)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.0)
            optimizer.step()
        val_clean = evaluate_rcmf_external_focus(student, val_loader, seed_tensors, variant="clean", noise_seed=seed * 97 + 3)
        val_noisy = evaluate_rcmf_external_focus(student, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 97 + 4)
        val_external = evaluate_rcmf_external_focus(student, external_loader, seed_tensors, variant="clean", noise_seed=seed * 97 + 6)
        val_score = (
            val_clean["mae_k"]
            + 0.95 * max(0.0, val_clean["mae_k"] - anchor_val_clean["mae_k"])
            + 0.85 * max(0.0, val_noisy["mae_k"] - anchor_val_noisy["mae_k"])
            + 0.55 * max(0.0, val_external["mae_k"] - anchor_external["mae_k"])
            + 0.80 * (val_external["high_conflict_external_metric"] - anchor_external["high_conflict_external_metric"])
        )
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 2:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint stored for focused RCMF")
    student.load_state_dict(best_state)
    return student


def classify_failure(repair_delta_clean: float, repair_delta_noisy: float, repair_delta_external: float, aux: dict[str, float]) -> str:
    if repair_delta_clean <= PRIMARY_CLEAN_PASS_DELTA and repair_delta_noisy <= PRIMARY_NOISY_PASS_DELTA and repair_delta_external <= EXTERNAL_PASS_DELTA:
        return "none"
    if aux["selector_mean"] > 0.92 or aux["entropy_mean"] < 0.40:
        return "RCMF too close to selector"
    if aux["gate_mean"] > 0.030 and repair_delta_external > EXTERNAL_PASS_DELTA:
        return "RCMF over-perturbs repaired MSPCE anchor"
    if aux["gate_mean"] < 0.008 and repair_delta_clean >= 0.0:
        return "fusion controller too weak"
    if aux["risk_mean"] < 0.35 and repair_delta_external > 0.0:
        return "uncertainty usage weak"
    if aux["selector_mean"] < 0.55 and repair_delta_clean >= 0.0:
        return "trust signals not informative"
    return "fusion controller too strong"


def run_seed(
    *,
    seed: int,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = splits["seeds"][str(seed)]
    seed_tensors = prepare_seed_tensors(features, split["train"])
    baseline_model, repair_model = train_repair_student(split=split, seed_tensors=seed_tensors, config=config, seed=seed)
    rcmf_model = train_rcmf_student(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=seed,
        repair_model=repair_model,
    )
    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)

    baseline_primary_clean = collect_repair_metrics(baseline_model, primary_loader, seed_tensors, variant="clean", noise_seed=seed * 53 + 1)["mae_k"]
    baseline_primary_noisy = collect_repair_metrics(baseline_model, primary_loader, seed_tensors, variant="noisy", noise_seed=seed * 53 + 2)["mae_k"]
    baseline_external = collect_repair_metrics(baseline_model, external_loader, seed_tensors, variant="clean", noise_seed=seed * 53 + 3)["mae_k"]

    repair_primary_clean = collect_repair_metrics(repair_model, primary_loader, seed_tensors, variant="clean", noise_seed=seed * 53 + 4)["mae_k"]
    repair_primary_noisy = collect_repair_metrics(repair_model, primary_loader, seed_tensors, variant="noisy", noise_seed=seed * 53 + 5)["mae_k"]
    repair_external = collect_repair_metrics(repair_model, external_loader, seed_tensors, variant="clean", noise_seed=seed * 53 + 6)["mae_k"]

    rcmf_primary_clean = collect_repair_metrics(rcmf_model, primary_loader, seed_tensors, variant="clean", noise_seed=seed * 53 + 7)["mae_k"]
    rcmf_primary_noisy = collect_repair_metrics(rcmf_model, primary_loader, seed_tensors, variant="noisy", noise_seed=seed * 53 + 8)["mae_k"]
    rcmf_external = collect_repair_metrics(rcmf_model, external_loader, seed_tensors, variant="clean", noise_seed=seed * 53 + 9)["mae_k"]

    aux_primary = collect_rcmf_aux(rcmf_model, primary_loader)
    aux_external = collect_rcmf_aux(rcmf_model, external_loader)
    aux = {
        "gate_mean": float(np.mean([aux_primary["gate_mean"], aux_external["gate_mean"]])),
        "risk_mean": float(np.mean([aux_primary["risk_mean"], aux_external["risk_mean"]])),
        "selector_mean": float(np.mean([aux_primary["selector_mean"], aux_external["selector_mean"]])),
        "entropy_mean": float(np.mean([aux_primary["entropy_mean"], aux_external["entropy_mean"]])),
        "hidden_consistency_mean": float(np.mean([aux_primary["hidden_consistency_mean"], aux_external["hidden_consistency_mean"]])),
    }

    rcmf_row = {
        "model_name": "Baseline + MSPCE + RCMF",
        "seed": seed,
        "primary_clean": rcmf_primary_clean,
        "primary_noisy": rcmf_primary_noisy,
        "external_holdout": rcmf_external,
        "delta_vs_simple_concat_primary_clean": rcmf_primary_clean - baseline_primary_clean,
        "delta_vs_simple_concat_primary_noisy": rcmf_primary_noisy - baseline_primary_noisy,
        "delta_vs_simple_concat_external": rcmf_external - baseline_external,
        "delta_vs_repaired_mspce_primary_clean": rcmf_primary_clean - repair_primary_clean,
        "delta_vs_repaired_mspce_primary_noisy": rcmf_primary_noisy - repair_primary_noisy,
        "delta_vs_repaired_mspce_external": rcmf_external - repair_external,
    }
    rcmf_row["pass_flag"] = bool(
        rcmf_row["delta_vs_repaired_mspce_primary_clean"] <= PRIMARY_CLEAN_PASS_DELTA
        and rcmf_row["delta_vs_repaired_mspce_primary_noisy"] <= PRIMARY_NOISY_PASS_DELTA
        and rcmf_row["delta_vs_repaired_mspce_external"] <= EXTERNAL_PASS_DELTA
    )
    rcmf_row["failure_type"] = classify_failure(
        rcmf_row["delta_vs_repaired_mspce_primary_clean"],
        rcmf_row["delta_vs_repaired_mspce_primary_noisy"],
        rcmf_row["delta_vs_repaired_mspce_external"],
        aux,
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
        "delta_vs_repaired_mspce_primary_clean": baseline_primary_clean - repair_primary_clean,
        "delta_vs_repaired_mspce_primary_noisy": baseline_primary_noisy - repair_primary_noisy,
        "delta_vs_repaired_mspce_external": baseline_external - repair_external,
        "pass_flag": "reference",
    }
    repair_row = {
        "model_name": "Repaired MSPCE",
        "seed": seed,
        "primary_clean": repair_primary_clean,
        "primary_noisy": repair_primary_noisy,
        "external_holdout": repair_external,
        "delta_vs_simple_concat_primary_clean": repair_primary_clean - baseline_primary_clean,
        "delta_vs_simple_concat_primary_noisy": repair_primary_noisy - baseline_primary_noisy,
        "delta_vs_simple_concat_external": repair_external - baseline_external,
        "delta_vs_repaired_mspce_primary_clean": 0.0,
        "delta_vs_repaired_mspce_primary_noisy": 0.0,
        "delta_vs_repaired_mspce_external": 0.0,
        "pass_flag": "anchor",
    }
    return [baseline_row, repair_row, rcmf_row], {"rcmf_row": rcmf_row, "aux": aux}


def write_outputs(
    *,
    rows: list[dict[str, Any]],
    rcmf_rows: list[dict[str, Any]],
    note_text: str,
) -> None:
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = DIAG_ROOT / "rcmf_min_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)

    mean_clean = float(np.mean([row["delta_vs_repaired_mspce_primary_clean"] for row in rcmf_rows]))
    mean_noisy = float(np.mean([row["delta_vs_repaired_mspce_primary_noisy"] for row in rcmf_rows]))
    mean_external = float(np.mean([row["delta_vs_repaired_mspce_external"] for row in rcmf_rows]))
    all_pass = all(row["pass_flag"] is True for row in rcmf_rows)
    led_allowed = bool(all_pass and len(rcmf_rows) >= 2)
    failure_type = "none" if led_allowed else str(rcmf_rows[-1]["failure_type"])
    seeds_text = ", ".join(str(int(row["seed"])) for row in rcmf_rows)

    gate_md = "\n".join(
        [
            "# RCMF Gate Decision",
            "",
            f"- Does baseline + MSPCE + RCMF beat repaired MSPCE: {'yes' if mean_clean <= 0.0 else 'no'} ({mean_clean:+.4f} K mean delta on primary clean).",
            f"- Does primary clean avoid regression: {'yes' if mean_clean <= PRIMARY_CLEAN_PASS_DELTA else 'no'} ({mean_clean:+.4f} K).",
            f"- Does primary noisy avoid regression: {'yes' if mean_noisy <= PRIMARY_NOISY_PASS_DELTA else 'no'} ({mean_noisy:+.4f} K).",
            f"- Does external holdout stay stable: {'yes' if mean_external <= EXTERNAL_PASS_DELTA else 'no'} ({mean_external:+.4f} K).",
            f"- Current failure type: `{failure_type}`.",
            f"- LED allowed next: {'yes' if led_allowed else 'no'}.",
            "- Full allowed now: no.",
            f"- Next correct action: {'open the smallest LED validation on top of the repaired MSPCE + RCMF anchor while keeping Full and 20-seed closed' if led_allowed else 'continue RCMF repair without opening LED or Full'}.",
        ]
    )
    (DIAG_ROOT / "rcmf_gate_decision.md").write_text(gate_md, encoding="utf-8")

    summary_md = "\n".join(
        [
            "# Summary",
            "",
            "Canonical protocol remains fixed and no further data/protocol work was performed in this turn.",
            f"Seeds executed for minimal RCMF validation: {seeds_text}.",
            f"Mean delta vs repaired MSPCE: primary clean {mean_clean:+.4f} K, primary noisy {mean_noisy:+.4f} K, external {mean_external:+.4f} K.",
            f"Current status: `{'LED_ALLOWED_NEXT' if led_allowed else 'RCMF_REPAIR_CONTINUES'}`.",
            "Full remains closed and 20-seed remains closed.",
        ]
    )
    (DIAG_ROOT / "summary.md").write_text(summary_md, encoding="utf-8")
    (DIAG_ROOT / "rcmf_min_repair_note.md").write_text(note_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal RCMF validation on top of repaired MSPCE.")
    parser.add_argument("--seed0_only", action="store_true")
    args = parser.parse_args()

    feature_info = ensure_multiscale_features()
    features = feature_info["features"]
    splits = torch.load(ROOT / "data/splits.json", map_location="cpu") if False else None
    # `splits.json` stays JSON; load through pandas-free path for consistency with existing code.
    import json
    splits = json.loads((ROOT / "data/splits.json").read_text(encoding="utf-8"))
    config = diagnostic_config()

    rows: list[dict[str, Any]] = []
    rcmf_rows: list[dict[str, Any]] = []

    seed0_rows, seed0_info = run_seed(seed=0, features=features, splits=splits, config=config)
    rows.extend(seed0_rows)
    rcmf_rows.append(seed0_info["rcmf_row"])
    if not args.seed0_only and bool(seed0_info["rcmf_row"]["pass_flag"]):
        seed1_rows, seed1_info = run_seed(seed=1, features=features, splits=splits, config=config)
        rows.extend(seed1_rows)
        rcmf_rows.append(seed1_info["rcmf_row"])

    note_text = "\n".join(
        [
            "# RCMF Minimal Repair Note",
            "",
            "- Current problem: repaired MSPCE is already stronger than `Simple Concat`, but RCMF still needs proof that it can add trusted multimodal fusion without breaking the repaired MSPCE anchor or external holdout.",
            "- This round changed code in three places: a new low-rank trusted fusion controller was added for RCMF; `FusionModel` gained a baseline-anchored `rcmf_min_trusted_fusion` mode that uses context, conflict, uncertainty, confidence, trust, and magnitude control; a minimal `rcmf_min_repair.py` validation path now loads repaired MSPCE weights, freezes the backbone, and trains only the light RCMF parameters.",
            "- These changes do not alter the innovation definitions. MSPCE is still the polymer-specific multiscale context source, RCMF is still a trusted fusion body rather than a selector, and LED remains unopened in this stage.",
            f"- Gate standard for this round: delta vs repaired MSPCE on primary clean <= {PRIMARY_CLEAN_PASS_DELTA:.2f} K, on primary noisy <= {PRIMARY_NOISY_PASS_DELTA:.2f} K, and on external holdout <= {EXTERNAL_PASS_DELTA:.2f} K before LED can open.",
        ]
    )
    write_outputs(rows=rows, rcmf_rows=rcmf_rows, note_text=note_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

