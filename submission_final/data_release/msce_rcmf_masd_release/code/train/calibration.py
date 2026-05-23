from __future__ import annotations

"""Calibration and teacher-anchoring helpers for the active experiment stack.

This module holds the fixed feature ordering and loss helpers used by the
repair, fusion, and current-full training stages.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AnchorLossWeights:
    teacher_nonbenef: float
    teacher_benef: float
    residual_benef: float
    improve_benef: float
    gate_nonbenef: float
    gate_benef: float
    gate_match_benef: float
    delta_nonbenef: float
    delta_benef: float


CERT_FEATURE_NAMES = (
    # Keep this order fixed because saved certification rules index by position.
    "conflict_level",
    "uncertainty",
    "context_strength",
    "teacher_student_gap",
    "teacher_candidate_gap",
)


def copy_shared_weights(teacher: nn.Module, student: nn.Module) -> None:
    """Copy matching parameters when a later stage is initialized from an earlier one."""
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    shared = {
        key: value
        for key, value in teacher_state.items()
        if key in student_state and student_state[key].shape == value.shape
    }
    student_state.update(shared)
    student.load_state_dict(student_state)


def freeze_teacher_anchored_stage(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module_name in ("ctx_encoder", "ctx_delta_head", "rcmf_gate", "led_prior"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def beneficial_threshold(errors: np.ndarray, quantile: float) -> float:
    clean = np.asarray(errors, dtype=np.float64).reshape(-1)
    if clean.size == 0:
        return 0.0
    return float(np.quantile(clean, quantile))


def beneficial_mask(
    y_true: torch.Tensor,
    teacher_pred: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_error = torch.abs(teacher_pred - y_true)
    beneficial = (teacher_error >= float(threshold)).float()
    nonbeneficial = 1.0 - beneficial
    return beneficial, nonbeneficial


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def teacher_anchor_loss(
    *,
    prediction: torch.Tensor,
    teacher_pred: torch.Tensor,
    y_true: torch.Tensor,
    innovation_gate: torch.Tensor,
    innovation_delta: torch.Tensor,
    threshold: float,
    delta_limit: float,
    weights: AnchorLossWeights,
) -> dict[str, torch.Tensor]:
    beneficial, nonbeneficial = beneficial_mask(y_true=y_true, teacher_pred=teacher_pred, threshold=threshold)
    teacher_distance = torch.abs(prediction - teacher_pred)
    target_residual = torch.clamp(y_true - teacher_pred, min=-float(delta_limit), max=float(delta_limit))
    gate_target = torch.clamp(torch.abs(target_residual) / max(float(delta_limit), 1e-6), max=1.0)
    beneficial_margin = torch.relu(
        torch.abs(prediction - y_true) - torch.abs(teacher_pred - y_true) + 0.01
    )

    losses = {
        "teacher_nonbenef": masked_mean(teacher_distance, nonbeneficial),
        "teacher_benef": masked_mean(teacher_distance, beneficial),
        "residual_benef": masked_mean(torch.abs(innovation_delta - target_residual), beneficial),
        "improve_benef": masked_mean(beneficial_margin, beneficial),
        "gate_nonbenef": masked_mean(innovation_gate, nonbeneficial),
        "gate_benef": masked_mean(innovation_gate, beneficial),
        "gate_match_benef": masked_mean(torch.relu(gate_target - innovation_gate), beneficial),
        "delta_nonbenef": masked_mean(torch.abs(innovation_delta), nonbeneficial),
        "delta_benef": masked_mean(torch.abs(innovation_delta), beneficial),
    }
    total = (
        weights.teacher_nonbenef * losses["teacher_nonbenef"]
        + weights.teacher_benef * losses["teacher_benef"]
        + weights.residual_benef * losses["residual_benef"]
        + weights.improve_benef * losses["improve_benef"]
        + weights.gate_nonbenef * losses["gate_nonbenef"]
        + weights.gate_benef * losses["gate_benef"]
        + weights.gate_match_benef * losses["gate_match_benef"]
        + weights.delta_nonbenef * losses["delta_nonbenef"]
        + weights.delta_benef * losses["delta_benef"]
    )
    losses["total"] = total
    return losses


def fit_certification_rule(oof_df: Any) -> dict[str, Any]:
    matrix = np.asarray(oof_df[list(CERT_FEATURE_NAMES)], dtype=np.float64)
    gain = np.asarray(oof_df["oof_gain"], dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (matrix - mean) / std
    design = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)
    weights, _, _, _ = np.linalg.lstsq(design, gain, rcond=None)
    score = design @ weights

    pos_threshold = float(np.quantile(score, 0.80))
    best_key: tuple[float, float, float] | None = None
    for thr in np.unique(np.quantile(score, [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93])):
        mask = score >= float(thr)
        if not np.any(mask):
            continue
        precision = float(np.mean(gain[mask] > 0.0))
        mean_gain = float(np.mean(gain[mask]))
        coverage = float(np.mean(mask))
        key = (
            1.0 if precision >= 0.55 and mean_gain >= 0.25 else 0.0,
            mean_gain,
            precision,
            coverage,
        )
        if best_key is None or key > best_key:
            best_key = key
            pos_threshold = float(thr)

    neg_threshold = float(np.quantile(score, 0.20))
    best_neg_key: tuple[float, float] | None = None
    for thr in np.unique(np.quantile(score, [0.10, 0.15, 0.20, 0.25, 0.30, 0.35])):
        mask = score <= float(thr)
        if not np.any(mask):
            continue
        mean_gain = float(np.mean(gain[mask]))
        coverage = float(np.mean(mask))
        key = (1.0 if mean_gain < 0.0 else 0.0, -mean_gain + 0.05 * coverage)
        if best_neg_key is None or key > best_neg_key:
            best_neg_key = key
            neg_threshold = float(thr)

    return {
        "feature_names": list(CERT_FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weights": weights[:-1].tolist(),
        "bias": float(weights[-1]),
        "positive_threshold": pos_threshold,
        "negative_threshold": neg_threshold,
        "switch_threshold": pos_threshold,
        "gain_margin": pos_threshold,
    }


def certification_score_from_features(
    feature_matrix: np.ndarray,
    rule: dict[str, Any],
) -> np.ndarray:
    mean = np.asarray(rule["feature_mean"], dtype=np.float64)
    std = np.asarray(rule["feature_std"], dtype=np.float64)
    weights = np.asarray(rule["weights"], dtype=np.float64)
    bias = float(rule["bias"])
    z = (np.asarray(feature_matrix, dtype=np.float64) - mean) / std
    return z @ weights + bias


def summarize_certification_rule(
    oof_df: Any,
    rule: dict[str, Any],
) -> dict[str, Any]:
    matrix = np.asarray(oof_df[list(CERT_FEATURE_NAMES)], dtype=np.float64)
    gain = np.asarray(oof_df["oof_gain"], dtype=np.float64)
    score = certification_score_from_features(matrix, rule)
    positive = score >= float(rule["positive_threshold"])
    negative = score <= float(rule["negative_threshold"])
    uncertain = ~(positive | negative)
    positive_coverage = float(np.mean(positive))
    positive_precision = float(np.mean(gain[positive] > 0.0)) if np.any(positive) else 0.0
    positive_gain = float(np.mean(gain[positive])) if np.any(positive) else 0.0
    negative_coverage = float(np.mean(negative))
    negative_gain = float(np.mean(gain[negative])) if np.any(negative) else 0.0
    uncertain_coverage = float(np.mean(uncertain))
    weight_terms = []
    for name, weight in zip(rule["feature_names"], rule["weights"], strict=True):
        weight_terms.append(f"{weight:+.4f}*z({name})")
    rule_text = (
        "estimated_gain_k = "
        + " ".join(weight_terms)
        + f" {float(rule['bias']):+.4f}; "
        + f"switch if estimated_gain_k >= {float(rule['positive_threshold']):.4f}; "
        + f"teacher if estimated_gain_k <= {float(rule['negative_threshold']):.4f}; "
        + "otherwise uncertain -> teacher fallback"
    )
    return {
        "positive_switch_region_coverage": positive_coverage,
        "positive_switch_region_precision": positive_precision,
        "positive_switch_region_gain": positive_gain,
        "teacher_region_coverage": negative_coverage,
        "teacher_region_gain": negative_gain,
        "uncertain_region_coverage": uncertain_coverage,
        "arbitration_rule": rule_text,
        "switch_threshold": float(rule["positive_threshold"]),
        "gain_margin": float(rule.get("gain_margin", rule["positive_threshold"])),
        "certified_positive_region_coverage": positive_coverage,
        "certified_positive_region_precision": positive_precision,
        "certified_positive_region_gain": positive_gain,
        "certified_negative_region_coverage": negative_coverage,
        "certified_negative_region_gain": negative_gain,
        "certification_rule": rule_text,
    }

