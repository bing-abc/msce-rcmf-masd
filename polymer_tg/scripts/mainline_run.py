from __future__ import annotations

"""Primary experiment driver for the locked MSCE-RCMF-MASD chain."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
DIAG_ROOT = ROOT / "outputs" / "exp" / "diagnostics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.calibration import copy_shared_weights  # noqa: E402
from train.full_train import (  # noqa: E402
    DEVICE,
    OTHER_SUBCLUSTER_COUNT,
    _to_device,
    build_model,
    diagnostic_config,
    gpu_info,
    load_artifacts,
    make_loader,
    prepare_seed_tensors,
    set_seed,
    stable_seed,
    train_standard_model,
)
from train.msce_stage import ensure_msce_features, train_msce_stage  # noqa: E402
from train.rcmf_stage import train_rcmf_external_focus_stage, train_rcmf_stage  # noqa: E402
import train.rcmf_min_repair as rcmf_min_repair  # noqa: E402

SMOKE_SEED = 18
# Stage aliases are retained only to compare the locked predecessor with the final defended mainline.
LEGACY_CURRENT_MODE = "main_core_sci2_masd_current_locked"
CURRENT_MODE = "main_core_sci2_masd_final"
LEGACY_CURRENT_STAGE_NAME = "strongest_baseline_plus_mspce_rcmf_masd_current_locked"
CURRENT_STAGE_NAME = "strongest_baseline_plus_mspce_rcmf_masd_final"
CURRENT_STAGE_ALIASES = (CURRENT_STAGE_NAME, LEGACY_CURRENT_STAGE_NAME)
DEFAULT_MAINLINE_SEEDS = tuple(range(10, 20))
DEFAULT_EXTERNAL_SUPPORTING_SEEDS = (15, 16, 17, 18, 19)
DEFAULT_ABLATION_SEEDS = (15, 16, 17, 18, 19)
TRISOUP_100RUN_PREFIX = "masd_final_trisoup_100run"
TRISOUP_100RUN_NUM_RUNS = 100
TRISOUP_WEIGHTLOCK_SCAN_PREFIX = "masd_final_trisoup_weightlock_scan"
TRISOUP_WEIGHTLOCK_100RUN_PREFIX = "masd_final_trisoup_weightlock_100run"
TRISOUP_WEIGHTLOCK_SCAN_NUM_RUNS = 20
TRISOUP_WEIGHTLOCK_CONFIRM_START_SEED = 30
SIMPLECONCAT_TERNARY_CONTROL_PREFIX = "masd_final_simpleconcat_ternary_control"
SIMPLECONCAT_TERNARY_MEMBER_COUNT = 3
SIMPLECONCAT_TERNARY_MEMBER_SEED_STRIDE = 1009
TRISOUP_LOCAL_EXTRA_COEFFS = (
    (0.70, 0.30, 0.00),
    (0.75, 0.25, 0.00),
    (0.80, 0.20, 0.00),
)
USE_EXTERNAL_HOLDOUT_FOR_SELECTION = False
USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION = False
MECHANISM_NAMES = (
    "rigidity_rotation",
    "intermolecular_polarity",
    "freevolume_packing",
    "sidechain_internalplasticization",
)
CHEMISTRY_CLUSTER_ORDER = (
    "aromatic_dense",
    "ester_or_carbonate",
    "fluorinated",
    "sulfone",
    "amide",
    "ether_oxygen",
    "imide_like",
    "other",
)
CHEMISTRY_CLUSTER_INDEX = {name: idx for idx, name in enumerate(CHEMISTRY_CLUSTER_ORDER)}
CHEMISTRY_TAIL_LOSS_WEIGHTS = {
    "imide_like": 1.00,
    "other": 0.85,
}
PRIMARY_CLEAN_PASS_DELTA = 0.10
PRIMARY_NOISY_PASS_DELTA = 0.05
EXTERNAL_PASS_DELTA = 0.30
TAILFIX_PRIMARY_EPSILON = 0.015
SIGNRATE_LOCK_PRIMARY_EPSILON = 0.020
STABILIZATION_PRIMARY_EPSILON = 0.020
SPLITHEAD_PRIMARY_EPSILON = 0.020
SELF_STABILIZATION_PRIMARY_EPSILON = 0.020
JTT_STABILIZATION_PRIMARY_EPSILON = 0.020
CTGF_PRIMARY_EPSILON = 0.020
TRISOUP_PRIMARY_EPSILON = 0.020
WEAK_CLUSTER_CODE = 1
UNSTABLE_CLUSTER_CODE = 2
LOCKED_FILES = (
    ROOT / "models" / "fusion.py",
    ROOT / "polymer_tg" / "scripts" / "mainline_run.py",
    ROOT / "polymer_tg" / "scripts" / "mainline_eval.py",
)


def is_weightlock_100run_prefix(output_prefix: str) -> bool:
    return output_prefix == TRISOUP_WEIGHTLOCK_100RUN_PREFIX or (
        output_prefix.startswith("masd_final_trisoup_weightlock_") and output_prefix.endswith("_100run")
    )


def is_trisoup_100run_prefix(output_prefix: str) -> bool:
    return output_prefix == TRISOUP_100RUN_PREFIX or (
        output_prefix.startswith("masd_final_trisoup_")
        and output_prefix.endswith("_100run")
        and not is_weightlock_100run_prefix(output_prefix)
    )


def trisoup_candidate_coefficients() -> list[tuple[float, float, float]]:
    coeffs = {tuple(float(item) for item in coeff) for coeff in simplex_coefficients(step=0.25)}
    coeffs.update(tuple(float(item) for item in coeff) for coeff in TRISOUP_LOCAL_EXTRA_COEFFS)
    ordered = list(coeffs)
    ordered.sort(key=lambda item: (sum(weight > 0.0 for weight in item), item[0], item[1], item[2]))
    return ordered
CHEMISTRY_TAG_LOOKUP: dict[int, tuple[str, ...]] = {}

def parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_weight_list(raw: str) -> tuple[float, float, float]:
    weights = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if len(weights) != 3:
        raise ValueError(f"trisoup weights must contain exactly 3 values, got {raw!r}")
    if float(sum(weights)) <= 0.0:
        raise ValueError(f"trisoup weights must sum to a positive value, got {raw!r}")
    return weights


def weight_key(weights: tuple[float, float, float] | list[float]) -> str:
    return ",".join(f"{float(weight):.2f}" for weight in weights)


def load_locked_weight_choice() -> tuple[str, tuple[float, float, float]] | None:
    path = DIAG_ROOT / TRISOUP_WEIGHTLOCK_SCAN_PREFIX / "best_candidate.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    mode = str(payload.get("fixed_mode", "weight"))
    weights = tuple(float(item) for item in payload.get("weights", []))
    if len(weights) != 3:
        raise RuntimeError(f"invalid locked weight choice in {path}")
    return mode, weights


def _split_bins(y: np.ndarray, n_bins: int = 14) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y, quantiles))
    if len(edges) < 3:
        return np.zeros_like(y, dtype=np.int64)
    return np.digitize(y, edges[1:-1], right=True).astype(np.int64)


def build_protocol_split(dataset: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    primary_idx = dataset.index[dataset["role"] == "primary_pool"].to_numpy()
    supplemental_idx = dataset.index[dataset["role"] == "supplemental_train"].to_numpy()
    external_idx = dataset.index[dataset["role"] == "external_holdout"].to_numpy()

    primary_y = dataset.loc[primary_idx, "tg_k"].to_numpy(dtype=np.float32)
    primary_bins = _split_bins(primary_y, n_bins=14)

    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=int(seed))
    train_val_pos, test_pos = next(outer.split(primary_idx, primary_bins))
    train_val_idx = primary_idx[train_val_pos]
    test_idx = primary_idx[test_pos]

    train_val_bins = primary_bins[train_val_pos]
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=int(seed))
    train_pos, val_pos = next(inner.split(train_val_idx, train_val_bins))
    train_idx = np.concatenate([train_val_idx[train_pos], supplemental_idx]).astype(np.int64)
    val_idx = train_val_idx[val_pos].astype(np.int64)
    test_idx = test_idx.astype(np.int64)
    external_idx = external_idx.astype(np.int64)
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
        "external": external_idx.tolist(),
        "counts": {
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_external": int(len(external_idx)),
        },
    }


def ensure_protocol_split(splits: dict[str, Any], dataset: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    seed_key = str(int(seed))
    seeds_payload = splits.setdefault("seeds", {})
    if seed_key not in seeds_payload:
        seeds_payload[seed_key] = build_protocol_split(dataset, seed=int(seed))
    return seeds_payload[seed_key]


def enable_determinism(*, strict: bool = False) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def current_stage_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in rows if row["model_name"] in CURRENT_STAGE_ALIASES)


def is_current_stage_name(name: str) -> bool:
    return name in CURRENT_STAGE_ALIASES


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_bundle(run_dir: Path, name: str, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / f"{name}.pt"
    tmp_path = run_dir / f".{name}.pt.tmp"
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)


def save_results_csv(run_dir: Path, output_prefix: str, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "results.csv"
    tmp_path = run_dir / ".results.csv.tmp"
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(final_path)
    if run_dir.name == output_prefix:
        legacy_path = DIAG_ROOT / f"{output_prefix}_results.csv"
        legacy_tmp = DIAG_ROOT / f".{output_prefix}_results.csv.tmp"
        pd.DataFrame(rows).to_csv(legacy_tmp, index=False)
        legacy_tmp.replace(legacy_path)


def lock_snapshot() -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "sha256": file_sha256(path),
            "mtime": path.stat().st_mtime,
        }
        for path in LOCKED_FILES
    }


def ensure_gpu() -> dict[str, Any]:
    payload = gpu_info()
    if not payload["gpu_used"]:
        raise RuntimeError("main_core_sci2_masd_final requires CUDA; CPU fallback is not allowed.")
    payload["device_count"] = int(torch.cuda.device_count())
    return payload


def _subgroup_mae(errors: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.float()
    denom = float(mask.sum().item())
    if denom <= 0.0:
        return float(errors.mean().item())
    return float((errors * mask).sum().item() / denom)


def _hard_mask(hard_score: torch.Tensor) -> torch.Tensor:
    threshold = torch.quantile(hard_score.squeeze(1), 0.80)
    return (hard_score.squeeze(1) >= threshold).float().unsqueeze(1)


rcmf_min_repair.DEVICE = DEVICE
rcmf_min_repair.subgroup_mae = _subgroup_mae


def stage_pass(curr: dict[str, float], prev: dict[str, float]) -> bool:
    return bool(
        (curr["primary_clean"] - prev["primary_clean"]) <= PRIMARY_CLEAN_PASS_DELTA
        and (curr["primary_noisy"] - prev["primary_noisy"]) <= PRIMARY_NOISY_PASS_DELTA
        and (curr["external_holdout"] - prev["external_holdout"]) <= EXTERNAL_PASS_DELTA
    )


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def chemistry_tags(smiles: str) -> list[str]:
    text = str(smiles or "")
    tags: list[str] = []
    if "S(=O)(=O)" in text:
        tags.append("sulfone")
    if "NC(=O)" in text:
        tags.append("amide")
    if "n(" in text and "=O" in text:
        tags.append("imide_like")
    if "OC(=O)" in text:
        tags.append("ester_or_carbonate")
    if "O" in text and "OC(=O)" not in text:
        tags.append("ether_oxygen")
    if "F" in text:
        tags.append("fluorinated")
    if text.count("c") >= 6:
        tags.append("aromatic_dense")
    if not tags:
        tags.append("other")
    return tags


def build_chemistry_tag_lookup(dataset: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    lookup: dict[int, tuple[str, ...]] = {}
    for idx, smiles in zip(dataset.index.tolist(), dataset["canonical_smiles"].tolist()):
        lookup[int(idx)] = tuple(chemistry_tags(str(smiles)))
    return lookup


def external_cluster_masks(dataset: pd.DataFrame) -> dict[str, np.ndarray]:
    external_df = dataset[dataset["role"] == "external_holdout"].reset_index(drop=True)
    cluster_map = {
        cluster_name: np.asarray(
            [cluster_name in chemistry_tags(str(smiles)) for smiles in external_df["canonical_smiles"].tolist()],
            dtype=bool,
        )
        for cluster_name in CHEMISTRY_CLUSTER_ORDER
    }
    return {name: mask for name, mask in cluster_map.items() if bool(mask.any())}


@torch.no_grad()
def evaluate_stage(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    seed_tensors: dict[str, Any],
    *,
    variant: str,
    noise_seed: int,
    return_payload: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    # Stage evaluation collects both headline metrics and the payload used by paper diagnostics.
    model.eval()
    generator = torch.Generator(device=DEVICE.type if DEVICE.type != "cpu" else "cpu")
    generator.manual_seed(int(noise_seed))

    bucket: dict[str, list[torch.Tensor]] = {
        "y": [],
        "pred": [],
        "error": [],
        "conflict": [],
        "uncertainty": [],
        "hard_score": [],
        "cluster_code": [],
        "cluster_support": [],
        "other_subcluster_code": [],
        "sample_index": [],
    }
    extra_keys = (
        "masd_alpha",
        "masd_delta",
        "masd_contribution",
        "masd_proxy_scores",
        "masd_proxy_target",
        "masd_signed_proxy_target",
        "masd_entropy",
        "masd_diversity",
        "masd_dominant_mechanism",
        "masd_slot_hidden",
        "masd_gate",
        "masd_thresholded_gate",
        "masd_thresholded_delta",
        "masd_applied_delta",
        "masd_alpha_max",
        "masd_alpha_margin",
        "masd_main_mag",
        "masd_mechanism_disagreement",
        "masd_gate_consistency",
        "masd_alpha_consistency",
    )
    has_masd = False

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
        error = torch.abs(pred - y)
        conflict = out["conflict_level"].detach().cpu()
        uncertainty = out["uncertainty_level"].detach().cpu()
        if "masd_alpha" in out:
            has_masd = True
            max_alpha = out["masd_alpha"].detach().cpu().max(dim=1, keepdim=True).values
            disagreement = out["masd_mechanism_disagreement"].detach().cpu()
            hard_score = conflict + 1.20 * uncertainty + 0.40 * (1.0 - max_alpha) + 0.30 * disagreement
        else:
            confidence = out.get("rcmf_min_confidence", torch.ones_like(out["pred"])).detach().cpu()
            hard_score = conflict + 1.35 * uncertainty + 0.40 * (1.0 - confidence)
        bucket["y"].append(y)
        bucket["pred"].append(pred)
        bucket["error"].append(error)
        bucket["conflict"].append(conflict)
        bucket["uncertainty"].append(uncertainty)
        bucket["hard_score"].append(hard_score)
        bucket["cluster_code"].append(batch["cluster_code"].detach().cpu())
        bucket["cluster_support"].append(batch["cluster_support"].detach().cpu())
        bucket["other_subcluster_code"].append(batch["other_subcluster_code"].detach().cpu())
        bucket["sample_index"].append(batch["sample_index"].detach().cpu())
        if return_payload and "masd_alpha" in out:
            model.train()
            out_stochastic = model(batch["graph"], desc, ctx, led=batch["led"], led_mask=batch["led_mask"])
            model.eval()
            out["masd_gate_consistency"] = (out["masd_gate"] - out_stochastic["masd_gate"]).abs()
            out["masd_alpha_consistency"] = (out["masd_alpha"] - out_stochastic["masd_alpha"]).abs().mean(dim=1, keepdim=True)
            for key in extra_keys:
                value = out[key].detach().cpu()
                bucket.setdefault(key, []).append(value)

    merged = {key: torch.cat(values, dim=0) for key, values in bucket.items()}
    hard_mask = _hard_mask(merged["hard_score"])
    metrics = {
        "mae_k": float(merged["error"].mean().item()),
        "hard_subgroup_mae_k": _subgroup_mae(merged["error"], hard_mask),
        "hard_mask_rate": float(hard_mask.mean().item()),
        "conflict_mean": float(merged["conflict"].mean().item()),
        "uncertainty_mean": float(merged["uncertainty"].mean().item()),
    }
    if not return_payload:
        return metrics, None
    payload: dict[str, np.ndarray] = {
        "y_true": merged["y"].numpy().squeeze(1),
        "pred": merged["pred"].numpy().squeeze(1),
        "error": merged["error"].numpy().squeeze(1),
        "hard_score": merged["hard_score"].numpy().squeeze(1),
        "hard_mask": hard_mask.numpy().squeeze(1),
        "conflict": merged["conflict"].numpy().squeeze(1),
        "uncertainty": merged["uncertainty"].numpy().squeeze(1),
        "cluster_code": merged["cluster_code"].numpy().squeeze(1),
        "cluster_support": merged["cluster_support"].numpy().squeeze(1),
        "other_subcluster_code": merged["other_subcluster_code"].numpy().squeeze(1),
        "sample_index": merged["sample_index"].numpy().reshape(-1),
    }
    if has_masd:
        for key in extra_keys:
            payload[key] = merged[key].numpy()
    return metrics, payload


def contribution_metrics_from_payload(
    clean_payload: dict[str, np.ndarray],
    noisy_payload: dict[str, np.ndarray] | None,
) -> dict[str, float]:
    alpha = np.asarray(clean_payload["masd_alpha"], dtype=np.float64)
    contribution = np.asarray(clean_payload["masd_contribution"], dtype=np.float64)
    proxy_scores = np.asarray(clean_payload["masd_proxy_scores"], dtype=np.float64)
    signed_proxy = np.asarray(clean_payload["masd_signed_proxy_target"], dtype=np.float64)
    entropy = np.asarray(clean_payload["masd_entropy"], dtype=np.float64).reshape(-1)
    diversity = np.asarray(clean_payload["masd_diversity"], dtype=np.float64).reshape(-1)
    slot_hidden = np.asarray(clean_payload["masd_slot_hidden"], dtype=np.float64)
    dominant_clean = np.asarray(clean_payload["masd_dominant_mechanism"], dtype=np.int64)
    slot_count = int(min(alpha.shape[1], contribution.shape[1], proxy_scores.shape[1], signed_proxy.shape[1]))
    if slot_count <= 0:
        raise RuntimeError("invalid MASD payload with zero slot count")
    dominant_proxy = proxy_scores.argmax(axis=1)
    head_corr = [safe_corr(alpha[:, idx], proxy_scores[:, idx]) for idx in range(slot_count)]
    contribution_corr = [safe_corr(contribution[:, idx], signed_proxy[:, idx]) for idx in range(slot_count)]
    sign_consistency = []
    for idx in range(slot_count):
        mask = dominant_proxy == idx
        if not np.any(mask):
            sign_consistency.append(0.5)
            continue
        sign_consistency.append(float(np.mean(np.sign(contribution[mask, idx]) == np.sign(signed_proxy[mask, idx]))))
    collapse_flags = []
    for sample_hidden in slot_hidden:
        norm = sample_hidden / np.linalg.norm(sample_hidden, axis=1, keepdims=True).clip(min=1e-6)
        pair = norm @ norm.T
        offdiag = pair[np.triu_indices(pair.shape[0], 1)]
        collapse_flags.append(float(np.max(np.abs(offdiag)) > 0.95))
    noisy_dominant = dominant_clean
    if noisy_payload is not None and "masd_dominant_mechanism" in noisy_payload:
        noisy_dominant = np.asarray(noisy_payload["masd_dominant_mechanism"], dtype=np.int64)
    gate = np.asarray(clean_payload["masd_gate"], dtype=np.float64).reshape(-1)
    alpha_max = np.asarray(clean_payload["masd_alpha_max"], dtype=np.float64).reshape(-1)
    disagreement = np.asarray(clean_payload["masd_mechanism_disagreement"], dtype=np.float64).reshape(-1)
    gate_consistency = np.asarray(clean_payload["masd_gate_consistency"], dtype=np.float64).reshape(-1)
    alpha_consistency = np.asarray(clean_payload["masd_alpha_consistency"], dtype=np.float64).reshape(-1)
    error = np.asarray(clean_payload["error"], dtype=np.float64).reshape(-1)
    conflict = np.asarray(clean_payload["conflict"], dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(clean_payload["uncertainty"], dtype=np.float64).reshape(-1)
    high_conflict = conflict >= np.quantile(conflict, 0.80)
    high_uncertainty = uncertainty >= np.quantile(uncertainty, 0.80)
    risk_score = conflict + 1.20 * uncertainty + 0.90 * entropy + 0.85 * disagreement + 0.65 * (1.0 - alpha_max)
    high_risk = risk_score >= np.quantile(risk_score, 0.80)
    low_risk = risk_score <= np.quantile(risk_score, 0.40)
    row = {
        "masd_slot_count": float(slot_count),
        "mechanism_anchor_alignment_corr": float(np.mean(head_corr)),
        "mechanism_anchor_alignment_rank": float(np.mean(dominant_clean == dominant_proxy)),
        "contribution_anchor_alignment_corr": float(np.mean(contribution_corr)),
        "contribution_sign_consistency": float(np.mean(sign_consistency)),
        "mechanism_weight_sparsity": float(np.mean(1.0 - entropy)),
        "alpha_sparsity_low_risk": float(np.mean(1.0 - entropy[low_risk])) if np.any(low_risk) else float(np.mean(1.0 - entropy)),
        "alpha_sparsity_high_risk": float(np.mean(1.0 - entropy[high_risk])) if np.any(high_risk) else float(np.mean(1.0 - entropy)),
        "dominant_mechanism_concentration": float(np.mean(alpha_max)),
        "dominant_slot_concentration_low_risk": float(np.mean(alpha_max[low_risk])) if np.any(low_risk) else float(np.mean(alpha_max)),
        "dominant_slot_concentration_high_risk": float(np.mean(alpha_max[high_risk])) if np.any(high_risk) else float(np.mean(alpha_max)),
        "mechanism_head_diversity": float(np.mean(diversity)),
        "dominant_mechanism_stability": float(np.mean(dominant_clean == noisy_dominant)),
        "head_collapse_ratio": float(np.mean(collapse_flags)),
        "high_conflict_gate_mean": float(gate[high_conflict].mean()) if np.any(high_conflict) else float(gate.mean()),
        "high_uncertainty_gate_mean": float(gate[high_uncertainty].mean()) if np.any(high_uncertainty) else float(gate.mean()),
        "low_risk_gate_mean": float(gate[low_risk].mean()) if np.any(low_risk) else float(gate.mean()),
        "gate_consistency_high_risk": float(gate_consistency[high_risk].mean()) if np.any(high_risk) else float(gate_consistency.mean()),
        "alpha_consistency_high_risk": float(alpha_consistency[high_risk].mean()) if np.any(high_risk) else float(alpha_consistency.mean()),
        "gate_vs_error_correlation": safe_corr(gate, error),
        "mechanism_disagreement_mean": float(np.mean(disagreement)),
    }
    row["mechanism_pass"] = bool(
        row["mechanism_anchor_alignment_corr"] >= 0.12
        and row["contribution_anchor_alignment_corr"] >= 0.10
        and row["contribution_sign_consistency"] >= 0.85
        and row["mechanism_weight_sparsity"] >= 0.22
        and row["dominant_mechanism_concentration"] >= 0.46
        and row["mechanism_head_diversity"] >= 0.18
        and row["dominant_mechanism_stability"] >= 0.45
        and row["head_collapse_ratio"] <= 0.35
    )
    return row


def common_mask_delta(student_payload: dict[str, np.ndarray], anchor_payload: dict[str, np.ndarray]) -> float:
    mask = np.asarray(anchor_payload["hard_mask"], dtype=bool).reshape(-1)
    if not np.any(mask):
        return 0.0
    student_error = np.asarray(student_payload["error"], dtype=np.float64).reshape(-1)
    anchor_error = np.asarray(anchor_payload["error"], dtype=np.float64).reshape(-1)
    return float(student_error[mask].mean() - anchor_error[mask].mean())


def set_masd_trainable(model: nn.Module, *, stage: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if bool(getattr(model, "pr_thresholded_masd_focus_only", False)):
        focus_modules = {
            "masd_core_proj",
            "masd_slot_bank",
            "masd_slot_proj",
            "masd_alpha_head",
            "masd_mag_head",
            "masd_res_head",
            "masd_calib_context_head",
            "masd_safety_gate_v3",
            "masd_gate_context_head",
        }
        for module_name in focus_modules:
            module = getattr(model, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True
        for param_name in (
            "masd_calib_proxy_scale",
            "masd_calib_proxy_bias",
            "masd_alpha_risk_weights",
            "masd_alpha_risk_bias",
            "masd_gate_risk_weights",
            "masd_gate_bias",
        ):
            param = getattr(model, param_name, None)
            if param is not None:
                param.requires_grad = True
        return
    stage_a = {
        "masd_core_proj",
        "masd_slot_bank",
        "masd_slot_proj",
        "masd_alpha_head",
        "masd_mag_head",
        "masd_res_head",
        "masd_calib_context_head",
        "masd_safety_gate_v3",
        "masd_gate_context_head",
    }
    stage_b_extra = {
        "ctx_scale_pool",
        "mspce_context_injector",
        "mspce_repair_gate",
        "rcmf_min_fusion",
        "rcmf_min_delta_head",
    }
    active = set(stage_a)
    if stage in {"B", "C"}:
        active.update(stage_b_extra)
    for module_name in active:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
    for param_name in (
        "masd_calib_proxy_scale",
        "masd_calib_proxy_bias",
        "masd_alpha_risk_weights",
        "masd_alpha_risk_bias",
        "masd_gate_risk_weights",
        "masd_gate_bias",
    ):
        param = getattr(model, param_name, None)
        if param is not None:
            param.requires_grad = True


def pairwise_rank_loss(contrib_abs: torch.Tensor, proxy_target: torch.Tensor) -> torch.Tensor:
    proxy_diff = proxy_target.unsqueeze(2) - proxy_target.unsqueeze(1)
    contrib_diff = contrib_abs.unsqueeze(2) - contrib_abs.unsqueeze(1)
    rank_mask = (proxy_diff > 0.02).float()
    if float(rank_mask.sum().item()) <= 0.0:
        return torch.zeros((), device=contrib_abs.device, dtype=contrib_abs.dtype)
    rank_penalty = torch.relu(0.004 - contrib_diff) * rank_mask
    return rank_penalty.sum() / rank_mask.sum().clamp_min(1.0)


def group_dro_lite_loss(
    sample_loss: torch.Tensor,
    *,
    uncertainty: torch.Tensor,
    conflict: torch.Tensor,
    entropy: torch.Tensor,
    disagreement: torch.Tensor,
    cluster_support: torch.Tensor,
    hard_like_mask: torch.Tensor,
    enabled: bool,
    cap: float = 0.48,
    temperature: float = 0.12,
    warmup: float = 1.0,
) -> torch.Tensor:
    if not enabled:
        return torch.zeros((), device=sample_loss.device, dtype=sample_loss.dtype)
    bins = []
    for tensor in (uncertainty, conflict, entropy, disagreement):
        threshold = torch.quantile(tensor.detach().reshape(-1), 0.5)
        bins.append((tensor >= threshold).long())
    cluster_id = torch.clamp(cluster_support.long(), min=0, max=2)
    hard_id = hard_like_mask.long()
    group_id = ((((bins[0] * 2 + bins[1]) * 2 + bins[2]) * 2 + bins[3]) * 2 + hard_id) * 3 + cluster_id
    group_losses = []
    for gid in range(96):
        mask = (group_id == gid).reshape(-1)
        if bool(mask.any()):
            group_losses.append(sample_loss.reshape(-1)[mask].mean())
    if not group_losses:
        return torch.zeros((), device=sample_loss.device, dtype=sample_loss.dtype)
    stacked = torch.stack(group_losses)
    capped = torch.clamp(stacked.detach(), max=cap)
    weights = torch.softmax(capped / temperature, dim=0)
    uniform = torch.full_like(weights, 1.0 / weights.numel())
    mix = float(max(0.0, min(1.0, warmup)))
    weights = mix * weights + (1.0 - mix) * uniform
    return (weights * stacked).sum()


def select_tailfix_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + TAILFIX_PRIMARY_EPSILON]
    eligible.sort(
        key=lambda item: (
            item["val_hard"],
            item["anchor_mask_delta"],
            item["gate_risk_penalty"],
            item["val_primary"],
            item["val_noisy"],
            float(item["val_external"]) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
        )
    )
    return eligible[0]


def candidate_proxy_hard_metrics(
    student_payload: dict[str, np.ndarray],
    anchor_payload: dict[str, np.ndarray],
) -> dict[str, float]:
    anchor_risk = np.asarray(anchor_payload["hard_score"], dtype=np.float64).reshape(-1)
    student_error = np.asarray(student_payload["error"], dtype=np.float64).reshape(-1)
    anchor_error = np.asarray(anchor_payload["error"], dtype=np.float64).reshape(-1)
    delta = student_error - anchor_error
    gate = np.asarray(student_payload.get("masd_gate", np.zeros_like(delta)), dtype=np.float64).reshape(-1)
    gate_consistency = np.asarray(
        student_payload.get("masd_gate_consistency", np.zeros_like(delta)),
        dtype=np.float64,
    ).reshape(-1)
    cluster_code = np.asarray(student_payload.get("cluster_code", np.zeros_like(delta)), dtype=np.int64).reshape(-1)
    cluster_support = np.asarray(student_payload.get("cluster_support", np.zeros_like(delta)), dtype=np.int64).reshape(-1)
    ambiguity = np.maximum(
        np.asarray(student_payload.get("masd_entropy", np.zeros_like(delta)), dtype=np.float64).reshape(-1),
        np.asarray(student_payload.get("masd_mechanism_disagreement", np.zeros_like(delta)), dtype=np.float64).reshape(-1),
    )
    if anchor_risk.size == 0:
        return {
            "proxy_hard_positive_bin_count": 0.0,
            "proxy_hard_worst_bin_delta": 0.0,
            "proxy_hard_bin_variance": 0.0,
            "proxy_hard_mean_delta": 0.0,
            "gate_volatility": 0.0,
            "weak_cluster_worst_delta": 0.0,
            "weak_cluster_mean_delta": 0.0,
            "weak_cluster_positive_count": 0.0,
            "ambiguity_variance": 0.0,
        }

    high_mask = anchor_risk >= np.quantile(anchor_risk, 0.60)
    if not np.any(high_mask):
        high_mask = np.ones_like(anchor_risk, dtype=bool)
    high_risk = anchor_risk[high_mask]
    high_delta = delta[high_mask]
    high_gate = gate[high_mask]
    high_gate_consistency = gate_consistency[high_mask]

    quantiles = np.quantile(high_risk, [0.25, 0.50, 0.75])
    bin_edges = [-np.inf, float(quantiles[0]), float(quantiles[1]), float(quantiles[2]), np.inf]
    bin_means: list[float] = []
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        if np.isposinf(high):
            mask = high_risk >= low
        else:
            mask = (high_risk >= low) & (high_risk < high)
        if np.any(mask):
            bin_means.append(float(np.mean(high_delta[mask])))
    if not bin_means:
        bin_means = [float(np.mean(high_delta))]

    weak_mask = cluster_support == WEAK_CLUSTER_CODE
    weak_cluster_means: list[float] = []
    if np.any(weak_mask):
        for code in sorted(set(cluster_code[weak_mask].tolist())):
            code_mask = weak_mask & (cluster_code == code)
            if np.any(code_mask):
                weak_cluster_means.append(float(np.mean(delta[code_mask])))

    return {
        "proxy_hard_positive_bin_count": float(sum(item > 0.0 for item in bin_means)),
        "proxy_hard_worst_bin_delta": float(max(bin_means)),
        "proxy_hard_bin_variance": float(np.var(np.asarray(bin_means, dtype=np.float64))),
        "proxy_hard_mean_delta": float(np.mean(high_delta)),
        "gate_volatility": float(np.mean(high_gate_consistency)) if high_gate_consistency.size else float(np.std(high_gate)),
        "weak_cluster_worst_delta": float(max(weak_cluster_means)) if weak_cluster_means else 0.0,
        "weak_cluster_mean_delta": float(np.mean(weak_cluster_means)) if weak_cluster_means else 0.0,
        "weak_cluster_positive_count": float(sum(item > 0.0 for item in weak_cluster_means)),
        "ambiguity_variance": float(np.var(ambiguity[high_mask])) if ambiguity[high_mask].size else 0.0,
    }


def candidate_chemistry_cluster_metrics(
    student_payload: dict[str, np.ndarray],
    anchor_payload: dict[str, np.ndarray],
) -> dict[str, float]:
    sample_index = np.asarray(student_payload.get("sample_index", []), dtype=np.int64).reshape(-1)
    student_error = np.asarray(student_payload.get("error", []), dtype=np.float64).reshape(-1)
    anchor_error = np.asarray(anchor_payload.get("error", []), dtype=np.float64).reshape(-1)
    if sample_index.size == 0 or student_error.size == 0 or student_error.size != anchor_error.size:
        return {
            "chem_cluster_worst_delta": 0.0,
            "chem_cluster_mean_delta": 0.0,
            "chem_cluster_positive_count": 0.0,
            "chem_cluster_count": 0.0,
            "other_subcluster_worst_delta": 0.0,
            "other_subcluster_mean_delta": 0.0,
            "other_subcluster_positive_count": 0.0,
            "other_subcluster_count": 0.0,
        }

    delta = student_error - anchor_error
    cluster_means: list[float] = []
    for cluster_name in CHEMISTRY_CLUSTER_ORDER:
        mask = np.asarray(
            [cluster_name in CHEMISTRY_TAG_LOOKUP.get(int(idx), ("other",)) for idx in sample_index.tolist()],
            dtype=bool,
        )
        if int(mask.sum()) < 3:
            continue
        cluster_means.append(float(np.mean(delta[mask])))

    if not cluster_means:
        return {
            "chem_cluster_worst_delta": 0.0,
            "chem_cluster_mean_delta": 0.0,
            "chem_cluster_positive_count": 0.0,
            "chem_cluster_count": 0.0,
            "other_subcluster_worst_delta": 0.0,
            "other_subcluster_mean_delta": 0.0,
            "other_subcluster_positive_count": 0.0,
            "other_subcluster_count": 0.0,
        }

    cluster_arr = np.asarray(cluster_means, dtype=np.float64)
    other_subcluster_code = np.asarray(student_payload.get("other_subcluster_code", np.full_like(delta, -1)), dtype=np.int64).reshape(-1)
    other_cluster_means: list[float] = []
    other_mask = other_subcluster_code >= 0
    if np.any(other_mask):
        for code in range(OTHER_SUBCLUSTER_COUNT):
            code_mask = other_mask & (other_subcluster_code == code)
            if int(code_mask.sum()) < 3:
                continue
            other_cluster_means.append(float(np.mean(delta[code_mask])))
    other_arr = np.asarray(other_cluster_means, dtype=np.float64) if other_cluster_means else np.zeros(0, dtype=np.float64)
    return {
        "chem_cluster_worst_delta": float(np.max(cluster_arr)),
        "chem_cluster_mean_delta": float(np.mean(cluster_arr)),
        "chem_cluster_positive_count": float(np.sum(cluster_arr > 0.0)),
        "chem_cluster_count": float(cluster_arr.size),
        "other_subcluster_worst_delta": float(np.max(other_arr)) if other_arr.size else 0.0,
        "other_subcluster_mean_delta": float(np.mean(other_arr)) if other_arr.size else 0.0,
        "other_subcluster_positive_count": float(np.sum(other_arr > 0.0)) if other_arr.size else 0.0,
        "other_subcluster_count": float(other_arr.size),
    }


def select_signrate_lock_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + SIGNRATE_LOCK_PRIMARY_EPSILON]
    eligible.sort(
        key=lambda item: (
            item["proxy_hard_positive_bin_count"],
            item["proxy_hard_worst_bin_delta"],
            item["proxy_hard_bin_variance"],
            item["gate_volatility"],
            item["proxy_hard_mean_delta"],
            item["val_hard"],
            item["val_primary"],
            item["val_noisy"],
            float(item["val_external"]) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
        )
    )
    return eligible[0]


def select_stabilization_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + STABILIZATION_PRIMARY_EPSILON]
    eligible.sort(
        key=lambda item: (
            item["proxy_hard_positive_bin_count"],
            item["weak_cluster_worst_delta"],
            item["gate_volatility"],
            item["ambiguity_variance"],
            item["proxy_hard_worst_bin_delta"],
            item["val_hard"],
            item["val_primary"],
            item["val_noisy"],
            float(item["val_external"]) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
        )
    )
    return eligible[0]


def select_checkpoint(candidates: list[dict[str, Any]], *, selection_policy: str) -> dict[str, Any]:
    if selection_policy == "trisoup":
        return select_trisoup_checkpoint(candidates)
    if selection_policy == "ctgf":
        return select_ctgf_checkpoint(candidates)
    if selection_policy == "jtt_stabilization":
        return select_jtt_stabilization_checkpoint(candidates)
    if selection_policy == "self_stabilization":
        return select_self_stabilization_checkpoint(candidates)
    if selection_policy == "splithead_stabilization":
        return select_splithead_checkpoint(candidates)
    if selection_policy == "stabilization":
        return select_stabilization_checkpoint(candidates)
    if selection_policy == "signrate_lock":
        return select_signrate_lock_checkpoint(candidates)
    return select_tailfix_checkpoint(candidates)


def splithead_priority_tuple(item: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(item["proxy_hard_positive_bin_count"]),
        float(max(0.0, item["weak_cluster_worst_delta"])),
        float(item["weak_cluster_positive_count"]),
        float(item["gate_volatility"]),
        float(item["ambiguity_variance"]),
        float(max(0.0, item["proxy_hard_worst_bin_delta"])),
        float(item["val_hard"]),
        float(item["val_primary"]),
        float(item["val_noisy"]),
        float(item["val_external"]) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
    )


def trisoup_priority_tuple(item: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(item.get("chem_cluster_positive_count", 0.0)),
        float(max(0.0, item.get("chem_cluster_worst_delta", 0.0) - 0.02)),
        float(item.get("other_subcluster_positive_count", 0.0)),
        float(max(0.0, item.get("other_subcluster_worst_delta", 0.0) - 0.02)),
        float(max(0.0, item["weak_cluster_worst_delta"] - 0.02)),
        float(max(0.0, item["external_proxy_delta"])) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
        float(max(0.0, item["full_proxy_delta"] - 0.02)),
        float(item["proxy_hard_positive_bin_count"]),
        float(max(0.0, item["proxy_hard_worst_bin_delta"])),
        float(item["proxy_hard_bin_variance"]),
        float(item["gate_volatility"]),
        float(item["ambiguity_variance"]),
        float(item["val_hard"]),
        float(item["val_primary"]),
        float(item["val_noisy"]),
        float(item["val_external"]) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
    )


def select_splithead_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + SPLITHEAD_PRIMARY_EPSILON]
    eligible.sort(key=splithead_priority_tuple)
    return eligible[0]


def select_self_stabilization_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + SELF_STABILIZATION_PRIMARY_EPSILON]
    eligible.sort(key=splithead_priority_tuple)
    return eligible[0]


def select_jtt_stabilization_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + JTT_STABILIZATION_PRIMARY_EPSILON]
    eligible.sort(
        key=lambda item: (
            float(item.get("chem_cluster_positive_count", 0.0)),
            float(max(0.0, item.get("chem_cluster_worst_delta", 0.0) - 0.02)),
            float(item["proxy_hard_positive_bin_count"]),
            float(max(0.0, item["weak_cluster_worst_delta"])),
            float(item["proxy_hard_bin_variance"]),
            float(item["gate_volatility"]),
            float(item["ambiguity_variance"]),
            float(item["val_hard"]),
            float(item["val_primary"]),
            float(item["val_noisy"]),
            float(item["val_external"]) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
        )
    )
    return eligible[0]


def select_ctgf_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + CTGF_PRIMARY_EPSILON]
    strict_guard = [
        item
        for item in eligible
        if float(item["weak_cluster_worst_delta"]) <= 0.05
        and float(item.get("chem_cluster_worst_delta", 0.0)) <= 0.05
        and float(item.get("chem_cluster_positive_count", 0.0)) <= 1.0
        and (
            not USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION
            or float(item["external_proxy_delta"]) <= 0.0
        )
        and float(item["full_proxy_delta"]) <= 0.02
    ]
    relaxed_guard = [
        item
        for item in eligible
        if float(item["weak_cluster_worst_delta"]) <= 0.08
        and float(item.get("chem_cluster_worst_delta", 0.0)) <= 0.10
        and float(item.get("chem_cluster_positive_count", 0.0)) <= 2.0
        and (
            not USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION
            or float(item["external_proxy_delta"]) <= 0.04
        )
        and float(item["full_proxy_delta"]) <= 0.03
    ]
    final_pool = strict_guard if strict_guard else (relaxed_guard if relaxed_guard else eligible)
    final_pool.sort(
        key=lambda item: (
            float(item.get("chem_cluster_positive_count", 0.0)),
            float(max(0.0, item.get("chem_cluster_worst_delta", 0.0) - 0.02)),
            float(max(0.0, item["weak_cluster_worst_delta"] - 0.05)),
            float(max(0.0, item["external_proxy_delta"])) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
            float(max(0.0, item["full_proxy_delta"] - 0.02)),
            float(item["proxy_hard_positive_bin_count"]),
            float(max(0.0, item["proxy_hard_worst_bin_delta"])),
            float(item["proxy_hard_bin_variance"]),
            float(item["gate_volatility"]),
            float(item["ambiguity_variance"]),
            float(item["val_hard"]),
            float(item["val_primary"]),
            float(item["val_noisy"]),
            float(item["val_external"]) if USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION else 0.0,
        )
    )
    return final_pool[0]


def select_trisoup_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + TRISOUP_PRIMARY_EPSILON]
    strict_guard = [
        item
        for item in eligible
        if float(item["weak_cluster_worst_delta"]) <= 0.02
        and float(item.get("chem_cluster_worst_delta", 0.0)) <= 0.05
        and float(item.get("chem_cluster_positive_count", 0.0)) <= 1.0
        and float(item.get("other_subcluster_worst_delta", 0.0)) <= 0.08
        and float(item.get("other_subcluster_positive_count", 0.0)) <= 1.0
        and (
            not USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION
            or float(item["external_proxy_delta"]) <= 0.0
        )
        and float(item["full_proxy_delta"]) <= 0.02
    ]
    relaxed_guard = [
        item
        for item in eligible
        if float(item["weak_cluster_worst_delta"]) <= 0.05
        and float(item.get("chem_cluster_worst_delta", 0.0)) <= 0.10
        and float(item.get("chem_cluster_positive_count", 0.0)) <= 2.0
        and float(item.get("other_subcluster_worst_delta", 0.0)) <= 0.12
        and float(item.get("other_subcluster_positive_count", 0.0)) <= 2.0
        and (
            not USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION
            or float(item["external_proxy_delta"]) <= 0.02
        )
        and float(item["full_proxy_delta"]) <= 0.03
    ]
    final_pool = strict_guard if strict_guard else (relaxed_guard if relaxed_guard else eligible)
    final_pool.sort(key=trisoup_priority_tuple)
    return final_pool[0]


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise RuntimeError("average_state_dicts requires at least one state")
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        values = [state[key] for state in states]
        if torch.is_floating_point(values[0]):
            stacked = torch.stack([value.to(torch.float32) for value in values], dim=0)
            averaged[key] = stacked.mean(dim=0).to(values[0].dtype)
        else:
            averaged[key] = values[0].clone()
    return averaged


def set_masd_split_head_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    active_modules = {
        "ctx_scale_pool",
        "masd_mag_head",
        "masd_res_head",
        "masd_calib_context_head",
        "masd_gate_context_head",
    }
    for module_name in active_modules:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
    for param_name in (
        "masd_calib_proxy_scale",
        "masd_calib_proxy_bias",
        "masd_alpha_risk_weights",
        "masd_alpha_risk_bias",
        "masd_gate_risk_weights",
        "masd_gate_bias",
    ):
        param = getattr(model, param_name, None)
        if param is not None:
            param.requires_grad = True


def set_masd_ctgf_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    active_modules = {
        "ctx_scale_pool",
        "mspce_context_injector",
        "mspce_repair_gate",
        "rcmf_min_fusion",
        "rcmf_min_delta_head",
        "baseline_head",
        "concat_head",
        "masd_core_proj",
        "masd_slot_bank",
        "masd_slot_proj",
        "masd_alpha_head",
        "masd_mag_head",
        "masd_res_head",
        "masd_calib_context_head",
        "masd_safety_gate_v3",
        "masd_gate_context_head",
    }
    for module_name in active_modules:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
    for param_name in (
        "masd_calib_proxy_scale",
        "masd_calib_proxy_bias",
        "masd_alpha_risk_weights",
        "masd_alpha_risk_bias",
        "masd_gate_risk_weights",
        "masd_gate_bias",
        "static_logits",
    ):
        param = getattr(model, param_name, None)
        if param is not None:
            param.requires_grad = True


def build_splithead_partitions(
    *,
    current_rcmf: nn.Module,
    seed_tensors: dict[str, Any],
    split: dict[str, Any],
    config: Any,
    seed: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    train_indices = list(split["train"])
    train_loader = make_loader(
        seed_tensors,
        train_indices,
        config.batch_size,
        shuffle=False,
        loader_seed=seed * 1009 + 17,
    )
    _, train_payload = evaluate_stage(
        current_rcmf,
        train_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1009 + 19,
        return_payload=True,
    )
    sample_index = np.asarray(train_payload["sample_index"], dtype=np.int64).reshape(-1)
    cluster_support = np.asarray(train_payload["cluster_support"], dtype=np.int64).reshape(-1)
    hard_score = np.asarray(train_payload["hard_score"], dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(train_payload["uncertainty"], dtype=np.float64).reshape(-1)
    conflict = np.asarray(train_payload["conflict"], dtype=np.float64).reshape(-1)
    risk_score = hard_score + 0.25 * uncertainty + 0.15 * conflict
    risk_edges = np.quantile(risk_score, [0.50, 0.80])
    risk_bin = np.digitize(risk_score, bins=np.asarray(risk_edges, dtype=np.float64), right=False)
    strata: dict[tuple[int, int], list[int]] = {}
    for idx, support, bucket in zip(sample_index.tolist(), cluster_support.tolist(), risk_bin.tolist()):
        strata.setdefault((int(support), int(bucket)), []).append(int(idx))

    rng = np.random.default_rng(94000 + seed * 131)
    split_a: list[int] = []
    split_b: list[int] = []
    for key in sorted(strata.keys()):
        members = list(strata[key])
        rng.shuffle(members)
        if len(members) <= 1:
            split_a.extend(members)
            continue
        cut = int(round(len(members) * 0.80))
        cut = min(max(cut, 1), len(members) - 1)
        split_a.extend(members[:cut])
        split_b.extend(members[cut:])

    target_b = max(32, int(round(len(train_indices) * 0.20)))
    if len(split_b) < target_b:
        current_a = np.asarray(split_a, dtype=np.int64)
        move_budget = target_b - len(split_b)
        risk_lookup = {int(idx): float(score) for idx, score in zip(sample_index.tolist(), risk_score.tolist())}
        move_candidates = sorted(current_a.tolist(), key=lambda idx: (-risk_lookup.get(int(idx), 0.0), int(idx)))
        moved: list[int] = []
        for idx in move_candidates:
            if move_budget <= 0:
                break
            moved.append(int(idx))
            move_budget -= 1
        if moved:
            moved_set = set(moved)
            split_a = [idx for idx in split_a if idx not in moved_set]
            split_b.extend(moved)

    split_a = sorted(set(split_a))
    split_b = sorted(set(split_b))
    splithead_meta = {
        "split_a_count": int(len(split_a)),
        "split_b_count": int(len(split_b)),
        "risk_edge_50": float(risk_edges[0]),
        "risk_edge_80": float(risk_edges[1]),
        "split_b_fraction": float(len(split_b) / max(len(train_indices), 1)),
    }
    return split_a, split_b, splithead_meta


def build_splithead_sample_weights(
    *,
    reference_model: nn.Module,
    seed_tensors: dict[str, Any],
    split_b_indices: list[int],
    config: Any,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    split_b_loader = make_loader(
        seed_tensors,
        split_b_indices,
        config.batch_size,
        shuffle=False,
        loader_seed=seed * 1013 + 29,
    )
    _, payload = evaluate_stage(
        reference_model,
        split_b_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1013 + 31,
        return_payload=True,
    )
    sample_index = np.asarray(payload["sample_index"], dtype=np.int64).reshape(-1)
    cluster_support = np.asarray(payload["cluster_support"], dtype=np.int64).reshape(-1)
    hard_score = np.asarray(payload["hard_score"], dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(payload["uncertainty"], dtype=np.float64).reshape(-1)
    conflict = np.asarray(payload["conflict"], dtype=np.float64).reshape(-1)
    risk_hi = float(np.quantile(hard_score, 0.80))
    risk_mid = float(np.quantile(hard_score, 0.60))
    unc_hi = float(np.quantile(uncertainty, 0.80))
    conf_hi = float(np.quantile(conflict, 0.80))
    weight_lookup: dict[int, float] = {}
    for idx, support, risk, unc, conf in zip(
        sample_index.tolist(),
        cluster_support.tolist(),
        hard_score.tolist(),
        uncertainty.tolist(),
        conflict.tolist(),
    ):
        weight = 1.0
        if int(support) == WEAK_CLUSTER_CODE:
            weight *= 1.80
        elif int(support) == UNSTABLE_CLUSTER_CODE:
            weight *= 1.45
        if float(risk) >= risk_hi:
            weight *= 1.65
        elif float(risk) >= risk_mid:
            weight *= 1.28
        if float(unc) >= unc_hi:
            weight *= 1.15
        if float(conf) >= conf_hi:
            weight *= 1.10
        weight_lookup[int(idx)] = float(weight)
    weights = [weight_lookup.get(int(idx), 1.0) for idx in split_b_indices]
    mean_weight = float(np.mean(weights)) if weights else 1.0
    if mean_weight > 0.0:
        weights = [float(weight / mean_weight) for weight in weights]
    meta = {
        "risk_hi": risk_hi,
        "risk_mid": risk_mid,
        "uncertainty_hi": unc_hi,
        "conflict_hi": conf_hi,
        "weight_mean": float(np.mean(weights)) if weights else 1.0,
        "weight_max": float(np.max(weights)) if weights else 1.0,
    }
    return weights, meta


def select_stage_a_reference_candidate(candidates: list[dict[str, Any]], *, epsilon: float) -> dict[str, Any]:
    mechanism_ok = [item for item in candidates if item["mechanism_pass"]]
    pool = mechanism_ok if mechanism_ok else candidates
    best_val_primary = min(item["val_primary"] for item in pool)
    eligible = [item for item in pool if item["val_primary"] <= best_val_primary + epsilon]
    eligible.sort(
        key=lambda item: (
            int(item["epoch"]),
            float(item["gate_volatility"]),
            float(item["ambiguity_variance"]),
            float(item["val_primary"]),
        )
    )
    return eligible[0]


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
    return rank


def build_self_split_b_selection(
    *,
    final_payload: dict[str, np.ndarray],
    reference_payload: dict[str, np.ndarray],
) -> tuple[list[int], list[float], dict[str, Any]]:
    sample_index = np.asarray(final_payload["sample_index"], dtype=np.int64).reshape(-1)
    if sample_index.size == 0:
        return [], [], {"selected_count": 0}
    cluster_support = np.asarray(final_payload["cluster_support"], dtype=np.int64).reshape(-1)
    hard_mask = np.asarray(final_payload["hard_mask"], dtype=bool).reshape(-1)
    uncertainty = np.asarray(final_payload["uncertainty"], dtype=np.float64).reshape(-1)
    conflict = np.asarray(final_payload["conflict"], dtype=np.float64).reshape(-1)
    ambiguity = np.asarray(final_payload["masd_mechanism_disagreement"], dtype=np.float64).reshape(-1)
    alpha_entropy = np.asarray(final_payload["masd_entropy"], dtype=np.float64).reshape(-1)
    error = np.asarray(final_payload["error"], dtype=np.float64).reshape(-1)
    y_true = np.asarray(final_payload["y_true"], dtype=np.float64).reshape(-1)
    disagreement = np.abs(
        np.asarray(final_payload["pred"], dtype=np.float64).reshape(-1)
        - np.asarray(reference_payload["pred"], dtype=np.float64).reshape(-1)
    )

    cluster_class = np.where(
        cluster_support == WEAK_CLUSTER_CODE,
        "weak",
        np.where(cluster_support == UNSTABLE_CLUSTER_CODE, "unstable", "supported"),
    )
    q25, q50, q75 = np.quantile(y_true, [0.25, 0.50, 0.75])
    y_bin = np.digitize(y_true, bins=np.asarray([q25, q50, q75], dtype=np.float64), right=False)

    score = (
        0.30 * percentile_rank(disagreement)
        + 0.24 * percentile_rank(error)
        + 0.20 * hard_mask.astype(np.float64)
        + 0.15 * percentile_rank(uncertainty)
        + 0.12 * percentile_rank(ambiguity)
        + 0.10 * percentile_rank(conflict)
        + 0.08 * percentile_rank(alpha_entropy)
        + 0.20 * np.where(cluster_support == WEAK_CLUSTER_CODE, 1.0, np.where(cluster_support == UNSTABLE_CLUSTER_CODE, 0.7, 0.2))
    )

    selection_df = pd.DataFrame(
        {
            "sample_index": sample_index,
            "cluster_class": cluster_class,
            "cluster_support": cluster_support,
            "y_bin": y_bin,
            "tail_score": score,
            "hard_flag": hard_mask.astype(np.int64),
            "uncertainty": uncertainty,
            "conflict": conflict,
        }
    ).sort_values(["tail_score", "sample_index"], ascending=[False, True]).reset_index(drop=True)

    total_k = min(len(selection_df), max(40, int(round(len(selection_df) * 0.45))))
    class_counts = selection_df["cluster_class"].value_counts().to_dict()
    quotas = {
        "weak": min(class_counts.get("weak", 0), max(8, int(round(total_k * 0.35)))),
        "unstable": min(class_counts.get("unstable", 0), max(4, int(round(total_k * 0.20)))),
        "supported": min(class_counts.get("supported", 0), max(6, int(round(total_k * 0.15)))),
    }
    while sum(quotas.values()) > total_k:
        for key in ("supported", "unstable", "weak"):
            if sum(quotas.values()) <= total_k:
                break
            if quotas[key] > 0:
                quotas[key] -= 1

    selected_rows: list[pd.DataFrame] = []
    chosen_indices: set[int] = set()
    for class_name, quota in quotas.items():
        if quota <= 0:
            continue
        class_df = selection_df[selection_df["cluster_class"] == class_name].copy()
        if class_df.empty:
            continue
        class_selected: list[pd.Series] = []
        for bin_id in sorted(class_df["y_bin"].unique().tolist()):
            if len(class_selected) >= quota:
                break
            bin_df = class_df[class_df["y_bin"] == bin_id]
            for _, row in bin_df.iterrows():
                sample_id = int(row["sample_index"])
                if sample_id in chosen_indices:
                    continue
                class_selected.append(row)
                chosen_indices.add(sample_id)
                break
        if len(class_selected) < quota:
            for _, row in class_df.iterrows():
                sample_id = int(row["sample_index"])
                if sample_id in chosen_indices:
                    continue
                class_selected.append(row)
                chosen_indices.add(sample_id)
                if len(class_selected) >= quota:
                    break
        if class_selected:
            selected_rows.append(pd.DataFrame(class_selected))

    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(columns=selection_df.columns)
    if len(selected_df) < total_k:
        for _, row in selection_df.iterrows():
            sample_id = int(row["sample_index"])
            if sample_id in chosen_indices:
                continue
            selected_df = pd.concat([selected_df, row.to_frame().T], ignore_index=True)
            chosen_indices.add(sample_id)
            if len(selected_df) >= total_k:
                break
    selected_df = selected_df.drop_duplicates(subset=["sample_index"], keep="first").reset_index(drop=True)

    bin_counts = selected_df["y_bin"].value_counts().to_dict()
    weights: list[float] = []
    for _, row in selected_df.iterrows():
        weight = 1.0
        if row["cluster_class"] == "weak":
            weight *= 1.75
        elif row["cluster_class"] == "unstable":
            weight *= 1.45
        if int(row["hard_flag"]) > 0:
            weight *= 1.35
        if float(row["uncertainty"]) >= float(np.quantile(uncertainty, 0.75)):
            weight *= 1.15
        if float(row["conflict"]) >= float(np.quantile(conflict, 0.75)):
            weight *= 1.08
        weight *= float(len(selected_df) / max(1, 4 * int(bin_counts.get(int(row["y_bin"]), 1))))
        weights.append(float(weight))
    mean_weight = float(np.mean(weights)) if weights else 1.0
    if mean_weight > 0.0:
        weights = [float(weight / mean_weight) for weight in weights]

    meta = {
        "candidate_count": int(len(selection_df)),
        "selected_count": int(len(selected_df)),
        "selected_fraction": float(len(selected_df) / max(len(selection_df), 1)),
        "quota_weak": int(quotas["weak"]),
        "quota_unstable": int(quotas["unstable"]),
        "quota_supported": int(quotas["supported"]),
        "selected_weak": int((selected_df["cluster_class"] == "weak").sum()) if len(selected_df) else 0,
        "selected_unstable": int((selected_df["cluster_class"] == "unstable").sum()) if len(selected_df) else 0,
        "selected_supported": int((selected_df["cluster_class"] == "supported").sum()) if len(selected_df) else 0,
        "y_bin_counts": {str(int(key)): int(val) for key, val in bin_counts.items()},
        "mean_tail_score": float(selected_df["tail_score"].mean()) if len(selected_df) else 0.0,
        "max_tail_score": float(selected_df["tail_score"].max()) if len(selected_df) else 0.0,
    }
    return selected_df["sample_index"].astype(int).tolist(), weights, meta


def build_jtt_error_set(
    *,
    payload: dict[str, np.ndarray],
) -> tuple[list[int], dict[int, float], dict[str, Any]]:
    sample_index = np.asarray(payload["sample_index"], dtype=np.int64).reshape(-1)
    if sample_index.size == 0:
        return [], {}, {"selected_count": 0}
    cluster_support = np.asarray(payload["cluster_support"], dtype=np.int64).reshape(-1)
    hard_mask = np.asarray(payload["hard_mask"], dtype=bool).reshape(-1)
    hard_score = np.asarray(payload["hard_score"], dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(payload["uncertainty"], dtype=np.float64).reshape(-1)
    conflict = np.asarray(payload["conflict"], dtype=np.float64).reshape(-1)
    ambiguity = np.asarray(payload["masd_mechanism_disagreement"], dtype=np.float64).reshape(-1)
    error = np.asarray(payload["error"], dtype=np.float64).reshape(-1)
    y_true = np.asarray(payload["y_true"], dtype=np.float64).reshape(-1)

    cluster_class = np.where(
        cluster_support == WEAK_CLUSTER_CODE,
        "weak",
        np.where(cluster_support == UNSTABLE_CLUSTER_CODE, "unstable", "supported"),
    )
    q25, q50, q75 = np.quantile(y_true, [0.25, 0.50, 0.75])
    y_bin = np.digitize(y_true, bins=np.asarray([q25, q50, q75], dtype=np.float64), right=False)
    tail_score = (
        0.34 * percentile_rank(error)
        + 0.18 * percentile_rank(hard_score)
        + 0.18 * hard_mask.astype(np.float64)
        + 0.14 * percentile_rank(uncertainty)
        + 0.10 * percentile_rank(ambiguity)
        + 0.08 * percentile_rank(conflict)
        + 0.20 * np.where(cluster_support == WEAK_CLUSTER_CODE, 1.0, np.where(cluster_support == UNSTABLE_CLUSTER_CODE, 0.75, 0.15))
    )
    selection_df = pd.DataFrame(
        {
            "sample_index": sample_index,
            "cluster_class": cluster_class,
            "cluster_support": cluster_support,
            "hard_flag": hard_mask.astype(np.int64),
            "y_bin": y_bin,
            "tail_score": tail_score,
            "error": error,
        }
    ).sort_values(["tail_score", "sample_index"], ascending=[False, True]).reset_index(drop=True)

    total_k = min(len(selection_df), max(96, int(round(len(selection_df) * 0.30))))
    class_counts = selection_df["cluster_class"].value_counts().to_dict()
    quotas = {
        "weak": min(class_counts.get("weak", 0), max(18, int(round(total_k * 0.28)))),
        "unstable": min(class_counts.get("unstable", 0), max(10, int(round(total_k * 0.16)))),
        "supported": min(class_counts.get("supported", 0), max(16, int(round(total_k * 0.14)))),
    }
    while sum(quotas.values()) > total_k:
        for key in ("supported", "unstable", "weak"):
            if sum(quotas.values()) <= total_k:
                break
            if quotas[key] > 0:
                quotas[key] -= 1
    hard_quota = min(int(selection_df["hard_flag"].sum()), max(24, int(round(total_k * 0.35))))

    chosen_indices: set[int] = set()
    selected_rows: list[pd.DataFrame] = []
    for class_name, quota in quotas.items():
        if quota <= 0:
            continue
        class_df = selection_df[selection_df["cluster_class"] == class_name].copy()
        if class_df.empty:
            continue
        class_selected: list[pd.Series] = []
        for bin_id in sorted(class_df["y_bin"].unique().tolist()):
            if len(class_selected) >= quota:
                break
            bin_df = class_df[class_df["y_bin"] == bin_id]
            for _, row in bin_df.iterrows():
                sample_id = int(row["sample_index"])
                if sample_id in chosen_indices:
                    continue
                class_selected.append(row)
                chosen_indices.add(sample_id)
                break
        if len(class_selected) < quota:
            for _, row in class_df.iterrows():
                sample_id = int(row["sample_index"])
                if sample_id in chosen_indices:
                    continue
                class_selected.append(row)
                chosen_indices.add(sample_id)
                if len(class_selected) >= quota:
                    break
        if class_selected:
            selected_rows.append(pd.DataFrame(class_selected))
    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(columns=selection_df.columns)

    current_hard = int(selected_df["hard_flag"].sum()) if len(selected_df) else 0
    if current_hard < hard_quota:
        hard_candidates = selection_df[selection_df["hard_flag"] == 1].copy()
        for _, row in hard_candidates.iterrows():
            sample_id = int(row["sample_index"])
            if sample_id in chosen_indices:
                continue
            selected_df = pd.concat([selected_df, row.to_frame().T], ignore_index=True)
            chosen_indices.add(sample_id)
            current_hard += 1
            if current_hard >= hard_quota:
                break

    if len(selected_df) < total_k:
        for _, row in selection_df.iterrows():
            sample_id = int(row["sample_index"])
            if sample_id in chosen_indices:
                continue
            selected_df = pd.concat([selected_df, row.to_frame().T], ignore_index=True)
            chosen_indices.add(sample_id)
            if len(selected_df) >= total_k:
                break

    selected_df = selected_df.drop_duplicates(subset=["sample_index"], keep="first").reset_index(drop=True)
    weight_lookup: dict[int, float] = {}
    y_bin_counts = selected_df["y_bin"].value_counts().to_dict() if len(selected_df) else {}
    for _, row in selection_df.iterrows():
        sample_id = int(row["sample_index"])
        cluster_name = str(row["cluster_class"])
        weight = 1.0
        if sample_id in set(selected_df["sample_index"].astype(int).tolist()):
            weight *= 2.10
        if cluster_name == "weak":
            weight *= 1.30
        elif cluster_name == "unstable":
            weight *= 1.18
        if int(row["hard_flag"]) > 0:
            weight *= 1.24
        if sample_id in set(selected_df["sample_index"].astype(int).tolist()):
            count = int(y_bin_counts.get(int(row["y_bin"]), 1))
            weight *= float(len(selected_df) / max(1, 4 * count))
        weight_lookup[sample_id] = float(weight)

    selected_ids = set(selected_df["sample_index"].astype(int).tolist())
    full_weights = np.asarray([weight_lookup[int(idx)] for idx in sample_index.tolist()], dtype=np.float64)
    mean_weight = float(full_weights.mean()) if full_weights.size else 1.0
    if mean_weight > 0.0:
        full_weights = full_weights / mean_weight
        weight_lookup = {int(idx): float(weight) for idx, weight in zip(sample_index.tolist(), full_weights.tolist())}

    meta = {
        "candidate_count": int(len(selection_df)),
        "selected_count": int(len(selected_df)),
        "selected_fraction": float(len(selected_df) / max(len(selection_df), 1)),
        "selected_weak": int((selected_df["cluster_class"] == "weak").sum()) if len(selected_df) else 0,
        "selected_unstable": int((selected_df["cluster_class"] == "unstable").sum()) if len(selected_df) else 0,
        "selected_supported": int((selected_df["cluster_class"] == "supported").sum()) if len(selected_df) else 0,
        "selected_hard": int((selected_df["hard_flag"] == 1).sum()) if len(selected_df) else 0,
        "hard_quota": int(hard_quota),
        "quota_weak": int(quotas["weak"]),
        "quota_unstable": int(quotas["unstable"]),
        "quota_supported": int(quotas["supported"]),
        "y_bin_counts": {str(int(key)): int(val) for key, val in y_bin_counts.items()},
        "mean_tail_score": float(selected_df["tail_score"].mean()) if len(selected_df) else 0.0,
        "max_tail_score": float(selected_df["tail_score"].max()) if len(selected_df) else 0.0,
        "mean_weight": float(full_weights.mean()) if full_weights.size else 1.0,
        "max_weight": float(full_weights.max()) if full_weights.size else 1.0,
    }
    return sorted(selected_ids), weight_lookup, meta


def build_checkpoint_candidate(
    *,
    student: nn.Module,
    state: dict[str, torch.Tensor],
    stage: str,
    epoch: int,
    val_loader: torch.utils.data.DataLoader,
    external_loader: torch.utils.data.DataLoader,
    seed_tensors: dict[str, Any],
    seed: int,
    eval_offset: int,
    anchor_clean: dict[str, float],
    anchor_noisy: dict[str, float],
    anchor_external: dict[str, float],
    anchor_clean_payload: dict[str, np.ndarray],
) -> dict[str, Any]:
    student.load_state_dict(state)
    val_clean, clean_payload = evaluate_stage(student, val_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 10 + eval_offset, return_payload=True)
    val_noisy, noisy_payload = evaluate_stage(student, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 503 + 20 + eval_offset, return_payload=True)
    if USE_EXTERNAL_HOLDOUT_FOR_SELECTION:
        val_external, _ = evaluate_stage(student, external_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 30 + eval_offset, return_payload=False)
        external_proxy_delta = float(val_external["mae_k"] - anchor_external["mae_k"])
    else:
        val_external = {
            "mae_k": float(anchor_external["mae_k"]),
            "hard_subgroup_mae_k": float(anchor_external.get("hard_subgroup_mae_k", anchor_clean["hard_subgroup_mae_k"])),
        }
        external_proxy_delta = 0.0
    mech = contribution_metrics_from_payload(clean_payload, noisy_payload)
    anchor_mask_delta = common_mask_delta(clean_payload, anchor_clean_payload)
    proxy_hard = candidate_proxy_hard_metrics(clean_payload, anchor_clean_payload)
    chemistry_proxy = candidate_chemistry_cluster_metrics(clean_payload, anchor_clean_payload)
    gate_risk_penalty = max(
        0.0,
        float(mech["high_uncertainty_gate_mean"] - mech["low_risk_gate_mean"] + 0.01),
    ) + max(0.0, float(anchor_mask_delta))
    full_proxy_delta = float(val_clean["mae_k"] - anchor_clean["mae_k"])
    val_score = (
        val_clean["mae_k"]
        + 0.95 * max(0.0, val_clean["mae_k"] - anchor_clean["mae_k"])
        + 0.90 * max(0.0, val_noisy["mae_k"] - anchor_noisy["mae_k"])
        + 0.80 * max(0.0, anchor_mask_delta)
        + 0.85 * max(0.0, val_clean["hard_subgroup_mae_k"] - anchor_clean["hard_subgroup_mae_k"] - 0.02)
        + 0.40 * proxy_hard["proxy_hard_positive_bin_count"]
        + 0.95 * max(0.0, proxy_hard["proxy_hard_worst_bin_delta"])
        + 0.80 * max(0.0, proxy_hard["weak_cluster_worst_delta"] - 0.02)
        + 0.14 * proxy_hard["weak_cluster_positive_count"]
        + 0.70 * max(0.0, chemistry_proxy["chem_cluster_worst_delta"] - 0.02)
        + 0.20 * max(0.0, chemistry_proxy["chem_cluster_mean_delta"])
        + 0.12 * chemistry_proxy["chem_cluster_positive_count"]
        + 0.55 * max(0.0, chemistry_proxy["other_subcluster_worst_delta"] - 0.02)
        + 0.12 * max(0.0, chemistry_proxy["other_subcluster_mean_delta"])
        + 0.08 * chemistry_proxy["other_subcluster_positive_count"]
        + 0.12 * proxy_hard["gate_volatility"]
        + 0.10 * proxy_hard["ambiguity_variance"]
        + 0.18 * max(0.0, 0.90 - mech["contribution_sign_consistency"])
        + 0.10 * max(0.0, mech["gate_vs_error_correlation"] + 0.02)
    )
    return {
        "stage": stage,
        "epoch": int(epoch),
        "val_primary": float(val_clean["mae_k"]),
        "val_noisy": float(val_noisy["mae_k"]),
        "val_hard": float(val_clean["hard_subgroup_mae_k"]),
        "val_external": float(val_external["mae_k"]),
        "anchor_mask_delta": float(anchor_mask_delta),
        "gate_risk_penalty": float(gate_risk_penalty),
        "external_proxy_delta": external_proxy_delta,
        "full_proxy_delta": full_proxy_delta,
        "selection_uses_external_holdout": bool(USE_EXTERNAL_HOLDOUT_FOR_SELECTION),
        "val_score": float(val_score),
        **proxy_hard,
        **chemistry_proxy,
        "mechanism_pass": bool(mech["mechanism_pass"]),
        "state": {key: value.detach().cpu() for key, value in state.items()},
    }


def payload_metrics(payload: dict[str, np.ndarray]) -> dict[str, float]:
    error = np.asarray(payload["error"], dtype=np.float64).reshape(-1)
    hard_score = np.asarray(payload["hard_score"], dtype=np.float64).reshape(-1)
    if hard_score.size == 0:
        return {
            "mae_k": 0.0,
            "hard_subgroup_mae_k": 0.0,
            "hard_mask_rate": 0.0,
            "conflict_mean": 0.0,
            "uncertainty_mean": 0.0,
        }
    threshold = float(np.quantile(hard_score, 0.80))
    hard_mask = hard_score >= threshold
    if not np.any(hard_mask):
        hard_mask = np.ones_like(hard_score, dtype=bool)
    payload["hard_mask"] = hard_mask.astype(np.float32)
    return {
        "mae_k": float(np.mean(error)),
        "hard_subgroup_mae_k": float(np.mean(error[hard_mask])),
        "hard_mask_rate": float(np.mean(hard_mask)),
        "conflict_mean": float(np.mean(np.asarray(payload["conflict"], dtype=np.float64).reshape(-1))),
        "uncertainty_mean": float(np.mean(np.asarray(payload["uncertainty"], dtype=np.float64).reshape(-1))),
    }


def interpolate_state_dicts(
    states: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    if not states:
        raise RuntimeError("interpolate_state_dicts requires at least one state")
    if len(states) != len(weights):
        raise RuntimeError("state / weight length mismatch")
    total = float(sum(weights))
    if total <= 0.0:
        raise RuntimeError("weights must sum to a positive value")
    norm_weights = [float(weight / total) for weight in weights]
    mixed: dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        ref = states[0][key]
        if torch.is_floating_point(ref):
            accumulator = None
            for state, weight in zip(states, norm_weights):
                value = state[key].to(torch.float32) * float(weight)
                accumulator = value if accumulator is None else accumulator + value
            mixed[key] = accumulator.to(ref.dtype)
        else:
            best_idx = int(np.argmax(norm_weights))
            mixed[key] = states[best_idx][key].clone()
    return mixed


def simplex_coefficients(step: float = 0.25) -> list[tuple[float, float, float]]:
    steps = int(round(1.0 / step))
    coeffs: list[tuple[float, float, float]] = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            coeff = (round(i * step, 10), round(j * step, 10), round(k * step, 10))
            coeffs.append(coeff)
    coeffs = [item for item in coeffs if any(weight > 0.0 for weight in item)]
    coeffs.sort(key=lambda item: (sum(weight > 0.0 for weight in item), item[0], item[1], item[2]))
    return coeffs


def blend_payloads(
    payloads: list[dict[str, np.ndarray]],
    weights: list[float],
) -> dict[str, np.ndarray]:
    if len(payloads) != len(weights):
        raise RuntimeError("payload / weight length mismatch")
    total = float(sum(weights))
    if total <= 0.0:
        raise RuntimeError("payload weights must sum to a positive value")
    norm_weights = [float(weight / total) for weight in weights]
    blended: dict[str, np.ndarray] = {}
    first = payloads[0]
    passthrough = {"y_true", "sample_index", "cluster_code", "cluster_support", "other_subcluster_code"}
    for key in first.keys():
        arrays = [np.asarray(payload[key]) for payload in payloads]
        if key in passthrough:
            blended[key] = np.asarray(arrays[0]).copy()
            continue
        if np.issubdtype(arrays[0].dtype, np.integer) and key not in {"masd_dominant_mechanism"}:
            blended[key] = np.asarray(arrays[0]).copy()
            continue
        stack = np.stack([array.astype(np.float64) for array in arrays], axis=0)
        blended[key] = np.tensordot(np.asarray(norm_weights, dtype=np.float64), stack, axes=(0, 0))

    if "masd_alpha" in blended:
        alpha = np.asarray(blended["masd_alpha"], dtype=np.float64)
        alpha = np.clip(alpha, 1e-8, None)
        alpha = alpha / np.clip(alpha.sum(axis=1, keepdims=True), 1e-8, None)
        blended["masd_alpha"] = alpha
        blended["masd_alpha_max"] = alpha.max(axis=1, keepdims=True)
        blended["masd_dominant_mechanism"] = alpha.argmax(axis=1).astype(np.int64)
        blended["masd_entropy"] = (-alpha * np.log(np.clip(alpha, 1e-8, 1.0))).sum(axis=1, keepdims=True) / np.log(alpha.shape[1])
    if "pred" in blended and "y_true" in blended:
        y_true = np.asarray(blended["y_true"], dtype=np.float64).reshape(-1)
        pred = np.asarray(blended["pred"], dtype=np.float64).reshape(-1)
        blended["error"] = np.abs(pred - y_true)
    return blended


def ensure_payload_alignment(payloads: list[dict[str, np.ndarray]]) -> None:
    if not payloads:
        raise RuntimeError("expected at least one payload for alignment check")
    ref_index = np.asarray(payloads[0]["sample_index"]).reshape(-1)
    ref_cluster = np.asarray(payloads[0]["cluster_code"]).reshape(-1)
    for idx, payload in enumerate(payloads[1:], start=1):
        sample_index = np.asarray(payload["sample_index"]).reshape(-1)
        cluster_code = np.asarray(payload["cluster_code"]).reshape(-1)
        if ref_index.shape != sample_index.shape or not np.array_equal(ref_index, sample_index):
            raise RuntimeError(f"payload sample_index mismatch for ensemble member {idx}")
        if ref_cluster.shape != cluster_code.shape or not np.array_equal(ref_cluster, cluster_code):
            raise RuntimeError(f"payload cluster_code mismatch for ensemble member {idx}")


def run_simpleconcat_ternary_seed(
    *,
    seed: int,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False, loader_seed=seed * 2003 + 1)
    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False, loader_seed=seed * 2003 + 3)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False, loader_seed=seed * 2003 + 5)

    member_summaries: list[dict[str, Any]] = []
    val_clean_payloads: list[dict[str, np.ndarray]] = []
    val_noisy_payloads: list[dict[str, np.ndarray]] = []
    primary_clean_payloads: list[dict[str, np.ndarray]] = []
    primary_noisy_payloads: list[dict[str, np.ndarray]] = []
    external_payloads: list[dict[str, np.ndarray]] = []
    member_seeds: list[int] = []

    for member_idx in range(SIMPLECONCAT_TERNARY_MEMBER_COUNT):
        member_seed = stable_seed(seed, "simple_concat", "main") + member_idx * SIMPLECONCAT_TERNARY_MEMBER_SEED_STRIDE
        member_seeds.append(int(member_seed))
        model = train_standard_model(
            mode="simple_concat",
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=member_seed,
        )
        val_clean_metrics, val_clean_payload = evaluate_stage(
            model,
            val_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 503 + member_idx * 100 + 11,
            return_payload=True,
        )
        val_noisy_metrics, val_noisy_payload = evaluate_stage(
            model,
            val_loader,
            seed_tensors,
            variant="noisy",
            noise_seed=seed * 503 + member_idx * 100 + 12,
            return_payload=True,
        )
        primary_clean_metrics, primary_clean_payload = evaluate_stage(
            model,
            primary_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 601 + member_idx * 100 + 1,
            return_payload=True,
        )
        primary_noisy_metrics, primary_noisy_payload = evaluate_stage(
            model,
            primary_loader,
            seed_tensors,
            variant="noisy",
            noise_seed=seed * 601 + member_idx * 100 + 2,
            return_payload=True,
        )
        external_metrics, external_payload = evaluate_stage(
            model,
            external_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 601 + member_idx * 100 + 3,
            return_payload=True,
        )
        member_summaries.append(
            {
                "member_idx": int(member_idx),
                "member_seed": int(member_seed),
                "val_primary": float(val_clean_metrics["mae_k"]),
                "val_noisy": float(val_noisy_metrics["mae_k"]),
                "val_hard": float(val_clean_metrics["hard_subgroup_mae_k"]),
                "primary_clean": float(primary_clean_metrics["mae_k"]),
                "primary_noisy": float(primary_noisy_metrics["mae_k"]),
                "primary_hard_subgroup": float(primary_clean_metrics["hard_subgroup_mae_k"]),
                "external_holdout": float(external_metrics["mae_k"]),
                "external_hard_subgroup": float(external_metrics["hard_subgroup_mae_k"]),
            }
        )
        val_clean_payloads.append(val_clean_payload)
        val_noisy_payloads.append(val_noisy_payload)
        primary_clean_payloads.append(primary_clean_payload)
        primary_noisy_payloads.append(primary_noisy_payload)
        external_payloads.append(external_payload)
        model.cpu()
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    ensure_payload_alignment(val_clean_payloads)
    ensure_payload_alignment(val_noisy_payloads)
    ensure_payload_alignment(primary_clean_payloads)
    ensure_payload_alignment(primary_noisy_payloads)
    ensure_payload_alignment(external_payloads)

    ensemble_weights = [1.0, 1.0, 1.0]
    ensemble_primary_clean_payload = blend_payloads(primary_clean_payloads, ensemble_weights)
    ensemble_primary_noisy_payload = blend_payloads(primary_noisy_payloads, ensemble_weights)
    ensemble_external_payload = blend_payloads(external_payloads, ensemble_weights)
    ensemble_val_clean_payload = blend_payloads(val_clean_payloads, ensemble_weights)
    ensemble_val_noisy_payload = blend_payloads(val_noisy_payloads, ensemble_weights)

    baseline_stage = {
        "primary_clean": float(member_summaries[0]["primary_clean"]),
        "primary_noisy": float(member_summaries[0]["primary_noisy"]),
        "primary_hard_subgroup": float(member_summaries[0]["primary_hard_subgroup"]),
        "external_holdout": float(member_summaries[0]["external_holdout"]),
        "external_hard_subgroup": float(member_summaries[0]["external_hard_subgroup"]),
    }
    ensemble_primary_clean_metrics = payload_metrics(ensemble_primary_clean_payload)
    ensemble_primary_noisy_metrics = payload_metrics(ensemble_primary_noisy_payload)
    ensemble_external_metrics = payload_metrics(ensemble_external_payload)
    ensemble_val_clean_metrics = payload_metrics(ensemble_val_clean_payload)
    ensemble_val_noisy_metrics = payload_metrics(ensemble_val_noisy_payload)
    ensemble_stage = {
        "primary_clean": float(ensemble_primary_clean_metrics["mae_k"]),
        "primary_noisy": float(ensemble_primary_noisy_metrics["mae_k"]),
        "primary_hard_subgroup": float(ensemble_primary_clean_metrics["hard_subgroup_mae_k"]),
        "external_holdout": float(ensemble_external_metrics["mae_k"]),
        "external_hard_subgroup": float(ensemble_external_metrics["hard_subgroup_mae_k"]),
    }

    rows = [
        {
            "seed": seed,
            "model_name": "strongest_baseline",
            **baseline_stage,
            "delta_vs_strongest_baseline_primary_clean": 0.0,
            "delta_vs_strongest_baseline_primary_noisy": 0.0,
            "delta_vs_strongest_baseline_primary_hard_subgroup": 0.0,
            "delta_vs_strongest_baseline_external_holdout": 0.0,
            "delta_vs_previous_primary_clean": 0.0,
            "delta_vs_previous_primary_noisy": 0.0,
            "delta_vs_previous_primary_hard_subgroup": 0.0,
            "delta_vs_previous_external_holdout": 0.0,
            "result_group": "mainline",
            "pass_flag": True,
            "ensemble_member_count": 1,
            "ensemble_weights": "",
        },
        {
            "seed": seed,
            "model_name": "strongest_baseline_ternary_ensemble",
            **ensemble_stage,
            "delta_vs_strongest_baseline_primary_clean": float(ensemble_stage["primary_clean"] - baseline_stage["primary_clean"]),
            "delta_vs_strongest_baseline_primary_noisy": float(ensemble_stage["primary_noisy"] - baseline_stage["primary_noisy"]),
            "delta_vs_strongest_baseline_primary_hard_subgroup": float(ensemble_stage["primary_hard_subgroup"] - baseline_stage["primary_hard_subgroup"]),
            "delta_vs_strongest_baseline_external_holdout": float(ensemble_stage["external_holdout"] - baseline_stage["external_holdout"]),
            "delta_vs_previous_primary_clean": float(ensemble_stage["primary_clean"] - baseline_stage["primary_clean"]),
            "delta_vs_previous_primary_noisy": float(ensemble_stage["primary_noisy"] - baseline_stage["primary_noisy"]),
            "delta_vs_previous_primary_hard_subgroup": float(ensemble_stage["primary_hard_subgroup"] - baseline_stage["primary_hard_subgroup"]),
            "delta_vs_previous_external_holdout": float(ensemble_stage["external_holdout"] - baseline_stage["external_holdout"]),
            "result_group": "mainline",
            "pass_flag": stage_pass(ensemble_stage, baseline_stage),
            "ensemble_member_count": SIMPLECONCAT_TERNARY_MEMBER_COUNT,
            "ensemble_weights": "0.333333,0.333333,0.333333",
        },
    ]
    bundle = {
        "seed": seed,
        "member_seeds": member_seeds,
        "member_summaries": member_summaries,
        "ensemble_member_count": SIMPLECONCAT_TERNARY_MEMBER_COUNT,
        "ensemble_weights": [1.0 / SIMPLECONCAT_TERNARY_MEMBER_COUNT] * SIMPLECONCAT_TERNARY_MEMBER_COUNT,
        "baseline_primary_clean": primary_clean_payloads[0],
        "baseline_primary_noisy": primary_noisy_payloads[0],
        "baseline_external": external_payloads[0],
        "ensemble_primary_clean": ensemble_primary_clean_payload,
        "ensemble_primary_noisy": ensemble_primary_noisy_payload,
        "ensemble_external": ensemble_external_payload,
        "ensemble_val_clean_metrics": ensemble_val_clean_metrics,
        "ensemble_val_noisy_metrics": ensemble_val_noisy_metrics,
        "selection_uses_external_holdout": False,
    }
    return rows, bundle


def build_payload_candidate(
    *,
    stage: str,
    weights: tuple[float, float, float],
    clean_payload: dict[str, np.ndarray],
    noisy_payload: dict[str, np.ndarray],
    external_payload: dict[str, np.ndarray],
    anchor_clean: dict[str, float],
    anchor_noisy: dict[str, float],
    anchor_external: dict[str, float],
    anchor_clean_payload: dict[str, np.ndarray],
) -> dict[str, Any]:
    val_clean = payload_metrics(clean_payload)
    val_noisy = payload_metrics(noisy_payload)
    if USE_EXTERNAL_HOLDOUT_FOR_SELECTION:
        val_external = payload_metrics(external_payload)
        external_proxy_delta = float(val_external["mae_k"] - anchor_external["mae_k"])
    else:
        val_external = {
            "mae_k": float(anchor_external["mae_k"]),
            "hard_subgroup_mae_k": float(anchor_external.get("hard_subgroup_mae_k", anchor_clean["hard_subgroup_mae_k"])),
        }
        external_proxy_delta = 0.0
    mech = contribution_metrics_from_payload(clean_payload, noisy_payload)
    anchor_mask_delta = common_mask_delta(clean_payload, anchor_clean_payload)
    proxy_hard = candidate_proxy_hard_metrics(clean_payload, anchor_clean_payload)
    chemistry_proxy = candidate_chemistry_cluster_metrics(clean_payload, anchor_clean_payload)
    gate_risk_penalty = max(
        0.0,
        float(mech["high_uncertainty_gate_mean"] - mech["low_risk_gate_mean"] + 0.01),
    ) + max(0.0, float(anchor_mask_delta))
    full_proxy_delta = float(val_clean["mae_k"] - anchor_clean["mae_k"])
    val_score = (
        val_clean["mae_k"]
        + 0.95 * max(0.0, val_clean["mae_k"] - anchor_clean["mae_k"])
        + 0.90 * max(0.0, val_noisy["mae_k"] - anchor_noisy["mae_k"])
        + 0.80 * max(0.0, anchor_mask_delta)
        + 0.85 * max(0.0, val_clean["hard_subgroup_mae_k"] - anchor_clean["hard_subgroup_mae_k"] - 0.02)
        + 0.40 * proxy_hard["proxy_hard_positive_bin_count"]
        + 0.95 * max(0.0, proxy_hard["proxy_hard_worst_bin_delta"])
        + 0.80 * max(0.0, proxy_hard["weak_cluster_worst_delta"] - 0.02)
        + 0.14 * proxy_hard["weak_cluster_positive_count"]
        + 0.70 * max(0.0, chemistry_proxy["chem_cluster_worst_delta"] - 0.02)
        + 0.20 * max(0.0, chemistry_proxy["chem_cluster_mean_delta"])
        + 0.12 * chemistry_proxy["chem_cluster_positive_count"]
        + 0.55 * max(0.0, chemistry_proxy["other_subcluster_worst_delta"] - 0.02)
        + 0.12 * max(0.0, chemistry_proxy["other_subcluster_mean_delta"])
        + 0.08 * chemistry_proxy["other_subcluster_positive_count"]
        + 0.12 * proxy_hard["gate_volatility"]
        + 0.10 * proxy_hard["ambiguity_variance"]
        + 0.18 * max(0.0, 0.90 - mech["contribution_sign_consistency"])
        + 0.10 * max(0.0, mech["gate_vs_error_correlation"] + 0.02)
    )
    return {
        "stage": stage,
        "epoch": 0,
        "val_primary": float(val_clean["mae_k"]),
        "val_noisy": float(val_noisy["mae_k"]),
        "val_hard": float(val_clean["hard_subgroup_mae_k"]),
        "val_external": float(val_external["mae_k"]),
        "anchor_mask_delta": float(anchor_mask_delta),
        "gate_risk_penalty": float(gate_risk_penalty),
        "external_proxy_delta": external_proxy_delta,
        "full_proxy_delta": full_proxy_delta,
        "selection_uses_external_holdout": bool(USE_EXTERNAL_HOLDOUT_FOR_SELECTION),
        "val_score": float(val_score),
        **proxy_hard,
        **chemistry_proxy,
        "mechanism_pass": bool(mech["mechanism_pass"]),
        "weights": tuple(float(item) for item in weights),
        "clean_payload": clean_payload,
        "noisy_payload": noisy_payload,
        "external_payload": external_payload,
    }


def evaluate_weight_soup_candidate(
    *,
    weights: tuple[float, float, float],
    endpoint_states: dict[str, dict[str, torch.Tensor]],
    endpoint_order: list[str],
    seed_tensors: dict[str, Any],
    config: Any,
    primary_loader: torch.utils.data.DataLoader,
    external_loader: torch.utils.data.DataLoader,
    seed: int,
    eval_offset: int,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    mixed_state = interpolate_state_dicts([endpoint_states[name] for name in endpoint_order], list(weights))
    model = build_model(CURRENT_MODE, seed_tensors, config)
    model.load_state_dict(mixed_state)
    model.to(DEVICE)
    clean_metrics, clean_payload = evaluate_stage(
        model,
        primary_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1701 + eval_offset * 10 + 1,
        return_payload=True,
    )
    noisy_metrics, noisy_payload = evaluate_stage(
        model,
        primary_loader,
        seed_tensors,
        variant="noisy",
        noise_seed=seed * 1701 + eval_offset * 10 + 2,
        return_payload=True,
    )
    external_metrics, external_payload = evaluate_stage(
        model,
        external_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1701 + eval_offset * 10 + 3,
        return_payload=True,
    )
    stage_metrics = {
        "primary_clean": clean_metrics["mae_k"],
        "primary_noisy": noisy_metrics["mae_k"],
        "primary_hard_subgroup": clean_metrics["hard_subgroup_mae_k"],
        "external_holdout": external_metrics["mae_k"],
        "external_hard_subgroup": external_metrics["hard_subgroup_mae_k"],
    }
    return stage_metrics, clean_payload, noisy_payload, external_payload


def external_cluster_reduction_row(
    *,
    baseline_external_payload: dict[str, np.ndarray],
    candidate_external_payload: dict[str, np.ndarray],
    cluster_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    baseline_error = np.asarray(baseline_external_payload["error"], dtype=np.float64).reshape(-1)
    candidate_error = np.asarray(candidate_external_payload["error"], dtype=np.float64).reshape(-1)
    row: dict[str, float] = {}
    for cluster_name, mask in cluster_masks.items():
        row[f"cluster_{cluster_name}_mae_reduction_k"] = float(baseline_error[mask].mean() - candidate_error[mask].mean())
    return row


def greedy_soup_candidate(
    *,
    student: nn.Module,
    candidates: list[dict[str, Any]],
    val_loader: torch.utils.data.DataLoader,
    external_loader: torch.utils.data.DataLoader,
    seed_tensors: dict[str, Any],
    seed: int,
    anchor_clean: dict[str, float],
    anchor_noisy: dict[str, float],
    anchor_external: dict[str, float],
    anchor_clean_payload: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    if len(candidates) < 2:
        return None
    ordered = sorted(candidates, key=splithead_priority_tuple)[:4]
    soup_members = [ordered[0]]
    soup_state = average_state_dicts([ordered[0]["state"]])
    best_candidate = build_checkpoint_candidate(
        student=student,
        state=soup_state,
        stage="HEAD_SOUP",
        epoch=int(ordered[0]["epoch"]),
        val_loader=val_loader,
        external_loader=external_loader,
        seed_tensors=seed_tensors,
        seed=seed,
        eval_offset=500,
        anchor_clean=anchor_clean,
        anchor_noisy=anchor_noisy,
        anchor_external=anchor_external,
        anchor_clean_payload=anchor_clean_payload,
    )
    best_candidate["aggregation"] = "greedy_soup"
    for idx, candidate in enumerate(ordered[1:], start=1):
        trial_members = soup_members + [candidate]
        trial_state = average_state_dicts([member["state"] for member in trial_members])
        trial_candidate = build_checkpoint_candidate(
            student=student,
            state=trial_state,
            stage="HEAD_SOUP",
            epoch=int(candidate["epoch"]),
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            eval_offset=500 + idx,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        trial_candidate["aggregation"] = "greedy_soup"
        if splithead_priority_tuple(trial_candidate) < splithead_priority_tuple(best_candidate):
            soup_members = trial_members
            best_candidate = trial_candidate
    return best_candidate if len(soup_members) > 1 else None


def masd_current_loss(
    out: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    *,
    model: nn.Module | None = None,
    mode: str,
    stage: str,
    second_out: dict[str, torch.Tensor] | None,
    cluster_support: torch.Tensor,
    chemistry_multihot: torch.Tensor | None,
    other_subcluster_code: torch.Tensor | None,
    stage_progress: float,
    dro_cap: float = 0.42,
    dro_temperature: float = 0.14,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_err = F.smooth_l1_loss(out["pred"], y_true, reduction="none")
    full_err = torch.abs(out["pred"] - y_true)
    anchor_err = torch.abs(out["masd_anchor_pred"] - y_true)
    hard_reweight_alpha = float(getattr(model, "pr_hard_reweight_alpha", 0.0)) if model is not None else 0.0
    hard_weights: torch.Tensor | None = None
    if hard_reweight_alpha > 0.0 and "pr_hard_score" in out:
        hard_weights = 1.0 + hard_reweight_alpha * out["pr_hard_score"].detach()
    pred_loss = sample_err.mean() if hard_weights is None else (sample_err * hard_weights).sum() / hard_weights.sum().clamp_min(1.0)
    anchor_margin_raw = torch.relu(full_err - anchor_err + 0.0015)
    anchor_margin = anchor_margin_raw.mean() if hard_weights is None else (anchor_margin_raw * hard_weights).sum() / hard_weights.sum().clamp_min(1.0)
    proxy_target = out["masd_proxy_target"].detach()
    signed_proxy = out["masd_signed_proxy_target"].detach()
    contrib = out["masd_contribution"]
    contrib_abs = contrib.abs()
    sign_target = torch.sign(signed_proxy)
    sign_loss = ((torch.relu(-contrib * sign_target) * proxy_target).sum(dim=1)).mean()
    rank_loss = pairwise_rank_loss(contrib_abs, proxy_target)
    contrib_center = contrib - contrib.mean(dim=1, keepdim=True)
    target_center = signed_proxy - signed_proxy.mean(dim=1, keepdim=True)
    corr_loss = (1.0 - F.cosine_similarity(contrib_center, target_center, dim=1)).mean()
    alpha_anchor = F.kl_div(torch.log(out["masd_alpha"].clamp_min(1e-8)), proxy_target, reduction="batchmean")
    sparse_penalty = (1.0 - out["masd_alpha_max"]).mean() + 0.25 * out["masd_entropy"].mean()
    mag_norm = out["masd_main_mag"] / out["masd_main_mag"].sum(dim=1, keepdim=True).clamp_min(1e-8)
    calib_align = F.smooth_l1_loss(mag_norm, proxy_target)
    diversity_penalty = (1.0 - out["masd_diversity"]).mean()
    disagreement = out["masd_mechanism_disagreement"]
    risk = torch.sigmoid(
        1.95
        * (
            out["conflict_level"]
            + 1.20 * out["uncertainty_level"]
            + 0.90 * out["masd_entropy"]
            + 0.85 * disagreement
            + 0.65 * (1.0 - out["masd_alpha_max"])
            - 1.95
        )
    )
    gate_high_penalty = (out["masd_gate"] * risk * contrib_abs.sum(dim=1, keepdim=True)).mean()
    gate_low_penalty = (torch.relu(0.18 - out["masd_gate"]) * (1.0 - risk)).mean()
    scale_floor = (torch.relu(0.011 - contrib_abs.sum(dim=1, keepdim=True)) * (1.0 - risk)).mean()
    hard_like_raw = full_err * risk
    hard_like_loss = hard_like_raw.mean() if hard_weights is None else (hard_like_raw * hard_weights).sum() / hard_weights.sum().clamp_min(1.0)

    gate_consistency = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    alpha_consistency = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    if second_out is not None:
        high_mask = (risk >= torch.quantile(risk.detach().reshape(-1), 0.80)).float()
        denom = high_mask.sum().clamp_min(1.0)
        gate_consistency = (((out["masd_gate"] - second_out["masd_gate"]).abs() * high_mask).sum() / denom)
        alpha_consistency = ((((out["masd_alpha"] - second_out["masd_alpha"]).abs().mean(dim=1, keepdim=True)) * high_mask).sum() / denom)

    weak_mask = (cluster_support == WEAK_CLUSTER_CODE).float()
    unstable_mask = (cluster_support == UNSTABLE_CLUSTER_CODE).float()
    hard_like_mask = (risk >= torch.quantile(risk.detach().reshape(-1), 0.60)).float()
    hi_uncertainty = (out["uncertainty_level"] >= torch.quantile(out["uncertainty_level"].detach().reshape(-1), 0.75)).float()
    chemistry_cluster_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    chemistry_focus_mask = torch.zeros_like(sample_err)
    other_subcluster_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    other_subcluster_mask = torch.zeros_like(sample_err)
    if chemistry_multihot is not None and chemistry_multihot.numel() > 0:
        chemistry_multihot = chemistry_multihot.float()
        for cluster_name in CHEMISTRY_TAIL_LOSS_WEIGHTS.keys():
            cluster_idx = CHEMISTRY_CLUSTER_INDEX[cluster_name]
            tag_mask = (chemistry_multihot[:, cluster_idx : cluster_idx + 1] > 0.5).float()
            if float(tag_mask.sum().detach().item()) > 0.0:
                chemistry_focus_mask = torch.maximum(chemistry_focus_mask, tag_mask)
        if float(chemistry_focus_mask.sum().detach().item()) > 0.0:
            chemistry_cluster_loss = (full_err * chemistry_focus_mask).sum() / chemistry_focus_mask.sum().clamp_min(1.0)
    if other_subcluster_code is not None and other_subcluster_code.numel() > 0:
        other_subcluster_mask = (other_subcluster_code >= 0).float()
        if float(other_subcluster_mask.sum().detach().item()) > 0.0:
            subcluster_terms: list[torch.Tensor] = []
            present_codes = torch.unique(other_subcluster_code[other_subcluster_code >= 0]).detach().cpu().tolist()
            for raw_code in present_codes:
                code = int(raw_code)
                code_mask = (other_subcluster_code == code).float()
                if float(code_mask.sum().detach().item()) <= 0.0:
                    continue
                subcluster_terms.append((full_err * code_mask).sum() / code_mask.sum().clamp_min(1.0))
            if subcluster_terms:
                other_subcluster_loss = torch.stack(subcluster_terms).mean()
    stage_weight_scale = 0.0 if stage == "A" else (0.45 if stage == "B" else 1.0)
    stabilization_weight = 1.0 + stage_weight_scale * (
        0.28 * weak_mask
        + 0.16 * unstable_mask
        + 0.22 * hard_like_mask
        + 0.10 * hi_uncertainty
        + 0.18 * chemistry_focus_mask
        + 0.08 * other_subcluster_mask
    )
    if hard_weights is not None:
        stabilization_weight = stabilization_weight * hard_weights
    weighted_sample_err = sample_err * stabilization_weight
    dro_loss = group_dro_lite_loss(
        weighted_sample_err,
        uncertainty=out["uncertainty_level"],
        conflict=out["conflict_level"],
        entropy=out["masd_entropy"],
        disagreement=disagreement,
        cluster_support=cluster_support,
        hard_like_mask=hard_like_mask,
        enabled=(stage == "C" and mode != "main_core_sci2_masd_current_no_group_dro_lite"),
        cap=dro_cap,
        temperature=dro_temperature,
        warmup=stage_progress,
    )

    if mode == "main_core_sci2_masd_current_no_risk_adaptive_alpha":
        sparse_penalty = sparse_penalty * 0.0
    if mode == "main_core_sci2_masd_current_no_monotonic_calibrator":
        calib_align = calib_align * 0.0
    if mode == "main_core_sci2_masd_current_no_monotone_risk_gate":
        gate_high_penalty = gate_high_penalty * 0.0
        gate_low_penalty = gate_low_penalty * 0.0
        gate_consistency = gate_consistency * 0.0
    mean_correction_penalty = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    sparse_gate_penalty = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    if bool(getattr(model, "pr_thresholded_masd_enabled", False)):
        thresholded_gate = out.get("masd_thresholded_gate")
        applied_delta = out.get("masd_applied_delta")
        if thresholded_gate is not None:
            sparse_gate_penalty = thresholded_gate.mean()
        if applied_delta is not None:
            mean_correction_penalty = torch.abs(applied_delta.mean())
    total = (
        pred_loss
        + 0.84 * anchor_margin
        + 0.14 * alpha_anchor
        + 0.22 * sign_loss
        + 0.14 * rank_loss
        + 0.16 * corr_loss
        + 0.10 * sparse_penalty
        + 0.10 * calib_align
        + 0.10 * diversity_penalty
        + 0.14 * gate_high_penalty
        + 0.05 * gate_low_penalty
        + 0.04 * scale_floor
        + 0.05 * gate_consistency
        + 0.04 * alpha_consistency
        + stage_weight_scale * (0.06 * chemistry_cluster_loss + 0.04 * other_subcluster_loss)
        + (float(getattr(model, "pr_thresholded_masd_mean_lambda", 0.0)) if model is not None else 0.0) * mean_correction_penalty
        + (float(getattr(model, "pr_thresholded_masd_sparse_lambda", 0.0)) if model is not None else 0.0) * sparse_gate_penalty
    )
    if stage == "C":
        warm = float(max(0.0, min(1.0, stage_progress)))
        weighted_focus_loss = weighted_sample_err.mean()
        total = total + (0.05 + 0.03 * warm) * hard_like_loss + (0.08 + 0.04 * warm) * dro_loss + (0.03 + 0.03 * warm) * weighted_focus_loss
    return total, {
        "mean_correction_penalty": float(mean_correction_penalty.detach().cpu()),
        "sparse_gate_penalty": float(sparse_gate_penalty.detach().cpu()),
    }


def train_masd_splithead_student(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
    current_rcmf: nn.Module,
    epoch_log: list[float],
) -> nn.Module:
    set_seed(93150 + seed * 131)
    student = build_model(CURRENT_MODE, seed_tensors, config)
    copy_shared_weights(current_rcmf, student)

    split_a, split_b, splithead_meta = build_splithead_partitions(
        current_rcmf=current_rcmf,
        seed_tensors=seed_tensors,
        split=split,
        config=config,
        seed=seed,
    )
    train_loader_a = make_loader(
        seed_tensors,
        split_a,
        config.batch_size,
        shuffle=True,
        loader_seed=seed * 1103 + 1,
    )
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False, loader_seed=seed * 1103 + 3)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False, loader_seed=seed * 1103 + 5)

    anchor_clean, anchor_clean_payload = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 1, return_payload=True)
    anchor_noisy, _ = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 503 + 2, return_payload=True)
    anchor_external, _ = evaluate_stage(current_rcmf, external_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 3, return_payload=False)

    stage_a_candidates: list[dict[str, Any]] = []
    stage_a_specs = (
        ("A", 4, 6.5e-5, 2, 1.0),
        ("B", 2, 1.6e-5, 1, 1.2),
        ("C", 2, 8.0e-6, 1, 1.8),
    )
    eval_counter = 0
    for stage_name, epochs, lr, patience, weight_decay_scale in stage_a_specs:
        set_masd_trainable(student, stage=stage_name)
        optimizer = torch.optim.AdamW(
            [param for param in student.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=config.weight_decay * weight_decay_scale,
        )
        bad_epochs = 0
        best_local = float("inf")
        for epoch_idx in range(epochs):
            tick = time.time()
            student.train()
            for batch in train_loader_a:
                batch = _to_device(batch)
                out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                second_out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                loss, _ = masd_current_loss(
                    out,
                    batch["y"],
                    model=student,
                    mode=CURRENT_MODE,
                    stage=stage_name,
                    second_out=second_out,
                    cluster_support=batch["cluster_support"],
                    chemistry_multihot=batch["chemistry_multihot"],
                    other_subcluster_code=batch["other_subcluster_code"],
                    stage_progress=float(epoch_idx + 1) / float(epochs),
                    dro_cap=0.42,
                    dro_temperature=0.14,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0 if stage_name == "A" else 2.5)
                optimizer.step()
            epoch_log.append(float(time.time() - tick))
            candidate = build_checkpoint_candidate(
                student=student,
                state={key: value.detach().cpu() for key, value in student.state_dict().items()},
                stage=f"SPLIT_A_{stage_name}",
                epoch=epoch_idx,
                val_loader=val_loader,
                external_loader=external_loader,
                seed_tensors=seed_tensors,
                seed=seed,
                eval_offset=eval_counter,
                anchor_clean=anchor_clean,
                anchor_noisy=anchor_noisy,
                anchor_external=anchor_external,
                anchor_clean_payload=anchor_clean_payload,
            )
            eval_counter += 1
            stage_a_candidates.append(candidate)
            if candidate["val_score"] < best_local:
                best_local = candidate["val_score"]
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if not stage_a_candidates:
        raise RuntimeError("split-head stage-A produced no candidates")
    selected_a = select_tailfix_checkpoint(stage_a_candidates)
    student.load_state_dict(selected_a["state"])

    split_b_weights, split_b_weight_meta = build_splithead_sample_weights(
        reference_model=student,
        seed_tensors=seed_tensors,
        split_b_indices=split_b,
        config=config,
        seed=seed,
    )
    train_loader_b = make_loader(
        seed_tensors,
        split_b,
        config.batch_size,
        shuffle=False,
        sample_weights=split_b_weights,
        loader_seed=seed * 1103 + 7,
    )
    late_candidates: list[dict[str, Any]] = []
    optimizer = None
    head_epochs = 6
    best_local = float("inf")
    bad_epochs = 0
    for epoch_idx in range(head_epochs):
        set_masd_split_head_trainable(student)
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [param for param in student.parameters() if param.requires_grad],
                lr=7.5e-6,
                weight_decay=config.weight_decay * 3.0,
            )
        tick = time.time()
        student.train()
        for batch in train_loader_b:
            batch = _to_device(batch)
            out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
            second_out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
            loss, _ = masd_current_loss(
                out,
                batch["y"],
                model=student,
                mode=CURRENT_MODE,
                stage="C",
                second_out=second_out,
                cluster_support=batch["cluster_support"],
                chemistry_multihot=batch["chemistry_multihot"],
                other_subcluster_code=batch["other_subcluster_code"],
                stage_progress=float(epoch_idx + 1) / float(head_epochs),
                dro_cap=0.36,
                dro_temperature=0.11,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.0)
            optimizer.step()
        epoch_log.append(float(time.time() - tick))
        candidate = build_checkpoint_candidate(
            student=student,
            state={key: value.detach().cpu() for key, value in student.state_dict().items()},
            stage="SPLIT_B_HEAD",
            epoch=epoch_idx,
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            eval_offset=100 + epoch_idx,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        candidate["aggregation"] = "raw"
        late_candidates.append(candidate)
        if candidate["val_score"] < best_local:
            best_local = candidate["val_score"]
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 2:
                break

    if not late_candidates:
        raise RuntimeError("split-head stage-B produced no candidates")

    aggregate_candidates: list[dict[str, Any]] = []
    late_pool = late_candidates[-min(4, len(late_candidates)) :]
    if len(late_pool) >= 2:
        swa_state = average_state_dicts([item["state"] for item in late_pool])
        swa_candidate = build_checkpoint_candidate(
            student=student,
            state=swa_state,
            stage="SPLIT_B_SWA",
            epoch=int(late_pool[-1]["epoch"]),
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            eval_offset=200,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        swa_candidate["aggregation"] = "swa"
        aggregate_candidates.append(swa_candidate)
        soup_candidate = greedy_soup_candidate(
            student=student,
            candidates=late_pool,
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        if soup_candidate is not None:
            aggregate_candidates.append(soup_candidate)

    selected_b = select_splithead_checkpoint(late_candidates + aggregate_candidates)
    student.load_state_dict(selected_b["state"])

    candidate_log = []
    for item in stage_a_candidates + late_candidates + aggregate_candidates:
        compact = {key: value for key, value in item.items() if key != "state"}
        candidate_log.append(compact)
    student._masd_checkpoint_meta = {
        "selection_policy": "splithead_stabilization",
        "selected_stage": selected_b["stage"],
        "selected_epoch": int(selected_b["epoch"]),
        "val_primary": float(selected_b["val_primary"]),
        "val_hard": float(selected_b["val_hard"]),
        "proxy_hard_positive_bin_count": float(selected_b["proxy_hard_positive_bin_count"]),
        "proxy_hard_worst_bin_delta": float(selected_b["proxy_hard_worst_bin_delta"]),
        "weak_cluster_worst_delta": float(selected_b["weak_cluster_worst_delta"]),
        "weak_cluster_mean_delta": float(selected_b["weak_cluster_mean_delta"]),
        "weak_cluster_positive_count": float(selected_b["weak_cluster_positive_count"]),
        "gate_volatility": float(selected_b["gate_volatility"]),
        "ambiguity_variance": float(selected_b["ambiguity_variance"]),
        "epsilon": float(SPLITHEAD_PRIMARY_EPSILON),
        "aggregation": str(selected_b.get("aggregation", "raw")),
        "splithead_meta": splithead_meta,
        "split_b_weight_meta": split_b_weight_meta,
    }
    student._masd_checkpoint_candidates = candidate_log
    return student


def train_masd_self_stabilization_student(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
    current_rcmf: nn.Module,
    epoch_log: list[float],
) -> nn.Module:
    set_seed(93240 + seed * 131)
    student = build_model(CURRENT_MODE, seed_tensors, config)
    copy_shared_weights(current_rcmf, student)

    split_a, candidate_pool, splithead_meta = build_splithead_partitions(
        current_rcmf=current_rcmf,
        seed_tensors=seed_tensors,
        split=split,
        config=config,
        seed=seed,
    )
    train_loader_a = make_loader(
        seed_tensors,
        split_a,
        config.batch_size,
        shuffle=True,
        loader_seed=seed * 1111 + 1,
    )
    candidate_loader = make_loader(
        seed_tensors,
        candidate_pool,
        config.batch_size,
        shuffle=False,
        loader_seed=seed * 1111 + 3,
    )
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False, loader_seed=seed * 1111 + 5)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False, loader_seed=seed * 1111 + 7)

    anchor_clean, anchor_clean_payload = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 1, return_payload=True)
    anchor_noisy, _ = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 503 + 2, return_payload=True)
    anchor_external, _ = evaluate_stage(current_rcmf, external_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 3, return_payload=False)

    stage_a_candidates: list[dict[str, Any]] = []
    stage_a_specs = (
        ("A", 4, 6.5e-5, 2, 1.0),
        ("B", 2, 1.6e-5, 1, 1.2),
        ("C", 2, 8.0e-6, 1, 1.8),
    )
    eval_counter = 0
    for stage_name, epochs, lr, patience, weight_decay_scale in stage_a_specs:
        set_masd_trainable(student, stage=stage_name)
        optimizer = torch.optim.AdamW(
            [param for param in student.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=config.weight_decay * weight_decay_scale,
        )
        best_local = float("inf")
        bad_epochs = 0
        for epoch_idx in range(epochs):
            tick = time.time()
            student.train()
            for batch in train_loader_a:
                batch = _to_device(batch)
                out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                second_out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                loss, _ = masd_current_loss(
                    out,
                    batch["y"],
                    model=student,
                    mode=CURRENT_MODE,
                    stage=stage_name,
                    second_out=second_out,
                    cluster_support=batch["cluster_support"],
                    chemistry_multihot=batch["chemistry_multihot"],
                    other_subcluster_code=batch["other_subcluster_code"],
                    stage_progress=float(epoch_idx + 1) / float(epochs),
                    dro_cap=0.42,
                    dro_temperature=0.14,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0 if stage_name == "A" else 2.5)
                optimizer.step()
            epoch_log.append(float(time.time() - tick))
            candidate = build_checkpoint_candidate(
                student=student,
                state={key: value.detach().cpu() for key, value in student.state_dict().items()},
                stage=f"SELF_A_{stage_name}",
                epoch=epoch_idx,
                val_loader=val_loader,
                external_loader=external_loader,
                seed_tensors=seed_tensors,
                seed=seed,
                eval_offset=eval_counter,
                anchor_clean=anchor_clean,
                anchor_noisy=anchor_noisy,
                anchor_external=anchor_external,
                anchor_clean_payload=anchor_clean_payload,
            )
            eval_counter += 1
            stage_a_candidates.append(candidate)
            if candidate["val_score"] < best_local:
                best_local = candidate["val_score"]
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if not stage_a_candidates:
        raise RuntimeError("self-style stage-A produced no candidates")
    selected_a = select_self_stabilization_checkpoint(stage_a_candidates)
    reference_a = select_stage_a_reference_candidate(stage_a_candidates, epsilon=SELF_STABILIZATION_PRIMARY_EPSILON)
    student.load_state_dict(selected_a["state"])

    _, final_pool_payload = evaluate_stage(
        student,
        candidate_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1111 + 11,
        return_payload=True,
    )
    student.load_state_dict(reference_a["state"])
    _, reference_pool_payload = evaluate_stage(
        student,
        candidate_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1111 + 13,
        return_payload=True,
    )
    student.load_state_dict(selected_a["state"])

    selective_split_b, selective_weights, selective_meta = build_self_split_b_selection(
        final_payload=final_pool_payload,
        reference_payload=reference_pool_payload,
    )
    if not selective_split_b:
        raise RuntimeError("self-style split-B selection produced no samples")
    train_loader_b = make_loader(
        seed_tensors,
        selective_split_b,
        config.batch_size,
        shuffle=False,
        sample_weights=selective_weights,
        loader_seed=seed * 1111 + 17,
    )

    late_candidates: list[dict[str, Any]] = []
    optimizer = None
    head_epochs = 7
    best_local = float("inf")
    bad_epochs = 0
    for epoch_idx in range(head_epochs):
        set_masd_split_head_trainable(student)
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [param for param in student.parameters() if param.requires_grad],
                lr=6.5e-6,
                weight_decay=config.weight_decay * 3.4,
            )
        tick = time.time()
        student.train()
        for batch in train_loader_b:
            batch = _to_device(batch)
            out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
            second_out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
            loss, _ = masd_current_loss(
                out,
                batch["y"],
                model=student,
                mode=CURRENT_MODE,
                stage="C",
                second_out=second_out,
                cluster_support=batch["cluster_support"],
                chemistry_multihot=batch["chemistry_multihot"],
                other_subcluster_code=batch["other_subcluster_code"],
                stage_progress=float(epoch_idx + 1) / float(head_epochs),
                dro_cap=0.34,
                dro_temperature=0.10,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.0)
            optimizer.step()
        epoch_log.append(float(time.time() - tick))
        candidate = build_checkpoint_candidate(
            student=student,
            state={key: value.detach().cpu() for key, value in student.state_dict().items()},
            stage="SELF_B_HEAD",
            epoch=epoch_idx,
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            eval_offset=120 + epoch_idx,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        candidate["aggregation"] = "raw"
        late_candidates.append(candidate)
        if candidate["val_score"] < best_local:
            best_local = candidate["val_score"]
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 2:
                break

    if not late_candidates:
        raise RuntimeError("self-style stage-B produced no candidates")

    aggregate_candidates: list[dict[str, Any]] = []
    late_pool = late_candidates[-min(4, len(late_candidates)) :]
    if len(late_pool) >= 2:
        swa_state = average_state_dicts([item["state"] for item in late_pool])
        swa_candidate = build_checkpoint_candidate(
            student=student,
            state=swa_state,
            stage="SELF_B_SWA",
            epoch=int(late_pool[-1]["epoch"]),
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            eval_offset=240,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        swa_candidate["aggregation"] = "swa"
        aggregate_candidates.append(swa_candidate)
        soup_candidate = greedy_soup_candidate(
            student=student,
            candidates=late_pool,
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        if soup_candidate is not None:
            soup_candidate["stage"] = "SELF_B_SOUP"
            aggregate_candidates.append(soup_candidate)

    selected_b = select_self_stabilization_checkpoint(late_candidates + aggregate_candidates)
    student.load_state_dict(selected_b["state"])

    candidate_log = []
    for item in stage_a_candidates + late_candidates + aggregate_candidates:
        compact = {key: value for key, value in item.items() if key != "state"}
        candidate_log.append(compact)
    student._masd_checkpoint_meta = {
        "selection_policy": "self_stabilization",
        "selected_stage": selected_b["stage"],
        "selected_epoch": int(selected_b["epoch"]),
        "val_primary": float(selected_b["val_primary"]),
        "val_hard": float(selected_b["val_hard"]),
        "proxy_hard_positive_bin_count": float(selected_b["proxy_hard_positive_bin_count"]),
        "proxy_hard_worst_bin_delta": float(selected_b["proxy_hard_worst_bin_delta"]),
        "weak_cluster_worst_delta": float(selected_b["weak_cluster_worst_delta"]),
        "weak_cluster_mean_delta": float(selected_b["weak_cluster_mean_delta"]),
        "weak_cluster_positive_count": float(selected_b["weak_cluster_positive_count"]),
        "gate_volatility": float(selected_b["gate_volatility"]),
        "ambiguity_variance": float(selected_b["ambiguity_variance"]),
        "epsilon": float(SELF_STABILIZATION_PRIMARY_EPSILON),
        "aggregation": str(selected_b.get("aggregation", "raw")),
        "splithead_meta": splithead_meta,
        "self_selection_meta": selective_meta,
    }
    student._masd_checkpoint_candidates = candidate_log
    return student


def train_masd_jtt_stabilization_student(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
    current_rcmf: nn.Module,
    epoch_log: list[float],
) -> nn.Module:
    set_seed(93340 + seed * 131)
    identification = build_model(CURRENT_MODE, seed_tensors, config)
    copy_shared_weights(current_rcmf, identification)

    train_loader_id = make_loader(
        seed_tensors,
        split["train"],
        config.batch_size,
        shuffle=True,
        loader_seed=seed * 1201 + 1,
    )
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False, loader_seed=seed * 1201 + 3)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False, loader_seed=seed * 1201 + 5)

    anchor_clean, anchor_clean_payload = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 1, return_payload=True)
    anchor_noisy, _ = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 503 + 2, return_payload=True)
    anchor_external, _ = evaluate_stage(current_rcmf, external_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 3, return_payload=False)

    id_candidates: list[dict[str, Any]] = []
    id_specs = (
        ("A", 4, 6.5e-5, 2, 1.0),
        ("B", 2, 1.5e-5, 1, 1.2),
        ("C", 2, 7.5e-6, 1, 1.7),
    )
    eval_counter = 0
    for stage_name, epochs, lr, patience, weight_decay_scale in id_specs:
        set_masd_trainable(identification, stage=stage_name)
        optimizer = torch.optim.AdamW(
            [param for param in identification.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=config.weight_decay * weight_decay_scale,
        )
        bad_epochs = 0
        best_local = float("inf")
        for epoch_idx in range(epochs):
            tick = time.time()
            identification.train()
            for batch in train_loader_id:
                batch = _to_device(batch)
                out = identification(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                second_out = identification(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                loss, _ = masd_current_loss(
                    out,
                    batch["y"],
                    model=identification,
                    mode=CURRENT_MODE,
                    stage=stage_name,
                    second_out=second_out,
                    cluster_support=batch["cluster_support"],
                    chemistry_multihot=batch["chemistry_multihot"],
                    other_subcluster_code=batch["other_subcluster_code"],
                    stage_progress=float(epoch_idx + 1) / float(epochs),
                    dro_cap=0.42,
                    dro_temperature=0.14,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(identification.parameters(), max_norm=3.0 if stage_name == "A" else 2.5)
                optimizer.step()
            epoch_log.append(float(time.time() - tick))
            candidate = build_checkpoint_candidate(
                student=identification,
                state={key: value.detach().cpu() for key, value in identification.state_dict().items()},
                stage=f"JTT_ID_{stage_name}",
                epoch=epoch_idx,
                val_loader=val_loader,
                external_loader=external_loader,
                seed_tensors=seed_tensors,
                seed=seed,
                eval_offset=eval_counter,
                anchor_clean=anchor_clean,
                anchor_noisy=anchor_noisy,
                anchor_external=anchor_external,
                anchor_clean_payload=anchor_clean_payload,
            )
            eval_counter += 1
            id_candidates.append(candidate)
            if candidate["val_score"] < best_local:
                best_local = candidate["val_score"]
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if not id_candidates:
        raise RuntimeError("jtt identification stage produced no candidates")
    selected_id = select_jtt_stabilization_checkpoint(id_candidates)
    identification.load_state_dict(selected_id["state"])

    train_eval_loader = make_loader(
        seed_tensors,
        split["train"],
        config.batch_size,
        shuffle=False,
        loader_seed=seed * 1201 + 7,
    )
    _, train_payload = evaluate_stage(
        identification,
        train_eval_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1201 + 11,
        return_payload=True,
    )
    error_set_ids, weight_lookup, error_set_meta = build_jtt_error_set(payload=train_payload)
    if not error_set_ids:
        raise RuntimeError("jtt error set is empty")

    full_train_weights = [weight_lookup.get(int(idx), 1.0) for idx in split["train"]]
    train_loader = make_loader(
        seed_tensors,
        split["train"],
        config.batch_size,
        shuffle=False,
        sample_weights=full_train_weights,
        loader_seed=seed * 1201 + 13,
    )

    student = build_model(CURRENT_MODE, seed_tensors, config)
    copy_shared_weights(current_rcmf, student)
    checkpoint_candidates: list[dict[str, Any]] = []
    stage_specs = (
        ("A", 5, 6.2e-5, 2, 1.20),
        ("B", 3, 1.4e-5, 1, 1.55),
        ("C", 3, 6.5e-6, 1, 2.20),
    )
    eval_counter = 0
    for stage_name, epochs, lr, patience, weight_decay_scale in stage_specs:
        for param in student.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=lr,
            weight_decay=config.weight_decay * weight_decay_scale,
        )
        bad_epochs = 0
        best_local = float("inf")
        for epoch_idx in range(epochs):
            tick = time.time()
            student.train()
            for batch in train_loader:
                batch = _to_device(batch)
                out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                second_out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                loss, _ = masd_current_loss(
                    out,
                    batch["y"],
                    model=student,
                    mode=CURRENT_MODE,
                    stage=stage_name,
                    second_out=second_out,
                    cluster_support=batch["cluster_support"],
                    chemistry_multihot=batch["chemistry_multihot"],
                    other_subcluster_code=batch["other_subcluster_code"],
                    stage_progress=float(epoch_idx + 1) / float(epochs),
                    dro_cap=0.40,
                    dro_temperature=0.13,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0 if stage_name == "A" else 2.5)
                optimizer.step()
            epoch_log.append(float(time.time() - tick))
            candidate = build_checkpoint_candidate(
                student=student,
                state={key: value.detach().cpu() for key, value in student.state_dict().items()},
                stage=f"JTT_{stage_name}",
                epoch=epoch_idx,
                val_loader=val_loader,
                external_loader=external_loader,
                seed_tensors=seed_tensors,
                seed=seed,
                eval_offset=300 + eval_counter,
                anchor_clean=anchor_clean,
                anchor_noisy=anchor_noisy,
                anchor_external=anchor_external,
                anchor_clean_payload=anchor_clean_payload,
            )
            eval_counter += 1
            checkpoint_candidates.append(candidate)
            if candidate["val_score"] < best_local:
                best_local = candidate["val_score"]
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if not checkpoint_candidates:
        raise RuntimeError("jtt full retraining produced no checkpoints")
    selected = select_jtt_stabilization_checkpoint(checkpoint_candidates)
    student.load_state_dict(selected["state"])

    candidate_log = []
    for item in id_candidates + checkpoint_candidates:
        compact = {key: value for key, value in item.items() if key != "state"}
        candidate_log.append(compact)
    student._masd_checkpoint_meta = {
        "selection_policy": "jtt_stabilization",
        "selected_stage": selected["stage"],
        "selected_epoch": int(selected["epoch"]),
        "val_primary": float(selected["val_primary"]),
        "val_hard": float(selected["val_hard"]),
        "proxy_hard_positive_bin_count": float(selected["proxy_hard_positive_bin_count"]),
        "proxy_hard_worst_bin_delta": float(selected["proxy_hard_worst_bin_delta"]),
        "weak_cluster_worst_delta": float(selected["weak_cluster_worst_delta"]),
        "weak_cluster_mean_delta": float(selected["weak_cluster_mean_delta"]),
        "weak_cluster_positive_count": float(selected["weak_cluster_positive_count"]),
        "gate_volatility": float(selected["gate_volatility"]),
        "ambiguity_variance": float(selected["ambiguity_variance"]),
        "epsilon": float(JTT_STABILIZATION_PRIMARY_EPSILON),
        "aggregation": "raw",
        "error_set_meta": error_set_meta,
    }
    student._masd_checkpoint_candidates = candidate_log
    return student


def train_masd_ctgf_student(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
    current_rcmf: nn.Module,
    epoch_log: list[float],
    base_final_override: nn.Module | None = None,
) -> nn.Module:
    if base_final_override is None:
        base_final = train_masd_current_student(
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=seed,
            current_rcmf=current_rcmf,
            mode=CURRENT_MODE,
            selection_policy="tailfix",
            epoch_log=epoch_log,
        )
    else:
        base_final = build_model(CURRENT_MODE, seed_tensors, config)
        base_final.load_state_dict(base_final_override.state_dict())
        base_final.to(DEVICE)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False, loader_seed=seed * 1301 + 3)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False, loader_seed=seed * 1301 + 5)
    train_eval_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=False, loader_seed=seed * 1301 + 7)

    anchor_clean, anchor_clean_payload = evaluate_stage(base_final, val_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 101, return_payload=True)
    anchor_noisy, _ = evaluate_stage(base_final, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 503 + 102, return_payload=True)
    anchor_external, _ = evaluate_stage(base_final, external_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 103, return_payload=False)
    _, train_payload = evaluate_stage(
        base_final,
        train_eval_loader,
        seed_tensors,
        variant="clean",
        noise_seed=seed * 1301 + 11,
        return_payload=True,
    )
    error_set_ids, weight_lookup, error_set_meta = build_jtt_error_set(payload=train_payload)
    if not error_set_ids:
        raise RuntimeError("ctgf error set is empty")

    weak_set = set(np.asarray(train_payload["sample_index"], dtype=np.int64).reshape(-1)[np.asarray(train_payload["cluster_support"], dtype=np.int64).reshape(-1) == WEAK_CLUSTER_CODE].tolist())
    unstable_set = set(np.asarray(train_payload["sample_index"], dtype=np.int64).reshape(-1)[np.asarray(train_payload["cluster_support"], dtype=np.int64).reshape(-1) == UNSTABLE_CLUSTER_CODE].tolist())
    full_train_weights: list[float] = []
    selected_ids = set(error_set_ids)
    for idx in split["train"]:
        base_weight = 1.0 + 0.55 * max(0.0, float(weight_lookup.get(int(idx), 1.0)) - 1.0)
        if int(idx) in selected_ids:
            base_weight *= 1.18
        if int(idx) in weak_set:
            base_weight *= 1.10
        elif int(idx) in unstable_set:
            base_weight *= 1.06
        full_train_weights.append(float(base_weight))
    mean_weight = float(np.mean(full_train_weights)) if full_train_weights else 1.0
    if mean_weight > 0.0:
        full_train_weights = [float(weight / mean_weight) for weight in full_train_weights]

    train_loader = make_loader(
        seed_tensors,
        split["train"],
        config.batch_size,
        shuffle=False,
        sample_weights=full_train_weights,
        loader_seed=seed * 1301 + 13,
    )

    student = build_model(CURRENT_MODE, seed_tensors, config)
    student.load_state_dict(base_final.state_dict())
    anchor_model = build_model(CURRENT_MODE, seed_tensors, config)
    anchor_model.load_state_dict(base_final.state_dict())
    anchor_model.to(DEVICE)
    anchor_model.eval()
    for param in anchor_model.parameters():
        param.requires_grad = False
    set_masd_ctgf_trainable(student)
    anchor_params = {
        name: param.detach().clone()
        for name, param in anchor_model.named_parameters()
        if param.requires_grad is False and name in dict(student.named_parameters()) and dict(student.named_parameters())[name].requires_grad
    }
    trainable_names = [name for name, param in student.named_parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=4.5e-6,
        weight_decay=config.weight_decay * 2.4,
    )
    checkpoint_candidates: list[dict[str, Any]] = []
    stage_epochs = 6
    bad_epochs = 0
    best_local = float("inf")
    for epoch_idx in range(stage_epochs):
        tick = time.time()
        student.train()
        for batch in train_loader:
            batch = _to_device(batch)
            out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
            second_out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
            with torch.no_grad():
                anchor_out = anchor_model(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
            main_loss, _ = masd_current_loss(
                out,
                batch["y"],
                model=student,
                mode=CURRENT_MODE,
                stage="C",
                second_out=second_out,
                cluster_support=batch["cluster_support"],
                chemistry_multihot=batch["chemistry_multihot"],
                other_subcluster_code=batch["other_subcluster_code"],
                stage_progress=float(epoch_idx + 1) / float(stage_epochs),
                dro_cap=0.34,
                dro_temperature=0.11,
            )
            sample_loss = F.smooth_l1_loss(out["pred"], batch["y"], reduction="none").reshape(-1)
            anchor_sample_loss = F.smooth_l1_loss(anchor_out["pred"], batch["y"], reduction="none").reshape(-1)
            sample_index = batch["sample_index"].reshape(-1)
            tail_mask = torch.as_tensor(
                [int(item) in selected_ids for item in sample_index.detach().cpu().tolist()],
                device=sample_loss.device,
                dtype=torch.bool,
            )
            weak_mask = (batch["cluster_support"].reshape(-1) > 0)
            uncertainty = out["uncertainty_level"].reshape(-1)
            conflict = out["conflict_level"].reshape(-1)
            ambiguity = out["masd_mechanism_disagreement"].reshape(-1)
            high_unc = uncertainty >= torch.quantile(uncertainty.detach(), 0.60)
            high_conf = conflict >= torch.quantile(conflict.detach(), 0.60)
            high_amb = ambiguity >= torch.quantile(ambiguity.detach(), 0.60)
            external_proxy_mask = weak_mask | (high_unc & high_conf) | high_amb
            tail_loss = sample_loss[tail_mask].mean() if bool(tail_mask.any()) else torch.zeros((), device=sample_loss.device, dtype=sample_loss.dtype)
            weak_guard = (
                torch.relu(sample_loss[weak_mask].mean() - anchor_sample_loss[weak_mask].mean() - 0.05)
                if bool(weak_mask.any())
                else torch.zeros((), device=sample_loss.device, dtype=sample_loss.dtype)
            )
            external_guard = (
                torch.relu(sample_loss[external_proxy_mask].mean() - anchor_sample_loss[external_proxy_mask].mean())
                if bool(external_proxy_mask.any())
                else torch.zeros((), device=sample_loss.device, dtype=sample_loss.dtype)
            )
            full_guard = torch.relu(sample_loss.mean() - anchor_sample_loss.mean() - 0.01)
            anchor_penalty_terms = []
            named_params = dict(student.named_parameters())
            for name in trainable_names:
                if name not in anchor_params:
                    continue
                anchor_penalty_terms.append(F.mse_loss(named_params[name], anchor_params[name].to(named_params[name].device), reduction="mean"))
            anchor_penalty = torch.stack(anchor_penalty_terms).mean() if anchor_penalty_terms else torch.zeros((), device=sample_loss.device, dtype=sample_loss.dtype)
            total_loss = main_loss + 0.40 * tail_loss + 0.18 * anchor_penalty + 0.90 * (weak_guard + external_guard + full_guard)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.0)
            optimizer.step()
        epoch_log.append(float(time.time() - tick))
        candidate = build_checkpoint_candidate(
            student=student,
            state={key: value.detach().cpu() for key, value in student.state_dict().items()},
            stage="CTGF",
            epoch=epoch_idx,
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            eval_offset=520 + epoch_idx,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        candidate["aggregation"] = "raw"
        checkpoint_candidates.append(candidate)
        if candidate["val_score"] < best_local:
            best_local = candidate["val_score"]
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 2:
                break

    if not checkpoint_candidates:
        raise RuntimeError("ctgf produced no checkpoints")

    aggregate_candidates: list[dict[str, Any]] = []
    late_pool = checkpoint_candidates[-min(4, len(checkpoint_candidates)) :]
    if len(late_pool) >= 2:
        swa_state = average_state_dicts([item["state"] for item in late_pool])
        swa_candidate = build_checkpoint_candidate(
            student=student,
            state=swa_state,
            stage="CTGF_SWA",
            epoch=int(late_pool[-1]["epoch"]),
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            eval_offset=620,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        swa_candidate["aggregation"] = "swa"
        aggregate_candidates.append(swa_candidate)
        soup_candidate = greedy_soup_candidate(
            student=student,
            candidates=late_pool,
            val_loader=val_loader,
            external_loader=external_loader,
            seed_tensors=seed_tensors,
            seed=seed,
            anchor_clean=anchor_clean,
            anchor_noisy=anchor_noisy,
            anchor_external=anchor_external,
            anchor_clean_payload=anchor_clean_payload,
        )
        if soup_candidate is not None:
            soup_candidate["stage"] = "CTGF_SOUP"
            aggregate_candidates.append(soup_candidate)

    selected = select_ctgf_checkpoint(checkpoint_candidates + aggregate_candidates)
    student.load_state_dict(selected["state"])

    candidate_log = []
    for item in checkpoint_candidates + aggregate_candidates:
        compact = {key: value for key, value in item.items() if key != "state"}
        candidate_log.append(compact)
    student._masd_checkpoint_meta = {
        "selection_policy": "ctgf",
        "selected_stage": selected["stage"],
        "selected_epoch": int(selected["epoch"]),
        "val_primary": float(selected["val_primary"]),
        "val_hard": float(selected["val_hard"]),
        "proxy_hard_positive_bin_count": float(selected["proxy_hard_positive_bin_count"]),
        "proxy_hard_worst_bin_delta": float(selected["proxy_hard_worst_bin_delta"]),
        "weak_cluster_worst_delta": float(selected["weak_cluster_worst_delta"]),
        "weak_cluster_mean_delta": float(selected["weak_cluster_mean_delta"]),
        "weak_cluster_positive_count": float(selected["weak_cluster_positive_count"]),
        "external_proxy_delta": float(selected["external_proxy_delta"]),
        "full_proxy_delta": float(selected["full_proxy_delta"]),
        "gate_volatility": float(selected["gate_volatility"]),
        "ambiguity_variance": float(selected["ambiguity_variance"]),
        "epsilon": float(CTGF_PRIMARY_EPSILON),
        "aggregation": str(selected.get("aggregation", "raw")),
        "error_set_meta": error_set_meta,
        "guardrail_meta": {
            "weak_guardrail": 0.05,
            "external_guardrail": 0.0,
            "full_guardrail": 0.02,
        },
    }
    student._masd_checkpoint_candidates = candidate_log
    return student


def train_masd_current_student(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
    current_rcmf: nn.Module,
    mode: str,
    selection_policy: str,
    epoch_log: list[float],
) -> nn.Module:
    if selection_policy == "ctgf":
        return train_masd_ctgf_student(
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=seed,
            current_rcmf=current_rcmf,
            epoch_log=epoch_log,
        )
    if selection_policy == "jtt_stabilization":
        return train_masd_jtt_stabilization_student(
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=seed,
            current_rcmf=current_rcmf,
            epoch_log=epoch_log,
        )
    if selection_policy == "self_stabilization":
        return train_masd_self_stabilization_student(
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=seed,
            current_rcmf=current_rcmf,
            epoch_log=epoch_log,
        )
    if selection_policy == "splithead_stabilization":
        return train_masd_splithead_student(
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=seed,
            current_rcmf=current_rcmf,
            epoch_log=epoch_log,
        )
    offsets = {
        "main_core_sci2_masd_current": 0,
        "main_core_sci2_masd_current_locked": 0,
        "main_core_sci2_masd_final": 0,
        "main_core_sci2_masd_final_stabilization": 61,
        "main_core_sci2_masd_current_no_risk_adaptive_alpha": 11,
        "main_core_sci2_masd_current_no_monotonic_calibrator": 23,
        "main_core_sci2_masd_current_no_monotone_risk_gate": 37,
        "main_core_sci2_masd_current_no_group_dro_lite": 43,
        "main_core_sci2_masd_no_competition": 53,
        "main_core_sci2_masd_v3_no_sparse_alpha": 59,
    }
    set_seed(93000 + seed * 131 + offsets[mode])
    student = build_model(mode, seed_tensors, config)
    copy_shared_weights(current_rcmf, student)

    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)
    use_external_holdout_in_selection = bool(USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION)

    anchor_clean, anchor_clean_payload = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 1, return_payload=True)
    anchor_noisy, _ = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 503 + 2, return_payload=True)
    anchor_external, _ = evaluate_stage(current_rcmf, external_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 3, return_payload=False)

    best_state: dict[str, Any] | None = None
    best_score = float("inf")
    checkpoint_candidates: list[dict[str, Any]] = []
    if selection_policy == "stabilization":
        stage_specs = (
            ("A", 5, 6.5e-5, 2, 1.10),
            ("B", 4, 1.3e-5, 2, 1.55),
            ("C", 5, 5.5e-6, 2, 2.60),
        )
        dro_cap = 0.34
        dro_temperature = 0.11
    elif selection_policy == "signrate_lock":
        stage_specs = (
            ("A", 5, 6.5e-5, 2, 1.05),
            ("B", 3, 1.5e-5, 2, 1.30),
            ("C", 4, 7.0e-6, 1, 2.20),
        )
        dro_cap = 0.36
        dro_temperature = 0.12
    else:
        stage_specs = (
            ("A", 5, 6.5e-5, 2, 1.0),
            ("B", 3, 1.6e-5, 2, 1.2),
            ("C", 3, 8.0e-6, 1, 1.8),
        )
        dro_cap = 0.42
        dro_temperature = 0.14
    for stage_name, epochs, lr, patience, weight_decay_scale in stage_specs:
        set_masd_trainable(student, stage=stage_name)
        optimizer = torch.optim.AdamW(
            [param for param in student.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=config.weight_decay * weight_decay_scale,
        )
        bad_epochs = 0
        for epoch_idx in range(epochs):
            tick = time.time()
            student.train()
            for batch in train_loader:
                batch = _to_device(batch)
                out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                second_out = student(batch["graph"], batch["desc"], batch["ctx"], led=batch["led"], led_mask=batch["led_mask"])
                loss, _ = masd_current_loss(
                    out,
                    batch["y"],
                    model=student,
                    mode=mode,
                    stage=stage_name,
                    second_out=second_out,
                    cluster_support=batch["cluster_support"],
                    chemistry_multihot=batch["chemistry_multihot"],
                    other_subcluster_code=batch["other_subcluster_code"],
                    stage_progress=float(epoch_idx + 1) / float(epochs),
                    dro_cap=dro_cap,
                    dro_temperature=dro_temperature,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=2.5 if stage_name == "C" else 3.0)
                optimizer.step()
            epoch_log.append(float(time.time() - tick))

            val_clean, clean_payload = evaluate_stage(student, val_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 10 + len(epoch_log), return_payload=True)
            val_noisy, noisy_payload = evaluate_stage(student, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 503 + 20 + len(epoch_log), return_payload=True)
            if use_external_holdout_in_selection:
                val_external, _ = evaluate_stage(
                    student,
                    external_loader,
                    seed_tensors,
                    variant="clean",
                    noise_seed=seed * 503 + 30 + len(epoch_log),
                    return_payload=False,
                )
            else:
                val_external = {
                    "mae_k": float(anchor_external["mae_k"]),
                    "hard_subgroup_mae_k": float(anchor_external["hard_subgroup_mae_k"]),
                }
            mech = contribution_metrics_from_payload(clean_payload, noisy_payload)
            anchor_mask_delta = common_mask_delta(clean_payload, anchor_clean_payload)
            proxy_hard = candidate_proxy_hard_metrics(clean_payload, anchor_clean_payload)
            gate_risk_penalty = max(
                0.0,
                float(mech["high_uncertainty_gate_mean"] - mech["low_risk_gate_mean"] + 0.01),
            ) + max(0.0, float(anchor_mask_delta))
            val_score = (
                val_clean["mae_k"]
                + 0.95 * max(0.0, val_clean["mae_k"] - anchor_clean["mae_k"])
                + 0.90 * max(0.0, val_noisy["mae_k"] - anchor_noisy["mae_k"])
                + 0.80 * max(0.0, anchor_mask_delta)
                + 0.75 * max(0.0, val_clean["hard_subgroup_mae_k"] - anchor_clean["hard_subgroup_mae_k"] - 0.03)
                + 0.30 * max(0.0, 0.90 - mech["contribution_sign_consistency"])
                + 0.18 * max(0.0, 0.10 - mech["contribution_anchor_alignment_corr"])
                + 0.08 * max(0.0, 0.22 - mech["mechanism_weight_sparsity"])
                + 0.06 * max(0.0, 0.42 - mech["dominant_mechanism_concentration"])
                + 0.08 * max(0.0, mech["gate_vs_error_correlation"] + 0.02)
                + 0.20 * max(0.0, mech["high_uncertainty_gate_mean"] - mech["low_risk_gate_mean"] + 0.02)
            )
            if use_external_holdout_in_selection:
                val_score = val_score + 0.60 * max(0.0, val_external["mae_k"] - anchor_external["mae_k"])
            if selection_policy == "signrate_lock":
                val_score = (
                    val_score
                    + 0.35 * proxy_hard["proxy_hard_positive_bin_count"]
                    + 0.85 * max(0.0, proxy_hard["proxy_hard_worst_bin_delta"])
                    + 0.22 * proxy_hard["proxy_hard_bin_variance"]
                    + 0.15 * proxy_hard["gate_volatility"]
                )
            elif selection_policy == "stabilization":
                val_score = (
                    val_score
                    + 0.45 * proxy_hard["proxy_hard_positive_bin_count"]
                    + 0.95 * max(0.0, proxy_hard["proxy_hard_worst_bin_delta"])
                    + 0.65 * max(0.0, proxy_hard["weak_cluster_worst_delta"] - 0.02)
                    + 0.12 * proxy_hard["weak_cluster_positive_count"]
                    + 0.12 * proxy_hard["gate_volatility"]
                    + 0.10 * proxy_hard["ambiguity_variance"]
                )
            checkpoint_candidates.append(
                {
                    "stage": stage_name,
                    "epoch": epoch_idx,
                    "val_primary": float(val_clean["mae_k"]),
                    "val_noisy": float(val_noisy["mae_k"]),
                    "val_hard": float(val_clean["hard_subgroup_mae_k"]),
                    "val_external": float(val_external["mae_k"]),
                    "anchor_mask_delta": float(anchor_mask_delta),
                    "gate_risk_penalty": float(gate_risk_penalty),
                    **proxy_hard,
                    "mechanism_pass": bool(mech["mechanism_pass"]),
                    "state": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                }
            )
            if val_score < best_score:
                best_score = val_score
                best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if best_state is None:
        raise RuntimeError(f"no checkpoint stored for {mode}")
    selected = select_checkpoint(checkpoint_candidates, selection_policy=selection_policy)
    best_state = selected["state"]
    student.load_state_dict(best_state)
    candidate_log = []
    for item in checkpoint_candidates:
        compact = {key: value for key, value in item.items() if key != "state"}
        candidate_log.append(compact)
    student._masd_checkpoint_meta = {
        "selection_policy": selection_policy,
        "selection_uses_external_holdout": bool(use_external_holdout_in_selection),
        "stage": selected["stage"],
        "epoch": int(selected["epoch"]),
        "val_primary": float(selected["val_primary"]),
        "val_hard": float(selected["val_hard"]),
        "anchor_mask_delta": float(selected["anchor_mask_delta"]),
        "gate_risk_penalty": float(selected["gate_risk_penalty"]),
        "proxy_hard_positive_bin_count": float(selected["proxy_hard_positive_bin_count"]),
        "proxy_hard_worst_bin_delta": float(selected["proxy_hard_worst_bin_delta"]),
                    "proxy_hard_bin_variance": float(selected["proxy_hard_bin_variance"]),
                    "proxy_hard_mean_delta": float(selected["proxy_hard_mean_delta"]),
                    "gate_volatility": float(selected["gate_volatility"]),
                    "weak_cluster_worst_delta": float(selected["weak_cluster_worst_delta"]),
                    "weak_cluster_mean_delta": float(selected["weak_cluster_mean_delta"]),
                    "weak_cluster_positive_count": float(selected["weak_cluster_positive_count"]),
                    "ambiguity_variance": float(selected["ambiguity_variance"]),
                    "epsilon": float(
                        STABILIZATION_PRIMARY_EPSILON
                        if selection_policy == "stabilization"
                        else (SIGNRATE_LOCK_PRIMARY_EPSILON if selection_policy == "signrate_lock" else TAILFIX_PRIMARY_EPSILON)
                    ),
                }
    student._masd_checkpoint_candidates = candidate_log
    return student


def run_mainline_seed(
    *,
    seed: int,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    selection_policy: str,
    epoch_log: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # This is the seed-wise paper pipeline: baseline -> MSCE -> RCMF bridge -> full chain.
    split = ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    baseline_model, msce_model = train_msce_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed)
    minimal_rcmf = train_rcmf_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed, repair_model=msce_model)
    current_rcmf = train_rcmf_external_focus_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed, minimal_rcmf=minimal_rcmf)
    masd_model = train_masd_current_student(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=seed,
        current_rcmf=current_rcmf,
        mode=CURRENT_MODE,
        selection_policy=selection_policy,
        epoch_log=epoch_log,
    )

    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)
    stages = [
        ("strongest_baseline", baseline_model),
        ("strongest_baseline_plus_mspce", msce_model),
        ("strongest_baseline_plus_mspce_rcmf", current_rcmf),
        (CURRENT_STAGE_NAME, masd_model),
    ]
    rows: list[dict[str, Any]] = []
    payload_bundle: dict[str, Any] = {"seed": seed}
    baseline_metrics: dict[str, float] | None = None
    prev_stage: dict[str, float] | None = None
    for idx, (name, model) in enumerate(stages):
        need_payload = (
            name in {"strongest_baseline", "strongest_baseline_plus_mspce"}
            or name.endswith("_rcmf")
            or is_current_stage_name(name)
            or name.endswith("_masd_current")
        )
        clean_metrics, clean_payload = evaluate_stage(model, primary_loader, seed_tensors, variant="clean", noise_seed=seed * 601 + idx * 10 + 1, return_payload=need_payload)
        noisy_metrics, noisy_payload = evaluate_stage(model, primary_loader, seed_tensors, variant="noisy", noise_seed=seed * 601 + idx * 10 + 2, return_payload=need_payload)
        external_metrics, external_payload = evaluate_stage(model, external_loader, seed_tensors, variant="clean", noise_seed=seed * 601 + idx * 10 + 3, return_payload=need_payload)
        stage_metrics = {
            "primary_clean": clean_metrics["mae_k"],
            "primary_noisy": noisy_metrics["mae_k"],
            "primary_hard_subgroup": clean_metrics["hard_subgroup_mae_k"],
            "external_holdout": external_metrics["mae_k"],
            "external_hard_subgroup": external_metrics["hard_subgroup_mae_k"],
        }
        if baseline_metrics is None:
            baseline_metrics = stage_metrics
        pass_flag = True if prev_stage is None else stage_pass(stage_metrics, prev_stage)
        rows.append(
            {
                "seed": seed,
                "model_name": name,
                **stage_metrics,
                "delta_vs_strongest_baseline_primary_clean": stage_metrics["primary_clean"] - baseline_metrics["primary_clean"],
                "delta_vs_strongest_baseline_primary_noisy": stage_metrics["primary_noisy"] - baseline_metrics["primary_noisy"],
                "delta_vs_strongest_baseline_primary_hard_subgroup": stage_metrics["primary_hard_subgroup"] - baseline_metrics["primary_hard_subgroup"],
                "delta_vs_strongest_baseline_external_holdout": stage_metrics["external_holdout"] - baseline_metrics["external_holdout"],
                "delta_vs_previous_primary_clean": 0.0 if prev_stage is None else stage_metrics["primary_clean"] - prev_stage["primary_clean"],
                "delta_vs_previous_primary_noisy": 0.0 if prev_stage is None else stage_metrics["primary_noisy"] - prev_stage["primary_noisy"],
                "delta_vs_previous_primary_hard_subgroup": 0.0 if prev_stage is None else stage_metrics["primary_hard_subgroup"] - prev_stage["primary_hard_subgroup"],
                "delta_vs_previous_external_holdout": 0.0 if prev_stage is None else stage_metrics["external_holdout"] - prev_stage["external_holdout"],
                "result_group": "mainline",
                "pass_flag": pass_flag,
            }
        )
        prev_stage = stage_metrics
        if is_current_stage_name(name) or name.endswith("_masd_current"):
            payload_bundle["masd_primary_clean"] = clean_payload
            payload_bundle["masd_primary_noisy"] = noisy_payload
            payload_bundle["masd_external"] = external_payload
            payload_bundle["masd_checkpoint_meta"] = getattr(model, "_masd_checkpoint_meta", {})
            payload_bundle["masd_checkpoint_candidates"] = getattr(model, "_masd_checkpoint_candidates", [])
        if name.endswith("_rcmf"):
            payload_bundle["rcmf_primary_clean"] = clean_payload
            payload_bundle["rcmf_primary_noisy"] = noisy_payload
            payload_bundle["rcmf_external"] = external_payload
        if name == "strongest_baseline":
            payload_bundle["baseline_primary_clean"] = clean_payload
            payload_bundle["baseline_primary_noisy"] = noisy_payload
            payload_bundle["baseline_external"] = external_payload
        if name == "strongest_baseline_plus_mspce":
            payload_bundle["mspce_primary_clean"] = clean_payload
            payload_bundle["mspce_primary_noisy"] = noisy_payload
            payload_bundle["mspce_external"] = external_payload
    return rows, payload_bundle


def run_mainline_seed_trisoup(
    *,
    seed: int,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    epoch_log: list[float],
    fixed_weights: tuple[float, float, float] | None = None,
    fixed_mode: str = "weight",
    scan_weight_grid: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    baseline_model, msce_model = train_msce_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed)
    minimal_rcmf = train_rcmf_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed, repair_model=msce_model)
    current_rcmf = train_rcmf_external_focus_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed, minimal_rcmf=minimal_rcmf)

    final_model = train_masd_current_student(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=seed,
        current_rcmf=current_rcmf,
        mode=CURRENT_MODE,
        selection_policy="tailfix",
        epoch_log=epoch_log,
    )
    jtt_model = train_masd_current_student(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=seed,
        current_rcmf=current_rcmf,
        mode=CURRENT_MODE,
        selection_policy="jtt_stabilization",
        epoch_log=epoch_log,
    )
    ctgf_model = train_masd_ctgf_student(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=seed,
        current_rcmf=current_rcmf,
        epoch_log=epoch_log,
        base_final_override=final_model,
    )

    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False, loader_seed=seed * 1401 + 1)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False, loader_seed=seed * 1401 + 3)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False, loader_seed=seed * 1401 + 5)

    anchor_clean, anchor_clean_payload = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 1, return_payload=True)
    anchor_noisy, _ = evaluate_stage(current_rcmf, val_loader, seed_tensors, variant="noisy", noise_seed=seed * 503 + 2, return_payload=True)
    anchor_external, _ = evaluate_stage(current_rcmf, external_loader, seed_tensors, variant="clean", noise_seed=seed * 503 + 3, return_payload=False)

    endpoint_order = ["final", "jtt", "ctgf"]
    endpoint_models = {
        "final": final_model,
        "jtt": jtt_model,
        "ctgf": ctgf_model,
    }
    endpoint_states = {
        name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
        for name, model in endpoint_models.items()
    }
    cluster_masks = external_cluster_masks(dataset)
    endpoint_val_payloads: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for idx, name in enumerate(endpoint_order):
        model = endpoint_models[name]
        clean_metrics, clean_payload = evaluate_stage(
            model,
            val_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 1401 + 11 + idx * 10,
            return_payload=True,
        )
        noisy_metrics, noisy_payload = evaluate_stage(
            model,
            val_loader,
            seed_tensors,
            variant="noisy",
            noise_seed=seed * 1401 + 12 + idx * 10,
            return_payload=True,
        )
        external_metrics, external_payload = evaluate_stage(
            model,
            external_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 1401 + 13 + idx * 10,
            return_payload=True,
        )
        endpoint_val_payloads[name] = {
            "clean": clean_payload,
            "noisy": noisy_payload,
            "external": external_payload,
            "clean_metrics": clean_metrics,
            "noisy_metrics": noisy_metrics,
            "external_metrics": external_metrics,
        }

    if scan_weight_grid:
        baseline_primary_metrics, _baseline_primary_payload = evaluate_stage(
            baseline_model,
            primary_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 1601 + 1,
            return_payload=True,
        )
        baseline_external_metrics, baseline_external_payload = evaluate_stage(
            baseline_model,
            external_loader,
            seed_tensors,
            variant="clean",
            noise_seed=seed * 1601 + 3,
            return_payload=True,
        )
        scan_rows: list[dict[str, Any]] = []
        for scan_idx, coeffs in enumerate(trisoup_candidate_coefficients()):
            try:
                stage_metrics, _clean_payload, _noisy_payload, external_payload = evaluate_weight_soup_candidate(
                    weights=coeffs,
                    endpoint_states=endpoint_states,
                    endpoint_order=endpoint_order,
                    seed_tensors=seed_tensors,
                    config=config,
                    primary_loader=primary_loader,
                    external_loader=external_loader,
                    seed=seed,
                    eval_offset=scan_idx,
                )
            except Exception:
                continue
            scan_rows.append(
                {
                    "seed": seed,
                    "weights": weight_key(coeffs),
                    "fixed_mode": "weight",
                    "primary_mae_k": float(stage_metrics["primary_clean"]),
                    "hard_subgroup_mae_k": float(stage_metrics["primary_hard_subgroup"]),
                    "external_mae_k": float(stage_metrics["external_holdout"]),
                    "primary_mae_reduction_k": float(baseline_primary_metrics["mae_k"] - stage_metrics["primary_clean"]),
                    "hard_subgroup_mae_reduction_k": float(baseline_primary_metrics["hard_subgroup_mae_k"] - stage_metrics["primary_hard_subgroup"]),
                    "external_mae_reduction_k": float(baseline_external_metrics["mae_k"] - stage_metrics["external_holdout"]),
                    "result_group": "weight_scan",
                    **external_cluster_reduction_row(
                        baseline_external_payload=baseline_external_payload,
                        candidate_external_payload=external_payload,
                        cluster_masks=cluster_masks,
                    ),
                }
            )
        return scan_rows, {"seed": seed, "scan_weight_grid": True}

    coeff_grid = trisoup_candidate_coefficients()
    template_model = build_model(CURRENT_MODE, seed_tensors, config)
    weight_candidates: list[dict[str, Any]] = []
    weight_selected: dict[str, Any] | None = None
    output_candidates: list[dict[str, Any]] = []
    output_selected: dict[str, Any] | None = None
    if fixed_weights is None:
        for coeffs in coeff_grid:
            try:
                mixed_state = interpolate_state_dicts([endpoint_states[name] for name in endpoint_order], list(coeffs))
                candidate = build_checkpoint_candidate(
                    student=template_model,
                    state=mixed_state,
                    stage="TRISOUP_WEIGHT",
                    epoch=0,
                    val_loader=val_loader,
                    external_loader=external_loader,
                    seed_tensors=seed_tensors,
                    seed=seed,
                    eval_offset=700 + int(round(sum((idx + 1) * weight * 100 for idx, weight in enumerate(coeffs)))),
                    anchor_clean=anchor_clean,
                    anchor_noisy=anchor_noisy,
                    anchor_external=anchor_external,
                    anchor_clean_payload=anchor_clean_payload,
                )
                candidate["coefficients"] = tuple(float(item) for item in coeffs)
                candidate["search_mode"] = "weight"
                candidate["aggregation"] = "weight_interp" if sum(weight > 0.0 for weight in coeffs) >= 2 else endpoint_order[int(np.argmax(coeffs))]
                weight_candidates.append(candidate)
            except Exception:
                continue

        weight_selected = select_trisoup_checkpoint(weight_candidates) if weight_candidates else None
        weight_interpolation_effective = bool(
            weight_selected is not None
            and sum(weight > 0.0 for weight in weight_selected["coefficients"]) >= 2
            and float(weight_selected.get("chem_cluster_worst_delta", 0.0)) <= 0.05
            and float(weight_selected.get("chem_cluster_positive_count", 0.0)) <= 1.0
            and float(weight_selected["weak_cluster_worst_delta"]) <= 0.05
            and float(weight_selected["external_proxy_delta"]) <= 0.0
            and float(weight_selected["full_proxy_delta"]) <= 0.02
        )
        need_output_candidates = bool(weight_selected is None or not weight_interpolation_effective)
        output_interpolation_used = False

        if need_output_candidates:
            for coeffs in coeff_grid:
                clean_payload = blend_payloads([endpoint_val_payloads[name]["clean"] for name in endpoint_order], list(coeffs))
                noisy_payload = blend_payloads([endpoint_val_payloads[name]["noisy"] for name in endpoint_order], list(coeffs))
                external_payload = blend_payloads([endpoint_val_payloads[name]["external"] for name in endpoint_order], list(coeffs))
                candidate = build_payload_candidate(
                    stage="TRISOUP_OUTPUT",
                    weights=coeffs,
                    clean_payload=clean_payload,
                    noisy_payload=noisy_payload,
                    external_payload=external_payload,
                    anchor_clean=anchor_clean,
                    anchor_noisy=anchor_noisy,
                    anchor_external=anchor_external,
                    anchor_clean_payload=anchor_clean_payload,
                )
                candidate["search_mode"] = "output"
                candidate["aggregation"] = "output_interp" if sum(weight > 0.0 for weight in coeffs) >= 2 else endpoint_order[int(np.argmax(coeffs))]
                output_candidates.append(candidate)
            output_selected = select_trisoup_checkpoint(output_candidates)

        if weight_selected is None and output_selected is None:
            raise RuntimeError("trisoup selection produced no valid weight or output candidate")
        if output_selected is None:
            chosen_mode = "weight"
        elif weight_selected is None:
            chosen_mode = "output"
        else:
            chosen_mode = "weight" if trisoup_priority_tuple(weight_selected) <= trisoup_priority_tuple(output_selected) else "output"
        chosen_weights = tuple(float(item) for item in (weight_selected["coefficients"] if chosen_mode == "weight" else output_selected["weights"]))
        weight_interpolation_effective = bool(
            chosen_mode == "weight"
            and weight_selected is not None
            and sum(weight > 0.0 for weight in weight_selected["coefficients"]) >= 2
            and float(weight_selected.get("chem_cluster_worst_delta", 0.0)) <= 0.05
            and float(weight_selected.get("chem_cluster_positive_count", 0.0)) <= 1.0
            and float(weight_selected["weak_cluster_worst_delta"]) <= 0.05
            and float(weight_selected["external_proxy_delta"]) <= 0.0
            and float(weight_selected["full_proxy_delta"]) <= 0.02
        )
        output_interpolation_used = bool(chosen_mode == "output" and sum(weight > 0.0 for weight in chosen_weights) >= 2)
    else:
        chosen_mode = str(fixed_mode)
        chosen_weights = tuple(float(item) for item in fixed_weights)
        weight_interpolation_effective = bool(chosen_mode == "weight")
        output_interpolation_used = bool(chosen_mode == "output")

    stages = [
        ("strongest_baseline", baseline_model),
        ("strongest_baseline_plus_mspce", msce_model),
        ("strongest_baseline_plus_mspce_rcmf", current_rcmf),
    ]
    rows: list[dict[str, Any]] = []
    payload_bundle: dict[str, Any] = {"seed": seed}
    baseline_metrics: dict[str, float] | None = None
    prev_stage: dict[str, float] | None = None
    for idx, (name, model) in enumerate(stages):
        need_payload = name in {"strongest_baseline", "strongest_baseline_plus_mspce_rcmf"}
        clean_metrics, clean_payload = evaluate_stage(model, primary_loader, seed_tensors, variant="clean", noise_seed=seed * 601 + idx * 10 + 1, return_payload=need_payload)
        noisy_metrics, noisy_payload = evaluate_stage(model, primary_loader, seed_tensors, variant="noisy", noise_seed=seed * 601 + idx * 10 + 2, return_payload=need_payload)
        external_metrics, external_payload = evaluate_stage(model, external_loader, seed_tensors, variant="clean", noise_seed=seed * 601 + idx * 10 + 3, return_payload=need_payload)
        stage_metrics = {
            "primary_clean": clean_metrics["mae_k"],
            "primary_noisy": noisy_metrics["mae_k"],
            "primary_hard_subgroup": clean_metrics["hard_subgroup_mae_k"],
            "external_holdout": external_metrics["mae_k"],
            "external_hard_subgroup": external_metrics["hard_subgroup_mae_k"],
        }
        if baseline_metrics is None:
            baseline_metrics = stage_metrics
        pass_flag = True if prev_stage is None else stage_pass(stage_metrics, prev_stage)
        rows.append(
            {
                "seed": seed,
                "model_name": name,
                **stage_metrics,
                "delta_vs_strongest_baseline_primary_clean": stage_metrics["primary_clean"] - baseline_metrics["primary_clean"],
                "delta_vs_strongest_baseline_primary_noisy": stage_metrics["primary_noisy"] - baseline_metrics["primary_noisy"],
                "delta_vs_strongest_baseline_primary_hard_subgroup": stage_metrics["primary_hard_subgroup"] - baseline_metrics["primary_hard_subgroup"],
                "delta_vs_strongest_baseline_external_holdout": stage_metrics["external_holdout"] - baseline_metrics["external_holdout"],
                "delta_vs_previous_primary_clean": 0.0 if prev_stage is None else stage_metrics["primary_clean"] - prev_stage["primary_clean"],
                "delta_vs_previous_primary_noisy": 0.0 if prev_stage is None else stage_metrics["primary_noisy"] - prev_stage["primary_noisy"],
                "delta_vs_previous_primary_hard_subgroup": 0.0 if prev_stage is None else stage_metrics["primary_hard_subgroup"] - prev_stage["primary_hard_subgroup"],
                "delta_vs_previous_external_holdout": 0.0 if prev_stage is None else stage_metrics["external_holdout"] - prev_stage["external_holdout"],
                "result_group": "mainline",
                "pass_flag": pass_flag,
            }
        )
        prev_stage = stage_metrics
        if name == "strongest_baseline":
            payload_bundle["baseline_primary_clean"] = clean_payload
            payload_bundle["baseline_primary_noisy"] = noisy_payload
            payload_bundle["baseline_external"] = external_payload
        if name.endswith("_rcmf"):
            payload_bundle["rcmf_primary_clean"] = clean_payload
            payload_bundle["rcmf_primary_noisy"] = noisy_payload
            payload_bundle["rcmf_external"] = external_payload

    if chosen_mode == "weight":
        if fixed_weights is None and weight_selected is not None:
            chosen_model = build_model(CURRENT_MODE, seed_tensors, config)
            chosen_model.load_state_dict(weight_selected["state"])
            chosen_model.to(DEVICE)
            clean_metrics, clean_payload = evaluate_stage(chosen_model, primary_loader, seed_tensors, variant="clean", noise_seed=seed * 601 + 91, return_payload=True)
            noisy_metrics, noisy_payload = evaluate_stage(chosen_model, primary_loader, seed_tensors, variant="noisy", noise_seed=seed * 601 + 92, return_payload=True)
            external_metrics, external_payload = evaluate_stage(chosen_model, external_loader, seed_tensors, variant="clean", noise_seed=seed * 601 + 93, return_payload=True)
        else:
            stage_metrics, clean_payload, noisy_payload, external_payload = evaluate_weight_soup_candidate(
                weights=chosen_weights,
                endpoint_states=endpoint_states,
                endpoint_order=endpoint_order,
                seed_tensors=seed_tensors,
                config=config,
                primary_loader=primary_loader,
                external_loader=external_loader,
                seed=seed,
                eval_offset=91,
            )
            clean_metrics = {
                "mae_k": float(stage_metrics["primary_clean"]),
                "hard_subgroup_mae_k": float(stage_metrics["primary_hard_subgroup"]),
            }
            noisy_metrics = payload_metrics(noisy_payload)
            external_metrics = {
                "mae_k": float(stage_metrics["external_holdout"]),
                "hard_subgroup_mae_k": float(stage_metrics["external_hard_subgroup"]),
            }
    else:
        endpoint_test_payloads: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for idx, name in enumerate(endpoint_order):
            model = endpoint_models[name]
            clean_metrics_raw, clean_payload_raw = evaluate_stage(
                model,
                primary_loader,
                seed_tensors,
                variant="clean",
                noise_seed=seed * 1501 + 11 + idx * 10,
                return_payload=True,
            )
            noisy_metrics_raw, noisy_payload_raw = evaluate_stage(
                model,
                primary_loader,
                seed_tensors,
                variant="noisy",
                noise_seed=seed * 1501 + 12 + idx * 10,
                return_payload=True,
            )
            external_metrics_raw, external_payload_raw = evaluate_stage(
                model,
                external_loader,
                seed_tensors,
                variant="clean",
                noise_seed=seed * 1501 + 13 + idx * 10,
                return_payload=True,
            )
            endpoint_test_payloads[name] = {
                "clean": clean_payload_raw,
                "noisy": noisy_payload_raw,
                "external": external_payload_raw,
                "clean_metrics": clean_metrics_raw,
                "noisy_metrics": noisy_metrics_raw,
                "external_metrics": external_metrics_raw,
            }
        clean_payload = blend_payloads([endpoint_test_payloads[name]["clean"] for name in endpoint_order], list(chosen_weights))
        noisy_payload = blend_payloads([endpoint_test_payloads[name]["noisy"] for name in endpoint_order], list(chosen_weights))
        external_payload = blend_payloads([endpoint_test_payloads[name]["external"] for name in endpoint_order], list(chosen_weights))
        clean_metrics = payload_metrics(clean_payload)
        noisy_metrics = payload_metrics(noisy_payload)
        external_metrics = payload_metrics(external_payload)

    stage_metrics = {
        "primary_clean": clean_metrics["mae_k"],
        "primary_noisy": noisy_metrics["mae_k"],
        "primary_hard_subgroup": clean_metrics["hard_subgroup_mae_k"],
        "external_holdout": external_metrics["mae_k"],
        "external_hard_subgroup": external_metrics["hard_subgroup_mae_k"],
    }
    pass_flag = stage_pass(stage_metrics, prev_stage)
    rows.append(
        {
            "seed": seed,
            "model_name": CURRENT_STAGE_NAME,
            **stage_metrics,
            "delta_vs_strongest_baseline_primary_clean": stage_metrics["primary_clean"] - baseline_metrics["primary_clean"],
            "delta_vs_strongest_baseline_primary_noisy": stage_metrics["primary_noisy"] - baseline_metrics["primary_noisy"],
            "delta_vs_strongest_baseline_primary_hard_subgroup": stage_metrics["primary_hard_subgroup"] - baseline_metrics["primary_hard_subgroup"],
            "delta_vs_strongest_baseline_external_holdout": stage_metrics["external_holdout"] - baseline_metrics["external_holdout"],
            "delta_vs_previous_primary_clean": stage_metrics["primary_clean"] - prev_stage["primary_clean"],
            "delta_vs_previous_primary_noisy": stage_metrics["primary_noisy"] - prev_stage["primary_noisy"],
            "delta_vs_previous_primary_hard_subgroup": stage_metrics["primary_hard_subgroup"] - prev_stage["primary_hard_subgroup"],
            "delta_vs_previous_external_holdout": stage_metrics["external_holdout"] - prev_stage["external_holdout"],
            "result_group": "mainline",
            "pass_flag": pass_flag,
        }
    )
    payload_bundle["masd_primary_clean"] = clean_payload
    payload_bundle["masd_primary_noisy"] = noisy_payload
    payload_bundle["masd_external"] = external_payload
    payload_bundle["masd_checkpoint_meta"] = {
        "selection_policy": "trisoup",
        "selected_mode": chosen_mode,
        "selected_weights": chosen_weights,
        "fixed_weights": tuple(float(item) for item in fixed_weights) if fixed_weights is not None else (),
        "fixed_mode": str(fixed_mode) if fixed_weights is not None else "",
        "weight_interpolation_effective": bool(weight_interpolation_effective),
        "output_interpolation_used": bool(output_interpolation_used),
        "weight_candidate_count": int(len(weight_candidates)),
        "output_candidate_count": int(len(output_candidates)),
        "selection_uses_external_holdout": bool(USE_EXTERNAL_HOLDOUT_FOR_SELECTION),
        "final_meta": getattr(final_model, "_masd_checkpoint_meta", {}),
        "jtt_meta": getattr(jtt_model, "_masd_checkpoint_meta", {}),
        "ctgf_meta": getattr(ctgf_model, "_masd_checkpoint_meta", {}),
    }
    candidate_log: list[dict[str, Any]] = []
    for item in weight_candidates:
        candidate_log.append({key: value for key, value in item.items() if key not in {"state"}})
    for item in output_candidates:
        candidate_log.append({key: value for key, value in item.items() if key not in {"clean_payload", "noisy_payload", "external_payload"}})
    payload_bundle["masd_checkpoint_candidates"] = candidate_log
    return rows, payload_bundle


def run_ablation_seed(
    *,
    seed: int,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    selection_policy: str,
    epoch_log: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = ensure_protocol_split(splits, dataset, seed=seed)
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    _baseline_model, msce_model = train_msce_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed)
    minimal_rcmf = train_rcmf_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed, repair_model=msce_model)
    current_rcmf = train_rcmf_external_focus_stage(split=split, seed_tensors=seed_tensors, config=config, repeat_id=seed, minimal_rcmf=minimal_rcmf)
    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)

    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {"seed": seed}
    current_clean, _ = evaluate_stage(current_rcmf, primary_loader, seed_tensors, variant="clean", noise_seed=seed * 709 + 1, return_payload=False)
    current_noisy, _ = evaluate_stage(current_rcmf, primary_loader, seed_tensors, variant="noisy", noise_seed=seed * 709 + 2, return_payload=False)
    current_external, _ = evaluate_stage(current_rcmf, external_loader, seed_tensors, variant="clean", noise_seed=seed * 709 + 3, return_payload=False)
    rows.append(
        {
            "seed": seed,
            "result_group": "ablation",
            "teacher_locked": True,
            "model_name": "strongest_baseline_plus_mspce_rcmf",
            "primary_clean": current_clean["mae_k"],
            "primary_noisy": current_noisy["mae_k"],
            "primary_hard_subgroup": current_clean["hard_subgroup_mae_k"],
            "external_holdout": current_external["mae_k"],
            "external_hard_subgroup": current_external["hard_subgroup_mae_k"],
            "pass_flag": True,
        }
    )
    rows.append(
        {
            "seed": seed,
            "result_group": "ablation",
            "model_name": "no_masd",
            "primary_clean": current_clean["mae_k"],
            "primary_noisy": current_noisy["mae_k"],
            "primary_hard_subgroup": current_clean["hard_subgroup_mae_k"],
            "external_holdout": current_external["mae_k"],
            "external_hard_subgroup": current_external["hard_subgroup_mae_k"],
            "pass_flag": True,
        }
    )
    mode_map = {"full_current": CURRENT_MODE}
    for row_name, mode in mode_map.items():
        model = train_masd_current_student(
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=seed,
            current_rcmf=current_rcmf,
            mode=mode,
            selection_policy=selection_policy,
            epoch_log=epoch_log,
        )
        clean_metrics, clean_payload = evaluate_stage(model, primary_loader, seed_tensors, variant="clean", noise_seed=seed * 709 + 10 + len(rows), return_payload=row_name == "full_current")
        noisy_metrics, _ = evaluate_stage(model, primary_loader, seed_tensors, variant="noisy", noise_seed=seed * 709 + 20 + len(rows), return_payload=False)
        external_metrics, _ = evaluate_stage(model, external_loader, seed_tensors, variant="clean", noise_seed=seed * 709 + 30 + len(rows), return_payload=False)
        rows.append(
            {
                "seed": seed,
                "result_group": "ablation",
                "model_name": row_name,
                "primary_clean": clean_metrics["mae_k"],
                "primary_noisy": noisy_metrics["mae_k"],
                "primary_hard_subgroup": clean_metrics["hard_subgroup_mae_k"],
                "external_holdout": external_metrics["mae_k"],
                "external_hard_subgroup": external_metrics["hard_subgroup_mae_k"],
                "pass_flag": True,
            }
        )
        if row_name == "full_current":
            payloads["full_current_primary_clean"] = clean_payload
    return rows, payloads


def smoke_passes(smoke_rows: list[dict[str, Any]], smoke_bundle: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    final_row = current_stage_row(smoke_rows)
    mech = contribution_metrics_from_payload(smoke_bundle["masd_primary_clean"], smoke_bundle["masd_primary_noisy"])
    anchor_mask_delta = common_mask_delta(smoke_bundle["masd_primary_clean"], smoke_bundle["rcmf_primary_clean"])
    payload = {
        "smoke_sign_consistency": mech["contribution_sign_consistency"],
        "smoke_contribution_corr": mech["contribution_anchor_alignment_corr"],
        "smoke_sparsity": mech["mechanism_weight_sparsity"],
        "smoke_concentration": mech["dominant_mechanism_concentration"],
        "smoke_hard_delta": float(final_row["delta_vs_previous_primary_hard_subgroup"]),
        "smoke_primary_clean_delta": float(final_row["delta_vs_previous_primary_clean"]),
        "smoke_primary_noisy_delta": float(final_row["delta_vs_previous_primary_noisy"]),
        "smoke_anchor_mask_delta": anchor_mask_delta,
    }
    passed = bool(
        mech["mechanism_pass"]
        and payload["smoke_sign_consistency"] >= 0.90
        and payload["smoke_hard_delta"] <= 0.12
        and payload["smoke_primary_clean_delta"] <= PRIMARY_CLEAN_PASS_DELTA + 0.02
        and payload["smoke_primary_noisy_delta"] <= PRIMARY_NOISY_PASS_DELTA + 0.02
        and payload["smoke_anchor_mask_delta"] <= 0.05
    )
    return passed, payload


def strict_deterministic_smoke(
    *,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    selection_policy: str,
) -> tuple[dict[str, Any], list[float]]:
    epoch_log: list[float] = []
    smoke_payload = {
        "strict_determinism_smoke": False,
        "deterministic_op_failure": "",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }
    try:
        enable_determinism(strict=True)
        rows, bundle = run_mainline_seed(
            seed=SMOKE_SEED,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            selection_policy=selection_policy,
            epoch_log=epoch_log,
        )
        smoke_ok, smoke_metrics = smoke_passes(rows, bundle)
        smoke_payload.update(
            {
                "strict_determinism_smoke": True,
                "strict_smoke_pass": smoke_ok,
                "strict_smoke_metrics": smoke_metrics,
            }
        )
    except RuntimeError as exc:
        smoke_payload["deterministic_op_failure"] = str(exc)
    finally:
        enable_determinism(strict=False)
    return smoke_payload, epoch_log


def run_replay_seed(
    *,
    seed: int,
    replay_id: int,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    selection_policy: str,
    epoch_log: list[float],
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    rows, _bundle = run_mainline_seed(
        seed=seed,
        dataset=dataset,
        features=features,
        splits=splits,
        config=config,
        selection_policy=selection_policy,
        epoch_log=epoch_log,
    )
    row = current_stage_row(rows).copy()
    row["replay_id"] = replay_id
    row["result_group"] = "replay"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked MSCE-RCMF-MASD mainline / confirmation workflows.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="masd_final")
    parser.add_argument("--mainline-seeds", type=str, default="10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--external-supporting-seeds", type=str, default="15,16,17,18,19")
    parser.add_argument("--ablation-seeds", type=str, default="15,16,17,18,19")
    parser.add_argument("--replay-seeds", type=str, default="")
    parser.add_argument("--replay-count", type=int, default=0)
    parser.add_argument("--strict-deterministic-smoke", action="store_true")
    parser.add_argument("--trisoup-fixed-weights", type=str, default="")
    parser.add_argument("--trisoup-fixed-mode", type=str, default="weight")
    parser.add_argument("--disable-external-holdout-selection", action="store_true")
    parser.add_argument("--enable-external-holdout-selection", action="store_true")
    args = parser.parse_args()

    global USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION
    USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION = False
    if args.enable_external_holdout_selection:
        USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION = True
    elif args.disable_external_holdout_selection:
        USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION = False

    enable_determinism(strict=False)
    ensure_msce_features()
    gpu_payload = ensure_gpu()
    dataset, features, splits = load_artifacts()
    global CHEMISTRY_TAG_LOOKUP
    CHEMISTRY_TAG_LOOKUP = build_chemistry_tag_lookup(dataset)
    config = diagnostic_config()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix
    if output_prefix == "masd_final_jtt_stabilization":
        selection_policy = "jtt_stabilization"
    elif output_prefix in {"masd_final_trisoup", TRISOUP_WEIGHTLOCK_SCAN_PREFIX} or is_trisoup_100run_prefix(output_prefix) or is_weightlock_100run_prefix(output_prefix):
        selection_policy = "trisoup"
    elif output_prefix == "masd_final_ctgf":
        selection_policy = "ctgf"
    elif output_prefix == "masd_final_self_stabilization":
        selection_policy = "self_stabilization"
    elif output_prefix == "masd_final_splithead_stabilization":
        selection_policy = "splithead_stabilization"
    elif output_prefix == "masd_final_stabilization":
        selection_policy = "stabilization"
    elif output_prefix == "masd_final_signrate_lock":
        selection_policy = "signrate_lock"
    else:
        selection_policy = "tailfix"
    mainline_seeds = parse_seed_list(args.mainline_seeds)
    external_supporting_seeds = parse_seed_list(args.external_supporting_seeds)
    ablation_seeds = parse_seed_list(args.ablation_seeds)
    trisoup_fixed_weights: tuple[float, float, float] | None = None
    trisoup_fixed_mode = str(args.trisoup_fixed_mode or "weight")
    if args.trisoup_fixed_weights:
        trisoup_fixed_weights = parse_weight_list(args.trisoup_fixed_weights)
    elif is_weightlock_100run_prefix(output_prefix):
        locked_choice = load_locked_weight_choice()
        if locked_choice is None:
            raise RuntimeError(
                f"{TRISOUP_WEIGHTLOCK_100RUN_PREFIX} requires either --trisoup-fixed-weights or "
                f"{DIAG_ROOT / TRISOUP_WEIGHTLOCK_SCAN_PREFIX / 'best_candidate.json'}"
            )
        trisoup_fixed_mode, trisoup_fixed_weights = locked_choice
    if is_trisoup_100run_prefix(output_prefix):
        if args.mainline_seeds == parser.get_default("mainline_seeds"):
            mainline_seeds = list(range(TRISOUP_100RUN_NUM_RUNS))
        if args.external_supporting_seeds == parser.get_default("external_supporting_seeds"):
            external_supporting_seeds = list(mainline_seeds)
        if args.ablation_seeds == parser.get_default("ablation_seeds"):
            ablation_seeds = []
    if output_prefix == TRISOUP_WEIGHTLOCK_SCAN_PREFIX:
        if args.mainline_seeds == parser.get_default("mainline_seeds"):
            mainline_seeds = list(range(TRISOUP_WEIGHTLOCK_SCAN_NUM_RUNS))
        if args.external_supporting_seeds == parser.get_default("external_supporting_seeds"):
            external_supporting_seeds = list(mainline_seeds)
        if args.ablation_seeds == parser.get_default("ablation_seeds"):
            ablation_seeds = []
    if is_weightlock_100run_prefix(output_prefix):
        if args.mainline_seeds == parser.get_default("mainline_seeds"):
            mainline_seeds = list(range(TRISOUP_WEIGHTLOCK_CONFIRM_START_SEED, TRISOUP_WEIGHTLOCK_CONFIRM_START_SEED + TRISOUP_100RUN_NUM_RUNS))
        if args.external_supporting_seeds == parser.get_default("external_supporting_seeds"):
            external_supporting_seeds = list(mainline_seeds)
        if args.ablation_seeds == parser.get_default("ablation_seeds"):
            ablation_seeds = []
    replay_seeds = parse_seed_list(args.replay_seeds) if args.replay_seeds else []
    locked_snapshot = lock_snapshot()

    if args.strict_deterministic_smoke or replay_seeds:
        strict_smoke_payload, smoke_epoch_log = strict_deterministic_smoke(
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            selection_policy=selection_policy,
        )
        audit_payload: dict[str, Any] = {
            "gpu_payload": gpu_payload,
            "strict_smoke_payload": strict_smoke_payload,
            "replay_seeds": replay_seeds,
            "replay_count": int(args.replay_count),
            "locked_snapshot": locked_snapshot,
            "output_prefix": output_prefix,
            "epoch_time_mean_sec": float(np.mean(smoke_epoch_log)) if smoke_epoch_log else float("nan"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        }
        if replay_seeds:
            epoch_log: list[float] = []
            replay_rows: list[dict[str, Any]] = []
            for seed in replay_seeds:
                for replay_id in range(int(args.replay_count)):
                    replay_rows.append(
                        run_replay_seed(
                            seed=seed,
                            replay_id=replay_id,
                            dataset=dataset,
                            features=features,
                            splits=splits,
                            config=config,
                            selection_policy=selection_policy,
                            epoch_log=epoch_log,
                        )
                    )
            audit_payload["replay_rows"] = replay_rows
            audit_payload["epoch_time_mean_sec"] = float(np.mean(epoch_log)) if epoch_log else audit_payload["epoch_time_mean_sec"]
        save_bundle(run_dir, "final_audit_bundle", audit_payload)
        return 0

    if output_prefix == SIMPLECONCAT_TERNARY_CONTROL_PREFIX:
        control_rows: list[dict[str, Any]] = []
        control_bundles: list[dict[str, Any]] = []
        for seed in mainline_seeds:
            ensure_protocol_split(splits, dataset, seed=seed)
            rows, bundle = run_simpleconcat_ternary_seed(
                seed=seed,
                dataset=dataset,
                features=features,
                splits=splits,
                config=config,
            )
            control_rows.extend(rows)
            control_bundles.append(bundle)
            save_bundle(
                run_dir,
                "mainline_bundle",
                {
                    "gpu_payload": gpu_payload,
                    "rows": control_rows,
                    "seed_bundles": control_bundles,
                    "mainline_seeds": mainline_seeds,
                    "completed_mainline_seeds": [int(seed_bundle["seed"]) for seed_bundle in control_bundles],
                    "locked_snapshot": locked_snapshot,
                    "output_prefix": output_prefix,
                    "selection_uses_external_holdout": False,
                    "ensemble_member_count": SIMPLECONCAT_TERNARY_MEMBER_COUNT,
                },
            )
            save_results_csv(run_dir, output_prefix, control_rows)
        save_bundle(
            run_dir,
            "mainline_bundle",
            {
                "gpu_payload": gpu_payload,
                "rows": control_rows,
                "seed_bundles": control_bundles,
                "mainline_seeds": mainline_seeds,
                "completed_mainline_seeds": [int(seed_bundle["seed"]) for seed_bundle in control_bundles],
                "locked_snapshot": locked_snapshot,
                "output_prefix": output_prefix,
                "selection_uses_external_holdout": False,
                "ensemble_member_count": SIMPLECONCAT_TERNARY_MEMBER_COUNT,
            },
        )
        save_results_csv(run_dir, output_prefix, control_rows)
        return 0

    epoch_log: list[float] = []
    skip_smoke = output_prefix in {
        "masd_final_splithead_stabilization",
        "masd_final_self_stabilization",
        "masd_final_jtt_stabilization",
        "masd_final_ctgf",
        "masd_final_trisoup",
        TRISOUP_WEIGHTLOCK_SCAN_PREFIX,
        TRISOUP_WEIGHTLOCK_100RUN_PREFIX,
    }
    skip_smoke = skip_smoke or is_trisoup_100run_prefix(output_prefix) or is_weightlock_100run_prefix(output_prefix)
    if not skip_smoke:
        smoke_rows, smoke_seed_bundle = run_mainline_seed(
            seed=SMOKE_SEED,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            selection_policy=selection_policy,
            epoch_log=epoch_log,
        )
        smoke_ok, smoke_metrics = smoke_passes(smoke_rows, smoke_seed_bundle)
        save_bundle(
            run_dir,
            "smoke_bundle",
            {
                "gpu_payload": gpu_payload,
                "rows": smoke_rows,
                "seed_bundle": smoke_seed_bundle,
                "smoke_pass": smoke_ok,
                "smoke_metrics": smoke_metrics,
                "smoke_seed": SMOKE_SEED,
                "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
                "locked_snapshot": locked_snapshot,
                "output_prefix": output_prefix,
                "selection_uses_external_holdout": bool(USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION),
                "mainline_seeds": mainline_seeds,
                "external_supporting_seeds": external_supporting_seeds,
                "ablation_seeds": ablation_seeds,
            },
        )
        if not smoke_ok:
            return 2

    all_rows: list[dict[str, Any]] = []
    mainline_bundles: list[dict[str, Any]] = []
    for seed in mainline_seeds:
        ensure_protocol_split(splits, dataset, seed=seed)
        if selection_policy == "trisoup":
            rows, bundle = run_mainline_seed_trisoup(
                seed=seed,
                dataset=dataset,
                features=features,
                splits=splits,
                config=config,
                epoch_log=epoch_log,
                fixed_weights=trisoup_fixed_weights,
                fixed_mode=trisoup_fixed_mode,
                scan_weight_grid=(output_prefix == TRISOUP_WEIGHTLOCK_SCAN_PREFIX),
            )
        else:
            rows, bundle = run_mainline_seed(
                seed=seed,
                dataset=dataset,
                features=features,
                splits=splits,
                config=config,
                selection_policy=selection_policy,
                epoch_log=epoch_log,
            )
        all_rows.extend(rows)
        mainline_bundles.append(bundle)
        save_bundle(
            run_dir,
            "mainline_bundle",
            {
                "gpu_payload": gpu_payload,
                "rows": all_rows,
                "seed_bundles": mainline_bundles,
                "mainline_seeds": mainline_seeds,
                "external_supporting_seeds": external_supporting_seeds,
                "completed_mainline_seeds": [int(seed_bundle["seed"]) for seed_bundle in mainline_bundles],
                "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
                "locked_snapshot": locked_snapshot,
                "output_prefix": output_prefix,
                "selection_uses_external_holdout": bool(USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION),
                "trisoup_fixed_weights": list(trisoup_fixed_weights) if trisoup_fixed_weights is not None else [],
                "trisoup_fixed_mode": trisoup_fixed_mode if trisoup_fixed_weights is not None else "",
            },
        )
        save_results_csv(run_dir, output_prefix, all_rows)
    save_bundle(
        run_dir,
        "mainline_bundle",
        {
            "gpu_payload": gpu_payload,
            "rows": all_rows,
            "seed_bundles": mainline_bundles,
            "mainline_seeds": mainline_seeds,
            "external_supporting_seeds": external_supporting_seeds,
            "completed_mainline_seeds": [int(seed_bundle["seed"]) for seed_bundle in mainline_bundles],
            "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
            "locked_snapshot": locked_snapshot,
            "output_prefix": output_prefix,
            "selection_uses_external_holdout": bool(USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION),
            "trisoup_fixed_weights": list(trisoup_fixed_weights) if trisoup_fixed_weights is not None else [],
            "trisoup_fixed_mode": trisoup_fixed_mode if trisoup_fixed_weights is not None else "",
        },
    )

    ablation_rows: list[dict[str, Any]] = []
    ablation_bundles: list[dict[str, Any]] = []
    for seed in ablation_seeds:
        ensure_protocol_split(splits, dataset, seed=seed)
        rows, bundle = run_ablation_seed(
            seed=seed,
            dataset=dataset,
            features=features,
            splits=splits,
            config=config,
            selection_policy=selection_policy,
            epoch_log=epoch_log,
        )
        ablation_rows.extend(rows)
        ablation_bundles.append(bundle)
        save_results_csv(run_dir, output_prefix, all_rows + ablation_rows)
        save_bundle(
            run_dir,
            "ablation_bundle",
            {
                "gpu_payload": gpu_payload,
                "rows": ablation_rows,
                "seed_bundles": ablation_bundles,
                "ablation_seeds": ablation_seeds,
                "completed_ablation_seeds": [int(seed_bundle["seed"]) for seed_bundle in ablation_bundles],
                "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
                "locked_snapshot": locked_snapshot,
                "output_prefix": output_prefix,
                "selection_uses_external_holdout": bool(USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION),
            },
        )
    results_df = pd.DataFrame(all_rows + ablation_rows)
    save_results_csv(run_dir, output_prefix, (all_rows + ablation_rows) if (all_rows or ablation_rows) else [])
    save_bundle(
        run_dir,
        "ablation_bundle",
        {
            "gpu_payload": gpu_payload,
            "rows": ablation_rows,
            "seed_bundles": ablation_bundles,
            "ablation_seeds": ablation_seeds,
            "completed_ablation_seeds": [int(seed_bundle["seed"]) for seed_bundle in ablation_bundles],
            "epoch_time_mean_sec": float(np.mean(epoch_log)) if epoch_log else float("nan"),
            "locked_snapshot": locked_snapshot,
            "output_prefix": output_prefix,
            "selection_uses_external_holdout": bool(USE_EXTERNAL_HOLDOUT_IN_MASD_CHECKPOINT_SELECTION),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
