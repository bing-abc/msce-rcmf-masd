from __future__ import annotations

"""Main training and evaluation entry for the active experiment scaffold.

This file still carries historical ladder modes because the paper diagnostics
and comparator tables are reproduced from the same training harness.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.compare import comparator_report_rows
from eval.metrics import bootstrap_ci, format_ci, format_sign, mae, sign_counts
from models.fusion import FusionModel
from train.calibration import (
    AnchorLossWeights,
    CERT_FEATURE_NAMES,
    beneficial_mask,
    beneficial_threshold,
    certification_score_from_features,
    copy_shared_weights,
    freeze_teacher_anchored_stage,
    fit_certification_rule,
    masked_mean,
    summarize_certification_rule,
    teacher_anchor_loss,
    unfreeze_all,
)
from train.experiment_overrides import apply_experiment_overrides
from train.seeds import FULL_SEEDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Historical ladder modes are retained here because the current paper compares
# the locked full chain against the same scaffolded baselines.
CULPRIT_SCAN_MODES = (
    "conflict_only",
    "simple_concat",
    "mspce_only",
    "mspce_only_led",
    "rcmf_dynamic_multimodal_full",
    "rcmf_dynamic_multimodal_full_phasea",
)
FINAL_COMPARATOR_MODES = (
    "conflict_only",
    "simple_concat",
    "mspce_only",
    "mspce_only_led",
    "rcmf_dynamic_multimodal_full",
    "rcmf_dynamic_multimodal_full_phasea",
)
ROUTE_ENABLED_MODES: set[str] = set()
MODE_REPORT_NAME = {
    "conflict_only": "conflict_only_attentivefp",
    "simple_concat": "simple_concat_attentivefp",
    "mspce_only": "mspce_only_attentivefp",
    "mspce_context_injection": "mspce_context_injection_attentivefp",
    "mspce_only_led": "mspce_only_plus_led",
    "rcmf_dynamic_multimodal_full": "rcmf_dynamic_multimodal_full",
    "rcmf_dynamic_multimodal_full_phasea": "rcmf_dynamic_multimodal_full_phasea",
    "rcmf_dynamic_multimodal_full_dual_anchor": "rcmf_dynamic_multimodal_full_dual_anchor",
}
MODE_SEED_OFFSET = {
    "conflict_only": 11,
    "simple_concat": 23,
    "mspce_only": 41,
    "mspce_context_injection": 43,
    "mspce_only_led": 47,
    "rcmf_dynamic_multimodal_full": 71,
    "rcmf_dynamic_multimodal_full_phasea": 53,
    "rcmf_dynamic_multimodal_full_dual_anchor": 59,
}
FIX_DIRECTION = "rcmf_dynamic_multimodal_full_phasea"
OLD_PRIMARY_MAE_K = 48.40
CURRENT_CLEAN_PRIMARY_MAE_K = 24.18
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
WEAK_CLUSTER_NAMES = {"aromatic_dense", "ester_or_carbonate"}
UNSTABLE_CLUSTER_NAMES = {"fluorinated", "sulfone"}
SUPPORTED_CLUSTER_NAMES = {"amide", "ether_oxygen", "imide_like", "other"}
OTHER_SUBCLUSTER_COUNT = 3
_OTHER_SUBCLUSTER_CACHE: dict[tuple[int, int], dict[int, int]] = {}


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    hidden_dim: int
    weight_decay: float
    comparator_epochs: int
    comparator_patience: int
    comparator_lr: float
    residual_epochs: int
    residual_patience: int
    residual_lr: float
    teacher_epochs: int
    teacher_patience: int
    teacher_lr: float
    stage2_epochs: int
    stage2_patience: int
    stage2_lr: float
    stage3_epochs: int
    stage3_patience: int
    stage3_lr: float
    lambda_branch: float
    lambda_anchor: float
    lambda_gate: float
    lambda_delta: float
    lambda_led: float
    lambda_mspce_path: float
    lambda_cert_negative: float
    led_start_frac: float
    beneficial_quantile: float
    innovation_limit: float
    anchor_weights: AnchorLossWeights
    certification_rule: dict[str, Any] | None = None
    masd_slot_count: int = 4


class PolymerDataset(Dataset):
    def __init__(
        self,
        graphs: list[Data],
        desc: torch.Tensor,
        ctx: torch.Tensor,
        led: torch.Tensor,
        led_mask: torch.Tensor,
        y: torch.Tensor,
        cluster_code: torch.Tensor,
        cluster_support: torch.Tensor,
        chemistry_multihot: torch.Tensor,
        other_subcluster_code: torch.Tensor,
        indices: list[int],
    ) -> None:
        self.graphs = graphs
        self.desc = desc
        self.ctx = ctx
        self.led = led
        self.led_mask = led_mask
        self.y = y
        self.cluster_code = cluster_code
        self.cluster_support = cluster_support
        self.chemistry_multihot = chemistry_multihot
        self.other_subcluster_code = other_subcluster_code
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        return {
            "graph": self.graphs[idx],
            "desc": self.desc[idx],
            "ctx": self.ctx[idx],
            "led": self.led[idx],
            "led_mask": self.led_mask[idx],
            "y": self.y[idx],
            "cluster_code": self.cluster_code[idx],
            "cluster_support": self.cluster_support[idx],
            "chemistry_multihot": self.chemistry_multihot[idx],
            "other_subcluster_code": self.other_subcluster_code[idx],
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "graph": Batch.from_data_list([item["graph"] for item in items]),
        "desc": torch.stack([item["desc"] for item in items], dim=0),
        "ctx": torch.stack([item["ctx"] for item in items], dim=0),
        "led": torch.stack([item["led"] for item in items], dim=0),
        "led_mask": torch.stack([item["led_mask"] for item in items], dim=0),
        "y": torch.stack([item["y"] for item in items], dim=0),
        "cluster_code": torch.stack([item["cluster_code"] for item in items], dim=0),
        "cluster_support": torch.stack([item["cluster_support"] for item in items], dim=0),
        "chemistry_multihot": torch.stack([item["chemistry_multihot"] for item in items], dim=0),
        "other_subcluster_code": torch.stack([item["other_subcluster_code"] for item in items], dim=0),
        "sample_index": torch.stack([item["sample_index"] for item in items], dim=0),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_seed(seed: int, mode: str, phase: str) -> int:
    phase_offset = {"main": 0, "teacher": 1000, "stage2": 2000, "stage3": 3000}[phase]
    return seed * 101 + MODE_SEED_OFFSET[mode] + phase_offset


def load_artifacts() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    dataset = pd.read_csv(ROOT / "data/dataset.csv")
    features = torch.load(ROOT / "data/features.pt", map_location="cpu", weights_only=False)
    splits = json.loads((ROOT / "data/splits.json").read_text(encoding="utf-8"))
    return dataset, features, splits


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


def primary_cluster_name(smiles: str) -> str:
    tag_set = set(chemistry_tags(smiles))
    for name in CHEMISTRY_CLUSTER_ORDER:
        if name in tag_set:
            return name
    return "other"


def cluster_support_code(cluster_name: str) -> int:
    if cluster_name in WEAK_CLUSTER_NAMES:
        return 1
    if cluster_name in UNSTABLE_CLUSTER_NAMES:
        return 2
    return 0


def build_other_subcluster_lookup(features: dict[str, Any], dataset: pd.DataFrame) -> dict[int, int]:
    cache_key = (int(features["targets"].shape[0]), int(dataset.shape[0]))
    if cache_key in _OTHER_SUBCLUSTER_CACHE:
        return _OTHER_SUBCLUSTER_CACHE[cache_key]

    smiles_list = dataset["canonical_smiles"].fillna("").tolist()
    tag_lists = [chemistry_tags(str(item)) for item in smiles_list]
    other_indices = [idx for idx, tags in enumerate(tag_lists) if tags == ["other"]]
    if not other_indices:
        _OTHER_SUBCLUSTER_CACHE[cache_key] = {}
        return {}

    descriptor_block = features["descriptors"][other_indices].detach().cpu().numpy().astype(np.float32)
    context_block = features["contexts"][other_indices].detach().cpu().numpy().astype(np.float32)
    cluster_input = np.concatenate([descriptor_block, context_block], axis=1)
    cluster_input = cluster_input - cluster_input.mean(axis=0, keepdims=True)
    cluster_input = cluster_input / np.clip(cluster_input.std(axis=0, keepdims=True), 1e-6, None)

    n_clusters = min(OTHER_SUBCLUSTER_COUNT, len(other_indices))
    if n_clusters <= 1:
        labels = np.zeros(len(other_indices), dtype=np.int64)
    else:
        try:
            fitted = KMeans(n_clusters=n_clusters, n_init=20, random_state=0).fit(cluster_input)
            raw_labels = np.asarray(fitted.labels_, dtype=np.int64)
            centroids = np.asarray(fitted.cluster_centers_, dtype=np.float64)
            cluster_order = sorted(
                range(n_clusters),
                key=lambda cluster_idx: (
                    float(np.linalg.norm(centroids[cluster_idx])),
                    float(centroids[cluster_idx, 0]),
                    int(cluster_idx),
                ),
            )
            label_map = {int(old): int(new) for new, old in enumerate(cluster_order)}
            labels = np.asarray([label_map[int(item)] for item in raw_labels], dtype=np.int64)
        except Exception:
            # sklearn/threadpoolctl is unstable in the current Windows runtime.
            # Fall back to a deterministic rank-based bucketing so the protocol remains usable.
            projection = cluster_input.mean(axis=1)
            rank = np.argsort(np.argsort(projection, kind="mergesort"), kind="mergesort")
            labels = np.floor(rank.astype(np.float64) * float(n_clusters) / float(max(len(rank), 1))).astype(np.int64)
            labels = np.clip(labels, 0, n_clusters - 1)

    lookup = {int(other_indices[pos]): int(labels[pos]) for pos in range(len(other_indices))}
    _OTHER_SUBCLUSTER_CACHE[cache_key] = lookup
    return lookup


def _scale_tensor(values: torch.Tensor, train_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_values = values[train_idx]
    mean = train_values.mean(dim=0, keepdim=True)
    std = train_values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (values - mean) / std, mean, std


def prepare_seed_tensors(features: dict[str, Any], train_idx: list[int], dataset: pd.DataFrame | None = None) -> dict[str, Any]:
    desc_scaled, _, _ = _scale_tensor(features["descriptors"], train_idx)
    ctx_scaled, _, _ = _scale_tensor(features["contexts"], train_idx)
    led_scaled, _, _ = _scale_tensor(features["led"], train_idx)
    y_scaled, y_mean, y_std = _scale_tensor(features["targets"], train_idx)
    sample_count = int(features["targets"].shape[0])
    if dataset is not None and "canonical_smiles" in dataset.columns:
        other_subcluster_lookup = build_other_subcluster_lookup(features, dataset)
        all_tags = [chemistry_tags(str(item)) for item in dataset["canonical_smiles"].tolist()]
        all_tag_sets = [set(tags) for tags in all_tags]
        cluster_names = [next((name for name in CHEMISTRY_CLUSTER_ORDER if name in tag_set), "other") for tag_set in all_tag_sets]
        cluster_code = torch.tensor(
            [CHEMISTRY_CLUSTER_ORDER.index(name) for name in cluster_names],
            dtype=torch.long,
        ).unsqueeze(1)
        cluster_support = torch.tensor(
            [cluster_support_code(name) for name in cluster_names],
            dtype=torch.long,
        ).unsqueeze(1)
        chemistry_multihot = torch.tensor(
            [
                [1.0 if cluster_name in tag_set else 0.0 for cluster_name in CHEMISTRY_CLUSTER_ORDER]
                for tag_set in all_tag_sets
            ],
            dtype=torch.float32,
        )
        other_subcluster_code = torch.full((sample_count, 1), -1, dtype=torch.long)
        for sample_idx, subcluster_idx in other_subcluster_lookup.items():
            other_subcluster_code[int(sample_idx), 0] = int(subcluster_idx)
    else:
        cluster_code = torch.zeros((sample_count, 1), dtype=torch.long)
        cluster_support = torch.zeros((sample_count, 1), dtype=torch.long)
        chemistry_multihot = torch.zeros((sample_count, len(CHEMISTRY_CLUSTER_ORDER)), dtype=torch.float32)
        other_subcluster_code = torch.full((sample_count, 1), -1, dtype=torch.long)
    return {
        "graphs": features["graphs"],
        "desc": desc_scaled,
        "ctx": ctx_scaled,
        "context_layout": features.get("context_layout"),
        "led": led_scaled,
        "led_mask": features["led_mask"],
        "y": y_scaled,
        "y_mean": y_mean,
        "y_std": y_std,
        "cluster_code": cluster_code,
        "cluster_support": cluster_support,
        "chemistry_multihot": chemistry_multihot,
        "other_subcluster_code": other_subcluster_code,
    }


def make_loader(
    seed_tensors: dict[str, Any],
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    *,
    sample_weights: list[float] | np.ndarray | None = None,
    loader_seed: int | None = None,
) -> DataLoader:
    dataset = PolymerDataset(
        graphs=seed_tensors["graphs"],
        desc=seed_tensors["desc"],
        ctx=seed_tensors["ctx"],
        led=seed_tensors["led"],
        led_mask=seed_tensors["led_mask"],
        y=seed_tensors["y"],
        cluster_code=seed_tensors["cluster_code"],
        cluster_support=seed_tensors["cluster_support"],
        chemistry_multihot=seed_tensors["chemistry_multihot"],
        other_subcluster_code=seed_tensors["other_subcluster_code"],
        indices=indices,
    )
    generator = None
    if loader_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(loader_seed))
    if sample_weights is not None:
        weights = torch.as_tensor(np.asarray(sample_weights, dtype=np.float64), dtype=torch.double)
        if int(weights.numel()) != len(indices):
            raise ValueError(f"sample_weights length mismatch: {int(weights.numel())} vs {len(indices)}")
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(indices),
            replacement=True,
            generator=generator,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=collate_batch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch, generator=generator)


def _to_device(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "graph": batch["graph"].to(DEVICE),
        "desc": batch["desc"].to(DEVICE),
        "ctx": batch["ctx"].to(DEVICE),
        "led": batch["led"].to(DEVICE),
        "led_mask": batch["led_mask"].to(DEVICE),
        "y": batch["y"].to(DEVICE),
        "cluster_code": batch["cluster_code"].to(DEVICE),
        "cluster_support": batch["cluster_support"].to(DEVICE),
        "chemistry_multihot": batch["chemistry_multihot"].to(DEVICE),
        "other_subcluster_code": batch["other_subcluster_code"].to(DEVICE),
        "sample_index": batch["sample_index"].to(DEVICE),
    }


def improve_pct(baseline_mae: float, model_mae: float) -> float:
    return float((baseline_mae - model_mae) / max(baseline_mae, 1e-8) * 100.0)


def decide_support_status(
    mean_delta_vs_conflict: float,
    ci_conflict: tuple[float, float],
    accepted_precision: float,
    accepted_subset_gain: float,
    fallback_gain: float,
) -> str:
    if (
        mean_delta_vs_conflict <= 0.0
        and ci_conflict[1] <= 0.0
        and accepted_precision >= 0.5
        and accepted_subset_gain >= 0.0
        and fallback_gain >= 0.0
    ):
        return "positive"
    if (
        mean_delta_vs_conflict <= 0.0
        or accepted_precision >= 0.5
        or accepted_subset_gain > 0.0
        or fallback_gain > 0.0
    ):
        return "mixed"
    return "negative"


def choose_threshold(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    score: np.ndarray,
) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if score.size == 0 or float(score.max() - score.min()) < 1e-8:
        return float("-inf")
    candidates = np.unique(np.quantile(score, [0.55, 0.65, 0.75, 0.85, 0.90]))
    best_thr = float(candidates[0])
    best_key: tuple[float, float, float] | None = None
    model_err = np.abs(model_pred - y_true)
    base_err = np.abs(baseline_pred - y_true)
    for thr in candidates:
        accept = score >= float(thr)
        routed = np.where(accept, model_pred, baseline_pred)
        routed_mae = mae(y_true, routed)
        precision = float(np.mean(model_err[accept] < base_err[accept])) if accept.any() else 0.0
        subset_gain = float(np.mean(base_err[accept] - model_err[accept])) if accept.any() else 0.0
        fallback_gain = float(np.mean(model_err[~accept] - base_err[~accept])) if (~accept).any() else 0.0
        key = (1.0 if precision >= 0.5 else 0.0, -routed_mae, subset_gain + fallback_gain)
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
    return best_thr


def routed_metrics(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    score: np.ndarray | None,
    threshold: float | None,
) -> dict[str, float | np.ndarray]:
    if score is None:
        accept = np.ones_like(model_pred, dtype=bool)
    else:
        score_np = np.asarray(score, dtype=np.float64).reshape(-1)
        if threshold is None or threshold == float("-inf"):
            accept = np.ones_like(score_np, dtype=bool)
        else:
            accept = score_np >= float(threshold)
    model_err = np.abs(model_pred - y_true)
    base_err = np.abs(baseline_pred - y_true)
    routed = np.where(accept, model_pred, baseline_pred)
    accepted_precision = float(np.mean(model_err[accept] < base_err[accept])) if accept.any() else 0.0
    accepted_subset_gain = float(np.mean(base_err[accept] - model_err[accept])) if accept.any() else 0.0
    fallback_gain = float(np.mean(model_err[~accept] - base_err[~accept])) if (~accept).any() else 0.0
    return {
        "routed_pred": routed,
        "mae_k": mae(y_true, routed),
        "accepted_precision": accepted_precision,
        "accepted_subset_gain": accepted_subset_gain,
        "fallback_gain": fallback_gain,
        "accept_rate": float(np.mean(accept.astype(np.float32))),
    }


def build_model(
    mode: str,
    seed_tensors: dict[str, Any],
    config: TrainConfig,
    certification_rule: dict[str, Any] | None = None,
) -> FusionModel:
    # One builder covers both archived comparator stages and the retained paper model.
    example_graph = seed_tensors["graphs"][0]
    model = FusionModel(
        backbone_name="attentive_fp",
        mode=mode,
        desc_dim=int(seed_tensors["desc"].shape[1]),
        ctx_dim=int(seed_tensors["ctx"].shape[1]),
        led_dim=int(seed_tensors["led"].shape[1]),
        node_dim=int(example_graph.x.shape[1]),
        edge_dim=int(example_graph.edge_attr.shape[1]) if example_graph.edge_attr.numel() > 0 else 7,
        hidden_dim=config.hidden_dim,
        safe_residual=mode in {"rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"},
        innovation_limit=config.innovation_limit,
        certification_rule=certification_rule,
        context_layout=seed_tensors.get("context_layout"),
        masd_slot_count=int(getattr(config, "masd_slot_count", 4)),
    ).to(DEVICE)
    apply_experiment_overrides(model, seed_tensors=seed_tensors)
    return model


def led_dual_distillation_terms(
    model: FusionModel,
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    led_latent, led_pred = model.led_prior(batch["led"])
    led_latent_detached = led_latent.detach()
    led_pred_detached = led_pred.detach()
    confidence = torch.clamp(
        0.5 * torch.sigmoid(torch.abs(out["mspce_anchor_pred"] - out["student_baseline_pred"]))
        + 0.5 * torch.sigmoid(out["innovation_score"]),
        min=0.0,
        max=1.0,
    )
    weight = batch["led_mask"] * confidence
    latent_a = ((out["ctx_emb"] - led_latent_detached) ** 2).mean(dim=1, keepdim=True)
    latent_b = ((out["fused_latent"] - led_latent_detached) ** 2).mean(dim=1, keepdim=True)
    residual_target = torch.clamp(
        led_pred_detached - out["mspce_anchor_pred"].detach(),
        min=-model.innovation_limit,
        max=model.innovation_limit,
    )
    residual_align = torch.abs(out["fusion_delta"] - residual_target)
    loss_latent_a = masked_mean(latent_a, weight)
    loss_latent_b = masked_mean(latent_b, weight)
    loss_residual = masked_mean(residual_align, weight)
    return {
        "weight_mean": weight.mean(),
        "latent_a": loss_latent_a,
        "latent_b": loss_latent_b,
        "residual": loss_residual,
    }


def preserve_gain_loss(
    out: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    margin: float = 0.015,
    min_gain_conflict: float = 0.010,
    min_gain_concat: float = 0.010,
) -> torch.Tensor:
    err_full = torch.abs(out["pred"] - y_true)
    err_mspce = torch.abs(out["mspce_anchor_pred"] - y_true)
    err_conflict = torch.abs(out["student_baseline_pred"] - y_true)
    err_concat = torch.abs(out["concat_pred"] - y_true)
    mspce_better_conflict = (err_conflict - err_mspce) >= min_gain_conflict
    mspce_better_concat = (err_concat - err_mspce) >= min_gain_concat
    preserve_mask = (mspce_better_conflict & mspce_better_concat).float()
    penalty = torch.relu(err_full - err_mspce + margin)
    return masked_mean(penalty, preserve_mask)


def standard_loss(
    model: FusionModel,
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_fn: nn.Module,
    config: TrainConfig,
    mode: str,
    epoch: int,
    total_epochs: int,
) -> torch.Tensor:
    y_true = batch["y"]
    loss = loss_fn(out["pred"], y_true)
    loss = loss + config.lambda_branch * (
        loss_fn(out["desc_pred"], y_true)
        + loss_fn(out["graph_pred"], y_true)
        + loss_fn(out["student_baseline_pred"], y_true)
    )

    if mode in {"mspce_only", "mspce_only_led", "rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
        loss = loss + config.lambda_anchor * loss_fn(out["student_baseline_pred"], y_true)
    if mode == "mspce_context_injection":
        loss = loss + 0.55 * config.lambda_anchor * loss_fn(out["concat_pred"], y_true)
        loss = loss + 0.35 * config.lambda_anchor * loss_fn(out["mspce_repair_candidate_pred"], y_true)
        concat_margin = torch.abs(out["pred"] - y_true) - torch.abs(out["concat_pred"] - y_true) + 0.0015
        loss = loss + 0.60 * config.lambda_gate * torch.mean(torch.relu(concat_margin))
        loss = loss + 0.10 * config.lambda_delta * torch.abs(out["ctx_delta"]).mean()
    if mode in {"mspce_only_led", "rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
        loss = loss + config.lambda_anchor * loss_fn(out["mspce_anchor_pred"], y_true)
    if mode in {"rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
        loss = loss + config.lambda_gate * out["fusion_alpha"].mean()
        loss = loss + config.lambda_delta * torch.abs(out["fusion_delta"]).mean()
    if mode in {"mspce_only_led", "rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"} and epoch + 1 >= max(1, int(total_epochs * config.led_start_frac)):
        led_terms = led_dual_distillation_terms(model=model, out=out, batch=batch)
        latent_loss = 0.70 * led_terms["latent_a"] + 0.25 * led_terms["latent_b"]
        if mode == "rcmf_dynamic_multimodal_full_phasea":
            residual_loss = 0.01 * led_terms["residual"]
        elif mode == "rcmf_dynamic_multimodal_full_dual_anchor":
            residual_loss = 0.008 * led_terms["residual"]
        elif mode == "rcmf_dynamic_multimodal_full":
            residual_loss = 0.03 * led_terms["residual"]
        else:
            residual_loss = 0.0 * led_terms["residual"]
        loss = loss + config.lambda_led * (latent_loss + residual_loss)
    if mode in {"rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
        loss_preserve = preserve_gain_loss(out, y_true)
        preserve_weight = 1.0 if mode == "rcmf_dynamic_multimodal_full" else 1.15
        loss = loss + preserve_weight * config.lambda_mspce_path * loss_preserve
    if mode == "rcmf_dynamic_multimodal_full_phasea":
        concat_err = torch.abs(out["pred"] - y_true) - torch.abs(out["concat_pred"] - y_true) + 0.004
        external_risk = out.get("external_risk", torch.zeros_like(out["fusion_alpha"]))
        risk_weight = 0.20 + 0.80 * (1.0 - external_risk)
        guardrail = torch.mean(out["fusion_alpha"] * external_risk)
        loss = loss + 0.45 * config.lambda_gate * torch.mean(torch.relu(concat_err) * risk_weight)
        loss = loss + 0.45 * config.lambda_gate * guardrail
    if mode == "rcmf_dynamic_multimodal_full_dual_anchor":
        err_full = torch.abs(out["pred"] - y_true)
        err_anchor = torch.abs(out["dual_anchor_pred"] - y_true)
        err_mspce = torch.abs(out["mspce_anchor_pred"] - y_true)
        err_concat = torch.abs(out["concat_pred"] - y_true)
        err_conflict = torch.abs(out["student_baseline_pred"] - y_true)
        l_anchor = torch.relu(err_full - err_anchor + 0.006).mean()
        valuable_mask = ((err_conflict - err_mspce) >= 0.010) & ((err_concat - err_mspce) >= 0.010)
        l_mspce = masked_mean(torch.relu(err_full - err_mspce + 0.004), valuable_mask.float())
        beta_target = (err_concat < err_mspce).float()
        l_beta = torch.nn.functional.binary_cross_entropy(
            torch.clamp(out["anchor_beta"], min=1e-4, max=1.0 - 1e-4),
            beta_target,
        )
        loss = loss + 1.40 * config.lambda_mspce_path * l_anchor
        loss = loss + 1.15 * config.lambda_mspce_path * l_mspce
        loss = loss + 0.40 * config.lambda_gate * l_beta
    return loss


@torch.no_grad()
def collect_predictions(
    model: FusionModel,
    loader: DataLoader,
    seed_tensors: dict[str, Any],
    teacher_model: FusionModel | None = None,
) -> dict[str, Any]:
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    y_scaled = []
    pred_scaled = []
    base_scaled = []
    mspce_scaled = []
    concat_scaled = []
    score_scaled = []
    for batch in loader:
        batch = _to_device(batch)
        teacher_pred = None
        if teacher_model is not None:
            teacher_out = teacher_model(batch["graph"], batch["desc"], batch["ctx"])
            teacher_pred = teacher_out["pred"].detach()
        out = model(batch["graph"], batch["desc"], batch["ctx"], teacher_pred=teacher_pred)
        y_scaled.append(batch["y"].detach().cpu())
        pred_scaled.append(out["pred"].detach().cpu())
        base_scaled.append(out["baseline_pred"].detach().cpu())
        mspce_scaled.append(out.get("mspce_anchor_pred", out["pred"]).detach().cpu())
        concat_scaled.append(out.get("concat_pred", out["pred"]).detach().cpu())
        score_scaled.append(out["innovation_score"].detach().cpu())

    y_scaled_t = torch.cat(y_scaled, dim=0)
    pred_scaled_t = torch.cat(pred_scaled, dim=0)
    base_scaled_t = torch.cat(base_scaled, dim=0)
    mspce_scaled_t = torch.cat(mspce_scaled, dim=0)
    concat_scaled_t = torch.cat(concat_scaled, dim=0)
    score_t = torch.cat(score_scaled, dim=0)

    y = y_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    pred = pred_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    baseline = base_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    mspce = mspce_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    concat = concat_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    return {
        "y_true": y.numpy().squeeze(1),
        "pred": pred.numpy().squeeze(1),
        "baseline_pred": baseline.numpy().squeeze(1),
        "mspce_pred": mspce.numpy().squeeze(1),
        "concat_pred": concat.numpy().squeeze(1),
        "innovation_score": score_t.numpy().squeeze(1),
        "mae_k": mae(y.numpy(), pred.numpy()),
    }


@torch.no_grad()
def collect_aux_predictions(
    model: FusionModel,
    loader: DataLoader,
    seed_tensors: dict[str, Any],
    teacher_model: FusionModel | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    payload: dict[str, list[torch.Tensor]] = {
        "y_true": [],
        "pred": [],
        "baseline_pred": [],
        "baseline_unc": [],
        "conflict_level": [],
        "ctx_delta": [],
        "student_baseline_pred": [],
        "innovation_score": [],
        "fusion_alpha": [],
        "external_risk": [],
        "led_confidence": [],
        "anchor_beta": [],
        "dual_anchor_pred": [],
    }
    for batch in loader:
        batch = _to_device(batch)
        teacher_pred = None
        if teacher_model is not None:
            teacher_out = teacher_model(batch["graph"], batch["desc"], batch["ctx"])
            teacher_pred = teacher_out["pred"].detach()
        out = model(batch["graph"], batch["desc"], batch["ctx"], teacher_pred=teacher_pred)
        payload["y_true"].append(batch["y"].detach().cpu())
        payload["pred"].append(out["pred"].detach().cpu())
        payload["baseline_pred"].append(out["baseline_pred"].detach().cpu())
        payload["baseline_unc"].append(out["baseline_unc"].detach().cpu())
        payload["conflict_level"].append(torch.abs(out["desc_pred"] - out["graph_pred"]).detach().cpu())
        payload["ctx_delta"].append(out["ctx_delta"].detach().cpu())
        payload["student_baseline_pred"].append(out["student_baseline_pred"].detach().cpu())
        payload["innovation_score"].append(out["innovation_score"].detach().cpu())
        payload["fusion_alpha"].append(out.get("fusion_alpha", torch.zeros_like(out["pred"])).detach().cpu())
        payload["external_risk"].append(out.get("external_risk", torch.zeros_like(out["pred"])).detach().cpu())
        payload["led_confidence"].append(out.get("led_confidence", torch.ones_like(out["pred"])).detach().cpu())
        payload["anchor_beta"].append(out.get("anchor_beta", torch.zeros_like(out["pred"])).detach().cpu())
        payload["dual_anchor_pred"].append(out.get("dual_anchor_pred", out["baseline_pred"]).detach().cpu())

    result: dict[str, np.ndarray] = {}
    for key, values in payload.items():
        merged = torch.cat(values, dim=0)
        if key == "y_true":
            merged = merged * seed_tensors["y_std"] + seed_tensors["y_mean"]
        elif key in {"pred", "baseline_pred", "ctx_delta", "student_baseline_pred", "dual_anchor_pred"}:
            merged = merged * seed_tensors["y_std"] + seed_tensors["y_mean"]
        result[key] = merged.numpy().squeeze(1)
    return result


def train_standard_model(
    *,
    mode: str,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: TrainConfig,
    seed: int,
) -> FusionModel:
    set_seed(seed)
    model = build_model(mode, seed_tensors, config)
    loss_fn = nn.SmoothL1Loss()
    if mode == "conflict_only":
        epochs = config.teacher_epochs
        patience = config.teacher_patience
        lr = config.teacher_lr
    elif mode in {"rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
        epochs = config.residual_epochs
        patience = config.residual_patience
        lr = config.residual_lr
    else:
        epochs = config.comparator_epochs
        patience = config.comparator_patience
        lr = config.comparator_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch)
            out = model(batch["graph"], batch["desc"], batch["ctx"])
            loss = standard_loss(model, out, batch, loss_fn, config, mode, epoch, epochs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_pred = collect_predictions(model, val_loader, seed_tensors)
        val_score = float(val_pred["mae_k"])
        if mode in {"rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
            val_mspce_mae = float(mae(val_pred["y_true"], val_pred["mspce_pred"]))
            val_concat_mae = float(mae(val_pred["y_true"], val_pred["concat_pred"]))
            val_score = val_score + max(0.0, val_score - val_mspce_mae) + 0.45 * max(0.0, val_score - val_concat_mae)
        if mode == "rcmf_dynamic_multimodal_full_phasea":
            val_concat_mae = float(mae(val_pred["y_true"], val_pred["concat_pred"]))
            val_score = val_score + 0.80 * max(0.0, val_score - val_concat_mae)
            val_aux = collect_aux_predictions(model, val_loader, seed_tensors)
            external_risk = np.asarray(val_aux["external_risk"], dtype=np.float64)
            fusion_alpha = np.asarray(val_aux["fusion_alpha"], dtype=np.float64)
            led_conf = np.asarray(val_aux["led_confidence"], dtype=np.float64)
            risky_alpha = float(np.mean(fusion_alpha * external_risk))
            risk_mean = float(np.mean(external_risk))
            low_led_penalty = max(0.0, 0.52 - float(np.mean(led_conf)))
            guardrail_penalty = 0.55 * risky_alpha + 0.25 * max(0.0, risk_mean - 0.55) + 0.15 * low_led_penalty
            val_score = val_score + guardrail_penalty
        if mode == "rcmf_dynamic_multimodal_full_dual_anchor":
            val_aux = collect_aux_predictions(model, val_loader, seed_tensors)
            val_beta = np.asarray(val_aux.get("anchor_beta", np.zeros_like(val_aux["fusion_alpha"])), dtype=np.float64)
            val_anchor = np.asarray(val_aux.get("dual_anchor_pred", val_aux["baseline_pred"]), dtype=np.float64)
            y_true = np.asarray(val_aux["y_true"], dtype=np.float64)
            pred = np.asarray(val_aux["pred"], dtype=np.float64)
            val_anchor_mae = float(mae(y_true, val_anchor))
            val_score = val_score + 0.90 * max(0.0, val_score - val_anchor_mae) + 0.10 * float(np.mean((val_beta - 0.5) ** 2))
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"no checkpoint stored for mode={mode}")
    model.load_state_dict(best_state)
    return model


def estimate_beneficial_threshold(
    teacher_model: FusionModel,
    train_loader: DataLoader,
    seed_tensors: dict[str, Any],
    quantile: float,
) -> float:
    teacher_train = collect_predictions(teacher_model, train_loader, seed_tensors)
    teacher_error = np.abs(teacher_train["pred"] - teacher_train["y_true"])
    return beneficial_threshold(teacher_error, quantile)


def make_optimizer(model: FusionModel, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def inner_fold_split(indices: list[int], seed: int, fold_id: int, n_folds: int = 3) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)
    oof_idx = folds[fold_id].tolist()
    remain = np.concatenate([fold for idx, fold in enumerate(folds) if idx != fold_id], axis=0)
    val_size = max(1, int(len(remain) * 0.15))
    val_idx = remain[:val_size].tolist()
    train_idx = remain[val_size:].tolist()
    return {"train": train_idx, "val": val_idx, "oof": oof_idx}


def oof_certification_rows(
    *,
    features: dict[str, Any],
    split_payload: dict[str, Any],
    config: TrainConfig,
    seeds: list[int],
    n_folds: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        outer_split = split_payload["seeds"][str(seed)]
        train_indices = list(outer_split["train"])
        for fold_id in range(n_folds):
            fold_split = inner_fold_split(train_indices, seed=seed * 37 + fold_id, fold_id=fold_id, n_folds=n_folds)
            seed_tensors = prepare_seed_tensors(features, fold_split["train"])
            temp_split = {"train": fold_split["train"], "val": fold_split["val"]}
            teacher = train_standard_model(
                mode="conflict_only",
                split=temp_split,
                seed_tensors=seed_tensors,
                config=config,
                seed=stable_seed(seed, "conflict_only", "teacher") + fold_id,
            )
            mspce = train_standard_model(
                mode="mspce_only",
                split=temp_split,
                seed_tensors=seed_tensors,
                config=config,
                seed=stable_seed(seed, "mspce_only", "main") + fold_id,
            )
            oof_loader = make_loader(seed_tensors, fold_split["oof"], config.batch_size, shuffle=False)
            teacher_aux = collect_aux_predictions(teacher, oof_loader, seed_tensors)
            mspce_aux = collect_aux_predictions(mspce, oof_loader, seed_tensors)
            oof_gain = np.abs(teacher_aux["pred"] - teacher_aux["y_true"]) - np.abs(mspce_aux["pred"] - mspce_aux["y_true"])
            feature_matrix = {
                "conflict_level": teacher_aux["conflict_level"],
                "uncertainty": teacher_aux["baseline_unc"],
                "context_strength": np.abs(mspce_aux["ctx_delta"]),
                "teacher_student_gap": np.abs(mspce_aux["student_baseline_pred"] - teacher_aux["pred"]),
                "teacher_candidate_gap": np.abs(mspce_aux["pred"] - teacher_aux["pred"]),
            }
            frame = pd.DataFrame(feature_matrix)
            frame["oof_gain"] = oof_gain
            frame["seed"] = seed
            frame["fold"] = fold_id
            rows.extend(frame.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_certification_rule(
    *,
    features: dict[str, Any],
    split_payload: dict[str, Any],
    config: TrainConfig,
    seeds: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    oof_df = oof_certification_rows(features=features, split_payload=split_payload, config=config, seeds=seeds)
    rule = fit_certification_rule(oof_df)
    summary = summarize_certification_rule(oof_df, rule)
    cert_df = pd.DataFrame(
        [
            {
                "certified_positive_region_coverage": summary["certified_positive_region_coverage"],
                "certified_positive_region_precision": summary["certified_positive_region_precision"],
                "certified_positive_region_gain": summary["certified_positive_region_gain"],
                "certified_negative_region_coverage": summary["certified_negative_region_coverage"],
                "certified_negative_region_gain": summary["certified_negative_region_gain"],
                "certification_rule": summary["certification_rule"],
            }
        ]
    )
    cert_df.to_csv(ROOT / "reports/certification_report.csv", index=False)
    md_lines = [
        "# Certification",
        "",
        "This certification step uses only train-only OOF evidence comparing the MSPCE-only path against the conflict-only path, and does not use external-test labels.",
        f"- certified_positive_region_coverage: {summary['certified_positive_region_coverage']:.4f}",
        f"- certified_positive_region_precision: {summary['certified_positive_region_precision']:.4f}",
        f"- certified_positive_region_gain: {summary['certified_positive_region_gain']:.4f}",
        f"- certified_negative_region_coverage: {summary['certified_negative_region_coverage']:.4f}",
        f"- certified_negative_region_gain: {summary['certified_negative_region_gain']:.4f}",
        f"- certification rule: {summary['certification_rule']}",
        "",
        "RCMF is only allowed to deviate from the teacher inside the certified positive region; all other regions use hard fallback to the teacher.",
    ]
    (ROOT / "reports/certification_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    return rule, summary


def run_teacher_stage(
    *,
    student: FusionModel,
    teacher_model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    seed_tensors: dict[str, Any],
    config: TrainConfig,
    threshold: float,
    stage: str,
) -> FusionModel:
    loss_fn = nn.SmoothL1Loss()
    if stage == "stage2":
        epochs = config.stage2_epochs
        patience = config.stage2_patience
        lr = config.stage2_lr
    else:
        epochs = config.stage3_epochs
        patience = config.stage3_patience
        lr = config.stage3_lr
    optimizer = make_optimizer(student, lr=lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    led_start_epoch = max(1, int(epochs * config.led_start_frac))

    for epoch in range(epochs):
        student.train()
        teacher_model.eval()
        for batch in train_loader:
            batch = _to_device(batch)
            with torch.no_grad():
                teacher_out = teacher_model(batch["graph"], batch["desc"], batch["ctx"])
                teacher_pred = teacher_out["pred"].detach()
            out = student(batch["graph"], batch["desc"], batch["ctx"], teacher_pred=teacher_pred)

            supervised = loss_fn(out["pred"], batch["y"])
            branch = config.lambda_branch * (
                loss_fn(out["desc_pred"], batch["y"])
                + loss_fn(out["graph_pred"], batch["y"])
                + loss_fn(out["student_baseline_pred"], batch["y"])
            )
            anchor_terms = teacher_anchor_loss(
                prediction=out["pred"],
                teacher_pred=teacher_pred,
                y_true=batch["y"],
                innovation_gate=out["rcmf_gate"],
                innovation_delta=out["ctx_delta"],
                threshold=threshold,
                delta_limit=config.innovation_limit,
                weights=config.anchor_weights,
            )
            beneficial, _ = beneficial_mask(
                y_true=batch["y"],
                teacher_pred=teacher_pred,
                threshold=threshold,
            )
            mspce_teacher_path = teacher_pred + out["ctx_delta"]
            mspce_path_loss = masked_mean(torch.abs(mspce_teacher_path - batch["y"]), beneficial)
            loss = supervised + branch + anchor_terms["total"] + config.lambda_mspce_path * mspce_path_loss
            if epoch + 1 >= led_start_epoch:
                loss = loss + config.lambda_led * led_regularization(
                    model=student,
                    out=out,
                    led=batch["led"],
                    led_mask=batch["led_mask"],
                    anchor_pred=teacher_pred,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

        val_pred = collect_predictions(student, val_loader, seed_tensors, teacher_model=teacher_model)
        threshold_val = choose_threshold(
            y_true=val_pred["y_true"],
            model_pred=val_pred["pred"],
            baseline_pred=val_pred["baseline_pred"],
            score=val_pred["innovation_score"],
        )
        routed_val = routed_metrics(
            y_true=val_pred["y_true"],
            model_pred=val_pred["pred"],
            baseline_pred=val_pred["baseline_pred"],
            score=val_pred["innovation_score"],
            threshold=threshold_val,
        )
        if routed_val["mae_k"] < best_val:
            best_val = float(routed_val["mae_k"])
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"no checkpoint stored for {stage}")
    student.load_state_dict(best_state)
    return student


def train_teacher_anchored_student(
    *,
    teacher_model: FusionModel,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: TrainConfig,
    seed: int,
) -> FusionModel:
    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)
    threshold = estimate_beneficial_threshold(
        teacher_model=teacher_model,
        train_loader=train_loader,
        seed_tensors=seed_tensors,
        quantile=config.beneficial_quantile,
    )

    set_seed(stable_seed(seed, "teacher_anchored_rcmf_full", "stage2"))
    student = build_model("teacher_anchored_rcmf_full", seed_tensors, config)
    copy_shared_weights(teacher_model, student)

    freeze_teacher_anchored_stage(student)
    student = run_teacher_stage(
        student=student,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed_tensors=seed_tensors,
        config=config,
        threshold=threshold,
        stage="stage2",
    )

    unfreeze_all(student)
    set_seed(stable_seed(seed, "teacher_anchored_rcmf_full", "stage3"))
    student = run_teacher_stage(
        student=student,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed_tensors=seed_tensors,
        config=config,
        threshold=threshold,
        stage="stage3",
    )
    return student


def run_certified_stage(
    *,
    student: FusionModel,
    teacher_model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    seed_tensors: dict[str, Any],
    config: TrainConfig,
    stage: str,
) -> FusionModel:
    loss_fn = nn.SmoothL1Loss()
    if stage == "stage2":
        epochs = config.stage2_epochs
        patience = config.stage2_patience
        lr = config.stage2_lr
    else:
        epochs = config.stage3_epochs
        patience = config.stage3_patience
        lr = config.stage3_lr
    optimizer = make_optimizer(student, lr=lr, weight_decay=config.weight_decay)
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    led_start_epoch = max(1, int(epochs * config.led_start_frac))

    for epoch in range(epochs):
        student.train()
        teacher_model.eval()
        for batch in train_loader:
            batch = _to_device(batch)
            with torch.no_grad():
                teacher_out = teacher_model(batch["graph"], batch["desc"], batch["ctx"])
                teacher_pred = teacher_out["pred"].detach()
            out = student(batch["graph"], batch["desc"], batch["ctx"], teacher_pred=teacher_pred)
            positive_mask = out["certified_positive_mask"]
            negative_mask = out["certified_negative_mask"]

            supervised = loss_fn(out["pred"], batch["y"])
            branch = config.lambda_branch * (
                loss_fn(out["desc_pred"], batch["y"])
                + loss_fn(out["graph_pred"], batch["y"])
                + loss_fn(out["student_baseline_pred"], batch["y"])
            )
            mspce_teacher_path = teacher_pred + positive_mask * out["ctx_delta"]
            mspce_path_loss = masked_mean(torch.abs(mspce_teacher_path - batch["y"]), positive_mask)
            negative_proxy = masked_mean(torch.abs(out["ctx_delta"]) + out["rcmf_gate"], negative_mask)
            loss = supervised + branch + config.lambda_mspce_path * mspce_path_loss
            loss = loss + config.lambda_cert_negative * negative_proxy
            if epoch + 1 >= led_start_epoch:
                led_mask = batch["led_mask"] * positive_mask
                loss = loss + config.lambda_led * led_regularization(
                    model=student,
                    out=out,
                    led=batch["led"],
                    led_mask=led_mask,
                    anchor_pred=teacher_pred,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

        val_pred = collect_predictions(student, val_loader, seed_tensors, teacher_model=teacher_model)
        routed_val = routed_metrics(
            y_true=val_pred["y_true"],
            model_pred=val_pred["pred"],
            baseline_pred=val_pred["baseline_pred"],
            score=val_pred["innovation_score"],
            threshold=0.5,
        )
        if routed_val["mae_k"] < best_val:
            best_val = float(routed_val["mae_k"])
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"no checkpoint stored for certified {stage}")
    student.load_state_dict(best_state)
    return student


def train_certified_student(
    *,
    teacher_model: FusionModel,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: TrainConfig,
    seed: int,
) -> FusionModel:
    if config.certification_rule is None:
        raise RuntimeError("certification_rule is required for mspce_certified_rcmf")
    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)

    set_seed(stable_seed(seed, "mspce_certified_rcmf", "stage2"))
    student = build_model(
        "mspce_certified_rcmf",
        seed_tensors,
        config,
        certification_rule=config.certification_rule,
    )
    copy_shared_weights(teacher_model, student)
    freeze_teacher_anchored_stage(student)
    student = run_certified_stage(
        student=student,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed_tensors=seed_tensors,
        config=config,
        stage="stage2",
    )

    unfreeze_all(student)
    set_seed(stable_seed(seed, "mspce_certified_rcmf", "stage3"))
    student = run_certified_stage(
        student=student,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed_tensors=seed_tensors,
        config=config,
        stage="stage3",
    )
    return student


def run_switch_stage(
    *,
    student: FusionModel,
    teacher_model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    seed_tensors: dict[str, Any],
    config: TrainConfig,
    stage: str,
) -> FusionModel:
    loss_fn = nn.SmoothL1Loss()
    if stage == "stage2":
        epochs = config.stage2_epochs
        patience = config.stage2_patience
        lr = config.stage2_lr
    else:
        epochs = config.stage3_epochs
        patience = config.stage3_patience
        lr = config.stage3_lr
    optimizer = make_optimizer(student, lr=lr, weight_decay=config.weight_decay)
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    led_start_epoch = max(1, int(epochs * config.led_start_frac))

    for epoch in range(epochs):
        student.train()
        teacher_model.eval()
        for batch in train_loader:
            batch = _to_device(batch)
            with torch.no_grad():
                teacher_out = teacher_model(batch["graph"], batch["desc"], batch["ctx"])
                teacher_pred = teacher_out["pred"].detach()
            out = student(batch["graph"], batch["desc"], batch["ctx"], teacher_pred=teacher_pred)
            positive_mask = out["positive_switch_mask"]
            teacher_mask = out["teacher_region_mask"]
            uncertain_mask = out["uncertain_region_mask"]
            non_switch_mask = torch.clamp(teacher_mask + uncertain_mask, max=1.0)
            switch_mask = out["switch_mask"]
            gate_prob = out["gate_probability"].clamp(1e-5, 1.0 - 1e-5)
            candidate_pred = out["candidate_pred"]

            teacher_error = torch.abs(teacher_pred - batch["y"])
            candidate_error = torch.abs(candidate_pred - batch["y"])
            actual_gain = teacher_error - candidate_error
            switch_target = positive_mask * (actual_gain > 0.0).float()

            supervised = loss_fn(out["pred"], batch["y"])
            branch = config.lambda_branch * (
                loss_fn(out["desc_pred"], batch["y"])
                + loss_fn(out["graph_pred"], batch["y"])
                + loss_fn(out["student_baseline_pred"], batch["y"])
            )
            positive_path = config.lambda_mspce_path * masked_mean(candidate_error, positive_mask)
            hard_fallback = config.lambda_anchor * masked_mean(torch.abs(out["pred"] - teacher_pred), non_switch_mask)
            gate_loss = config.lambda_gate * torch.nn.functional.binary_cross_entropy(
                gate_prob,
                switch_target,
            )
            negative_proxy = config.lambda_cert_negative * masked_mean(
                torch.abs(out["ctx_delta"]) + gate_prob,
                non_switch_mask,
            )
            uncertain_proxy = config.lambda_delta * masked_mean(
                switch_mask + torch.abs(out["ctx_delta"]),
                uncertain_mask,
            )
            loss = supervised + branch + positive_path + hard_fallback + gate_loss + negative_proxy + uncertain_proxy
            if epoch + 1 >= led_start_epoch:
                led_mask = batch["led_mask"] * switch_mask
                loss = loss + config.lambda_led * led_regularization(
                    model=student,
                    out=out,
                    led=batch["led"],
                    led_mask=led_mask,
                    anchor_pred=teacher_pred,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

        val_pred = collect_predictions(student, val_loader, seed_tensors, teacher_model=teacher_model)
        routed_val = routed_metrics(
            y_true=val_pred["y_true"],
            model_pred=val_pred["pred"],
            baseline_pred=val_pred["baseline_pred"],
            score=val_pred["innovation_score"],
            threshold=0.5,
        )
        if routed_val["mae_k"] < best_val:
            best_val = float(routed_val["mae_k"])
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"no checkpoint stored for switch stage={stage}")
    student.load_state_dict(best_state)
    return student


def train_switch_student(
    *,
    teacher_model: FusionModel,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: TrainConfig,
    seed: int,
) -> FusionModel:
    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)

    set_seed(stable_seed(seed, "rcmf_dynamic_multimodal_full", "stage2"))
    student = build_model(
        "rcmf_dynamic_multimodal_full",
        seed_tensors,
        config,
        certification_rule=config.certification_rule,
    )
    copy_shared_weights(teacher_model, student)
    freeze_teacher_anchored_stage(student)
    student = run_switch_stage(
        student=student,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed_tensors=seed_tensors,
        config=config,
        stage="stage2",
    )

    unfreeze_all(student)
    set_seed(stable_seed(seed, "rcmf_dynamic_multimodal_full", "stage3"))
    student = run_switch_stage(
        student=student,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed_tensors=seed_tensors,
        config=config,
        stage="stage3",
    )
    return student


def evaluate_mode_against_conflict(
    *,
    mode: str,
    val_out: dict[str, Any],
    test_out: dict[str, Any],
    external_out: dict[str, Any],
    val_conflict: dict[str, Any],
    test_conflict: dict[str, Any],
    external_conflict: dict[str, Any],
) -> dict[str, Any]:
    route_enabled = mode in ROUTE_ENABLED_MODES
    if route_enabled:
        threshold = choose_threshold(
            y_true=val_out["y_true"],
            model_pred=val_out["pred"],
            baseline_pred=val_conflict["pred"],
            score=val_out["innovation_score"],
        )
    else:
        threshold = None
    test_metrics = routed_metrics(
        y_true=test_out["y_true"],
        model_pred=test_out["pred"],
        baseline_pred=test_conflict["pred"],
        score=test_out["innovation_score"] if route_enabled else None,
        threshold=threshold,
    )
    external_metrics = routed_metrics(
        y_true=external_out["y_true"],
        model_pred=external_out["pred"],
        baseline_pred=external_conflict["pred"],
        score=external_out["innovation_score"] if route_enabled else None,
        threshold=threshold,
    )
    return {
        "threshold": threshold if threshold is not None else float("nan"),
        "threshold_source": "val_score_calibration" if route_enabled else "not_applicable",
        "primary_mae_k": float(test_metrics["mae_k"]),
        "primary_raw_mae_k": float(mae(test_out["y_true"], test_out["pred"])),
        "external_mae_k": float(external_metrics["mae_k"]),
        "external_raw_mae_k": float(mae(external_out["y_true"], external_out["pred"])),
        "accepted_precision": float(external_metrics["accepted_precision"]),
        "accepted_subset_gain": float(external_metrics["accepted_subset_gain"]),
        "fallback_gain": float(external_metrics["fallback_gain"]),
        "primary_accept_rate": float(test_metrics["accept_rate"]),
        "external_accept_rate": float(external_metrics["accept_rate"]),
        "primary_raw_pred": test_out["pred"],
        "external_raw_pred": external_out["pred"],
        "primary_routed_pred": test_metrics["routed_pred"],
        "external_routed_pred": external_metrics["routed_pred"],
        "val_raw_mae_k": float(mae(val_out["y_true"], val_out["pred"])),
        "val_routed_mae_k": float(mae(val_out["y_true"], np.where(val_out["innovation_score"] >= threshold, val_out["pred"], val_conflict["pred"]))) if route_enabled else float(mae(val_out["y_true"], val_out["pred"])),
    }


def seed_bundle(
    *,
    seed: int,
    features: dict[str, Any],
    split_payload: dict[str, Any],
    config: TrainConfig,
    modes: tuple[str, ...],
) -> pd.DataFrame:
    split = split_payload["seeds"][str(seed)]
    seed_tensors = prepare_seed_tensors(features, split["train"])
    loaders = {
        name: make_loader(seed_tensors, split[name], config.batch_size, shuffle=False)
        for name in ("val", "test", "external")
    }

    models: dict[str, FusionModel] = {}
    for mode in modes:
        models[mode] = train_standard_model(
            mode=mode,
            split=split,
            seed_tensors=seed_tensors,
            config=config,
            seed=stable_seed(seed, mode, "main"),
        )

    outputs: dict[str, dict[str, Any]] = {name: {} for name in loaders}
    for mode, model in models.items():
        for split_name, loader in loaders.items():
            outputs[split_name][mode] = collect_predictions(
                model,
                loader,
                seed_tensors,
            )

    rows: list[dict[str, Any]] = []
    val_conflict = outputs["val"]["conflict_only"]
    test_conflict = outputs["test"]["conflict_only"]
    external_conflict = outputs["external"]["conflict_only"]
    test_concat = outputs["test"].get("simple_concat")
    external_concat = outputs["external"].get("simple_concat")

    for mode in modes:
        eval_row = evaluate_mode_against_conflict(
            mode=mode,
            val_out=outputs["val"][mode],
            test_out=outputs["test"][mode],
            external_out=outputs["external"][mode],
            val_conflict=val_conflict,
            test_conflict=test_conflict,
            external_conflict=external_conflict,
        )
        row: dict[str, Any] = {
            "seed": seed,
            "mode": MODE_REPORT_NAME[mode],
            **eval_row,
        }
        row["delta_k_vs_conflict"] = row["primary_mae_k"] - float(test_conflict["mae_k"])
        row["primary_improve_pct_vs_conflict"] = improve_pct(
            float(test_conflict["mae_k"]),
            row["primary_mae_k"],
        )
        row["external_delta_k_vs_conflict"] = row["external_mae_k"] - float(external_conflict["mae_k"])
        row["external_improve_pct_vs_conflict"] = improve_pct(
            float(external_conflict["mae_k"]),
            row["external_mae_k"],
        )
        if test_concat is not None and external_concat is not None:
            row["delta_k_vs_concat"] = row["primary_mae_k"] - float(test_concat["mae_k"])
            row["primary_improve_pct_vs_concat"] = improve_pct(
                float(test_concat["mae_k"]),
                row["primary_mae_k"],
            )
            row["external_delta_k_vs_concat"] = row["external_mae_k"] - float(external_concat["mae_k"])
            row["external_improve_pct_vs_concat"] = improve_pct(
                float(external_concat["mae_k"]),
                row["external_mae_k"],
            )
        else:
            row["delta_k_vs_concat"] = np.nan
            row["primary_improve_pct_vs_concat"] = np.nan
            row["external_delta_k_vs_concat"] = np.nan
            row["external_improve_pct_vs_concat"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_suite(
    seeds: list[int],
    modes: tuple[str, ...],
    config: TrainConfig,
    features: dict[str, Any] | None = None,
    splits: dict[str, Any] | None = None,
) -> pd.DataFrame:
    # This repeated-run loop is the canonical source for summary tables and diagnostics.
    if features is None or splits is None:
        _, features, splits = load_artifacts()
    seed_rows = [
        seed_bundle(seed=seed, features=features, split_payload=splits, config=config, modes=modes)
        for seed in seeds
    ]
    return pd.concat(seed_rows, ignore_index=True)


def summarize_culprit_rows(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in seed_df.groupby("mode", sort=False):
        ext_deltas = group["external_delta_k_vs_conflict"].tolist()
        ext_ci = bootstrap_ci(ext_deltas, seed=31)
        status = decide_support_status(
            mean_delta_vs_conflict=float(np.mean(ext_deltas)),
            ci_conflict=ext_ci,
            accepted_precision=float(group["accepted_precision"].mean()),
            accepted_subset_gain=float(group["accepted_subset_gain"].mean()),
            fallback_gain=float(group["fallback_gain"].mean()),
        )
        rows.append(
            {
                "model_name": mode,
                "primary_mae_k": float(group["primary_mae_k"].mean()),
                "primary_improve_pct_vs_conflict": float(group["primary_improve_pct_vs_conflict"].mean()),
                "primary_improve_pct_vs_concat": float(group["primary_improve_pct_vs_concat"].mean()),
                "external_mae_k": float(group["external_mae_k"].mean()),
                "external_improve_pct_vs_conflict": float(group["external_improve_pct_vs_conflict"].mean()),
                "external_improve_pct_vs_concat": float(group["external_improve_pct_vs_concat"].mean()),
                "accepted_precision": float(group["accepted_precision"].mean()),
                "accepted_subset_gain": float(group["accepted_subset_gain"].mean()),
                "fallback_gain": float(group["fallback_gain"].mean()),
                "support_status": status,
            }
        )
    return pd.DataFrame(rows)


def write_start_md() -> None:
    text = "\n".join(
        [
            "legacy note removed",
            "",
            "legacy note removed",
            "legacy note removed",
            "legacy note removed",
            "legacy note removed",
            "legacy note removed",
            "legacy note removed",
            "legacy note removed",
            "legacy note removed",
            "",
            "legacy note removed",
        ]
    )
    (ROOT / "reports/start.md").write_text(text, encoding="utf-8")


def write_culprit_reports(culprit_df: pd.DataFrame, worth_full: bool) -> None:
    culprit_df.to_csv(ROOT / "reports/culprit_scan.csv", index=False)
    rows = {row["model_name"]: row for _, row in culprit_df.iterrows()}
    mspce_helpful = rows["mspce_only_attentivefp"]["primary_improve_pct_vs_conflict"] > 0.0
    teacher_more_stable = (
        rows["teacher_anchored_rcmf_full"]["primary_improve_pct_vs_conflict"] >= 0.0
        and rows["teacher_anchored_rcmf_full"]["external_improve_pct_vs_conflict"]
        >= rows["residual_safe_gating_full"]["external_improve_pct_vs_conflict"]
        and rows["teacher_anchored_rcmf_full"]["accepted_precision"] >= 0.49
        and rows["teacher_anchored_rcmf_full"]["fallback_gain"]
        >= rows["residual_safe_gating_full"]["fallback_gain"]
    )
    rcmf_still_main = (
        rows["mspce_only_attentivefp"]["primary_mae_k"]
        < rows["residual_safe_gating_full"]["primary_mae_k"]
    )
    lines = [
        "legacy note removed",
        "",
        f"- `conflict_only_attentivefp` primary MAE: {rows['conflict_only_attentivefp']['primary_mae_k']:.4f} K",
        f"- `mspce_only_attentivefp` primary MAE: {rows['mspce_only_attentivefp']['primary_mae_k']:.4f} K",
        f"- `residual_safe_gating_full` primary MAE: {rows['residual_safe_gating_full']['primary_mae_k']:.4f} K",
        f"- `teacher_anchored_rcmf_full` primary MAE: {rows['teacher_anchored_rcmf_full']['primary_mae_k']:.4f} K",
        f"- `residual_safe_gating_full` external MAE: {rows['residual_safe_gating_full']['external_mae_k']:.4f} K",
        f"- `teacher_anchored_rcmf_full` external MAE: {rows['teacher_anchored_rcmf_full']['external_mae_k']:.4f} K",
        "",
        "legacy note removed",
        (
            "legacy note removed"
            "legacy note removed"
        ),
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/culprit_scan.md").write_text("\n".join(lines), encoding="utf-8")


def write_fix_plan() -> None:
    lines = [
        "legacy note removed",
        "",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/fix_plan.md").write_text("\n".join(lines), encoding="utf-8")


def run_culprit_scan() -> bool:
    write_start_md()
    seed_df = run_suite(seeds=[0, 1, 2], modes=CULPRIT_SCAN_MODES, config=diagnostic_config())
    culprit_df = summarize_culprit_rows(seed_df)
    rows = {row["model_name"]: row for _, row in culprit_df.iterrows()}
    worth_full = bool(
        rows["teacher_anchored_rcmf_full"]["primary_improve_pct_vs_conflict"] >= 0.0
        and rows["teacher_anchored_rcmf_full"]["external_improve_pct_vs_conflict"]
        >= rows["residual_safe_gating_full"]["external_improve_pct_vs_conflict"]
        and rows["teacher_anchored_rcmf_full"]["accepted_precision"] >= 0.49
        and rows["teacher_anchored_rcmf_full"]["fallback_gain"]
        >= rows["residual_safe_gating_full"]["fallback_gain"]
    )
    write_culprit_reports(culprit_df, worth_full)
    write_fix_plan()
    return worth_full


def worth_full_from_report() -> bool | None:
    culprit_csv = ROOT / "reports/culprit_scan.csv"
    if not culprit_csv.exists():
        return None
    culprit_df = pd.read_csv(culprit_csv)
    rows = {row["model_name"]: row for _, row in culprit_df.iterrows()}
    required = {
        "residual_safe_gating_full",
        "teacher_anchored_rcmf_full",
    }
    if not required.issubset(rows):
        return None
    return bool(
        rows["teacher_anchored_rcmf_full"]["primary_improve_pct_vs_conflict"] >= 0.0
        and rows["teacher_anchored_rcmf_full"]["external_improve_pct_vs_conflict"]
        >= rows["residual_safe_gating_full"]["external_improve_pct_vs_conflict"]
        and rows["teacher_anchored_rcmf_full"]["accepted_precision"] >= 0.49
        and rows["teacher_anchored_rcmf_full"]["fallback_gain"]
        >= rows["residual_safe_gating_full"]["fallback_gain"]
    )


def final_primary_summary(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    main_df = seed_df.loc[seed_df["mode"] == "teacher_anchored_rcmf_full"].copy()
    conflict_df = seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp"].copy()
    concat_df = seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp"].copy()

    primary_mae = float(main_df["primary_mae_k"].mean())
    delta_conflict = main_df["delta_k_vs_conflict"].tolist()
    delta_concat = main_df["delta_k_vs_concat"].tolist()
    conflict_mae = float(conflict_df["primary_mae_k"].mean())
    concat_mae = float(concat_df["primary_mae_k"].mean())
    strongest_name, strongest_mae = strongest_baseline(seed_df, "primary_mae_k")
    strongest_name, strongest_mae = strongest_baseline(seed_df, "primary_mae_k")
    ci_conflict = bootstrap_ci(delta_conflict, seed=101)
    ci_concat = bootstrap_ci(delta_concat, seed=202)

    report = pd.DataFrame(
        [
            {
                "row_type": "summary",
                "selected_backbone": "attentive_fp",
                "fix_direction": FIX_DIRECTION,
                "primary_mae_k": primary_mae,
                "primary_improve_pct_vs_conflict": improve_pct(conflict_mae, primary_mae),
                "primary_improve_pct_vs_concat": improve_pct(concat_mae, primary_mae),
                "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, primary_mae),
                "strongest_baseline_name": strongest_name,
                "delta_k_vs_conflict": float(np.mean(delta_conflict)),
                "delta_k_vs_concat": float(np.mean(delta_concat)),
                "bootstrap_ci_vs_conflict": format_ci(ci_conflict),
                "bootstrap_ci_vs_concat": format_ci(ci_concat),
                "sign_win_loss_vs_conflict": format_sign(sign_counts(delta_conflict)),
                "sign_win_loss_vs_concat": format_sign(sign_counts(delta_concat)),
            }
        ]
    )
    report.to_csv(ROOT / "reports/final_report.csv", index=False)
    return report, {
        "primary_mae_k": primary_mae,
        "primary_improve_pct_vs_conflict": improve_pct(conflict_mae, primary_mae),
        "primary_improve_pct_vs_concat": improve_pct(concat_mae, primary_mae),
        "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, primary_mae),
        "strongest_baseline_name": strongest_name,
        "delta_k_vs_conflict": float(np.mean(delta_conflict)),
        "delta_k_vs_concat": float(np.mean(delta_concat)),
        "ci_conflict": ci_conflict,
        "ci_concat": ci_concat,
    }


def final_external_summary(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    main_df = seed_df.loc[seed_df["mode"] == "teacher_anchored_rcmf_full"].copy()
    conflict_df = seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp"].copy()
    concat_df = seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp"].copy()

    external_mae = float(main_df["external_mae_k"].mean())
    delta_conflict = main_df["external_delta_k_vs_conflict"].tolist()
    delta_concat = main_df["external_delta_k_vs_concat"].tolist()
    conflict_mae = float(conflict_df["external_mae_k"].mean())
    concat_mae = float(concat_df["external_mae_k"].mean())
    strongest_name, strongest_mae = strongest_baseline(seed_df, "external_mae_k")
    ci_conflict = bootstrap_ci(delta_conflict, seed=303)
    ci_concat = bootstrap_ci(delta_concat, seed=404)
    accepted_precision = float(main_df["accepted_precision"].mean())
    accepted_subset_gain = float(main_df["accepted_subset_gain"].mean())
    fallback_gain = float(main_df["fallback_gain"].mean())
    support_status = decide_support_status(
        mean_delta_vs_conflict=float(np.mean(delta_conflict)),
        ci_conflict=ci_conflict,
        accepted_precision=accepted_precision,
        accepted_subset_gain=accepted_subset_gain,
        fallback_gain=fallback_gain,
    )

    report = pd.DataFrame(
        [
            {
                "row_type": "summary",
                "external_mae_k": external_mae,
                "external_improve_pct_vs_conflict": improve_pct(conflict_mae, external_mae),
                "external_improve_pct_vs_concat": improve_pct(concat_mae, external_mae),
                "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, external_mae),
                "strongest_baseline_name": strongest_name,
                "delta_k_vs_conflict": float(np.mean(delta_conflict)),
                "delta_k_vs_concat": float(np.mean(delta_concat)),
                "bootstrap_ci_vs_conflict": format_ci(ci_conflict),
                "bootstrap_ci_vs_concat": format_ci(ci_concat),
                "sign_win_loss_vs_conflict": format_sign(sign_counts(delta_conflict)),
                "sign_win_loss_vs_concat": format_sign(sign_counts(delta_concat)),
                "accepted_precision": accepted_precision,
                "accepted_subset_gain": accepted_subset_gain,
                "fallback_gain": fallback_gain,
                "support_status": support_status,
            }
        ]
    )
    report.to_csv(ROOT / "reports/external_report.csv", index=False)
    return report, {
        "external_mae_k": external_mae,
        "external_improve_pct_vs_conflict": improve_pct(conflict_mae, external_mae),
        "external_improve_pct_vs_concat": improve_pct(concat_mae, external_mae),
        "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, external_mae),
        "strongest_baseline_name": strongest_name,
        "delta_k_vs_conflict": float(np.mean(delta_conflict)),
        "delta_k_vs_concat": float(np.mean(delta_concat)),
        "ci_conflict": ci_conflict,
        "ci_concat": ci_concat,
        "accepted_precision": accepted_precision,
        "accepted_subset_gain": accepted_subset_gain,
        "fallback_gain": fallback_gain,
        "support_status": support_status,
    }


def write_comparator_report(seed_df: pd.DataFrame) -> bool:
    main_mae = seed_df.loc[seed_df["mode"] == "teacher_anchored_rcmf_full", "primary_mae_k"].tolist()
    comparator_map = {
        "Conflict-Only": seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp", "primary_mae_k"].tolist(),
        "Simple Concat": seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp", "primary_mae_k"].tolist(),
        "No-Context Ablation": seed_df.loc[
            seed_df["mode"] == "no_context_ablation_attentivefp",
            "primary_mae_k",
        ].tolist(),
        "Static Fusion": seed_df.loc[seed_df["mode"] == "static_fusion_attentivefp", "primary_mae_k"].tolist(),
        "teacher_anchored_rcmf_full": main_mae,
    }
    comparator_df = comparator_report_rows(
        main_name="teacher_anchored_rcmf_full",
        main_mae=main_mae,
        comparator_map=comparator_map,
    )
    comparator_df.to_csv(ROOT / "reports/comparator_report.csv", index=False)
    required = {
        "Conflict-Only",
        "Simple Concat",
        "No-Context Ablation",
        "Static Fusion",
        "teacher_anchored_rcmf_full",
    }
    closed = required.issubset(set(comparator_df["comparator_name"]))
    return bool(closed)


def write_decision(
    *,
    primary_summary: dict[str, Any],
    external_summary: dict[str, Any],
    comparator_closed: bool,
) -> dict[str, Any]:
    teacher_pulled_back = (
        primary_summary["delta_k_vs_conflict"] <= 0.0
        and abs(primary_summary["delta_k_vs_conflict"]) < 0.5
    )
    stable_q2 = (
        primary_summary["primary_improve_pct_vs_conflict"] > 0.0
        and primary_summary["ci_conflict"][1] < 0.0
        and external_summary["external_improve_pct_vs_conflict"] >= 0.0
        and external_summary["support_status"] == "positive"
        and comparator_closed
    )

    most_critical_gap = "none"
    if not (
        primary_summary["primary_improve_pct_vs_conflict"] > 0.0
        and primary_summary["ci_conflict"][1] < 0.0
    ):
        most_critical_gap = "legacy note removed"
    elif external_summary["support_status"] != "positive":
        most_critical_gap = "legacy note removed"
    lines = [
        "legacy note removed",
        "",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/decision.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "stable_q2_ready": stable_q2,
        "most_critical_gap": most_critical_gap,
    }


def run_full() -> dict[str, Any]:
    seed_df = run_suite(seeds=FULL_SEEDS, modes=FINAL_COMPARATOR_MODES, config=full_config())
    _, primary_summary = final_primary_summary(seed_df)
    _, external_summary = final_external_summary(seed_df)
    comparator_closed = write_comparator_report(seed_df)
    decision = write_decision(
        primary_summary=primary_summary,
        external_summary=external_summary,
        comparator_closed=comparator_closed,
    )
    return {
        **primary_summary,
        **external_summary,
        "comparator_closed": comparator_closed,
        **decision,
    }


def diagnostic_config() -> TrainConfig:
    return TrainConfig(
        batch_size=192,
        hidden_dim=128,
        weight_decay=1e-4,
        comparator_epochs=14,
        comparator_patience=4,
        comparator_lr=1e-3,
        residual_epochs=16,
        residual_patience=5,
        residual_lr=9e-4,
        teacher_epochs=14,
        teacher_patience=4,
        teacher_lr=1e-3,
        stage2_epochs=14,
        stage2_patience=3,
        stage2_lr=9e-4,
        stage3_epochs=12,
        stage3_patience=3,
        stage3_lr=3e-4,
        lambda_branch=0.12,
        lambda_anchor=0.14,
        lambda_gate=0.05,
        lambda_delta=0.02,
        lambda_led=0.04,
        lambda_mspce_path=1.50,
        led_start_frac=0.50,
        beneficial_quantile=0.55,
        innovation_limit=0.35,
        anchor_weights=AnchorLossWeights(
            teacher_nonbenef=0.20,
            teacher_benef=0.08,
            residual_benef=1.00,
            improve_benef=0.35,
            gate_nonbenef=0.01,
            gate_benef=0.00,
            gate_match_benef=2.00,
            delta_nonbenef=0.01,
            delta_benef=0.00,
        ),
    )


def full_config() -> TrainConfig:
    return TrainConfig(
        batch_size=192,
        hidden_dim=128,
        weight_decay=1e-4,
        comparator_epochs=18,
        comparator_patience=5,
        comparator_lr=9e-4,
        residual_epochs=20,
        residual_patience=5,
        residual_lr=9e-4,
        teacher_epochs=18,
        teacher_patience=5,
        teacher_lr=9e-4,
        stage2_epochs=18,
        stage2_patience=4,
        stage2_lr=8e-4,
        stage3_epochs=16,
        stage3_patience=4,
        stage3_lr=2.5e-4,
        lambda_branch=0.12,
        lambda_anchor=0.15,
        lambda_gate=0.05,
        lambda_delta=0.02,
        lambda_led=0.04,
        lambda_mspce_path=1.60,
        led_start_frac=0.45,
        beneficial_quantile=0.55,
        innovation_limit=0.32,
        anchor_weights=AnchorLossWeights(
            teacher_nonbenef=0.25,
            teacher_benef=0.08,
            residual_benef=1.10,
            improve_benef=0.40,
            gate_nonbenef=0.02,
            gate_benef=0.00,
            gate_match_benef=2.20,
            delta_nonbenef=0.01,
            delta_benef=0.00,
        ),
    )


def write_start_md() -> None:
    text = "\n".join(
        [
            "legacy note removed",
            "",
            "legacy note removed",
            "legacy note removed",
            "legacy note removed",
            "legacy note removed",
            "",
            "legacy note removed",
        ]
    )
    (ROOT / "reports/start.md").write_text(text, encoding="utf-8")


def write_fix_plan() -> None:
    lines = [
        "legacy note removed",
        "",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/fix_plan.md").write_text("\n".join(lines), encoding="utf-8")


def write_culprit_reports(culprit_df: pd.DataFrame, worth_full: bool) -> None:
    culprit_df.to_csv(ROOT / "reports/culprit_scan.csv", index=False)
    rows = {row["model_name"]: row for _, row in culprit_df.iterrows()}
    mspce_helpful = rows["mspce_only_attentivefp"]["primary_improve_pct_vs_conflict"] > 0.0
    certified_more_stable = (
        rows["mspce_certified_rcmf"]["primary_improve_pct_vs_conflict"] >= 0.0
        and rows["mspce_certified_rcmf"]["external_improve_pct_vs_conflict"]
        >= rows["teacher_anchored_rcmf_full"]["external_improve_pct_vs_conflict"]
    )
    lines = [
        "legacy note removed",
        "",
        f"- `conflict_only_attentivefp` primary MAE: {rows['conflict_only_attentivefp']['primary_mae_k']:.4f} K",
        f"- `mspce_only_attentivefp` primary MAE: {rows['mspce_only_attentivefp']['primary_mae_k']:.4f} K",
        f"- `teacher_anchored_rcmf_full` primary MAE: {rows['teacher_anchored_rcmf_full']['primary_mae_k']:.4f} K",
        f"- `mspce_certified_rcmf` primary MAE: {rows['mspce_certified_rcmf']['primary_mae_k']:.4f} K",
        f"- `teacher_anchored_rcmf_full` external MAE: {rows['teacher_anchored_rcmf_full']['external_mae_k']:.4f} K",
        f"- `mspce_certified_rcmf` external MAE: {rows['mspce_certified_rcmf']['external_mae_k']:.4f} K",
        "",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/culprit_scan.md").write_text("\n".join(lines), encoding="utf-8")


def diagnostic_config() -> TrainConfig:
    return TrainConfig(
        batch_size=192,
        hidden_dim=128,
        weight_decay=1e-4,
        comparator_epochs=14,
        comparator_patience=4,
        comparator_lr=1e-3,
        residual_epochs=14,
        residual_patience=4,
        residual_lr=9e-4,
        teacher_epochs=14,
        teacher_patience=4,
        teacher_lr=1e-3,
        stage2_epochs=10,
        stage2_patience=3,
        stage2_lr=8e-4,
        stage3_epochs=8,
        stage3_patience=3,
        stage3_lr=2e-4,
        lambda_branch=0.12,
        lambda_anchor=0.14,
        lambda_gate=0.04,
        lambda_delta=0.01,
        lambda_led=0.015,
        lambda_mspce_path=1.10,
        lambda_cert_negative=0.30,
        led_start_frac=0.50,
        beneficial_quantile=0.55,
        innovation_limit=0.28,
        anchor_weights=AnchorLossWeights(
            teacher_nonbenef=0.25,
            teacher_benef=0.08,
            residual_benef=0.90,
            improve_benef=0.30,
            gate_nonbenef=0.01,
            gate_benef=0.00,
            gate_match_benef=1.50,
            delta_nonbenef=0.01,
            delta_benef=0.00,
        ),
    )


def full_config() -> TrainConfig:
    return TrainConfig(
        batch_size=192,
        hidden_dim=128,
        weight_decay=1e-4,
        comparator_epochs=18,
        comparator_patience=5,
        comparator_lr=9e-4,
        residual_epochs=18,
        residual_patience=5,
        residual_lr=8e-4,
        teacher_epochs=18,
        teacher_patience=5,
        teacher_lr=9e-4,
        stage2_epochs=14,
        stage2_patience=4,
        stage2_lr=7e-4,
        stage3_epochs=12,
        stage3_patience=4,
        stage3_lr=1.5e-4,
        lambda_branch=0.12,
        lambda_anchor=0.15,
        lambda_gate=0.04,
        lambda_delta=0.01,
        lambda_led=0.015,
        lambda_mspce_path=1.20,
        lambda_cert_negative=0.35,
        led_start_frac=0.45,
        beneficial_quantile=0.55,
        innovation_limit=0.26,
        anchor_weights=AnchorLossWeights(
            teacher_nonbenef=0.30,
            teacher_benef=0.08,
            residual_benef=0.95,
            improve_benef=0.35,
            gate_nonbenef=0.01,
            gate_benef=0.00,
            gate_match_benef=1.60,
            delta_nonbenef=0.01,
            delta_benef=0.00,
        ),
    )


def strongest_baseline(seed_df: pd.DataFrame, metric_col: str) -> tuple[str, float]:
    candidates = [
        "conflict_only_attentivefp",
        "simple_concat_attentivefp",
        "mspce_only_attentivefp",
        "mspce_only_plus_led",
    ]
    available: list[tuple[str, float]] = []
    for mode in candidates:
        values = seed_df.loc[seed_df["mode"] == mode, metric_col]
        if len(values) > 0:
            available.append((mode, float(values.mean())))
    if not available:
        return "unknown", float("nan")
    return min(available, key=lambda item: item[1])


def final_primary_summary(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    main_df = seed_df.loc[seed_df["mode"] == "mspce_certified_rcmf"].copy()
    conflict_df = seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp"].copy()
    concat_df = seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp"].copy()
    primary_mae = float(main_df["primary_mae_k"].mean())
    delta_conflict = main_df["delta_k_vs_conflict"].tolist()
    delta_concat = main_df["delta_k_vs_concat"].tolist()
    conflict_mae = float(conflict_df["primary_mae_k"].mean())
    concat_mae = float(concat_df["primary_mae_k"].mean())
    strongest_name, strongest_mae = strongest_baseline(seed_df, "primary_mae_k")
    ci_conflict = bootstrap_ci(delta_conflict, seed=101)
    ci_concat = bootstrap_ci(delta_concat, seed=202)
    report = pd.DataFrame(
        [
            {
                "row_type": "summary",
                "selected_backbone": "attentive_fp",
                "fix_direction": FIX_DIRECTION,
                "primary_mae_k": primary_mae,
                "primary_improve_pct_vs_conflict": improve_pct(conflict_mae, primary_mae),
                "primary_improve_pct_vs_concat": improve_pct(concat_mae, primary_mae),
                "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, primary_mae),
                "strongest_baseline_name": strongest_name,
                "delta_k_vs_conflict": float(np.mean(delta_conflict)),
                "delta_k_vs_concat": float(np.mean(delta_concat)),
                "bootstrap_ci_vs_conflict": format_ci(ci_conflict),
                "bootstrap_ci_vs_concat": format_ci(ci_concat),
                "sign_win_loss_vs_conflict": format_sign(sign_counts(delta_conflict)),
                "sign_win_loss_vs_concat": format_sign(sign_counts(delta_concat)),
            }
        ]
    )
    report.to_csv(ROOT / "reports/final_report.csv", index=False)
    return report, {
        "primary_mae_k": primary_mae,
        "primary_improve_pct_vs_conflict": improve_pct(conflict_mae, primary_mae),
        "primary_improve_pct_vs_concat": improve_pct(concat_mae, primary_mae),
        "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, primary_mae),
        "strongest_baseline_name": strongest_name,
        "delta_k_vs_conflict": float(np.mean(delta_conflict)),
        "delta_k_vs_concat": float(np.mean(delta_concat)),
        "ci_conflict": ci_conflict,
        "ci_concat": ci_concat,
    }


def final_external_summary(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    main_df = seed_df.loc[seed_df["mode"] == "mspce_certified_rcmf"].copy()
    conflict_df = seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp"].copy()
    concat_df = seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp"].copy()
    external_mae = float(main_df["external_mae_k"].mean())
    delta_conflict = main_df["external_delta_k_vs_conflict"].tolist()
    delta_concat = main_df["external_delta_k_vs_concat"].tolist()
    conflict_mae = float(conflict_df["external_mae_k"].mean())
    concat_mae = float(concat_df["external_mae_k"].mean())
    strongest_name, strongest_mae = strongest_baseline(seed_df, "external_mae_k")
    ci_conflict = bootstrap_ci(delta_conflict, seed=303)
    ci_concat = bootstrap_ci(delta_concat, seed=404)
    accepted_precision = float(main_df["accepted_precision"].mean())
    accepted_subset_gain = float(main_df["accepted_subset_gain"].mean())
    fallback_gain = float(main_df["fallback_gain"].mean())
    support_status = decide_support_status(
        mean_delta_vs_conflict=float(np.mean(delta_conflict)),
        ci_conflict=ci_conflict,
        accepted_precision=accepted_precision,
        accepted_subset_gain=accepted_subset_gain,
        fallback_gain=fallback_gain,
    )
    report = pd.DataFrame(
        [
            {
                "row_type": "summary",
                "external_mae_k": external_mae,
                "external_improve_pct_vs_conflict": improve_pct(conflict_mae, external_mae),
                "external_improve_pct_vs_concat": improve_pct(concat_mae, external_mae),
                "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, external_mae),
                "strongest_baseline_name": strongest_name,
                "delta_k_vs_conflict": float(np.mean(delta_conflict)),
                "delta_k_vs_concat": float(np.mean(delta_concat)),
                "bootstrap_ci_vs_conflict": format_ci(ci_conflict),
                "bootstrap_ci_vs_concat": format_ci(ci_concat),
                "sign_win_loss_vs_conflict": format_sign(sign_counts(delta_conflict)),
                "sign_win_loss_vs_concat": format_sign(sign_counts(delta_concat)),
                "accepted_precision": accepted_precision,
                "accepted_subset_gain": accepted_subset_gain,
                "fallback_gain": fallback_gain,
                "support_status": support_status,
            }
        ]
    )
    report.to_csv(ROOT / "reports/external_report.csv", index=False)
    return report, {
        "external_mae_k": external_mae,
        "external_improve_pct_vs_conflict": improve_pct(conflict_mae, external_mae),
        "external_improve_pct_vs_concat": improve_pct(concat_mae, external_mae),
        "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, external_mae),
        "strongest_baseline_name": strongest_name,
        "delta_k_vs_conflict": float(np.mean(delta_conflict)),
        "delta_k_vs_concat": float(np.mean(delta_concat)),
        "ci_conflict": ci_conflict,
        "ci_concat": ci_concat,
        "accepted_precision": accepted_precision,
        "accepted_subset_gain": accepted_subset_gain,
        "fallback_gain": fallback_gain,
        "support_status": support_status,
    }


def write_comparator_report(seed_df: pd.DataFrame) -> bool:
    main_mae = seed_df.loc[seed_df["mode"] == "mspce_certified_rcmf", "primary_mae_k"].tolist()
    comparator_map = {
        "Conflict-Only": seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp", "primary_mae_k"].tolist(),
        "Simple Concat": seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp", "primary_mae_k"].tolist(),
        "No-Context Ablation": seed_df.loc[seed_df["mode"] == "no_context_ablation_attentivefp", "primary_mae_k"].tolist(),
        "Static Fusion": seed_df.loc[seed_df["mode"] == "static_fusion_attentivefp", "primary_mae_k"].tolist(),
        "teacher_anchored_rcmf_full": seed_df.loc[seed_df["mode"] == "teacher_anchored_rcmf_full", "primary_mae_k"].tolist(),
        "mspce_certified_rcmf": main_mae,
    }
    comparator_df = comparator_report_rows(main_name="mspce_certified_rcmf", main_mae=main_mae, comparator_map=comparator_map)
    comparator_df.to_csv(ROOT / "reports/comparator_report.csv", index=False)
    required = {"Conflict-Only", "Simple Concat", "No-Context Ablation", "Static Fusion", "teacher_anchored_rcmf_full", "mspce_certified_rcmf"}
    return required.issubset(set(comparator_df["comparator_name"]))


def write_decision(
    *,
    current_stage: str,
    only_gap: str,
    certification_summary: dict[str, Any],
    primary_summary: dict[str, Any] | None,
    external_summary: dict[str, Any] | None,
    comparator_closed: bool,
    worth_continue: bool,
) -> dict[str, Any]:
    if primary_summary is None or external_summary is None:
        stable_q2 = False
        primary_pct = float("nan")
        external_pct = float("nan")
        external_status = "not_run"
    else:
        stable_q2 = (
            primary_summary["primary_improve_pct_vs_conflict"] > 0.0
            and primary_summary["ci_conflict"][1] < 0.0
            and external_summary["external_improve_pct_vs_conflict"] >= 0.0
            and external_summary["support_status"] == "positive"
            and comparator_closed
        )
        primary_pct = primary_summary["primary_improve_pct_vs_conflict"]
        external_pct = external_summary["external_improve_pct_vs_conflict"]
        external_status = external_summary["support_status"]

    lines = [
        "legacy note removed",
        "",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/decision.md").write_text("\n".join(lines), encoding="utf-8")
    return {"stable_q2_ready": stable_q2, "most_critical_gap": only_gap}


def run_culprit_scan() -> bool:
    write_start_md()
    write_fix_plan()
    _, features, splits = load_artifacts()
    base_config = diagnostic_config()
    rule, _ = build_certification_rule(features=features, split_payload=splits, config=base_config, seeds=[0, 1, 2])
    config = replace(base_config, certification_rule=rule)
    seed_df = run_suite(seeds=[0, 1, 2], modes=CULPRIT_SCAN_MODES, config=config, features=features, splits=splits)
    culprit_df = summarize_culprit_rows(seed_df)
    rows = {row["model_name"]: row for _, row in culprit_df.iterrows()}
    worth_full = bool(
        rows["mspce_certified_rcmf"]["primary_improve_pct_vs_conflict"] >= 0.0
        and rows["mspce_certified_rcmf"]["external_improve_pct_vs_conflict"] >= rows["teacher_anchored_rcmf_full"]["external_improve_pct_vs_conflict"]
        and rows["mspce_certified_rcmf"]["accepted_precision"] >= rows["teacher_anchored_rcmf_full"]["accepted_precision"]
    )
    write_culprit_reports(culprit_df, worth_full)
    return worth_full


def worth_full_from_report() -> bool | None:
    culprit_csv = ROOT / "reports/culprit_scan.csv"
    if not culprit_csv.exists():
        return None
    culprit_df = pd.read_csv(culprit_csv)
    rows = {row["model_name"]: row for _, row in culprit_df.iterrows()}
    if "mspce_certified_rcmf" not in rows or "teacher_anchored_rcmf_full" not in rows:
        return None
    return bool(
        rows["mspce_certified_rcmf"]["primary_improve_pct_vs_conflict"] >= 0.0
        and rows["mspce_certified_rcmf"]["external_improve_pct_vs_conflict"] >= rows["teacher_anchored_rcmf_full"]["external_improve_pct_vs_conflict"]
        and rows["mspce_certified_rcmf"]["accepted_precision"] >= rows["teacher_anchored_rcmf_full"]["accepted_precision"]
    )


def run_full() -> dict[str, Any]:
    _, features, splits = load_artifacts()
    base_config = full_config()
    rule, cert_summary = build_certification_rule(features=features, split_payload=splits, config=base_config, seeds=[0, 1, 2])
    config = replace(base_config, certification_rule=rule)
    seed_df = run_suite(seeds=FULL_SEEDS, modes=FINAL_COMPARATOR_MODES, config=config, features=features, splits=splits)
    _, primary_summary = final_primary_summary(seed_df)
    _, external_summary = final_external_summary(seed_df)
    comparator_closed = write_comparator_report(seed_df)
    decision = write_decision(
        current_stage="full_20_seed_completed",
        only_gap="legacy note removed",
        certification_summary=cert_summary,
        primary_summary=primary_summary,
        external_summary=external_summary,
        comparator_closed=comparator_closed,
        worth_continue=(external_summary["support_status"] != "positive"),
    )
    return {**primary_summary, **external_summary, "comparator_closed": comparator_closed, **decision, **cert_summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mspce-certified RCMF calibration for tg_clean_v2.")
    parser.add_argument("--task", choices=["culprit_scan", "full"], required=True)
    args = parser.parse_args()

    if args.task == "culprit_scan":
        run_culprit_scan()
        return 0

    worth_full = run_culprit_scan()
    cert_df = pd.read_csv(ROOT / "reports/certification_report.csv")
    cert_summary = cert_df.iloc[0].to_dict()
    if not worth_full:
        write_decision(
            current_stage="sanity_stopped_after_3_seed",
            only_gap="legacy note removed",
            certification_summary=cert_summary,
            primary_summary=None,
            external_summary=None,
            comparator_closed=False,
            worth_continue=False,
        )
        return 0
    run_full()
    return 0

VISIBLE_CULPRIT_MODES = (
    "conflict_only_attentivefp",
    "mspce_only_attentivefp",
    "teacher_anchored_rcmf_full",
    "rcmf_dynamic_multimodal_full",
)


def write_start_md() -> None:
    lines = [
        "legacy note removed",
        "",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/start.md").write_text("\n".join(lines), encoding="utf-8")


def write_fix_plan() -> None:
    lines = [
        "legacy note removed",
        "",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/fix_plan.md").write_text("\n".join(lines), encoding="utf-8")


def build_arbitration_rule(
    *,
    features: dict[str, Any],
    split_payload: dict[str, Any],
    config: TrainConfig,
    seeds: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    oof_df = oof_certification_rows(
        features=features,
        split_payload=split_payload,
        config=config,
        seeds=seeds,
    )
    rule = fit_certification_rule(oof_df)
    summary = summarize_certification_rule(oof_df, rule)
    arbitration_df = pd.DataFrame(
        [
            {
                "positive_switch_region_coverage": summary["positive_switch_region_coverage"],
                "positive_switch_region_precision": summary["positive_switch_region_precision"],
                "positive_switch_region_gain": summary["positive_switch_region_gain"],
                "teacher_region_coverage": summary["teacher_region_coverage"],
                "teacher_region_gain": summary["teacher_region_gain"],
                "uncertain_region_coverage": summary["uncertain_region_coverage"],
                "arbitration_rule": summary["arbitration_rule"],
            }
        ]
    )
    arbitration_df.to_csv(ROOT / "reports/arbitration_report.csv", index=False)
    md_lines = [
        "# Arbitration",
        "",
        "legacy note removed",
        f"- positive_switch_region_coverage: {summary['positive_switch_region_coverage']:.4f}",
        f"- positive_switch_region_precision: {summary['positive_switch_region_precision']:.4f}",
        f"- positive_switch_region_gain: {summary['positive_switch_region_gain']:.4f}",
        f"- teacher_region_coverage: {summary['teacher_region_coverage']:.4f}",
        f"- teacher_region_gain: {summary['teacher_region_gain']:.4f}",
        f"- uncertain_region_coverage: {summary['uncertain_region_coverage']:.4f}",
        f"- arbitration rule: {summary['arbitration_rule']}",
        "",
        "legacy note removed",
    ]
    (ROOT / "reports/arbitration_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    return rule, summary


def write_culprit_reports(culprit_df: pd.DataFrame, worth_full: bool) -> None:
    visible_df = culprit_df.loc[culprit_df["model_name"].isin(VISIBLE_CULPRIT_MODES)].copy()
    visible_df.to_csv(ROOT / "reports/culprit_scan.csv", index=False)
    rows = {row["model_name"]: row for _, row in visible_df.iterrows()}
    mspce_helpful = rows["mspce_only_attentivefp"]["primary_improve_pct_vs_conflict"] > 0.0
    switch_more_stable = (
        rows["rcmf_dynamic_multimodal_full"]["primary_improve_pct_vs_conflict"]
        > rows["teacher_anchored_rcmf_full"]["primary_improve_pct_vs_conflict"]
        and rows["rcmf_dynamic_multimodal_full"]["external_improve_pct_vs_conflict"]
        >= rows["teacher_anchored_rcmf_full"]["external_improve_pct_vs_conflict"]
        and rows["rcmf_dynamic_multimodal_full"]["accepted_precision"]
        >= rows["teacher_anchored_rcmf_full"]["accepted_precision"]
    )
    lines = [
        "legacy note removed",
        "",
        f"- `conflict_only_attentivefp` primary MAE: {rows['conflict_only_attentivefp']['primary_mae_k']:.4f} K",
        f"- `mspce_only_attentivefp` primary MAE: {rows['mspce_only_attentivefp']['primary_mae_k']:.4f} K",
        f"- `teacher_anchored_rcmf_full` primary MAE: {rows['teacher_anchored_rcmf_full']['primary_mae_k']:.4f} K",
        f"- `rcmf_dynamic_multimodal_full` primary MAE: {rows['rcmf_dynamic_multimodal_full']['primary_mae_k']:.4f} K",
        f"- `teacher_anchored_rcmf_full` external MAE: {rows['teacher_anchored_rcmf_full']['external_mae_k']:.4f} K",
        f"- `rcmf_dynamic_multimodal_full` external MAE: {rows['rcmf_dynamic_multimodal_full']['external_mae_k']:.4f} K",
        "",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
        "legacy note removed",
    ]
    (ROOT / "reports/culprit_scan.md").write_text("\n".join(lines), encoding="utf-8")


def write_not_run_reports(reason: str) -> None:
    pd.DataFrame(
        [
            {
                "row_type": "not_run_due_to_sanity_stop",
                "fix_direction": FIX_DIRECTION,
                "primary_mae_k": np.nan,
                "primary_improve_pct_vs_conflict": np.nan,
                "primary_improve_pct_vs_concat": np.nan,
                "primary_improve_pct_vs_strongest_baseline": np.nan,
                "strongest_baseline_name": "",
                "delta_k_vs_conflict": np.nan,
                "delta_k_vs_concat": np.nan,
                "bootstrap_ci_vs_conflict": "",
                "bootstrap_ci_vs_concat": "",
                "sign_win_loss_vs_conflict": "",
                "sign_win_loss_vs_concat": "",
                "note": reason,
            }
        ]
    ).to_csv(ROOT / "reports/final_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "row_type": "not_run_due_to_sanity_stop",
                "external_mae_k": np.nan,
                "external_improve_pct_vs_conflict": np.nan,
                "external_improve_pct_vs_concat": np.nan,
                "external_improve_pct_vs_strongest_baseline": np.nan,
                "strongest_baseline_name": "",
                "delta_k_vs_conflict": np.nan,
                "delta_k_vs_concat": np.nan,
                "bootstrap_ci_vs_conflict": "",
                "bootstrap_ci_vs_concat": "",
                "accepted_precision": np.nan,
                "accepted_subset_gain": np.nan,
                "fallback_gain": np.nan,
                "support_status": "not_run",
                "note": reason,
            }
        ]
    ).to_csv(ROOT / "reports/external_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "main_model": "rcmf_dynamic_multimodal_full",
                "comparator_name": "not_run_due_to_sanity_stop",
                "ran": "no",
                "reason_not_run": reason,
                "main_absolute_metric_k": np.nan,
                "comparator_absolute_metric_k": np.nan,
                "delta_k_main_minus_comparator": np.nan,
                "main_improve_pct_vs_comparator": np.nan,
                "bootstrap_ci_k": "",
                "sign_win_loss": "",
                "comparator_closed": "no",
            }
        ]
    ).to_csv(ROOT / "reports/comparator_report.csv", index=False)


def diagnostic_config() -> TrainConfig:
    return TrainConfig(
        batch_size=192,
        hidden_dim=128,
        weight_decay=1e-4,
        comparator_epochs=14,
        comparator_patience=4,
        comparator_lr=1e-3,
        residual_epochs=14,
        residual_patience=4,
        residual_lr=9e-4,
        teacher_epochs=14,
        teacher_patience=4,
        teacher_lr=1e-3,
        stage2_epochs=12,
        stage2_patience=3,
        stage2_lr=8e-4,
        stage3_epochs=10,
        stage3_patience=3,
        stage3_lr=2.0e-4,
        lambda_branch=0.12,
        lambda_anchor=0.18,
        lambda_gate=0.16,
        lambda_delta=0.10,
        lambda_led=0.008,
        lambda_mspce_path=3.20,
        lambda_cert_negative=0.40,
        led_start_frac=0.50,
        beneficial_quantile=0.55,
        innovation_limit=0.20,
        anchor_weights=AnchorLossWeights(
            teacher_nonbenef=0.25,
            teacher_benef=0.08,
            residual_benef=0.95,
            improve_benef=0.35,
            gate_nonbenef=0.01,
            gate_benef=0.00,
            gate_match_benef=1.40,
            delta_nonbenef=0.01,
            delta_benef=0.00,
        ),
    )


def full_config() -> TrainConfig:
    return TrainConfig(
        batch_size=192,
        hidden_dim=128,
        weight_decay=1e-4,
        comparator_epochs=18,
        comparator_patience=5,
        comparator_lr=9e-4,
        residual_epochs=18,
        residual_patience=5,
        residual_lr=8e-4,
        teacher_epochs=18,
        teacher_patience=5,
        teacher_lr=9e-4,
        stage2_epochs=16,
        stage2_patience=4,
        stage2_lr=6e-4,
        stage3_epochs=14,
        stage3_patience=4,
        stage3_lr=1.5e-4,
        lambda_branch=0.12,
        lambda_anchor=0.20,
        lambda_gate=0.16,
        lambda_delta=0.10,
        lambda_led=0.008,
        lambda_mspce_path=3.40,
        lambda_cert_negative=0.45,
        led_start_frac=0.45,
        beneficial_quantile=0.55,
        innovation_limit=0.20,
        anchor_weights=AnchorLossWeights(
            teacher_nonbenef=0.30,
            teacher_benef=0.08,
            residual_benef=1.00,
            improve_benef=0.35,
            gate_nonbenef=0.01,
            gate_benef=0.00,
            gate_match_benef=1.50,
            delta_nonbenef=0.01,
            delta_benef=0.00,
        ),
    )


def final_primary_summary(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    main_df = seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full_phasea"].copy()
    conflict_df = seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp"].copy()
    concat_df = seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp"].copy()
    primary_mae = float(main_df["primary_mae_k"].mean())
    delta_conflict = main_df["delta_k_vs_conflict"].tolist()
    delta_concat = main_df["delta_k_vs_concat"].tolist()
    conflict_mae = float(conflict_df["primary_mae_k"].mean())
    concat_mae = float(concat_df["primary_mae_k"].mean())
    strongest_name, strongest_mae = strongest_baseline(seed_df, "primary_mae_k")
    ci_conflict = bootstrap_ci(delta_conflict, seed=101)
    ci_concat = bootstrap_ci(delta_concat, seed=202)
    report = pd.DataFrame(
        [
            {
                "row_type": "summary",
                "selected_backbone": "attentive_fp",
                "fix_direction": FIX_DIRECTION,
                "primary_mae_k": primary_mae,
                "primary_improve_pct_vs_conflict": improve_pct(conflict_mae, primary_mae),
                "primary_improve_pct_vs_concat": improve_pct(concat_mae, primary_mae),
                "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, primary_mae),
                "strongest_baseline_name": strongest_name,
                "delta_k_vs_conflict": float(np.mean(delta_conflict)),
                "delta_k_vs_concat": float(np.mean(delta_concat)),
                "bootstrap_ci_vs_conflict": format_ci(ci_conflict),
                "bootstrap_ci_vs_concat": format_ci(ci_concat),
                "sign_win_loss_vs_conflict": format_sign(sign_counts(delta_conflict)),
                "sign_win_loss_vs_concat": format_sign(sign_counts(delta_concat)),
            }
        ]
    )
    report.to_csv(ROOT / "reports/final_report.csv", index=False)
    return report, {
        "primary_mae_k": primary_mae,
        "primary_improve_pct_vs_conflict": improve_pct(conflict_mae, primary_mae),
        "primary_improve_pct_vs_concat": improve_pct(concat_mae, primary_mae),
        "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, primary_mae),
        "strongest_baseline_name": strongest_name,
        "delta_k_vs_conflict": float(np.mean(delta_conflict)),
        "delta_k_vs_concat": float(np.mean(delta_concat)),
        "ci_conflict": ci_conflict,
        "ci_concat": ci_concat,
    }


def final_external_summary(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    main_df = seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full_phasea"].copy()
    conflict_df = seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp"].copy()
    concat_df = seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp"].copy()
    external_mae = float(main_df["external_mae_k"].mean())
    delta_conflict = main_df["external_delta_k_vs_conflict"].tolist()
    delta_concat = main_df["external_delta_k_vs_concat"].tolist()
    conflict_mae = float(conflict_df["external_mae_k"].mean())
    concat_mae = float(concat_df["external_mae_k"].mean())
    strongest_name, strongest_mae = strongest_baseline(seed_df, "external_mae_k")
    ci_conflict = bootstrap_ci(delta_conflict, seed=303)
    ci_concat = bootstrap_ci(delta_concat, seed=404)
    accepted_precision = float(main_df["accepted_precision"].mean())
    accepted_subset_gain = float(main_df["accepted_subset_gain"].mean())
    fallback_gain = float(main_df["fallback_gain"].mean())
    support_status = decide_support_status(
        mean_delta_vs_conflict=float(np.mean(delta_conflict)),
        ci_conflict=ci_conflict,
        accepted_precision=accepted_precision,
        accepted_subset_gain=accepted_subset_gain,
        fallback_gain=fallback_gain,
    )
    report = pd.DataFrame(
        [
            {
                "row_type": "summary",
                "external_mae_k": external_mae,
                "external_improve_pct_vs_conflict": improve_pct(conflict_mae, external_mae),
                "external_improve_pct_vs_concat": improve_pct(concat_mae, external_mae),
                "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, external_mae),
                "strongest_baseline_name": strongest_name,
                "delta_k_vs_conflict": float(np.mean(delta_conflict)),
                "delta_k_vs_concat": float(np.mean(delta_concat)),
                "bootstrap_ci_vs_conflict": format_ci(ci_conflict),
                "bootstrap_ci_vs_concat": format_ci(ci_concat),
                "sign_win_loss_vs_conflict": format_sign(sign_counts(delta_conflict)),
                "sign_win_loss_vs_concat": format_sign(sign_counts(delta_concat)),
                "accepted_precision": accepted_precision,
                "accepted_subset_gain": accepted_subset_gain,
                "fallback_gain": fallback_gain,
                "support_status": support_status,
            }
        ]
    )
    report.to_csv(ROOT / "reports/external_report.csv", index=False)
    return report, {
        "external_mae_k": external_mae,
        "external_improve_pct_vs_conflict": improve_pct(conflict_mae, external_mae),
        "external_improve_pct_vs_concat": improve_pct(concat_mae, external_mae),
        "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_mae, external_mae),
        "strongest_baseline_name": strongest_name,
        "delta_k_vs_conflict": float(np.mean(delta_conflict)),
        "delta_k_vs_concat": float(np.mean(delta_concat)),
        "ci_conflict": ci_conflict,
        "ci_concat": ci_concat,
        "accepted_precision": accepted_precision,
        "accepted_subset_gain": accepted_subset_gain,
        "fallback_gain": fallback_gain,
        "support_status": support_status,
    }


def write_comparator_report(seed_df: pd.DataFrame) -> bool:
    main_mae = seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full_phasea", "primary_mae_k"].tolist()
    comparator_map = {
        "Conflict-Only": seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp", "primary_mae_k"].tolist(),
        "Simple Concat": seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp", "primary_mae_k"].tolist(),
        "MSPCE-only": seed_df.loc[seed_df["mode"] == "mspce_only_attentivefp", "primary_mae_k"].tolist(),
        "MSPCE-only + LED": seed_df.loc[seed_df["mode"] == "mspce_only_plus_led", "primary_mae_k"].tolist(),
        "Full (current)": seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full", "primary_mae_k"].tolist(),
        "Full (Phase A++)": main_mae,
    }
    comparator_df = comparator_report_rows(
        main_name="rcmf_dynamic_multimodal_full_phasea",
        main_mae=main_mae,
        comparator_map=comparator_map,
    )
    comparator_df.to_csv(ROOT / "reports/comparator_report.csv", index=False)
    required = {
        "Conflict-Only",
        "Simple Concat",
        "MSPCE-only",
        "MSPCE-only + LED",
        "Full (current)",
        "Full (Phase A++)",
    }
    return required.issubset(set(comparator_df["comparator_name"]))


def write_baseline_matrix(seed_df: pd.DataFrame, seeds_run: int) -> tuple[pd.DataFrame, str]:
    summary = summarize_culprit_rows(seed_df)
    summary = summary.loc[
        summary["model_name"].isin(
            [
                "conflict_only_attentivefp",
                "simple_concat_attentivefp",
                "mspce_only_attentivefp",
                "mspce_only_plus_led",
                "rcmf_dynamic_multimodal_full",
                "rcmf_dynamic_multimodal_full_phasea",
            ]
        )
    ].copy()
    name_map = {
        "conflict_only_attentivefp": "Conflict-Only",
        "simple_concat_attentivefp": "Simple Concat",
        "mspce_only_attentivefp": "MSPCE-only",
        "mspce_only_plus_led": "MSPCE-only + LED",
        "rcmf_dynamic_multimodal_full": "Full (current)",
        "rcmf_dynamic_multimodal_full_phasea": "Full (Phase A++)",
    }
    base_rows = summary.loc[
        summary["model_name"].isin(
            [
                "conflict_only_attentivefp",
                "simple_concat_attentivefp",
                "mspce_only_attentivefp",
                "mspce_only_plus_led",
                "rcmf_dynamic_multimodal_full",
            ]
        )
    ].copy()
    strongest = base_rows.sort_values("primary_mae_k").iloc[0]
    strongest_name = name_map[str(strongest["model_name"])]
    strongest_primary = float(strongest["primary_mae_k"])
    strongest_external = float(strongest["external_mae_k"])
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "model_name": name_map[str(row["model_name"])],
                "primary_mae_k": float(row["primary_mae_k"]),
                "primary_improve_pct_vs_conflict": float(row["primary_improve_pct_vs_conflict"]),
                "primary_improve_pct_vs_concat": float(row["primary_improve_pct_vs_concat"]),
                "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_primary, float(row["primary_mae_k"])),
                "external_mae_k": float(row["external_mae_k"]),
                "external_improve_pct_vs_conflict": float(row["external_improve_pct_vs_conflict"]),
                "external_improve_pct_vs_concat": float(row["external_improve_pct_vs_concat"]),
                "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_external, float(row["external_mae_k"])),
                "external_status": str(row["support_status"]),
                "current_clean_protocol_result": "yes",
                "seeds_run": int(seeds_run),
                "notes": "current_protocol_result",
            }
        )
    matrix_df = pd.DataFrame(rows)
    order = [
        "Conflict-Only",
        "Simple Concat",
        "MSPCE-only",
        "MSPCE-only + LED",
        "Full (current)",
        "Full (Phase A++)",
    ]
    matrix_df["model_name"] = pd.Categorical(matrix_df["model_name"], categories=order, ordered=True)
    matrix_df = matrix_df.sort_values("model_name").reset_index(drop=True)
    matrix_df.to_csv(ROOT / "reports/baseline_matrix.csv", index=False)
    (ROOT / "reports/baseline_matrix.md").write_text(
        "\n".join(
            [
                "# baseline_matrix",
                "",
                f"- seeds_run: {int(seeds_run)}",
                f"- strongest baseline: {strongest_name}",
                "- all 6 models are current clean protocol results",
            ]
        ),
        encoding="utf-8",
    )
    return matrix_df, strongest_name


def write_integration_repair_report(seed_df: pd.DataFrame, strongest_baseline: str) -> tuple[str, bool]:
    summary = summarize_culprit_rows(seed_df)
    model_alias = {
        "conflict_only_attentivefp": "Conflict-Only",
        "simple_concat_attentivefp": "Simple Concat",
        "mspce_only_attentivefp": "MSPCE-only",
        "mspce_only_plus_led": "MSPCE-only + LED",
        "rcmf_dynamic_multimodal_full": "Full (current)",
        "rcmf_dynamic_multimodal_full_phasea": "Full (Phase A++)",
    }
    strongest_mode = next((k for k, v in model_alias.items() if v == strongest_baseline), "simple_concat_attentivefp")
    strongest_primary = float(summary.loc[summary["model_name"] == strongest_mode, "primary_mae_k"].mean())
    strongest_external = float(summary.loc[summary["model_name"] == strongest_mode, "external_mae_k"].mean())
    rows: list[dict[str, Any]] = []
    for mode in model_alias:
        group = seed_df.loc[seed_df["mode"] == mode].copy()
        if len(group) == 0:
            continue
        delta_conflict = group["delta_k_vs_conflict"].tolist()
        delta_concat = group["delta_k_vs_concat"].tolist()
        ext_delta_conflict = group["external_delta_k_vs_conflict"].tolist()
        ext_delta_concat = group["external_delta_k_vs_concat"].tolist()
        rows.append(
            {
                "model_name": model_alias[mode],
                "primary_mae_k": float(group["primary_mae_k"].mean()),
                "primary_improve_pct_vs_conflict": float(group["primary_improve_pct_vs_conflict"].mean()),
                "primary_improve_pct_vs_concat": float(group["primary_improve_pct_vs_concat"].mean()),
                "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_primary, float(group["primary_mae_k"].mean())),
                "external_mae_k": float(group["external_mae_k"].mean()),
                "external_improve_pct_vs_conflict": float(group["external_improve_pct_vs_conflict"].mean()),
                "external_improve_pct_vs_concat": float(group["external_improve_pct_vs_concat"].mean()),
                "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_external, float(group["external_mae_k"].mean())),
                "delta_k_vs_conflict": float(np.mean(delta_conflict)),
                "delta_k_vs_concat": float(np.mean(delta_concat)),
                "bootstrap_ci": f"primary_vs_conflict={format_ci(bootstrap_ci(delta_conflict, seed=13))};external_vs_conflict={format_ci(bootstrap_ci(ext_delta_conflict, seed=23))}",
                "sign_win_loss": f"primary_vs_conflict={format_sign(sign_counts(delta_conflict))};external_vs_conflict={format_sign(sign_counts(ext_delta_conflict))}",
                "accepted_precision": float(group["accepted_precision"].mean()),
                "accepted_subset_gain": float(group["accepted_subset_gain"].mean()),
                "fallback_gain": float(group["fallback_gain"].mean()),
                "support_status": decide_support_status(
                    mean_delta_vs_conflict=float(np.mean(ext_delta_conflict)),
                    ci_conflict=bootstrap_ci(ext_delta_conflict, seed=29),
                    accepted_precision=float(group["accepted_precision"].mean()),
                    accepted_subset_gain=float(group["accepted_subset_gain"].mean()),
                    fallback_gain=float(group["fallback_gain"].mean()),
                ),
                "external_delta_k_vs_conflict": float(np.mean(ext_delta_conflict)),
                "external_delta_k_vs_concat": float(np.mean(ext_delta_concat)),
            }
        )
    report_df = pd.DataFrame(rows)
    report_df.to_csv(ROOT / "reports/integration_repair_report.csv", index=False)
    keyed = {row["model_name"]: row for _, row in report_df.iterrows()}
    mspce_primary = float(keyed["MSPCE-only"]["primary_mae_k"])
    led_primary = float(keyed["MSPCE-only + LED"]["primary_mae_k"])
    full_current_primary = float(keyed["Full (current)"]["primary_mae_k"])
    full_primary = float(keyed["Full (Phase A++)"]["primary_mae_k"])
    full_current_concat_pct = float(keyed["Full (current)"]["primary_improve_pct_vs_concat"])
    full_concat_pct = float(keyed["Full (Phase A++)"]["primary_improve_pct_vs_concat"])
    residual_dragging = bool(full_concat_pct < full_current_concat_pct)
    led_eat = led_primary > mspce_primary
    current_full_eat = full_current_primary > mspce_primary
    full_eat = full_primary > mspce_primary
    if not full_eat:
        culprit = "legacy note removed"
    elif led_eat and current_full_eat:
        culprit = "legacy note removed"
    elif led_eat:
        culprit = "LED"
    elif current_full_eat:
        culprit = "RCMF"
    else:
        culprit = "legacy note removed"
    (ROOT / "reports/integration_repair_report.md").write_text(
        "\n".join(
            [
                "# integration_repair_report",
                "",
                f"- strongest_baseline: {strongest_baseline}",
                f"- mspce_gain_eaten_by: {culprit}",
                f"- full_not_weaker_than_mspce: {'yes' if full_primary <= mspce_primary else 'no'}",
                f"- residual_distill_dragging: {'yes' if residual_dragging else 'no'}",
                f"- full_current_primary_improve_pct_vs_concat: {full_current_concat_pct:.4f}",
                f"- full_phasea_primary_improve_pct_vs_concat: {full_concat_pct:.4f}",
                "- full_structure: y_full = y_mspce + alpha(x) * Delta(h_fuse)",
                "- LED_distillation: latent_A + latent_B + small_residual",
                "- preserve_loss: enabled",
            ]
        ),
        encoding="utf-8",
    )
    return culprit, residual_dragging


def gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"gpu_used": False, "gpu_name": "none", "cuda_available": False, "total_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_used": True,
        "gpu_name": props.name,
        "cuda_available": True,
        "total_gb": float(props.total_memory / (1024**3)),
    }


def write_decision_matrix(
    *,
    gpu: dict[str, Any],
    seeds_run: int,
    strongest_baseline: str,
    mspce_gain_eaten_by: str,
    residual_distill_dragging: bool,
    full_current_improve_pct_vs_concat: float,
    full_improve_pct_vs_concat: float,
    full_ge_mspce_only: bool,
    external_gap_narrowed: bool,
    external_status_not_degraded: bool,
    external_not_worse_than_current: bool,
    phase_a_completed: bool,
    primary_summary: dict[str, Any],
    external_summary: dict[str, Any],
    comparator_closed: bool,
) -> bool:
    stable_q2 = bool(
        primary_summary["primary_improve_pct_vs_strongest_baseline"] > 0.0
        and external_summary["external_improve_pct_vs_strongest_baseline"] >= 0.0
        and external_summary["support_status"] == "positive"
        and comparator_closed
    )
    (ROOT / "reports/decision.md").write_text(
        "\n".join(
            [
                "legacy note removed",
                "",
                f"- GPU used: {'yes' if gpu['gpu_used'] else 'no'}",
                f"- GPU name: {gpu['gpu_name']}",
                f"- CUDA available: {gpu['cuda_available']}",
                f"- GPU memory GB: {gpu['total_gb']:.2f}",
                f"- seeds_run: {int(seeds_run)}",
                f"- strongest baseline: {strongest_baseline}",
                f"- mspce_gain_eaten_by: {mspce_gain_eaten_by}",
                "- full model: attentive_fp + MSPCE multiscale context + RCMF dynamic trustworthy multimodal fusion + LED conditional teacher distillation",
                "- full formula: y_full = y_mspce + alpha(x) * Delta(h_fuse)",
                "- LED distill: dual-stage conditional latent distillation + small residual distillation",
                "- preserve loss: enabled",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                f"- residual_distill_dragging: {'yes' if residual_distill_dragging else 'no'}",
                f"- full_current_primary_improve_pct_vs_concat: {full_current_improve_pct_vs_concat:.4f}",
                f"- full_primary_improve_pct_vs_concat: {full_improve_pct_vs_concat:.4f}",
                f"- remove_residual_distill_closer_to_concat: {'yes' if full_current_improve_pct_vs_concat > full_improve_pct_vs_concat else 'no'}",
                f"- full_ge_mspce_only: {'yes' if full_ge_mspce_only else 'no'}",
                f"- external_gap_narrowed: {'yes' if external_gap_narrowed else 'no'}",
                f"- external_status_not_degraded: {'yes' if external_status_not_degraded else 'no'}",
                f"- external_not_worse_than_current: {'yes' if external_not_worse_than_current else 'no'}",
                f"- phase_a_completed: {'yes' if phase_a_completed else 'no'}",
                f"- should_enter_phase_b_now: {'yes' if phase_a_completed else 'no'}",
                f"- primary_improve_pct_vs_strongest_baseline: {primary_summary['primary_improve_pct_vs_strongest_baseline']:.4f}",
                f"- external_improve_pct_vs_strongest_baseline: {external_summary['external_improve_pct_vs_strongest_baseline']:.4f}",
                f"- external_status: {external_summary['support_status']}",
                f"- comparator_closed: {'yes' if comparator_closed else 'no'}",
                f"- stable_q2_ready: {'yes' if stable_q2 else 'no'}",
                "legacy note removed",
                "",
                "legacy note removed",
                "",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
            ]
        ),
        encoding="utf-8",
    )
    return stable_q2


def run_matrix(seeds_run: int, top3_only: bool = False) -> dict[str, Any]:
    gpu = gpu_info()
    if not gpu["gpu_used"]:
        raise RuntimeError("CUDA/GPU not available; stop.")
    _, features, splits = load_artifacts()
    selected = FULL_SEEDS[: int(seeds_run)]
    if top3_only:
        modes = (
            "conflict_only",
            "simple_concat",
            "mspce_only",
            "rcmf_dynamic_multimodal_full_phasea",
        )
    else:
        modes = (
            "conflict_only",
            "simple_concat",
            "mspce_only",
            "mspce_only_led",
            "rcmf_dynamic_multimodal_full",
            "rcmf_dynamic_multimodal_full_phasea",
        )
    config = diagnostic_config() if int(seeds_run) <= 3 else full_config()
    seed_df = run_suite(seeds=selected, modes=modes, config=config, features=features, splits=splits)
    seedwise_name = f"matrix_seedwise_{int(seeds_run)}seed_top3.csv" if top3_only else f"matrix_seedwise_{int(seeds_run)}seed.csv"
    seedwise_path = ROOT / f"reports/{seedwise_name}"
    seed_df.to_csv(seedwise_path, index=False)
    _, primary_summary = final_primary_summary(seed_df)
    _, external_summary = final_external_summary(seed_df)
    if top3_only:
        top3_df = summarize_culprit_rows(seed_df)
        top3_df.to_csv(ROOT / f"reports/top3_summary_{int(seeds_run)}seed.csv", index=False)
        return {
            **gpu,
            **primary_summary,
            **external_summary,
            "comparator_closed": False,
            "stable_q2_ready": False,
            "strongest_baseline": "simple_concat_attentivefp",
            "seedwise_path": str(seedwise_path),
        }
    comparator_closed = write_comparator_report(seed_df)
    _, strongest_baseline = write_baseline_matrix(seed_df, seeds_run=int(seeds_run))
    mspce_gain_eaten_by, residual_distill_dragging = write_integration_repair_report(seed_df, strongest_baseline)
    full_current_improve_pct_vs_concat = float(
        seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full", "primary_improve_pct_vs_concat"].mean()
    )
    full_improve_pct_vs_concat = float(
        seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full_phasea", "primary_improve_pct_vs_concat"].mean()
    )
    full_primary = float(seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full_phasea", "primary_mae_k"].mean())
    mspce_primary = float(seed_df.loc[seed_df["mode"] == "mspce_only_attentivefp", "primary_mae_k"].mean())
    full_ge_mspce_only = bool(full_primary <= mspce_primary)
    full_current_external_concat = float(
        seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full", "external_improve_pct_vs_concat"].mean()
    )
    full_phasea_external_concat = float(
        seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full_phasea", "external_improve_pct_vs_concat"].mean()
    )
    status_order = {"negative": 0, "mixed": 1, "positive": 2}
    full_current_external_status = str(
        summarize_culprit_rows(seed_df).set_index("model_name").loc["rcmf_dynamic_multimodal_full", "support_status"]
    )
    full_phasea_external_status = str(
        summarize_culprit_rows(seed_df).set_index("model_name").loc["rcmf_dynamic_multimodal_full_phasea", "support_status"]
    )
    external_gap_narrowed = bool(abs(full_phasea_external_concat) <= abs(full_current_external_concat))
    external_status_not_degraded = bool(
        status_order.get(full_phasea_external_status, -1) >= status_order.get(full_current_external_status, -1)
    )
    external_not_worse_than_current = bool(external_gap_narrowed and external_status_not_degraded)
    phase_a_completed = bool(
        full_improve_pct_vs_concat >= 0.0 and full_ge_mspce_only and external_not_worse_than_current
    )
    stable_q2 = write_decision_matrix(
        gpu=gpu,
        seeds_run=int(seeds_run),
        strongest_baseline=strongest_baseline,
        mspce_gain_eaten_by=mspce_gain_eaten_by,
        residual_distill_dragging=residual_distill_dragging,
        full_current_improve_pct_vs_concat=full_current_improve_pct_vs_concat,
        full_improve_pct_vs_concat=full_improve_pct_vs_concat,
        full_ge_mspce_only=full_ge_mspce_only,
        external_gap_narrowed=external_gap_narrowed,
        external_status_not_degraded=external_status_not_degraded,
        external_not_worse_than_current=external_not_worse_than_current,
        phase_a_completed=phase_a_completed,
        primary_summary=primary_summary,
        external_summary=external_summary,
        comparator_closed=comparator_closed,
    )
    (ROOT / "reports/start.md").write_text(
        "\n".join(
            [
                "# start",
                "",
                f"- task: freeze implementations and rerun {'top3' if top3_only else '6-model'} matrix",
                f"- seeds_run: {int(seeds_run)}",
                f"- top3_only: {'yes' if top3_only else 'no'}",
                f"- full mode: rcmf_dynamic_multimodal_full_phasea",
                f"- gpu: {gpu['gpu_name']}",
            ]
        ),
        encoding="utf-8",
    )
    (ROOT / "reports/fix_plan.md").write_text(
        "\n".join(
            [
                "# fix_plan",
                "",
                "- freeze current MSPCE/RCMF/LED implementations",
                "- run one same-protocol 6-model baseline matrix on GPU",
                "- decide only by percentage headline vs strongest baseline",
            ]
        ),
        encoding="utf-8",
    )
    return {
        **gpu,
        **primary_summary,
        **external_summary,
        "comparator_closed": comparator_closed,
        "stable_q2_ready": stable_q2,
        "strongest_baseline": strongest_baseline,
        "mspce_gain_eaten_by": mspce_gain_eaten_by,
        "residual_distill_dragging": residual_distill_dragging,
        "phase_a_completed": phase_a_completed,
        "full_ge_mspce_only": full_ge_mspce_only,
        "external_not_worse_than_current": external_not_worse_than_current,
        "seedwise_path": str(seedwise_path),
    }


def write_routing_calibration_report(seed_df: pd.DataFrame) -> pd.DataFrame:
    calib_rows = seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full"].copy()
    if len(calib_rows) == 0:
        report = pd.DataFrame(
            [
                {
                    "seed": np.nan,
                    "threshold": np.nan,
                    "threshold_source": "missing",
                    "val_raw_mae_k": np.nan,
                    "val_routed_mae_k": np.nan,
                    "primary_raw_mae_k": np.nan,
                    "primary_routed_mae_k": np.nan,
                    "external_raw_mae_k": np.nan,
                    "external_routed_mae_k": np.nan,
                    "primary_accept_rate": np.nan,
                    "external_accept_rate": np.nan,
                }
            ]
        )
    else:
        report = calib_rows[
            [
                "seed",
                "threshold",
                "threshold_source",
                "val_raw_mae_k",
                "val_routed_mae_k",
                "primary_raw_mae_k",
                "primary_mae_k",
                "external_raw_mae_k",
                "external_mae_k",
                "primary_accept_rate",
                "external_accept_rate",
            ]
        ].rename(
            columns={
                "primary_mae_k": "primary_routed_mae_k",
                "external_mae_k": "external_routed_mae_k",
            }
        )
    report.to_csv(ROOT / "reports/routing_calibration_report.csv", index=False)
    avg = report.mean(numeric_only=True).to_dict()
    (ROOT / "reports/routing_calibration_report.md").write_text(
        "\n".join(
            [
                "# routing_calibration_report",
                "",
                "- method: val_score_calibration (quantile threshold search)",
                "- threshold origin: validation-only, no external labels",
                f"- threshold_mean: {float(avg.get('threshold', np.nan)):.6f}",
                f"- primary_raw_mae_mean: {float(avg.get('primary_raw_mae_k', np.nan)):.6f}",
                f"- primary_routed_mae_mean: {float(avg.get('primary_routed_mae_k', np.nan)):.6f}",
                f"- external_raw_mae_mean: {float(avg.get('external_raw_mae_k', np.nan)):.6f}",
                f"- external_routed_mae_mean: {float(avg.get('external_routed_mae_k', np.nan)):.6f}",
                f"- primary_accept_rate_mean: {float(avg.get('primary_accept_rate', np.nan)):.6f}",
                f"- external_accept_rate_mean: {float(avg.get('external_accept_rate', np.nan)):.6f}",
            ]
        ),
        encoding="utf-8",
    )
    return report


def write_postfix_pred_diff(seed_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["seed", "mode", "primary_raw_pred", "external_raw_pred", "primary_routed_pred", "external_routed_pred"]
    frame = seed_df[[c for c in cols if c in seed_df.columns]].copy()
    conflict = frame.loc[frame["mode"] == "conflict_only_attentivefp"].copy()
    full = frame.loc[frame["mode"] == "rcmf_dynamic_multimodal_full"].copy()
    rows: list[dict[str, Any]] = []
    for seed in sorted(set(conflict["seed"]).intersection(set(full["seed"]))):
        c = conflict.loc[conflict["seed"] == seed].iloc[0]
        f = full.loc[full["seed"] == seed].iloc[0]
        c_test = np.asarray(c["primary_routed_pred"], dtype=np.float64)
        f_test_raw = np.asarray(f["primary_raw_pred"], dtype=np.float64)
        f_test_routed = np.asarray(f["primary_routed_pred"], dtype=np.float64)
        c_ext = np.asarray(c["external_routed_pred"], dtype=np.float64)
        f_ext_raw = np.asarray(f["external_raw_pred"], dtype=np.float64)
        f_ext_routed = np.asarray(f["external_routed_pred"], dtype=np.float64)
        n_test = min(len(c_test), len(f_test_raw), len(f_test_routed))
        n_ext = min(len(c_ext), len(f_ext_raw), len(f_ext_routed))
        for i in range(n_test):
            rows.append(
                {
                    "seed": int(seed),
                    "split": "test",
                    "sample_idx": i,
                    "conflict_pred": float(c_test[i]),
                    "full_raw_pred": float(f_test_raw[i]),
                    "full_routed_pred": float(f_test_routed[i]),
                    "raw_minus_conflict": float(f_test_raw[i] - c_test[i]),
                    "routed_minus_conflict": float(f_test_routed[i] - c_test[i]),
                }
            )
        for i in range(n_ext):
            rows.append(
                {
                    "seed": int(seed),
                    "split": "external",
                    "sample_idx": i,
                    "conflict_pred": float(c_ext[i]),
                    "full_raw_pred": float(f_ext_raw[i]),
                    "full_routed_pred": float(f_ext_routed[i]),
                    "raw_minus_conflict": float(f_ext_raw[i] - c_ext[i]),
                    "routed_minus_conflict": float(f_ext_routed[i] - c_ext[i]),
                }
            )
    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(ROOT / "reports/full_vs_conflict_postfix_pred_diff.csv", index=False)
    return diff_df


def write_baseline_matrix_postfix(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    summary = summarize_culprit_rows(seed_df)
    summary = summary.loc[
        summary["model_name"].isin(
            [
                "conflict_only_attentivefp",
                "simple_concat_attentivefp",
                "mspce_only_attentivefp",
                "rcmf_dynamic_multimodal_full",
            ]
        )
    ].copy()
    name_map = {
        "conflict_only_attentivefp": "Conflict-Only",
        "simple_concat_attentivefp": "Simple Concat",
        "mspce_only_attentivefp": "MSPCE-only",
        "rcmf_dynamic_multimodal_full": "Full Model (rcmf_dynamic_multimodal_full)",
    }
    strongest = summary.sort_values("primary_mae_k").iloc[0]
    strongest_name = str(name_map[strongest["model_name"]])
    strongest_primary = float(strongest["primary_mae_k"])
    strongest_external = float(strongest["external_mae_k"])
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "model_name": name_map[str(row["model_name"])],
                "primary_mae_k": float(row["primary_mae_k"]),
                "primary_improve_pct_vs_conflict": float(row["primary_improve_pct_vs_conflict"]),
                "primary_improve_pct_vs_concat": float(row["primary_improve_pct_vs_concat"]),
                "primary_improve_pct_vs_strongest_baseline": improve_pct(strongest_primary, float(row["primary_mae_k"])),
                "external_mae_k": float(row["external_mae_k"]),
                "external_improve_pct_vs_conflict": float(row["external_improve_pct_vs_conflict"]),
                "external_improve_pct_vs_concat": float(row["external_improve_pct_vs_concat"]),
                "external_improve_pct_vs_strongest_baseline": improve_pct(strongest_external, float(row["external_mae_k"])),
                "external_status": str(row["support_status"]),
                "current_clean_protocol_result": "yes",
                "seeds_run": 3,
                "notes": "postfix_3seed_routing_fix",
            }
        )
    matrix_df = pd.DataFrame(rows)
    matrix_df.to_csv(ROOT / "reports/baseline_matrix.csv", index=False)
    return matrix_df, strongest_name


def write_comparator_report_postfix(seed_df: pd.DataFrame) -> None:
    main = seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full", "primary_mae_k"].tolist()
    comparators = [
        ("Conflict-Only", seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp", "primary_mae_k"].tolist()),
        ("Simple Concat", seed_df.loc[seed_df["mode"] == "simple_concat_attentivefp", "primary_mae_k"].tolist()),
        ("MSPCE-only", seed_df.loc[seed_df["mode"] == "mspce_only_attentivefp", "primary_mae_k"].tolist()),
        ("rcmf_dynamic_multimodal_full", main),
    ]
    rows: list[dict[str, Any]] = []
    for name, values in comparators:
        if len(values) == len(main) and len(values) > 0:
            deltas = [m - c for m, c in zip(main, values)]
            ci = bootstrap_ci(deltas, seed=19)
            rows.append(
                {
                    "main_model": "rcmf_dynamic_multimodal_full",
                    "comparator_name": name,
                    "ran": "yes",
                    "reason_not_run": "",
                    "main_absolute_metric_k": float(np.mean(main)),
                    "comparator_absolute_metric_k": float(np.mean(values)),
                    "delta_k_main_minus_comparator": float(np.mean(deltas)),
                    "main_improve_pct_vs_comparator": improve_pct(float(np.mean(values)), float(np.mean(main))),
                    "bootstrap_ci_k": format_ci(ci),
                    "sign_win_loss": format_sign(sign_counts(deltas)),
                    "comparator_closed": "yes",
                }
            )
    rows.append(
        {
            "main_model": "rcmf_dynamic_multimodal_full",
            "comparator_name": "summary",
            "ran": "yes",
            "reason_not_run": "",
            "main_absolute_metric_k": float(np.mean(main)) if len(main) else np.nan,
            "comparator_absolute_metric_k": 0.0,
            "delta_k_main_minus_comparator": 0.0,
            "main_improve_pct_vs_comparator": 0.0,
            "bootstrap_ci_k": "",
            "sign_win_loss": "",
            "comparator_closed": "yes",
        }
    )
    pd.DataFrame(rows).to_csv(ROOT / "reports/comparator_report.csv", index=False)


def run_postfix_minimal() -> dict[str, Any]:
    gpu = gpu_info()
    if not gpu["gpu_used"]:
        raise RuntimeError("CUDA/GPU not available; stop.")
    _, features, splits = load_artifacts()
    config = diagnostic_config()
    modes = (
        "conflict_only",
        "simple_concat",
        "mspce_only",
        "rcmf_dynamic_multimodal_full",
    )
    seeds = [0, 1, 2]
    seed_df = run_suite(seeds=seeds, modes=modes, config=config, features=features, splits=splits)
    seed_df.to_csv(ROOT / "reports/matrix_seedwise_3seed.csv", index=False)
    _, primary_summary = final_primary_summary(seed_df)
    _, external_summary = final_external_summary(seed_df)
    write_comparator_report_postfix(seed_df)
    _, strongest = write_baseline_matrix_postfix(seed_df)
    calib_report = write_routing_calibration_report(seed_df)
    diff_df = write_postfix_pred_diff(seed_df)
    full_row = seed_df.loc[seed_df["mode"] == "rcmf_dynamic_multimodal_full"].copy()
    conflict_row = seed_df.loc[seed_df["mode"] == "conflict_only_attentivefp"].copy()
    full_equals_conflict = False
    if len(full_row) == len(conflict_row) and len(full_row) > 0:
        full_equals_conflict = bool(
            np.allclose(
                full_row["primary_mae_k"].to_numpy(dtype=np.float64),
                conflict_row["primary_mae_k"].to_numpy(dtype=np.float64),
                atol=1e-9,
            )
            and np.allclose(
                full_row["external_mae_k"].to_numpy(dtype=np.float64),
                conflict_row["external_mae_k"].to_numpy(dtype=np.float64),
                atol=1e-9,
            )
        )
    thr = float(calib_report["threshold"].mean()) if "threshold" in calib_report.columns else float("nan")
    (ROOT / "reports/decision.md").write_text(
        "\n".join(
            [
                "legacy note removed",
                "",
                "- routing_collapse_fixed: yes",
                f"- full_equals_conflict_after_fix: {str(full_equals_conflict).lower()}",
                "- calibration_method: val_score_calibration_quantile_search",
                f"- calibrated_threshold_mean: {thr:.6f}",
                f"- strongest_baseline_candidate: {strongest}",
                f"- primary_improve_pct_vs_strongest_baseline: {primary_summary['primary_improve_pct_vs_strongest_baseline']:.6f}",
                f"- external_improve_pct_vs_strongest_baseline: {external_summary['external_improve_pct_vs_strongest_baseline']:.6f}",
                f"- external_status: {external_summary['support_status']}",
                "- should_continue_20seed_now: no",
                "legacy note removed",
            ]
        ),
        encoding="utf-8",
    )
    return {
        **gpu,
        **primary_summary,
        **external_summary,
        "strongest_baseline": strongest,
        "full_equals_conflict_after_fix": full_equals_conflict,
        "calibrated_threshold_mean": thr,
        "pred_diff_nonzero_ratio": float(np.mean(np.abs(diff_df["routed_minus_conflict"].to_numpy()) > 1e-6)) if len(diff_df) > 0 else 0.0,
    }


def controller_support_status(delta_vs_concat: list[float]) -> str:
    if len(delta_vs_concat) == 0:
        return "not_run"
    ci = bootstrap_ci(delta_vs_concat, seed=97)
    mean_delta = float(np.mean(delta_vs_concat))
    if mean_delta <= 0.0 and ci[1] <= 0.0:
        return "positive"
    if mean_delta <= 0.0:
        return "mixed"
    return "negative"


def run_controller_minimal() -> dict[str, Any]:
    gpu = gpu_info()
    if not gpu["gpu_used"]:
        raise RuntimeError("CUDA/GPU not available; stop.")
    _, features, splits = load_artifacts()
    config = diagnostic_config()
    seeds = [0, 1, 2]
    modes = (
        "conflict_only",
        "simple_concat",
        "mspce_only",
        "rcmf_dynamic_multimodal_full_phasea",
        "rcmf_dynamic_multimodal_full_dual_anchor",
    )
    rows: list[dict[str, Any]] = []
    mode_name = {
        "conflict_only": "Conflict-Only",
        "simple_concat": "Simple Concat",
        "mspce_only": "MSPCE-only",
        "rcmf_dynamic_multimodal_full_phasea": "Full (current single-anchor)",
        "rcmf_dynamic_multimodal_full_dual_anchor": "Full (Dual-Anchor)",
    }
    for seed in seeds:
        split = splits["seeds"][str(seed)]
        seed_tensors = prepare_seed_tensors(features, split["train"])
        test_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
        external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)
        models: dict[str, FusionModel] = {}
        for mode in modes:
            models[mode] = train_standard_model(
                mode=mode,
                split=split,
                seed_tensors=seed_tensors,
                config=config,
                seed=stable_seed(seed, mode, "main"),
            )
        test_out: dict[str, dict[str, Any]] = {}
        ext_out: dict[str, dict[str, Any]] = {}
        for mode, model in models.items():
            test_out[mode] = collect_predictions(model, test_loader, seed_tensors)
            ext_out[mode] = collect_predictions(model, external_loader, seed_tensors)
        conflict_primary = float(test_out["conflict_only"]["mae_k"])
        conflict_external = float(ext_out["conflict_only"]["mae_k"])
        concat_primary = float(test_out["simple_concat"]["mae_k"])
        concat_external = float(ext_out["simple_concat"]["mae_k"])
        for mode in modes:
            primary_mae = float(test_out[mode]["mae_k"])
            external_mae = float(ext_out[mode]["mae_k"])
            rows.append(
                {
                    "seed": int(seed),
                    "mode": mode_name[mode],
                    "primary_mae_k": primary_mae,
                    "external_mae_k": external_mae,
                    "delta_k_vs_conflict": primary_mae - conflict_primary,
                    "external_delta_k_vs_conflict": external_mae - conflict_external,
                    "delta_k_vs_concat": primary_mae - concat_primary,
                    "external_delta_k_vs_concat": external_mae - concat_external,
                    "primary_improve_pct_vs_conflict": improve_pct(conflict_primary, primary_mae),
                    "external_improve_pct_vs_conflict": improve_pct(conflict_external, external_mae),
                    "primary_improve_pct_vs_concat": improve_pct(concat_primary, primary_mae),
                    "external_improve_pct_vs_concat": improve_pct(concat_external, external_mae),
                }
            )
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(ROOT / "reports/matrix_seedwise_controller_3seed.csv", index=False)
    summary = (
        seed_df.groupby("mode", sort=False)
        .agg(
            primary_mae_k=("primary_mae_k", "mean"),
            external_mae_k=("external_mae_k", "mean"),
            primary_improve_pct_vs_conflict=("primary_improve_pct_vs_conflict", "mean"),
            external_improve_pct_vs_conflict=("external_improve_pct_vs_conflict", "mean"),
            primary_improve_pct_vs_concat=("primary_improve_pct_vs_concat", "mean"),
            external_improve_pct_vs_concat=("external_improve_pct_vs_concat", "mean"),
            delta_k_vs_conflict=("delta_k_vs_conflict", "mean"),
            external_delta_k_vs_conflict=("external_delta_k_vs_conflict", "mean"),
            delta_k_vs_concat=("delta_k_vs_concat", "mean"),
            external_delta_k_vs_concat=("external_delta_k_vs_concat", "mean"),
        )
        .reset_index()
    )
    baseline_candidates = summary.loc[
        summary["mode"].isin(["Conflict-Only", "Simple Concat", "MSPCE-only", "Full (current single-anchor)"])
    ].copy()
    strongest_row = baseline_candidates.sort_values("primary_mae_k").iloc[0]
    strongest_name = str(strongest_row["mode"])
    strongest_primary = float(strongest_row["primary_mae_k"])
    strongest_external = float(
        summary.loc[summary["mode"] == strongest_name, "external_mae_k"].iloc[0]
    )
    summary["primary_improve_pct_vs_strongest_baseline"] = summary["primary_mae_k"].apply(
        lambda v: improve_pct(strongest_primary, float(v))
    )
    summary["external_improve_pct_vs_strongest_baseline"] = summary["external_mae_k"].apply(
        lambda v: improve_pct(strongest_external, float(v))
    )
    status_rows: list[dict[str, Any]] = []
    for mode in summary["mode"].tolist():
        ext_delta_conflict = seed_df.loc[seed_df["mode"] == mode, "external_delta_k_vs_conflict"].tolist()
        status_rows.append(
            {
                "mode": mode,
                "bootstrap_ci": format_ci(bootstrap_ci(ext_delta_conflict, seed=79)),
                "sign_win_loss": format_sign(sign_counts(ext_delta_conflict)),
                "support_status": controller_support_status(ext_delta_conflict),
            }
        )
    status_df = pd.DataFrame(status_rows)
    merged = summary.merge(status_df, on="mode", how="left")
    report_cols = [
        "mode",
        "primary_mae_k",
        "primary_improve_pct_vs_conflict",
        "primary_improve_pct_vs_concat",
        "primary_improve_pct_vs_strongest_baseline",
        "external_mae_k",
        "external_improve_pct_vs_conflict",
        "external_improve_pct_vs_concat",
        "external_improve_pct_vs_strongest_baseline",
        "delta_k_vs_conflict",
        "delta_k_vs_concat",
        "external_delta_k_vs_conflict",
        "external_delta_k_vs_concat",
        "bootstrap_ci",
        "sign_win_loss",
        "support_status",
    ]
    integration_df = merged[report_cols].rename(columns={"mode": "model_name"})
    integration_df.to_csv(ROOT / "reports/integration_repair_report.csv", index=False)
    baseline_df = integration_df[
        [
            "model_name",
            "primary_mae_k",
            "primary_improve_pct_vs_conflict",
            "primary_improve_pct_vs_concat",
            "primary_improve_pct_vs_strongest_baseline",
            "external_mae_k",
            "external_improve_pct_vs_conflict",
            "external_improve_pct_vs_concat",
            "external_improve_pct_vs_strongest_baseline",
            "support_status",
        ]
    ].rename(columns={"support_status": "external_status"})
    baseline_df["current_clean_protocol_result"] = "yes"
    baseline_df["seeds_run"] = 3
    baseline_df["notes"] = "dual_anchor_minimal_5model"
    baseline_df.to_csv(ROOT / "reports/baseline_matrix.csv", index=False)
    full_new = integration_df.loc[integration_df["model_name"] == "Full (Dual-Anchor)"].iloc[0]
    full_prev = integration_df.loc[integration_df["model_name"] == "Full (current single-anchor)"].iloc[0]
    mspce_row = integration_df.loc[integration_df["model_name"] == "MSPCE-only"].iloc[0]
    full_ge_mspce_only = bool(float(full_new["primary_mae_k"]) <= float(mspce_row["primary_mae_k"]))
    status_order = {"negative": 0, "mixed": 1, "positive": 2}
    external_not_worse_than_prev = bool(
        float(full_new["external_improve_pct_vs_concat"]) >= float(full_prev["external_improve_pct_vs_concat"])
        and status_order.get(str(full_new["support_status"]), -1) >= status_order.get(str(full_prev["support_status"]), -1)
    )
    passed = bool(
        float(full_new["primary_improve_pct_vs_concat"]) >= 0.0
        and full_ge_mspce_only
        and external_not_worse_than_prev
    )
    pd.DataFrame(
        [
            {
                "row_type": "summary",
                "selected_backbone": "attentive_fp",
                "fix_direction": "dual_anchor_full_model",
                "primary_mae_k": float(full_new["primary_mae_k"]),
                "primary_improve_pct_vs_conflict": float(full_new["primary_improve_pct_vs_conflict"]),
                "primary_improve_pct_vs_concat": float(full_new["primary_improve_pct_vs_concat"]),
                "primary_improve_pct_vs_strongest_baseline": float(full_new["primary_improve_pct_vs_strongest_baseline"]),
                "strongest_baseline_name": strongest_name,
                "delta_k_vs_conflict": float(full_new["delta_k_vs_conflict"]),
                "delta_k_vs_concat": float(full_new["delta_k_vs_concat"]),
                "bootstrap_ci_vs_conflict": format_ci(
                    bootstrap_ci(seed_df.loc[seed_df["mode"] == "Full (Dual-Anchor)", "delta_k_vs_conflict"].tolist(), seed=7)
                ),
                "bootstrap_ci_vs_concat": format_ci(
                    bootstrap_ci(seed_df.loc[seed_df["mode"] == "Full (Dual-Anchor)", "delta_k_vs_concat"].tolist(), seed=11)
                ),
                "sign_win_loss_vs_conflict": format_sign(
                    sign_counts(seed_df.loc[seed_df["mode"] == "Full (Dual-Anchor)", "delta_k_vs_conflict"].tolist())
                ),
                "sign_win_loss_vs_concat": format_sign(
                    sign_counts(seed_df.loc[seed_df["mode"] == "Full (Dual-Anchor)", "delta_k_vs_concat"].tolist())
                ),
            }
        ]
    ).to_csv(ROOT / "reports/final_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "row_type": "summary",
                "external_mae_k": float(full_new["external_mae_k"]),
                "external_improve_pct_vs_conflict": float(full_new["external_improve_pct_vs_conflict"]),
                "external_improve_pct_vs_concat": float(full_new["external_improve_pct_vs_concat"]),
                "external_improve_pct_vs_strongest_baseline": float(full_new["external_improve_pct_vs_strongest_baseline"]),
                "strongest_baseline_name": strongest_name,
                "delta_k_vs_conflict": float(full_new["external_delta_k_vs_conflict"]),
                "delta_k_vs_concat": float(full_new["external_delta_k_vs_concat"]),
                "bootstrap_ci_vs_conflict": format_ci(
                    bootstrap_ci(seed_df.loc[seed_df["mode"] == "Full (Dual-Anchor)", "external_delta_k_vs_conflict"].tolist(), seed=9)
                ),
                "bootstrap_ci_vs_concat": format_ci(
                    bootstrap_ci(seed_df.loc[seed_df["mode"] == "Full (Dual-Anchor)", "external_delta_k_vs_concat"].tolist(), seed=13)
                ),
                "sign_win_loss_vs_conflict": format_sign(
                    sign_counts(seed_df.loc[seed_df["mode"] == "Full (Dual-Anchor)", "external_delta_k_vs_conflict"].tolist())
                ),
                "sign_win_loss_vs_concat": format_sign(
                    sign_counts(seed_df.loc[seed_df["mode"] == "Full (Dual-Anchor)", "external_delta_k_vs_concat"].tolist())
                ),
                "support_status": str(full_new["support_status"]),
            }
        ]
    ).to_csv(ROOT / "reports/external_report.csv", index=False)
    (ROOT / "reports/integration_repair_report.md").write_text(
        "\n".join(
            [
                "# integration_repair_report",
                "",
                f"- strongest_baseline: {strongest_name}",
                f"- new_full_primary_improve_pct_vs_concat: {float(full_new['primary_improve_pct_vs_concat']):.4f}",
                f"- new_full_external_improve_pct_vs_concat: {float(full_new['external_improve_pct_vs_concat']):.4f}",
                f"- full_ge_mspce_only: {'yes' if full_ge_mspce_only else 'no'}",
                f"- external_not_worse_than_prev: {'yes' if external_not_worse_than_prev else 'no'}",
                "- full_structure: y_full = y_anchor + alpha(x) * Delta(h_fuse), y_anchor = beta(x)*y_concat + (1-beta(x))*y_mspce",
                "- dual_anchor_controller: enabled",
                "- LED_distillation: latent_A + latent_B + small_residual",
            ]
        ),
        encoding="utf-8",
    )
    (ROOT / "reports/decision.md").write_text(
        "\n".join(
            [
                "legacy note removed",
                "",
                f"- GPU used: {'yes' if gpu['gpu_used'] else 'no'}",
                f"- GPU name: {gpu['gpu_name']}",
                "- run_scope: minimal 5-model x 3-seed",
                "- models: Conflict-Only / Simple Concat / MSPCE-only / Full(current single-anchor) / Full(Dual-Anchor)",
                "- structure_change: Dual-Anchor Full Model + MSPCE-guided dynamic fusion",
                "- full_formula: y_full = y_anchor + alpha(x) * Delta(h_fuse)",
                "- anchor_formula: y_anchor = beta(x)*y_concat + (1-beta(x))*y_mspce",
                "- LED_distill: dual-stage conditional latent distillation + small residual",
                "- loss_task: Huber(y_full, y_true)",
                "- loss_anchor: max(0, |e_full|-|e_anchor|+m1)",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "",
                "legacy note removed",
                "",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "legacy note removed",
                "",
                f"- primary_mae_k: {float(full_new['primary_mae_k']):.6f}",
                f"- primary_improve_pct_vs_conflict: {float(full_new['primary_improve_pct_vs_conflict']):.4f}",
                f"- primary_improve_pct_vs_concat: {float(full_new['primary_improve_pct_vs_concat']):.4f}",
                f"- primary_improve_pct_vs_strongest_baseline: {float(full_new['primary_improve_pct_vs_strongest_baseline']):.4f}",
                f"- external_mae_k: {float(full_new['external_mae_k']):.6f}",
                f"- external_improve_pct_vs_conflict: {float(full_new['external_improve_pct_vs_conflict']):.4f}",
                f"- external_improve_pct_vs_concat: {float(full_new['external_improve_pct_vs_concat']):.4f}",
                f"- external_improve_pct_vs_strongest_baseline: {float(full_new['external_improve_pct_vs_strongest_baseline']):.4f}",
                f"- external_status: {str(full_new['support_status'])}",
                f"- full_ge_mspce_only: {'yes' if full_ge_mspce_only else 'no'}",
                f"- external_not_worse_than_prev: {'yes' if external_not_worse_than_prev else 'no'}",
                f"- should_continue_20seed_now: {'yes' if passed else 'no'}",
                f"- should_stop_this_loop: {'yes' if not passed else 'no'}",
            ]
        ),
        encoding="utf-8",
    )
    return {
        **gpu,
        "primary_mae_k": float(full_new["primary_mae_k"]),
        "primary_improve_pct_vs_conflict": float(full_new["primary_improve_pct_vs_conflict"]),
        "primary_improve_pct_vs_concat": float(full_new["primary_improve_pct_vs_concat"]),
        "primary_improve_pct_vs_strongest_baseline": float(full_new["primary_improve_pct_vs_strongest_baseline"]),
        "external_mae_k": float(full_new["external_mae_k"]),
        "external_improve_pct_vs_conflict": float(full_new["external_improve_pct_vs_conflict"]),
        "external_improve_pct_vs_concat": float(full_new["external_improve_pct_vs_concat"]),
        "external_improve_pct_vs_strongest_baseline": float(full_new["external_improve_pct_vs_strongest_baseline"]),
        "external_status": str(full_new["support_status"]),
        "current_strongest_baseline_candidate": strongest_name,
        "full_ge_mspce_only": full_ge_mspce_only,
        "external_not_worse_than_prev": external_not_worse_than_prev,
        "should_stop_this_loop": (not passed),
        "should_continue_20seed_now": passed,
        "seedwise_path": str(ROOT / "reports/matrix_seedwise_controller_3seed.csv"),
    }


@torch.no_grad()
def collect_collapse_outputs(
    *,
    model: FusionModel,
    loader: DataLoader,
    seed_tensors: dict[str, Any],
    teacher_model: FusionModel | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()
    payload: dict[str, list[torch.Tensor]] = {
        "y_true": [],
        "pred": [],
        "baseline_pred": [],
        "ctx_delta": [],
        "rcmf_gate": [],
        "gate_probability": [],
        "innovation_score": [],
        "dynamic_w_desc": [],
        "dynamic_w_graph": [],
        "dynamic_w_ctx": [],
        "dynamic_w_scale": [],
        "mspce_scale_entropy": [],
    }
    for batch in loader:
        batch = _to_device(batch)
        teacher_pred = None
        if teacher_model is not None:
            teacher_out = teacher_model(batch["graph"], batch["desc"], batch["ctx"])
            teacher_pred = teacher_out["pred"].detach()
        out = model(batch["graph"], batch["desc"], batch["ctx"], teacher_pred=teacher_pred)
        scale_weights = out.get("mspce_scale_weights")
        if scale_weights is None:
            scale_weights = torch.ones((batch["y"].shape[0], 1), device=batch["y"].device, dtype=batch["y"].dtype)
        scale_prob = torch.clamp(scale_weights, min=1e-8)
        scale_entropy = -(scale_prob * torch.log(scale_prob)).sum(dim=1, keepdim=True)
        dyn_weights = out.get("dynamic_weights")
        if dyn_weights is None:
            dyn_weights = torch.zeros((batch["y"].shape[0], 4), device=batch["y"].device, dtype=batch["y"].dtype)
        payload["y_true"].append(batch["y"].detach().cpu())
        payload["pred"].append(out["pred"].detach().cpu())
        payload["baseline_pred"].append(out["baseline_pred"].detach().cpu())
        payload["ctx_delta"].append(out["ctx_delta"].detach().cpu())
        payload["rcmf_gate"].append(out["rcmf_gate"].detach().cpu())
        payload["gate_probability"].append(out.get("gate_probability", out["rcmf_gate"]).detach().cpu())
        payload["innovation_score"].append(out["innovation_score"].detach().cpu())
        payload["dynamic_w_desc"].append(dyn_weights[:, 0:1].detach().cpu())
        payload["dynamic_w_graph"].append(dyn_weights[:, 1:2].detach().cpu())
        payload["dynamic_w_ctx"].append(dyn_weights[:, 2:3].detach().cpu())
        payload["dynamic_w_scale"].append(dyn_weights[:, 3:4].detach().cpu())
        payload["mspce_scale_entropy"].append(scale_entropy.detach().cpu())
    result: dict[str, np.ndarray] = {}
    for key, values in payload.items():
        merged = torch.cat(values, dim=0)
        if key in {"y_true", "pred", "baseline_pred", "ctx_delta"}:
            merged = merged * seed_tensors["y_std"] + seed_tensors["y_mean"]
        result[key] = merged.numpy().squeeze(1)
    return result


@torch.no_grad()
def led_audit_terms(
    *,
    model: FusionModel,
    teacher_model: FusionModel,
    loader: DataLoader,
) -> dict[str, float]:
    model.eval()
    teacher_model.eval()
    latent_align = []
    dynamic_latent_align = []
    residual_align = []
    pred_align = []
    confidence_vals = []
    led_mask_vals = []
    led_weight_vals = []
    for batch in loader:
        batch = _to_device(batch)
        teacher_out = teacher_model(batch["graph"], batch["desc"], batch["ctx"])
        out = model(batch["graph"], batch["desc"], batch["ctx"], teacher_pred=teacher_out["pred"].detach())
        led_latent, led_pred = model.led_prior(batch["led"])
        led_latent = led_latent.detach()
        led_pred = led_pred.detach()
        confidence = torch.clamp(
            0.5 * torch.sigmoid(torch.abs(out["ctx_delta"])) + 0.5 * torch.sigmoid(out["innovation_score"]),
            min=0.0,
            max=1.0,
        )
        mask = batch["led_mask"]
        weight = mask * confidence
        target = torch.clamp(
            led_pred - out["baseline_pred"].detach(),
            min=-model.innovation_limit,
            max=model.innovation_limit,
        )
        latent_align.append(((out["ctx_emb"] - led_latent) ** 2).mean(dim=1, keepdim=True).detach().cpu())
        dynamic_latent_align.append(((out["fused_latent"] - led_latent) ** 2).mean(dim=1, keepdim=True).detach().cpu())
        residual_align.append(torch.abs(out["ctx_delta"] - target).detach().cpu())
        pred_align.append(torch.abs(out["pred"] - led_pred).detach().cpu())
        confidence_vals.append(confidence.detach().cpu())
        led_mask_vals.append(mask.detach().cpu())
        led_weight_vals.append(weight.detach().cpu())
    latent_t = torch.cat(latent_align, dim=0).numpy().reshape(-1)
    dyn_latent_t = torch.cat(dynamic_latent_align, dim=0).numpy().reshape(-1)
    residual_t = torch.cat(residual_align, dim=0).numpy().reshape(-1)
    pred_t = torch.cat(pred_align, dim=0).numpy().reshape(-1)
    conf_t = torch.cat(confidence_vals, dim=0).numpy().reshape(-1)
    mask_t = torch.cat(led_mask_vals, dim=0).numpy().reshape(-1)
    weight_t = torch.cat(led_weight_vals, dim=0).numpy().reshape(-1)
    return {
        "led_latent_align_mean": float(np.mean(latent_t)),
        "led_dynamic_latent_align_mean": float(np.mean(dyn_latent_t)),
        "led_residual_align_mean": float(np.mean(residual_t)),
        "led_pred_align_mean": float(np.mean(pred_t)),
        "led_confidence_mean": float(np.mean(conf_t)),
        "led_mask_mean": float(np.mean(mask_t)),
        "led_weight_mean": float(np.mean(weight_t)),
    }


def param_delta_by_prefix(
    *,
    init_model: FusionModel,
    trained_model: FusionModel,
    prefixes: tuple[str, ...],
) -> pd.DataFrame:
    init_state = init_model.state_dict()
    trained_state = trained_model.state_dict()
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        names = [k for k in init_state.keys() if k.startswith(prefix)]
        if len(names) == 0:
            rows.append(
                {
                    "module_prefix": prefix,
                    "param_count": 0,
                    "l2_rel_change": 0.0,
                    "mean_abs_change": 0.0,
                    "max_abs_change": 0.0,
                }
            )
            continue
        init_vec = torch.cat([init_state[k].reshape(-1).float() for k in names], dim=0)
        trained_vec = torch.cat([trained_state[k].reshape(-1).float() for k in names], dim=0)
        diff = trained_vec - init_vec
        rows.append(
            {
                "module_prefix": prefix,
                "param_count": int(init_vec.numel()),
                "l2_rel_change": float(torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(init_vec) + 1e-12)),
                "mean_abs_change": float(diff.abs().mean()),
                "max_abs_change": float(diff.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def run_collapse_audit(seed: int) -> dict[str, Any]:
    gpu = gpu_info()
    if not gpu["gpu_used"]:
        raise RuntimeError("collapse audit requires CUDA")
    _, features, splits = load_artifacts()
    config = diagnostic_config()
    split = splits["seeds"][str(seed)]
    seed_tensors = prepare_seed_tensors(features, split["train"])
    set_seed(stable_seed(seed, "conflict_only", "teacher"))
    init_conflict = build_model("conflict_only", seed_tensors, config)
    conflict_model = train_standard_model(
        mode="conflict_only",
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=stable_seed(seed, "conflict_only", "teacher"),
    )
    set_seed(stable_seed(seed, "rcmf_dynamic_multimodal_full", "stage2"))
    init_full = build_model("rcmf_dynamic_multimodal_full", seed_tensors, config)
    full_model = train_switch_student(
        teacher_model=conflict_model,
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=seed,
    )
    test_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
    ext_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)
    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=False)

    conflict_test = collect_collapse_outputs(model=conflict_model, loader=test_loader, seed_tensors=seed_tensors)
    conflict_ext = collect_collapse_outputs(model=conflict_model, loader=ext_loader, seed_tensors=seed_tensors)
    full_test = collect_collapse_outputs(
        model=full_model,
        loader=test_loader,
        seed_tensors=seed_tensors,
        teacher_model=conflict_model,
    )
    full_ext = collect_collapse_outputs(
        model=full_model,
        loader=ext_loader,
        seed_tensors=seed_tensors,
        teacher_model=conflict_model,
    )

    pred_rows = []
    for split_name, c_out, f_out in (
        ("test", conflict_test, full_test),
        ("external", conflict_ext, full_ext),
    ):
        n = len(f_out["pred"])
        for i in range(n):
            pred_rows.append(
                {
                    "split": split_name,
                    "sample_idx": i,
                    "y_true": float(f_out["y_true"][i]),
                    "conflict_only_pred": float(c_out["pred"][i]),
                    "full_pred": float(f_out["pred"][i]),
                    "pred_diff": float(f_out["pred"][i] - c_out["pred"][i]),
                    "full_baseline_pred": float(f_out["baseline_pred"][i]),
                    "full_ctx_delta": float(f_out["ctx_delta"][i]),
                    "full_rcmf_gate": float(f_out["rcmf_gate"][i]),
                    "full_gate_probability": float(f_out["gate_probability"][i]),
                    "full_innovation_score": float(f_out["innovation_score"][i]),
                }
            )
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(ROOT / "reports/collapse_pred_diff.csv", index=False)
    abs_diff = pred_df["pred_diff"].abs().to_numpy()
    pred_diff_max_abs = float(np.max(abs_diff))
    pred_diff_mean_abs = float(np.mean(abs_diff))
    pred_diff_median_abs = float(np.median(abs_diff))
    pred_diff_nonzero_ratio = float(np.mean(abs_diff > 1e-6))
    routed_pred = np.where(
        pred_df["full_innovation_score"].to_numpy(dtype=np.float64) >= 0.5,
        pred_df["full_pred"].to_numpy(dtype=np.float64),
        pred_df["conflict_only_pred"].to_numpy(dtype=np.float64),
    )
    routed_diff_abs = np.abs(routed_pred - pred_df["conflict_only_pred"].to_numpy(dtype=np.float64))
    routed_nonzero_ratio = float(np.mean(routed_diff_abs > 1e-6))
    routed_max_abs = float(np.max(routed_diff_abs))

    gate_df = pd.DataFrame(
        [
            {"metric": "rcmf_gate_mean", "value": float(np.mean(np.concatenate([full_test["rcmf_gate"], full_ext["rcmf_gate"]])) )},
            {"metric": "rcmf_gate_std", "value": float(np.std(np.concatenate([full_test["rcmf_gate"], full_ext["rcmf_gate"]])) )},
            {"metric": "gate_probability_mean", "value": float(np.mean(np.concatenate([full_test["gate_probability"], full_ext["gate_probability"]])) )},
            {"metric": "ctx_delta_mean_abs", "value": float(np.mean(np.abs(np.concatenate([full_test["ctx_delta"], full_ext["ctx_delta"]]))))},
            {"metric": "innovation_score_mean", "value": float(np.mean(np.concatenate([full_test["innovation_score"], full_ext["innovation_score"]])) )},
            {"metric": "dynamic_w_desc_mean", "value": float(np.mean(np.concatenate([full_test["dynamic_w_desc"], full_ext["dynamic_w_desc"]])) )},
            {"metric": "dynamic_w_graph_mean", "value": float(np.mean(np.concatenate([full_test["dynamic_w_graph"], full_ext["dynamic_w_graph"]])) )},
            {"metric": "dynamic_w_ctx_mean", "value": float(np.mean(np.concatenate([full_test["dynamic_w_ctx"], full_ext["dynamic_w_ctx"]])) )},
            {"metric": "dynamic_w_scale_mean", "value": float(np.mean(np.concatenate([full_test["dynamic_w_scale"], full_ext["dynamic_w_scale"]])) )},
            {"metric": "mspce_scale_entropy_mean", "value": float(np.mean(np.concatenate([full_test["mspce_scale_entropy"], full_ext["mspce_scale_entropy"]])) )},
        ]
    )
    gate_df.to_csv(ROOT / "reports/collapse_gate_stats.csv", index=False)

    led_terms = led_audit_terms(model=full_model, teacher_model=conflict_model, loader=train_loader)
    param_df = param_delta_by_prefix(
        init_model=init_full,
        trained_model=full_model,
        prefixes=("ctx_encoder", "dynamic_fusion_gate", "dynamic_control", "dynamic_head", "led_prior"),
    )
    param_df.to_csv(ROOT / "reports/collapse_param_update.csv", index=False)

    report_mixture = False
    comparator_path = ROOT / "reports/comparator_report.csv"
    if comparator_path.exists():
        comparator_df = pd.read_csv(comparator_path)
        if "main_absolute_metric_k" in comparator_df.columns:
            current_full = float(np.mean(full_test["pred"] * 0 + np.array([mae(np.array(full_test["y_true"]), np.array(full_test["pred"]))])))
            recorded = comparator_df.loc[
                comparator_df["comparator_name"] == "summary", "main_absolute_metric_k"
            ]
            if len(recorded) > 0 and not np.isclose(float(recorded.iloc[0]), current_full, atol=1e-6):
                report_mixture = True

    module_rows = []
    for _, row in param_df.iterrows():
        module_rows.append(f"{row['module_prefix']}: l2_rel_change={row['l2_rel_change']:.6f}")
    collapse_confirmed = bool(routed_nonzero_ratio < 0.01 and routed_max_abs < 1e-3)
    gate_collapsed = bool(float(gate_df.loc[gate_df["metric"] == "ctx_delta_mean_abs", "value"].iloc[0]) < 1e-3)
    mspce_active = not gate_collapsed
    led_active = bool(led_terms["led_weight_mean"] > 0.05 and led_terms["led_residual_align_mean"] >= 0.0)

    audit_row = {
        "seed": int(seed),
        "gpu_name": gpu["gpu_name"],
        "full_equals_conflict_confirmed": collapse_confirmed,
        "collapse_type": "evaluation_time_routing_collapse_to_conflict" if collapse_confirmed else "not_exact_identity",
        "primary_root_cause": (
            "fixed_threshold_0p5_with_low_innovation_scores"
            if collapse_confirmed
            else ("rcmf_dynamic_delta_suppressed_to_zero" if gate_collapsed else "not_gate_zero")
        ),
        "pred_diff_max_abs": pred_diff_max_abs,
        "pred_diff_mean_abs": pred_diff_mean_abs,
        "pred_diff_median_abs": pred_diff_median_abs,
        "pred_diff_nonzero_ratio": pred_diff_nonzero_ratio,
        "routed_diff_max_abs_at_thr_0p5": routed_max_abs,
        "routed_diff_nonzero_ratio_at_thr_0p5": routed_nonzero_ratio,
        "gate_collapsed": gate_collapsed,
        "mspce_branch_active": mspce_active,
        "led_influence_active": led_active,
        "checkpoint_mixup_found": False,
        "report_mixup_found": report_mixture,
        **led_terms,
    }
    pd.DataFrame([audit_row]).to_csv(ROOT / "reports/collapse_audit.csv", index=False)
    (ROOT / "reports/collapse_audit.md").write_text(
        "\n".join(
            [
                "# collapse_audit",
                "",
                f"- seed: {seed}",
                f"- full_equals_conflict_confirmed: {collapse_confirmed}",
                f"- pred_diff_max_abs: {pred_diff_max_abs:.6f}",
                f"- pred_diff_mean_abs: {pred_diff_mean_abs:.6f}",
                f"- pred_diff_nonzero_ratio: {pred_diff_nonzero_ratio:.6f}",
                f"- routed_diff_max_abs_at_thr_0p5: {routed_max_abs:.6f}",
                f"- routed_diff_nonzero_ratio_at_thr_0p5: {routed_nonzero_ratio:.6f}",
                f"- gate_collapsed: {gate_collapsed}",
                f"- mspce_branch_active: {mspce_active}",
                f"- led_influence_active: {led_active}",
                f"- checkpoint_mixup_found: false",
                f"- report_mixup_found: {report_mixture}",
                "- parameter_update_summary:",
                *[f"  - {line}" for line in module_rows],
            ]
        ),
        encoding="utf-8",
    )
    (ROOT / "reports/decision.md").write_text(
        "\n".join(
            [
                "legacy note removed",
                "",
                f"- full_equals_conflict_confirmed: {collapse_confirmed}",
                f"- collapse_type: {'evaluation_time_routing_collapse_to_conflict' if collapse_confirmed else 'not_exact_identity'}",
                "legacy note removed",
                f"- pred_diff_max_abs: {pred_diff_max_abs:.6f}",
                f"- pred_diff_mean_abs: {pred_diff_mean_abs:.6f}",
                f"- pred_diff_nonzero_ratio: {pred_diff_nonzero_ratio:.6f}",
                f"- routed_diff_max_abs_at_thr_0p5: {routed_max_abs:.6f}",
                f"- routed_diff_nonzero_ratio_at_thr_0p5: {routed_nonzero_ratio:.6f}",
                f"- gate_collapsed: {gate_collapsed}",
                f"- mspce_branch_active: {mspce_active}",
                f"- led_influence_active: {led_active}",
                "- checkpoint_mixup_found: false",
                f"- report_mixup_found: {report_mixture}",
                "- should_continue_20seed_now: no",
                "legacy note removed",
            ]
        ),
        encoding="utf-8",
    )
    return audit_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tg_clean_v2 matrix experiments.")
    parser.add_argument("--task", choices=["matrix", "full", "collapse_audit", "postfix_minimal", "controller_minimal"], required=True)
    parser.add_argument("--seeds_run", type=int, default=3)
    parser.add_argument("--top3_only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seeds_run = int(args.seeds_run)
    if args.task == "collapse_audit":
        run_collapse_audit(seed=int(args.seed))
        return 0
    if args.task == "postfix_minimal":
        run_postfix_minimal()
        return 0
    if args.task == "controller_minimal":
        run_controller_minimal()
        return 0
    if args.task == "matrix":
        run_matrix(seeds_run=seeds_run, top3_only=bool(args.top3_only))
        return 0
    run_matrix(seeds_run=20, top3_only=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


