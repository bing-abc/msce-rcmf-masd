from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass(frozen=True)
class MASDConfig:
    slot_count: int = 4
    hidden_dim: int = 128
    slot_hidden_dim: int = 64
    dropout: float = 0.10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 120
    patience: int = 16
    late_epoch_candidate_count: int = 6
    late_epoch_top_soup_count: int = 4
    tail_focus_epochs: int = 4
    tail_focus_lr_scale: float = 0.35
    tail_focus_weight_scale: float = 2.0
    enable_split_b: bool = False
    split_b_fraction: float = 0.20
    split_b_min_size: int = 32
    split_b_head_epochs: int = 6
    split_b_patience: int = 2
    split_b_lr_scale: float = 0.12
    split_b_weight_decay_scale: float = 3.0
    split_b_primary_epsilon: float = 0.02
    blend_grid_size: int = 21
    blend_alpha_max: float = 3.00
    delta_bound_k: float = 20.0
    slot_min_magnitude_k: float = 0.5
    slot_max_magnitude_k: float = 10.0
    residual_cap_k: float = 3.0
    alpha_temperature: float = 0.70
    gate_low: float = 0.02
    gate_high: float = 0.55
    hard_threshold_tau: float = 0.70
    hard_threshold_gamma: float = 8.0
    calibration_tau_candidates: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70)
    calibration_gate_floor_candidates: tuple[float, ...] = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
    calibration_focus_gain_candidates: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
    calibration_cap_candidates: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 12.0)
    activation_floor: float = 0.05
    activation_ceiling: float = 0.45
    activation_delta_threshold_k: float = 0.25
    anchor_margin_weight: float = 0.84
    proxy_alignment_weight: float = 0.14
    sign_weight: float = 0.22
    sparse_weight: float = 0.10
    calibrator_weight: float = 0.10
    diversity_weight: float = 0.10
    gate_high_penalty_weight: float = 0.14
    gate_low_penalty_weight: float = 0.05
    delta_weight: float = 0.04
    hard_focus_weight: float = 0.08
    residual_target_weight: float = 0.18


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float32).std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return std


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits, _ = torch.sort(shifted, dim=dim, descending=True)
    rank_shape = [1] * shifted.ndim
    rank_shape[dim] = shifted.shape[dim]
    ranks = torch.arange(1, shifted.shape[dim] + 1, device=shifted.device, dtype=shifted.dtype).view(rank_shape)
    support = ranks * sorted_logits > (sorted_logits.cumsum(dim=dim) - 1.0)
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (
        (sorted_logits.cumsum(dim=dim) - 1.0)
        .gather(dim, support_size.to(dtype=torch.long) - 1)
        / support_size.to(dtype=shifted.dtype)
    )
    return torch.clamp(shifted - tau, min=0.0)


def _build_sign_prior(slot_count: int) -> np.ndarray:
    sign_prior_map = {
        2: [1.0, -1.0],
        3: [1.0, 1.0, -1.0],
        4: [1.0, 0.65, -1.0, -1.0],
        6: [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
    }
    if int(slot_count) not in sign_prior_map:
        raise ValueError(f"unsupported MASD slot count: {slot_count}")
    return np.asarray(sign_prior_map[int(slot_count)], dtype=np.float32)


def _expand_proxy_scores(base_scores: np.ndarray, slot_count: int) -> np.ndarray:
    if int(slot_count) == 4:
        return np.asarray(base_scores, dtype=np.float32)
    pos_1 = base_scores[:, 0:1]
    pos_2 = base_scores[:, 1:2]
    neg_1 = base_scores[:, 2:3]
    neg_2 = base_scores[:, 3:4]
    pos_mean = 0.5 * (pos_1 + pos_2)
    neg_mean = 0.5 * (neg_1 + neg_2)
    if int(slot_count) == 2:
        return np.concatenate([pos_mean, neg_mean], axis=1).astype(np.float32)
    if int(slot_count) == 3:
        return np.concatenate([pos_1, pos_2, neg_mean], axis=1).astype(np.float32)
    if int(slot_count) == 6:
        return np.concatenate([pos_1, pos_2, pos_mean, neg_1, neg_2, neg_mean], axis=1).astype(np.float32)
    raise ValueError(f"unsupported MASD slot count: {slot_count}")


def _mechanism_proxy_scores(descriptor_matrix: np.ndarray, context_matrix: np.ndarray, slot_count: int) -> np.ndarray:
    descriptor = np.asarray(descriptor_matrix, dtype=np.float32)
    context = np.asarray(context_matrix, dtype=np.float32)
    if descriptor.ndim != 2 or descriptor.shape[1] < 10:
        raise ValueError("descriptor matrix must contain at least the first 10 descriptor features")
    if context.ndim != 2 or context.shape[1] < 12:
        raise ValueError("context matrix must contain the interpretable context block")

    interpretable = context[:, -12:]
    molwt = descriptor[:, 0:1]
    logp = descriptor[:, 1:2]
    tpsa = descriptor[:, 2:3]
    hba = descriptor[:, 4:5]
    hbd = descriptor[:, 5:6]
    rot = descriptor[:, 6:7]
    ring = descriptor[:, 7:8]
    aromatic_ring = descriptor[:, 8:9]
    fraction_csp3 = descriptor[:, 9:10]

    token_count = interpretable[:, 0:1]
    branch_count = 0.5 * (interpretable[:, 2:3] + interpretable[:, 3:4])
    atom_count = interpretable[:, 6:7]
    bond_count = interpretable[:, 7:8]
    ring_ctx = interpretable[:, 8:9]
    aromatic_ratio = interpretable[:, 9:10]
    hetero_ratio = interpretable[:, 10:11]
    csp3_ctx = interpretable[:, 11:12]

    branch_ratio = branch_count / np.maximum(token_count, 1.0)
    mean_degree = 2.0 * bond_count / np.maximum(atom_count, 1.0)

    rigidity_rotation = (
        0.42 * aromatic_ring
        + 0.28 * ring
        + 0.24 * ring_ctx
        + 0.24 * aromatic_ratio
        + 0.18 * mean_degree
        - 0.52 * rot
        - 0.18 * fraction_csp3
    )
    intermolecular_polarity = (
        0.46 * tpsa
        + 0.30 * hba
        + 0.26 * hbd
        + 0.28 * hetero_ratio
        - 0.10 * logp
    )
    freevolume_packing = (
        0.32 * molwt
        + 0.24 * atom_count
        + 0.26 * ring_ctx
        + 0.16 * mean_degree
        + 0.16 * aromatic_ratio
        - 0.26 * branch_ratio
        - 0.20 * csp3_ctx
    )
    sidechain_internalplasticization = (
        0.52 * rot
        + 0.34 * branch_ratio
        + 0.28 * csp3_ctx
        + 0.12 * logp
        - 0.28 * ring
        - 0.22 * aromatic_ratio
    )
    base_scores = np.concatenate(
        [
            rigidity_rotation,
            intermolecular_polarity,
            freevolume_packing,
            sidechain_internalplasticization,
        ],
        axis=1,
    ).astype(np.float32)
    expanded = _expand_proxy_scores(base_scores, slot_count=int(slot_count))
    centered = expanded - expanded.mean(axis=1, keepdims=True)
    scaled = centered / np.maximum(expanded.std(axis=1, keepdims=True), 1e-6)
    return scaled.astype(np.float32)


def _softmax_rows(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float32) / float(max(temperature, 1e-6))
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    averaged: dict[str, torch.Tensor] = {}
    keys = list(state_dicts[0].keys())
    for key in keys:
        values = [state[key] for state in state_dicts]
        reference = values[0]
        if torch.is_floating_point(reference):
            stacked = torch.stack([value.to(dtype=torch.float32) for value in values], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=reference.dtype)
        else:
            averaged[key] = reference.clone()
    return averaged


def _quantile_edges(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64).reshape(-1)
    if vec.size == 0:
        return np.zeros((len(quantiles),), dtype=np.float64)
    edges = np.asarray([np.quantile(vec, quantile) for quantile in quantiles], dtype=np.float64)
    if edges.size <= 1:
        return edges
    for index in range(1, edges.size):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1e-6
    return edges


class _MASDNet(nn.Module):
    def __init__(self, *, descriptor_dim: int, context_dim: int, anchor_feature_dim: int, config: MASDConfig) -> None:
        super().__init__()
        self.slot_count = int(config.slot_count)
        self.slot_min_magnitude_k = float(config.slot_min_magnitude_k)
        self.slot_max_magnitude_k = float(config.slot_max_magnitude_k)
        self.residual_cap_k = float(config.residual_cap_k)
        self.delta_bound_k = float(config.delta_bound_k)
        self.alpha_temperature = float(config.alpha_temperature)
        self.gate_low = float(config.gate_low)
        self.gate_high = float(config.gate_high)
        self.hard_threshold_tau = float(config.hard_threshold_tau)
        self.hard_threshold_gamma = float(config.hard_threshold_gamma)

        self.register_buffer("sign_prior", torch.tensor(_build_sign_prior(self.slot_count), dtype=torch.float32).reshape(1, -1))

        self.descriptor_encoder = nn.Sequential(
            nn.Linear(int(descriptor_dim), int(config.hidden_dim)),
            nn.LayerNorm(int(config.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(int(context_dim), int(config.hidden_dim)),
            nn.LayerNorm(int(config.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(int(anchor_feature_dim), int(config.hidden_dim // 2)),
            nn.LayerNorm(int(config.hidden_dim // 2)),
            nn.GELU(),
        )
        self.core = nn.Sequential(
            nn.Linear(int(2 * config.hidden_dim + config.hidden_dim // 2 + 1), int(config.hidden_dim)),
            nn.LayerNorm(int(config.hidden_dim)),
            nn.GELU(),
        )
        self.slot_bank = nn.Embedding(self.slot_count, int(config.slot_hidden_dim))
        self.slot_projector = nn.Sequential(
            nn.Linear(int(config.hidden_dim + config.slot_hidden_dim), int(config.slot_hidden_dim)),
            nn.LayerNorm(int(config.slot_hidden_dim)),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(int(config.slot_hidden_dim), 1)
        self.magnitude_head = nn.Linear(int(config.slot_hidden_dim), 1)
        self.residual_head = nn.Linear(int(config.slot_hidden_dim), 1)
        self.gate_context = nn.Sequential(
            nn.Linear(int(config.hidden_dim + 4), int(config.hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(int(config.hidden_dim // 2), 1),
        )
        self.risk_weights = nn.Parameter(torch.tensor([1.20, 1.05, 0.85], dtype=torch.float32))
        self.risk_bias = nn.Parameter(torch.tensor(1.05, dtype=torch.float32))
        self.alpha_risk_weights = nn.Parameter(torch.tensor([1.10, 1.00, 0.70], dtype=torch.float32))
        self.alpha_risk_bias = nn.Parameter(torch.tensor(-1.95, dtype=torch.float32))

    def forward(
        self,
        descriptor: torch.Tensor,
        context: torch.Tensor,
        anchor_pred: torch.Tensor,
        anchor_features: torch.Tensor,
        proxy_scores: torch.Tensor,
        hard_proxy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        descriptor_hidden = self.descriptor_encoder(descriptor)
        context_hidden = self.context_encoder(context)
        anchor_hidden = self.anchor_encoder(anchor_features)
        core = self.core(torch.cat([descriptor_hidden, context_hidden, anchor_hidden, anchor_pred], dim=1))

        slot_bank = self.slot_bank.weight.unsqueeze(0).expand(core.shape[0], -1, -1)
        slot_hidden = self.slot_projector(
            torch.cat([core.unsqueeze(1).expand(-1, self.slot_count, -1), slot_bank], dim=2)
        )

        alpha_logits = self.alpha_head(slot_hidden).squeeze(-1) + 0.38 * proxy_scores
        diversity_seed = 1.0 - torch.std(slot_hidden, dim=1).mean(dim=1, keepdim=True)
        alpha_risk = torch.sigmoid(
            F.softplus(self.alpha_risk_weights[0]) * hard_proxy
            + F.softplus(self.alpha_risk_weights[1]) * proxy_scores.std(dim=1, keepdim=True)
            + F.softplus(self.alpha_risk_weights[2]) * (1.0 - diversity_seed)
            + self.alpha_risk_bias
        )
        alpha = _sparsemax(alpha_logits / (self.alpha_temperature + 0.42 * alpha_risk), dim=1)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        alpha_sorted, _ = torch.sort(alpha, dim=1, descending=True)
        alpha_max = alpha_sorted[:, :1]
        alpha_margin = alpha_sorted[:, :1] - alpha_sorted[:, 1:2]
        entropy = -(
            alpha.clamp_min(1e-8) * torch.log(alpha.clamp_min(1e-8))
        ).sum(dim=1, keepdim=True) / math.log(float(self.slot_count))

        magnitude_logits = self.magnitude_head(slot_hidden).squeeze(-1) + 0.24 * proxy_scores
        main_mag = self.slot_min_magnitude_k + (self.slot_max_magnitude_k - self.slot_min_magnitude_k) * torch.sigmoid(magnitude_logits)
        residual = self.residual_cap_k * torch.tanh(self.residual_head(slot_hidden).squeeze(-1))
        delta = self.sign_prior.to(descriptor.device) * main_mag + residual
        contribution = alpha * delta
        contribution_sum = contribution.sum(dim=1, keepdim=True)
        mechanism_disagreement = torch.std(contribution, dim=1, keepdim=True) / max(self.delta_bound_k, 1e-6)

        risk_linear = (
            F.softplus(self.risk_weights[0]) * hard_proxy
            + F.softplus(self.risk_weights[1]) * entropy
            + F.softplus(self.risk_weights[2]) * mechanism_disagreement
        )
        gate_input = torch.cat([core, entropy, 1.0 - alpha_max, mechanism_disagreement, hard_proxy], dim=1)
        gate = self.gate_low + (self.gate_high - self.gate_low) * torch.sigmoid(self.risk_bias + self.gate_context(gate_input) - risk_linear)
        thresholded_gate = torch.sigmoid(self.hard_threshold_gamma * (hard_proxy - self.hard_threshold_tau))
        raw_applied_delta = gate * thresholded_gate * contribution_sum
        bounded_delta = torch.clamp(raw_applied_delta, min=-self.delta_bound_k, max=self.delta_bound_k)
        pred = anchor_pred + bounded_delta

        slot_norm = F.normalize(slot_hidden, dim=2)
        pairwise = torch.matmul(slot_norm, slot_norm.transpose(1, 2))
        eye = torch.eye(self.slot_count, dtype=slot_hidden.dtype, device=slot_hidden.device).unsqueeze(0)
        offdiag = pairwise * (1.0 - eye)
        diversity = 1.0 - offdiag.abs().sum(dim=(1, 2), keepdim=True) / float(self.slot_count * (self.slot_count - 1))

        return {
            "pred": pred,
            "anchor_pred": anchor_pred,
            "bounded_delta": bounded_delta,
            "raw_bounded_delta": raw_applied_delta,
            "gate": gate,
            "thresholded_gate": thresholded_gate,
            "hard_proxy": hard_proxy,
            "alpha": alpha,
            "alpha_max": alpha_max,
            "alpha_margin": alpha_margin,
            "entropy": entropy,
            "delta": delta,
            "contribution_sum": contribution_sum,
            "main_mag": main_mag,
            "residual": residual,
            "contribution": contribution,
            "mechanism_disagreement": mechanism_disagreement,
            "diversity": diversity,
            "dominant_mechanism": alpha.argmax(dim=1),
        }


class MASDRegressor:
    def __init__(self, config: MASDConfig, *, train_seed: int) -> None:
        self.config = config
        self.train_seed = int(train_seed)
        self.device = _device()
        self.model: _MASDNet | None = None
        self.descriptor_mean: np.ndarray | None = None
        self.descriptor_std: np.ndarray | None = None
        self.context_mean: np.ndarray | None = None
        self.context_std: np.ndarray | None = None
        self.anchor_feature_mean: np.ndarray | None = None
        self.anchor_feature_std: np.ndarray | None = None
        self.hard_raw_mean: float = 0.0
        self.hard_raw_std: float = 1.0
        self.blend_alpha: float = 0.0
        self.calibrated_tau: float = float(self.config.hard_threshold_tau)
        self.calibrated_gate_floor: float = 0.0
        self.calibrated_focus_gain: float = 1.0
        self.calibrated_cap_k: float = float(self.config.delta_bound_k)
        self.aggregation_name: str = "raw_best"
        self.split_b_meta: dict[str, float] = {}

    def _prepare_stats(
        self,
        *,
        descriptor_train: np.ndarray,
        context_train: np.ndarray,
        anchor_feature_train: np.ndarray,
        hard_raw_train: np.ndarray,
    ) -> None:
        self.descriptor_mean = np.asarray(descriptor_train, dtype=np.float32).mean(axis=0, keepdims=True)
        self.descriptor_std = _safe_std(descriptor_train)
        self.context_mean = np.asarray(context_train, dtype=np.float32).mean(axis=0, keepdims=True)
        self.context_std = _safe_std(context_train)
        self.anchor_feature_mean = np.asarray(anchor_feature_train, dtype=np.float32).mean(axis=0, keepdims=True)
        self.anchor_feature_std = _safe_std(anchor_feature_train)
        hard_vec = np.asarray(hard_raw_train, dtype=np.float32).reshape(-1)
        self.hard_raw_mean = float(hard_vec.mean())
        self.hard_raw_std = float(max(hard_vec.std(), 1e-6))

    def _standardize_inputs(
        self,
        *,
        descriptor: np.ndarray,
        context: np.ndarray,
        anchor_features: np.ndarray,
        hard_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.descriptor_mean is None or self.descriptor_std is None:
            raise RuntimeError("MASDRegressor has not been fit yet")
        if self.context_mean is None or self.context_std is None:
            raise RuntimeError("MASDRegressor has not been fit yet")
        if self.anchor_feature_mean is None or self.anchor_feature_std is None:
            raise RuntimeError("MASDRegressor has not been fit yet")
        descriptor_z = (np.asarray(descriptor, dtype=np.float32) - self.descriptor_mean) / self.descriptor_std
        context_z = (np.asarray(context, dtype=np.float32) - self.context_mean) / self.context_std
        anchor_feature_z = (np.asarray(anchor_features, dtype=np.float32) - self.anchor_feature_mean) / self.anchor_feature_std
        hard_proxy = 1.0 / (
            1.0
            + np.exp(
                -(
                    (np.asarray(hard_raw, dtype=np.float32).reshape(-1, 1) - float(self.hard_raw_mean))
                    / float(self.hard_raw_std)
                )
            )
        )
        return (
            descriptor_z.astype(np.float32),
            context_z.astype(np.float32),
            anchor_feature_z.astype(np.float32),
            hard_proxy.astype(np.float32),
        )

    def _anchor_features(self, *, msce_details: dict[str, np.ndarray], rcmf_details: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        msce_pred = np.asarray(msce_details["pred"], dtype=np.float32).reshape(-1, 1)
        rcmf_pred = np.asarray(rcmf_details["pred"], dtype=np.float32).reshape(-1, 1)
        rcmf_delta = rcmf_pred - msce_pred
        trust_gate = np.asarray(rcmf_details["trust_gate"], dtype=np.float32).reshape(-1, 1)
        joint_gate = np.asarray(rcmf_details["joint_gate"], dtype=np.float32).reshape(-1, 1)
        conflict = np.asarray(rcmf_details["conflict"], dtype=np.float32).reshape(-1, 1)
        msce_delta = np.asarray(msce_details["bounded_delta"], dtype=np.float32).reshape(-1, 1)
        selection_entropy = np.asarray(msce_details["selection_entropy"], dtype=np.float32).reshape(-1, 1)
        dominant_weight = np.asarray(msce_details["dominant_scale_weight"], dtype=np.float32).reshape(-1, 1)
        scale_weights = np.asarray(msce_details["scale_weights"], dtype=np.float32)
        features = np.concatenate(
            [
                rcmf_delta,
                trust_gate,
                joint_gate,
                conflict,
                np.abs(msce_delta),
                selection_entropy,
                dominant_weight,
                scale_weights,
            ],
            axis=1,
        ).astype(np.float32)
        hard_raw = (
            conflict
            + (1.0 - trust_gate)
            + selection_entropy
            + (1.0 - dominant_weight)
            + 0.5 * np.abs(joint_gate)
        ).astype(np.float32)
        return features, hard_raw

    def _hard_target_from_anchor(self, *, anchor_pred: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        anchor_vec = np.asarray(anchor_pred, dtype=np.float32).reshape(-1)
        target_vec = np.asarray(y_true, dtype=np.float32).reshape(-1)
        error = np.abs(anchor_vec - target_vec)
        q60 = float(np.quantile(error, 0.60))
        q80 = float(np.quantile(error, 0.80))
        q90 = float(np.quantile(error, 0.90))
        denom = max(q90 - q60, 1e-6)
        soft_target = np.clip((error - q60) / denom, 0.0, 1.0)
        hard_mask = error >= q80
        soft_target = np.where(hard_mask, 1.0, soft_target).astype(np.float32)
        return soft_target.reshape(-1, 1), hard_mask.astype(bool)

    def _resolve_hard_target(
        self,
        *,
        benchmark_hard_mask: np.ndarray | None,
        anchor_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if benchmark_hard_mask is not None:
            mask = np.asarray(benchmark_hard_mask, dtype=bool).reshape(-1)
            return mask.astype(np.float32).reshape(-1, 1), mask
        return self._hard_target_from_anchor(anchor_pred=anchor_pred, y_true=y_true)

    def _build_loader(
        self,
        *,
        descriptor: np.ndarray,
        context: np.ndarray,
        anchor_pred: np.ndarray,
        anchor_features: np.ndarray,
        proxy_scores: np.ndarray,
        hard_proxy: np.ndarray,
        hard_target: np.ndarray,
        target: np.ndarray,
        shuffle: bool,
        sample_weights: np.ndarray | None = None,
    ) -> DataLoader:
        tensors = [
            torch.tensor(np.asarray(descriptor, dtype=np.float32), dtype=torch.float32),
            torch.tensor(np.asarray(context, dtype=np.float32), dtype=torch.float32),
            torch.tensor(np.asarray(anchor_pred, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
            torch.tensor(np.asarray(anchor_features, dtype=np.float32), dtype=torch.float32),
            torch.tensor(np.asarray(proxy_scores, dtype=np.float32), dtype=torch.float32),
            torch.tensor(np.asarray(hard_proxy, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
            torch.tensor(np.asarray(hard_target, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
            torch.tensor(np.asarray(target, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
        ]
        dataset = TensorDataset(*tensors)
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
            sampler = WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.double),
                num_samples=int(weights.shape[0]),
                replacement=True,
            )
            return DataLoader(dataset, batch_size=int(self.config.batch_size), sampler=sampler)
        return DataLoader(dataset, batch_size=int(self.config.batch_size), shuffle=bool(shuffle))

    def _set_split_b_trainable(self) -> None:
        if self.model is None:
            raise RuntimeError("MASDRegressor has not been initialized")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for module_name in ("slot_bank", "slot_projector", "alpha_head", "magnitude_head", "residual_head", "gate_context"):
            module = getattr(self.model, module_name)
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter_name in ("risk_weights", "risk_bias", "alpha_risk_weights", "alpha_risk_bias"):
            parameter = getattr(self.model, parameter_name)
            parameter.requires_grad = True

    def _build_split_b_partition(
        self,
        *,
        hard_target: np.ndarray,
        hard_raw: np.ndarray,
        msce_details: dict[str, np.ndarray],
        rcmf_details: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        train_size = int(np.asarray(hard_target).shape[0])
        if train_size <= 3:
            indices = np.arange(train_size, dtype=np.int64)
            return indices, np.zeros((0,), dtype=np.int64), {
                "split_a_count": float(train_size),
                "split_b_count": 0.0,
                "split_b_fraction": 0.0,
                "risk_edge_50": 0.0,
                "risk_edge_80": 0.0,
            }

        hard_mask = np.asarray(hard_target, dtype=np.float32).reshape(-1) >= 0.5
        selection_entropy = np.asarray(msce_details["selection_entropy"], dtype=np.float32).reshape(-1)
        conflict = np.asarray(rcmf_details["conflict"], dtype=np.float32).reshape(-1)
        risk_score = (
            np.asarray(hard_raw, dtype=np.float32).reshape(-1)
            + 0.25 * selection_entropy
            + 0.15 * conflict
        ).astype(np.float32)
        risk_edges = _quantile_edges(risk_score, (0.50, 0.80))
        risk_bin = np.digitize(risk_score, bins=risk_edges, right=False)

        strata: dict[tuple[int, int], list[int]] = {}
        for idx, is_hard, bucket in zip(range(train_size), hard_mask.tolist(), risk_bin.tolist()):
            strata.setdefault((int(is_hard), int(bucket)), []).append(int(idx))

        rng = np.random.default_rng(self.train_seed * 131 + 94000)
        split_a: list[int] = []
        split_b: list[int] = []
        for key in sorted(strata.keys()):
            members = list(strata[key])
            rng.shuffle(members)
            if len(members) <= 1:
                split_a.extend(members)
                continue
            cut = int(round(len(members) * (1.0 - float(self.config.split_b_fraction))))
            cut = min(max(cut, 1), len(members) - 1)
            split_a.extend(members[:cut])
            split_b.extend(members[cut:])

        target_b = int(round(train_size * float(self.config.split_b_fraction)))
        target_b = max(1, target_b)
        target_b = max(target_b, min(int(self.config.split_b_min_size), max(train_size - 1, 1)))
        target_b = min(target_b, max(train_size - 1, 1))
        if len(split_b) < target_b:
            move_budget = target_b - len(split_b)
            move_candidates = sorted(split_a, key=lambda idx: (-float(risk_score[idx]), -float(hard_mask[idx]), int(idx)))
            moved = move_candidates[:move_budget]
            if moved:
                moved_set = set(int(idx) for idx in moved)
                split_a = [int(idx) for idx in split_a if int(idx) not in moved_set]
                split_b.extend(int(idx) for idx in moved)

        split_a_arr = np.asarray(sorted(set(int(idx) for idx in split_a)), dtype=np.int64)
        split_b_arr = np.asarray(sorted(set(int(idx) for idx in split_b)), dtype=np.int64)
        meta = {
            "split_a_count": float(split_a_arr.shape[0]),
            "split_b_count": float(split_b_arr.shape[0]),
            "split_b_fraction": float(split_b_arr.shape[0] / max(train_size, 1)),
            "risk_edge_50": float(risk_edges[0]) if risk_edges.size >= 1 else 0.0,
            "risk_edge_80": float(risk_edges[1]) if risk_edges.size >= 2 else 0.0,
        }
        return split_a_arr, split_b_arr, meta

    @torch.no_grad()
    def _split_b_signal_summary(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("MASDRegressor has not been initialized")
        self.model.eval()
        hard_parts: list[np.ndarray] = []
        entropy_parts: list[np.ndarray] = []
        disagreement_parts: list[np.ndarray] = []
        for descriptor_tensor, context_tensor, anchor_pred_tensor, anchor_feature_tensor, proxy_tensor, hard_tensor, _hard_target_tensor, _y_tensor in loader:
            out = self.model(
                descriptor_tensor.to(self.device),
                context_tensor.to(self.device),
                anchor_pred_tensor.to(self.device),
                anchor_feature_tensor.to(self.device),
                proxy_tensor.to(self.device),
                hard_tensor.to(self.device),
            )
            hard_parts.append(out["hard_proxy"].detach().cpu().numpy().reshape(-1))
            entropy_parts.append(out["entropy"].detach().cpu().numpy().reshape(-1))
            disagreement_parts.append(out["mechanism_disagreement"].detach().cpu().numpy().reshape(-1))
        return (
            np.concatenate(hard_parts, axis=0),
            np.concatenate(entropy_parts, axis=0),
            np.concatenate(disagreement_parts, axis=0),
        )

    def _build_split_b_sample_weights(
        self,
        *,
        descriptor: np.ndarray,
        context: np.ndarray,
        anchor_pred: np.ndarray,
        anchor_features: np.ndarray,
        proxy_scores: np.ndarray,
        hard_proxy: np.ndarray,
        hard_target: np.ndarray,
        target: np.ndarray,
        split_b_indices: np.ndarray,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, dict[str, float]]:
        if split_b_indices.size == 0:
            return np.zeros((0,), dtype=np.float32), {
                "risk_hi": 0.0,
                "risk_mid": 0.0,
                "entropy_hi": 0.0,
                "disagreement_hi": 0.0,
                "weight_mean": 1.0,
                "weight_max": 1.0,
            }
        if self.model is None:
            raise RuntimeError("MASDRegressor has not been initialized")
        self.model.load_state_dict(state_dict)
        split_loader = self._build_loader(
            descriptor=np.asarray(descriptor, dtype=np.float32)[split_b_indices],
            context=np.asarray(context, dtype=np.float32)[split_b_indices],
            anchor_pred=np.asarray(anchor_pred, dtype=np.float32)[split_b_indices],
            anchor_features=np.asarray(anchor_features, dtype=np.float32)[split_b_indices],
            proxy_scores=np.asarray(proxy_scores, dtype=np.float32)[split_b_indices],
            hard_proxy=np.asarray(hard_proxy, dtype=np.float32)[split_b_indices],
            hard_target=np.asarray(hard_target, dtype=np.float32)[split_b_indices],
            target=np.asarray(target, dtype=np.float32)[split_b_indices],
            shuffle=False,
        )
        stage_hard, stage_entropy, stage_disagreement = self._split_b_signal_summary(split_loader)
        risk_hi = float(np.quantile(stage_hard, 0.80))
        risk_mid = float(np.quantile(stage_hard, 0.60))
        entropy_hi = float(np.quantile(stage_entropy, 0.80))
        disagreement_hi = float(np.quantile(stage_disagreement, 0.80))
        target_mask = np.asarray(hard_target, dtype=np.float32).reshape(-1)[split_b_indices] >= 0.5
        weights: list[float] = []
        for is_hard, risk, entropy, disagreement in zip(
            target_mask.tolist(),
            stage_hard.tolist(),
            stage_entropy.tolist(),
            stage_disagreement.tolist(),
        ):
            weight = 1.0
            if bool(is_hard):
                weight *= 1.80
            if float(risk) >= risk_hi:
                weight *= 1.65
            elif float(risk) >= risk_mid:
                weight *= 1.28
            if float(entropy) >= entropy_hi:
                weight *= 1.15
            if float(disagreement) >= disagreement_hi:
                weight *= 1.10
            weights.append(float(weight))
        weight_vec = np.asarray(weights, dtype=np.float32)
        mean_weight = float(weight_vec.mean()) if weight_vec.size else 1.0
        if mean_weight > 0.0:
            weight_vec = weight_vec / mean_weight
        meta = {
            "risk_hi": risk_hi,
            "risk_mid": risk_mid,
            "entropy_hi": entropy_hi,
            "disagreement_hi": disagreement_hi,
            "weight_mean": float(weight_vec.mean()) if weight_vec.size else 1.0,
            "weight_max": float(weight_vec.max()) if weight_vec.size else 1.0,
        }
        return weight_vec.astype(np.float32), meta

    def _loss(self, out: dict[str, torch.Tensor], target: torch.Tensor, proxy_scores: torch.Tensor, hard_target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        proxy_target = torch.softmax(proxy_scores / 0.80, dim=1)
        sign_prior = self.model.sign_prior.to(proxy_scores.device)  # type: ignore[union-attr]
        signed_proxy = proxy_target * sign_prior
        sample_loss = F.smooth_l1_loss(out["pred"], target, reduction="none")
        anchor_err = torch.abs(out["anchor_pred"] - target)
        pred_err = torch.abs(out["pred"] - target)
        focus_weight = torch.maximum(hard_target, out["hard_proxy"])
        hard_weight = 1.0 + float(self.config.hard_focus_weight) * focus_weight

        pred_loss = (sample_loss * hard_weight).sum() / hard_weight.sum().clamp_min(1.0)
        anchor_margin = (torch.relu(pred_err - anchor_err + 0.0015) * hard_weight).sum() / hard_weight.sum().clamp_min(1.0)
        residual_target = torch.clamp(target - out["anchor_pred"], min=-self.config.delta_bound_k, max=self.config.delta_bound_k)
        residual_target_loss = (F.smooth_l1_loss(out["bounded_delta"], residual_target, reduction="none") * hard_weight).sum() / hard_weight.sum().clamp_min(1.0)
        alpha_alignment = F.kl_div(torch.log(out["alpha"].clamp_min(1e-8)), proxy_target, reduction="batchmean")
        sign_target = torch.sign(signed_proxy)
        sign_loss = ((torch.relu(-out["contribution"] * sign_target) * proxy_target).sum(dim=1, keepdim=True) * hard_weight).sum() / hard_weight.sum().clamp_min(1.0)
        sparse_penalty = (1.0 - out["alpha_max"]).mean() + 0.25 * out["entropy"].mean()
        mag_norm = out["main_mag"] / out["main_mag"].sum(dim=1, keepdim=True).clamp_min(1e-8)
        calibrator_align = F.smooth_l1_loss(mag_norm, proxy_target)
        diversity_penalty = (1.0 - out["diversity"]).mean()
        risk = focus_weight * (1.0 + 0.40 * out["mechanism_disagreement"])
        gate_high_penalty = (out["gate"] * risk * out["contribution"].abs().sum(dim=1, keepdim=True)).mean()
        gate_low_penalty = (torch.relu(0.18 - out["gate"]) * (1.0 - risk)).mean()
        delta_penalty = torch.abs(out["bounded_delta"]).mean()
        total = (
            pred_loss
            + float(self.config.anchor_margin_weight) * anchor_margin
            + float(self.config.residual_target_weight) * residual_target_loss
            + float(self.config.proxy_alignment_weight) * alpha_alignment
            + float(self.config.sign_weight) * sign_loss
            + float(self.config.sparse_weight) * sparse_penalty
            + float(self.config.calibrator_weight) * calibrator_align
            + float(self.config.diversity_weight) * diversity_penalty
            + float(self.config.gate_high_penalty_weight) * gate_high_penalty
            + float(self.config.gate_low_penalty_weight) * gate_low_penalty
            + float(self.config.delta_weight) * delta_penalty
        )
        return total, {
            "pred_loss": float(pred_loss.detach().cpu()),
            "anchor_margin": float(anchor_margin.detach().cpu()),
            "residual_target_loss": float(residual_target_loss.detach().cpu()),
            "alpha_alignment": float(alpha_alignment.detach().cpu()),
            "sign_loss": float(sign_loss.detach().cpu()),
            "sparse_penalty": float(sparse_penalty.detach().cpu()),
            "calibrator_align": float(calibrator_align.detach().cpu()),
            "diversity_penalty": float(diversity_penalty.detach().cpu()),
            "gate_high_penalty": float(gate_high_penalty.detach().cpu()),
            "gate_low_penalty": float(gate_low_penalty.detach().cpu()),
            "delta_penalty": float(delta_penalty.detach().cpu()),
        }

    def _apply_calibrated_delta(
        self,
        *,
        gate: np.ndarray,
        contribution_sum: np.ndarray,
        hard_proxy: np.ndarray,
        tau: float,
        gate_floor: float,
        focus_gain: float,
        cap_k: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        gate_vec = np.asarray(gate, dtype=np.float32).reshape(-1)
        contribution_vec = np.asarray(contribution_sum, dtype=np.float32).reshape(-1)
        hard_vec = np.asarray(hard_proxy, dtype=np.float32).reshape(-1)
        thresholded_gate = 1.0 / (1.0 + np.exp(-float(self.config.hard_threshold_gamma) * (hard_vec - float(tau))))
        effective_gate = np.maximum(gate_vec, float(gate_floor))
        applied_delta = effective_gate * float(focus_gain) * thresholded_gate * contribution_vec
        bounded_delta = np.clip(applied_delta, -float(cap_k), float(cap_k)).astype(np.float32)
        return bounded_delta, thresholded_gate.astype(np.float32)

    def _select_calibration(
        self,
        *,
        anchor_pred: np.ndarray,
        gate: np.ndarray,
        contribution_sum: np.ndarray,
        hard_proxy: np.ndarray,
        y_true: np.ndarray,
        hard_mask: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        anchor_vec = np.asarray(anchor_pred, dtype=np.float32).reshape(-1)
        target_vec = np.asarray(y_true, dtype=np.float32).reshape(-1)
        hard_mask = np.asarray(hard_mask, dtype=bool).reshape(-1)
        anchor_mae = float(np.mean(np.abs(anchor_vec - target_vec)))
        anchor_hard_mae = float(np.mean(np.abs(anchor_vec[hard_mask] - target_vec[hard_mask]))) if hard_mask.any() else anchor_mae
        tau_candidates = sorted(set(float(value) for value in (*self.config.calibration_tau_candidates, self.config.hard_threshold_tau)))
        gate_floor_candidates = sorted(set(float(value) for value in self.config.calibration_gate_floor_candidates))
        focus_gain_candidates = sorted(set(float(value) for value in self.config.calibration_focus_gain_candidates))
        cap_candidates = sorted(set(float(value) for value in (*self.config.calibration_cap_candidates, self.config.delta_bound_k)))
        alpha_candidates = np.linspace(
            0.0,
            float(max(self.config.blend_alpha_max, 0.0)),
            num=max(2, int(self.config.blend_grid_size)),
            dtype=np.float32,
        )

        best_alpha = 0.0
        best_tau = float(self.config.hard_threshold_tau)
        best_gate_floor = 0.0
        best_focus_gain = 1.0
        best_cap_k = float(self.config.delta_bound_k)
        best_score = float("inf")

        for tau in tau_candidates:
            for gate_floor in gate_floor_candidates:
                for focus_gain in focus_gain_candidates:
                    bounded_delta, _thresholded_gate = self._apply_calibrated_delta(
                        gate=gate,
                        contribution_sum=contribution_sum,
                        hard_proxy=hard_proxy,
                        tau=tau,
                        gate_floor=gate_floor,
                        focus_gain=focus_gain,
                        cap_k=max(cap_candidates),
                    )
                    for cap_k in cap_candidates:
                        candidate_delta = np.clip(bounded_delta, -float(cap_k), float(cap_k)).astype(np.float32)
                        activation_rate = float((np.abs(candidate_delta) >= float(self.config.activation_delta_threshold_k)).mean())
                        correction_mean = float(np.abs(candidate_delta).mean())
                        for alpha in alpha_candidates:
                            pred = anchor_vec + float(alpha) * candidate_delta
                            mae = float(np.mean(np.abs(pred - target_vec)))
                            hard_mae = float(np.mean(np.abs(pred[hard_mask] - target_vec[hard_mask]))) if hard_mask.any() else mae
                            score = (
                                mae / max(anchor_mae, 1e-6)
                                + 3.00 * hard_mae / max(anchor_hard_mae, 1e-6)
                                + 0.15 * max(0.0, mae - anchor_mae)
                                + 0.60 * max(0.0, activation_rate - float(self.config.activation_ceiling))
                                + 0.40 * max(0.0, float(self.config.activation_floor) - activation_rate)
                                + 0.03 * correction_mean
                            )
                            if score + 1e-8 < best_score:
                                best_score = float(score)
                                best_alpha = float(alpha)
                                best_tau = float(tau)
                                best_gate_floor = float(gate_floor)
                                best_focus_gain = float(focus_gain)
                                best_cap_k = float(cap_k)
        return best_alpha, best_tau, best_gate_floor, best_focus_gain, best_cap_k, best_score

    @torch.no_grad()
    def _validation_signals(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("MASDRegressor has not been initialized")
        self.model.eval()
        gate_parts: list[np.ndarray] = []
        contribution_sum_parts: list[np.ndarray] = []
        hard_parts: list[np.ndarray] = []
        for descriptor_tensor, context_tensor, anchor_pred_tensor, anchor_feature_tensor, proxy_tensor, hard_tensor, _hard_target_tensor, _y_tensor in loader:
            out = self.model(
                descriptor_tensor.to(self.device),
                context_tensor.to(self.device),
                anchor_pred_tensor.to(self.device),
                anchor_feature_tensor.to(self.device),
                proxy_tensor.to(self.device),
                hard_tensor.to(self.device),
            )
            gate_parts.append(out["gate"].detach().cpu().numpy().reshape(-1))
            contribution_sum_parts.append(out["contribution_sum"].detach().cpu().numpy().reshape(-1))
            hard_parts.append(out["hard_proxy"].detach().cpu().numpy().reshape(-1))
        return (
            np.concatenate(gate_parts, axis=0),
            np.concatenate(contribution_sum_parts, axis=0),
            np.concatenate(hard_parts, axis=0),
        )

    @torch.no_grad()
    def _evaluate_candidate_state(
        self,
        *,
        state_dict: dict[str, torch.Tensor],
        val_loader: DataLoader,
        anchor_pred: np.ndarray,
        y_true: np.ndarray,
        hard_mask: np.ndarray,
        aggregation_name: str,
    ) -> dict[str, object]:
        if self.model is None:
            raise RuntimeError("MASDRegressor has not been initialized")
        self.model.load_state_dict(state_dict)
        gate, contribution_sum, hard_proxy = self._validation_signals(val_loader)
        blend_alpha, tau, gate_floor, focus_gain, cap_k, score = self._select_calibration(
            anchor_pred=anchor_pred,
            gate=gate,
            contribution_sum=contribution_sum,
            hard_proxy=hard_proxy,
            y_true=y_true,
            hard_mask=hard_mask,
        )
        raw_delta, thresholded_gate = self._apply_calibrated_delta(
            gate=gate,
            contribution_sum=contribution_sum,
            hard_proxy=hard_proxy,
            tau=tau,
            gate_floor=gate_floor,
            focus_gain=focus_gain,
            cap_k=cap_k,
        )
        pred = np.asarray(anchor_pred, dtype=np.float32).reshape(-1) + float(blend_alpha) * raw_delta
        target_vec = np.asarray(y_true, dtype=np.float32).reshape(-1)
        anchor_vec = np.asarray(anchor_pred, dtype=np.float32).reshape(-1)
        hard_mask_vec = np.asarray(hard_mask, dtype=bool).reshape(-1)
        entropy = np.clip(
            -thresholded_gate * np.log(np.clip(thresholded_gate, 1e-8, 1.0))
            - (1.0 - thresholded_gate) * np.log(np.clip(1.0 - thresholded_gate, 1e-8, 1.0)),
            0.0,
            1.0,
        )
        risk_edges = _quantile_edges(hard_proxy, (0.50, 0.80))
        risk_bins = np.digitize(hard_proxy, bins=risk_edges, right=False)
        bin_deltas: list[float] = []
        for bucket in sorted(set(int(value) for value in risk_bins.tolist())):
            mask = risk_bins == int(bucket)
            if int(mask.sum()) == 0:
                continue
            anchor_mae = float(np.mean(np.abs(anchor_vec[mask] - target_vec[mask])))
            candidate_mae = float(np.mean(np.abs(pred[mask] - target_vec[mask])))
            bin_deltas.append(float(candidate_mae - anchor_mae))
        activation_rate = float((np.abs(float(blend_alpha) * raw_delta) >= float(self.config.activation_delta_threshold_k)).mean())
        return {
            "state": _clone_state_dict(state_dict),
            "score": float(score),
            "blend_alpha": float(blend_alpha),
            "tau": float(tau),
            "gate_floor": float(gate_floor),
            "focus_gain": float(focus_gain),
            "cap_k": float(cap_k),
            "aggregation_name": aggregation_name,
            "val_primary": float(np.mean(np.abs(pred - target_vec))),
            "val_hard": float(np.mean(np.abs(pred[hard_mask_vec] - target_vec[hard_mask_vec]))) if hard_mask_vec.any() else float(np.mean(np.abs(pred - target_vec))),
            "proxy_hard_positive_bin_count": float(sum(delta > 0.0 for delta in bin_deltas)),
            "proxy_hard_worst_bin_delta": float(max([0.0, *bin_deltas])),
            "proxy_hard_bin_variance": float(np.var(bin_deltas)) if len(bin_deltas) >= 2 else 0.0,
            "gate_volatility": float(np.std(gate)),
            "ambiguity_variance": float(np.var(entropy)),
            "activation_rate": activation_rate,
        }

    def _select_split_b_candidate(self, candidates: list[dict[str, object]]) -> dict[str, object]:
        if not candidates:
            raise ValueError("candidates must not be empty")
        best_primary = min(float(item["val_primary"]) for item in candidates)
        eligible = [
            item
            for item in candidates
            if float(item["val_primary"]) <= best_primary + float(self.config.split_b_primary_epsilon)
        ]
        eligible.sort(
            key=lambda item: (
                float(item["proxy_hard_positive_bin_count"]),
                float(max(0.0, float(item["proxy_hard_worst_bin_delta"]))),
                float(item["gate_volatility"]),
                float(item["ambiguity_variance"]),
                float(item["val_hard"]),
                float(item["val_primary"]),
            )
        )
        return eligible[0]

    def fit(
        self,
        *,
        descriptor_train: np.ndarray,
        context_train: np.ndarray,
        msce_train_details: dict[str, np.ndarray],
        rcmf_train_details: dict[str, np.ndarray],
        y_train: np.ndarray,
        descriptor_val: np.ndarray,
        context_val: np.ndarray,
        msce_val_details: dict[str, np.ndarray],
        rcmf_val_details: dict[str, np.ndarray],
        y_val: np.ndarray,
        benchmark_hard_train_mask: np.ndarray | None = None,
        benchmark_hard_val_mask: np.ndarray | None = None,
    ) -> "MASDRegressor":
        _set_seed(self.train_seed)
        train_anchor_features, train_hard_raw = self._anchor_features(msce_details=msce_train_details, rcmf_details=rcmf_train_details)
        val_anchor_features, val_hard_raw = self._anchor_features(msce_details=msce_val_details, rcmf_details=rcmf_val_details)
        train_hard_target, _ = self._resolve_hard_target(
            benchmark_hard_mask=benchmark_hard_train_mask,
            anchor_pred=np.asarray(rcmf_train_details["pred"], dtype=np.float32),
            y_true=y_train,
        )
        val_hard_target, val_hard_mask = self._resolve_hard_target(
            benchmark_hard_mask=benchmark_hard_val_mask,
            anchor_pred=np.asarray(rcmf_val_details["pred"], dtype=np.float32),
            y_true=y_val,
        )
        train_proxy_scores = _mechanism_proxy_scores(descriptor_train, context_train, int(self.config.slot_count))
        val_proxy_scores = _mechanism_proxy_scores(descriptor_val, context_val, int(self.config.slot_count))
        self._prepare_stats(
            descriptor_train=descriptor_train,
            context_train=context_train,
            anchor_feature_train=train_anchor_features,
            hard_raw_train=train_hard_raw,
        )
        descriptor_train_z, context_train_z, anchor_train_z, hard_train = self._standardize_inputs(
            descriptor=descriptor_train,
            context=context_train,
            anchor_features=train_anchor_features,
            hard_raw=train_hard_raw,
        )
        descriptor_val_z, context_val_z, anchor_val_z, hard_val = self._standardize_inputs(
            descriptor=descriptor_val,
            context=context_val,
            anchor_features=val_anchor_features,
            hard_raw=val_hard_raw,
        )
        split_a_indices = np.arange(descriptor_train_z.shape[0], dtype=np.int64)
        split_b_indices = np.zeros((0,), dtype=np.int64)
        split_b_meta = {
            "split_a_count": float(split_a_indices.shape[0]),
            "split_b_count": 0.0,
            "split_b_fraction": 0.0,
            "risk_edge_50": 0.0,
            "risk_edge_80": 0.0,
        }
        if bool(self.config.enable_split_b):
            split_a_indices, split_b_indices, split_b_meta = self._build_split_b_partition(
                hard_target=train_hard_target,
                hard_raw=train_hard_raw,
                msce_details=msce_train_details,
                rcmf_details=rcmf_train_details,
            )
            if split_a_indices.size < max(4, int(self.config.batch_size // 2)):
                split_a_indices = np.arange(descriptor_train_z.shape[0], dtype=np.int64)
                split_b_indices = np.zeros((0,), dtype=np.int64)
                split_b_meta = {
                    "split_a_count": float(split_a_indices.shape[0]),
                    "split_b_count": 0.0,
                    "split_b_fraction": 0.0,
                    "risk_edge_50": float(split_b_meta.get("risk_edge_50", 0.0)),
                    "risk_edge_80": float(split_b_meta.get("risk_edge_80", 0.0)),
                }
        train_loader = self._build_loader(
            descriptor=descriptor_train_z[split_a_indices],
            context=context_train_z[split_a_indices],
            anchor_pred=np.asarray(rcmf_train_details["pred"], dtype=np.float32)[split_a_indices],
            anchor_features=anchor_train_z[split_a_indices],
            proxy_scores=train_proxy_scores[split_a_indices],
            hard_proxy=hard_train[split_a_indices],
            hard_target=train_hard_target[split_a_indices],
            target=np.asarray(y_train, dtype=np.float32)[split_a_indices],
            shuffle=True,
        )
        val_loader = self._build_loader(
            descriptor=descriptor_val_z,
            context=context_val_z,
            anchor_pred=np.asarray(rcmf_val_details["pred"], dtype=np.float32),
            anchor_features=anchor_val_z,
            proxy_scores=val_proxy_scores,
            hard_proxy=hard_val,
            hard_target=val_hard_target,
            target=y_val,
            shuffle=False,
        )
        self.model = _MASDNet(
            descriptor_dim=int(descriptor_train_z.shape[1]),
            context_dim=int(context_train_z.shape[1]),
            anchor_feature_dim=int(anchor_train_z.shape[1]),
            config=self.config,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_blend_alpha = 0.0
        best_tau = float(self.config.hard_threshold_tau)
        best_gate_floor = 0.0
        best_focus_gain = 1.0
        best_cap_k = float(self.config.delta_bound_k)
        best_score = float("inf")
        best_aggregation_name = "raw_best"
        bad_epochs = 0
        late_candidates: list[dict[str, object]] = []
        val_anchor_pred = np.asarray(rcmf_val_details["pred"], dtype=np.float32)
        for epoch_index in range(int(self.config.max_epochs)):
            self.model.train()
            for descriptor_tensor, context_tensor, anchor_pred_tensor, anchor_feature_tensor, proxy_tensor, hard_tensor, hard_target_tensor, y_tensor in train_loader:
                out = self.model(
                    descriptor_tensor.to(self.device),
                    context_tensor.to(self.device),
                    anchor_pred_tensor.to(self.device),
                    anchor_feature_tensor.to(self.device),
                    proxy_tensor.to(self.device),
                    hard_tensor.to(self.device),
                )
                loss, _ = self._loss(out, y_tensor.to(self.device), proxy_tensor.to(self.device), hard_target_tensor.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                optimizer.step()

            current_state = _clone_state_dict(self.model.state_dict())
            candidate = self._evaluate_candidate_state(
                state_dict=current_state,
                val_loader=val_loader,
                anchor_pred=val_anchor_pred,
                y_true=y_val,
                hard_mask=val_hard_mask,
                aggregation_name=f"raw_epoch_{epoch_index}",
            )
            late_candidates.append(candidate)
            if len(late_candidates) > max(2, int(self.config.late_epoch_candidate_count)):
                late_candidates.pop(0)
            blend_alpha = float(candidate["blend_alpha"])
            tau = float(candidate["tau"])
            gate_floor = float(candidate["gate_floor"])
            focus_gain = float(candidate["focus_gain"])
            cap_k = float(candidate["cap_k"])
            val_score = float(candidate["score"])
            if val_score < best_score:
                best_score = val_score
                best_blend_alpha = blend_alpha
                best_tau = tau
                best_gate_floor = gate_floor
                best_focus_gain = focus_gain
                best_cap_k = cap_k
                best_state = current_state
                best_aggregation_name = str(candidate["aggregation_name"])
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.config.patience):
                    break

        if best_state is None or self.model is None:
            raise RuntimeError("failed to store a MASD checkpoint")

        if len(late_candidates) >= 2:
            soup_pool = sorted(late_candidates, key=lambda item: float(item["score"]))
            averaged_candidates: list[dict[str, object]] = []
            max_recent = len(late_candidates)
            max_top = min(len(soup_pool), max(2, int(self.config.late_epoch_top_soup_count)))
            for size in range(2, max_recent + 1):
                averaged_state = _average_state_dicts([record["state"] for record in late_candidates[-size:]])  # type: ignore[index]
                averaged_candidates.append(
                    self._evaluate_candidate_state(
                        state_dict=averaged_state,
                        val_loader=val_loader,
                        anchor_pred=val_anchor_pred,
                        y_true=y_val,
                        hard_mask=val_hard_mask,
                        aggregation_name=f"recent_avg_{size}",
                    )
                )
            for size in range(2, max_top + 1):
                averaged_state = _average_state_dicts([record["state"] for record in soup_pool[:size]])  # type: ignore[index]
                averaged_candidates.append(
                    self._evaluate_candidate_state(
                        state_dict=averaged_state,
                        val_loader=val_loader,
                        anchor_pred=val_anchor_pred,
                        y_true=y_val,
                        hard_mask=val_hard_mask,
                        aggregation_name=f"top_avg_{size}",
                    )
                )
            for candidate in averaged_candidates:
                val_score = float(candidate["score"])
                if val_score < best_score:
                    best_score = val_score
                    best_blend_alpha = float(candidate["blend_alpha"])
                    best_tau = float(candidate["tau"])
                    best_gate_floor = float(candidate["gate_floor"])
                    best_focus_gain = float(candidate["focus_gain"])
                    best_cap_k = float(candidate["cap_k"])
                    best_state = candidate["state"]  # type: ignore[assignment]
                    best_aggregation_name = str(candidate["aggregation_name"])

        if int(self.config.tail_focus_epochs) > 0:
            weighted_loader = self._build_loader(
                descriptor=descriptor_train_z[split_a_indices],
                context=context_train_z[split_a_indices],
                anchor_pred=np.asarray(rcmf_train_details["pred"], dtype=np.float32)[split_a_indices],
                anchor_features=anchor_train_z[split_a_indices],
                proxy_scores=train_proxy_scores[split_a_indices],
                hard_proxy=hard_train[split_a_indices],
                hard_target=train_hard_target[split_a_indices],
                target=np.asarray(y_train, dtype=np.float32)[split_a_indices],
                shuffle=False,
                sample_weights=1.0 + float(self.config.tail_focus_weight_scale) * np.asarray(train_hard_target, dtype=np.float32).reshape(-1)[split_a_indices],
            )
            self.model.load_state_dict(best_state)
            tail_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.learning_rate) * float(self.config.tail_focus_lr_scale),
                weight_decay=float(self.config.weight_decay),
            )
            for tail_epoch in range(int(self.config.tail_focus_epochs)):
                self.model.train()
                for descriptor_tensor, context_tensor, anchor_pred_tensor, anchor_feature_tensor, proxy_tensor, hard_tensor, hard_target_tensor, y_tensor in weighted_loader:
                    out = self.model(
                        descriptor_tensor.to(self.device),
                        context_tensor.to(self.device),
                        anchor_pred_tensor.to(self.device),
                        anchor_feature_tensor.to(self.device),
                        proxy_tensor.to(self.device),
                        hard_tensor.to(self.device),
                    )
                    loss, _ = self._loss(out, y_tensor.to(self.device), proxy_tensor.to(self.device), hard_target_tensor.to(self.device))
                    tail_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                    tail_optimizer.step()

                candidate = self._evaluate_candidate_state(
                    state_dict=_clone_state_dict(self.model.state_dict()),
                    val_loader=val_loader,
                    anchor_pred=val_anchor_pred,
                    y_true=y_val,
                    hard_mask=val_hard_mask,
                    aggregation_name=f"tail_focus_epoch_{tail_epoch}",
                )
                val_score = float(candidate["score"])
                if val_score < best_score:
                    best_score = val_score
                    best_blend_alpha = float(candidate["blend_alpha"])
                    best_tau = float(candidate["tau"])
                    best_gate_floor = float(candidate["gate_floor"])
                    best_focus_gain = float(candidate["focus_gain"])
                    best_cap_k = float(candidate["cap_k"])
                    best_state = candidate["state"]  # type: ignore[assignment]
                    best_aggregation_name = str(candidate["aggregation_name"])

        split_b_meta.update({"weight_mean": 1.0, "weight_max": 1.0})
        if bool(self.config.enable_split_b) and best_state is not None and self.model is not None:
            if split_b_indices.size >= 2 and split_a_indices.size >= 1:
                split_b_weights, weight_meta = self._build_split_b_sample_weights(
                    descriptor=descriptor_train_z,
                    context=context_train_z,
                    anchor_pred=np.asarray(rcmf_train_details["pred"], dtype=np.float32),
                    anchor_features=anchor_train_z,
                    proxy_scores=train_proxy_scores,
                    hard_proxy=hard_train,
                    hard_target=train_hard_target,
                    target=y_train,
                    split_b_indices=split_b_indices,
                    state_dict=best_state,
                )
                split_b_meta.update(weight_meta)
                split_b_loader = self._build_loader(
                    descriptor=descriptor_train_z[split_b_indices],
                    context=context_train_z[split_b_indices],
                    anchor_pred=np.asarray(rcmf_train_details["pred"], dtype=np.float32)[split_b_indices],
                    anchor_features=anchor_train_z[split_b_indices],
                    proxy_scores=train_proxy_scores[split_b_indices],
                    hard_proxy=hard_train[split_b_indices],
                    hard_target=train_hard_target[split_b_indices],
                    target=np.asarray(y_train, dtype=np.float32)[split_b_indices],
                    shuffle=False,
                    sample_weights=split_b_weights,
                )
                self.model.load_state_dict(best_state)
                self._set_split_b_trainable()
                split_b_optimizer = torch.optim.AdamW(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    lr=float(self.config.learning_rate) * float(self.config.split_b_lr_scale),
                    weight_decay=float(self.config.weight_decay) * float(self.config.split_b_weight_decay_scale),
                )
                split_b_candidates: list[dict[str, object]] = [
                    self._evaluate_candidate_state(
                        state_dict=best_state,
                        val_loader=val_loader,
                        anchor_pred=val_anchor_pred,
                        y_true=y_val,
                        hard_mask=val_hard_mask,
                        aggregation_name=f"{best_aggregation_name}_pre_split_b",
                    )
                ]
                best_local = min(float(item["score"]) for item in split_b_candidates)
                bad_head_epochs = 0
                for head_epoch in range(int(self.config.split_b_head_epochs)):
                    self.model.train()
                    for descriptor_tensor, context_tensor, anchor_pred_tensor, anchor_feature_tensor, proxy_tensor, hard_tensor, hard_target_tensor, y_tensor in split_b_loader:
                        out = self.model(
                            descriptor_tensor.to(self.device),
                            context_tensor.to(self.device),
                            anchor_pred_tensor.to(self.device),
                            anchor_feature_tensor.to(self.device),
                            proxy_tensor.to(self.device),
                            hard_tensor.to(self.device),
                        )
                        loss, _ = self._loss(out, y_tensor.to(self.device), proxy_tensor.to(self.device), hard_target_tensor.to(self.device))
                        split_b_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.5)
                        split_b_optimizer.step()

                    candidate = self._evaluate_candidate_state(
                        state_dict=_clone_state_dict(self.model.state_dict()),
                        val_loader=val_loader,
                        anchor_pred=val_anchor_pred,
                        y_true=y_val,
                        hard_mask=val_hard_mask,
                        aggregation_name=f"split_b_head_epoch_{head_epoch}",
                    )
                    split_b_candidates.append(candidate)
                    if float(candidate["score"]) + 1e-8 < best_local:
                        best_local = float(candidate["score"])
                        bad_head_epochs = 0
                    else:
                        bad_head_epochs += 1
                        if bad_head_epochs >= int(self.config.split_b_patience):
                            break

                if len(split_b_candidates) >= 3:
                    recent_pool = split_b_candidates[-min(4, len(split_b_candidates)) :]
                    averaged_state = _average_state_dicts([record["state"] for record in recent_pool])  # type: ignore[index]
                    split_b_candidates.append(
                        self._evaluate_candidate_state(
                            state_dict=averaged_state,
                            val_loader=val_loader,
                            anchor_pred=val_anchor_pred,
                            y_true=y_val,
                            hard_mask=val_hard_mask,
                            aggregation_name="split_b_recent_avg",
                        )
                    )

                selected_split_b = self._select_split_b_candidate(split_b_candidates)
                best_score = float(selected_split_b["score"])
                best_blend_alpha = float(selected_split_b["blend_alpha"])
                best_tau = float(selected_split_b["tau"])
                best_gate_floor = float(selected_split_b["gate_floor"])
                best_focus_gain = float(selected_split_b["focus_gain"])
                best_cap_k = float(selected_split_b["cap_k"])
                best_state = selected_split_b["state"]  # type: ignore[assignment]
                best_aggregation_name = str(selected_split_b["aggregation_name"])

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        self.blend_alpha = float(best_blend_alpha)
        self.calibrated_tau = float(best_tau)
        self.calibrated_gate_floor = float(best_gate_floor)
        self.calibrated_focus_gain = float(best_focus_gain)
        self.calibrated_cap_k = float(best_cap_k)
        self.aggregation_name = str(best_aggregation_name)
        self.split_b_meta = {key: float(value) for key, value in split_b_meta.items()}
        return self

    @torch.no_grad()
    def predict_details(
        self,
        *,
        descriptor_matrix: np.ndarray,
        context_matrix: np.ndarray,
        msce_details: dict[str, np.ndarray],
        rcmf_details: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("MASDRegressor must be fit before predict")
        anchor_features, hard_raw = self._anchor_features(msce_details=msce_details, rcmf_details=rcmf_details)
        proxy_scores = _mechanism_proxy_scores(descriptor_matrix, context_matrix, int(self.config.slot_count))
        descriptor_z, context_z, anchor_feature_z, hard_proxy = self._standardize_inputs(
            descriptor=descriptor_matrix,
            context=context_matrix,
            anchor_features=anchor_features,
            hard_raw=hard_raw,
        )
        out = self.model(
            torch.tensor(descriptor_z, dtype=torch.float32, device=self.device),
            torch.tensor(context_z, dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(rcmf_details["pred"], dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device),
            torch.tensor(anchor_feature_z, dtype=torch.float32, device=self.device),
            torch.tensor(proxy_scores, dtype=torch.float32, device=self.device),
            torch.tensor(hard_proxy, dtype=torch.float32, device=self.device),
        )
        raw_delta, calibrated_thresholded_gate = self._apply_calibrated_delta(
            gate=out["gate"].detach().cpu().numpy().reshape(-1),
            contribution_sum=out["contribution_sum"].detach().cpu().numpy().reshape(-1),
            hard_proxy=out["hard_proxy"].detach().cpu().numpy().reshape(-1),
            tau=float(self.calibrated_tau),
            gate_floor=float(self.calibrated_gate_floor),
            focus_gain=float(self.calibrated_focus_gain),
            cap_k=float(self.calibrated_cap_k),
        )
        blended_delta = raw_delta * float(self.blend_alpha)
        anchor_pred = out["anchor_pred"].detach().cpu().numpy().reshape(-1)
        alpha = out["alpha"].detach().cpu().numpy()
        return {
            "pred": anchor_pred + blended_delta,
            "rcmf_anchor_pred": anchor_pred,
            "bounded_delta": blended_delta.astype(np.float32),
            "raw_bounded_delta": raw_delta.astype(np.float32),
            "gate": out["gate"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "thresholded_gate": calibrated_thresholded_gate.astype(np.float32),
            "hard_proxy": out["hard_proxy"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "alpha": alpha.astype(np.float32),
            "alpha_max": out["alpha_max"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "alpha_margin": out["alpha_margin"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "entropy": out["entropy"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "main_mag": out["main_mag"].detach().cpu().numpy().astype(np.float32),
            "residual": out["residual"].detach().cpu().numpy().astype(np.float32),
            "contribution": out["contribution"].detach().cpu().numpy().astype(np.float32),
            "mechanism_disagreement": out["mechanism_disagreement"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "dominant_mechanism": out["dominant_mechanism"].detach().cpu().numpy().reshape(-1).astype(np.int64),
            "proxy_scores": proxy_scores.astype(np.float32),
            "proxy_target": _softmax_rows(proxy_scores, temperature=0.80),
            "sign_prior": np.repeat(_build_sign_prior(int(self.config.slot_count)).reshape(1, -1), alpha.shape[0], axis=0).astype(np.float32),
            "blend_alpha": np.full((alpha.shape[0],), float(self.blend_alpha), dtype=np.float32),
            "calibrated_tau": np.full((alpha.shape[0],), float(self.calibrated_tau), dtype=np.float32),
            "gate_floor": np.full((alpha.shape[0],), float(self.calibrated_gate_floor), dtype=np.float32),
            "focus_gain": np.full((alpha.shape[0],), float(self.calibrated_focus_gain), dtype=np.float32),
            "cap_k": np.full((alpha.shape[0],), float(self.calibrated_cap_k), dtype=np.float32),
            "activation_flag": (np.abs(blended_delta) >= float(self.config.activation_delta_threshold_k)).astype(np.float32),
            "aggregation_name": np.asarray([self.aggregation_name] * alpha.shape[0], dtype=object),
            "split_b_fraction": np.full((alpha.shape[0],), float(self.split_b_meta.get("split_b_fraction", 0.0)), dtype=np.float32),
            "backend_name": np.asarray(["masd_signed_decomposition"] * alpha.shape[0], dtype=object),
        }

    def predict(
        self,
        *,
        descriptor_matrix: np.ndarray,
        context_matrix: np.ndarray,
        msce_details: dict[str, np.ndarray],
        rcmf_details: dict[str, np.ndarray],
    ) -> np.ndarray:
        return self.predict_details(
            descriptor_matrix=descriptor_matrix,
            context_matrix=context_matrix,
            msce_details=msce_details,
            rcmf_details=rcmf_details,
        )["pred"]
