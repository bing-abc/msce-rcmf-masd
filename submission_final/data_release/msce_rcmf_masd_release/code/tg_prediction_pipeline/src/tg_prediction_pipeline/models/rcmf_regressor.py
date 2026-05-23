from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class RCMFConfig:
    hidden_dim: int = 96
    dropout: float = 0.10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 120
    patience: int = 16
    delta_bound_k: float = 40.0
    blend_grid_size: int = 21
    residual_alpha: float = 3.0
    trust_margin_weight: float = 0.15
    delta_weight: float = 0.01


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


class _RCMFGateNet(nn.Module):
    def __init__(self, input_dim: int, config: RCMFConfig) -> None:
        super().__init__()
        self.delta_bound_k = float(config.delta_bound_k)
        self.gate = nn.Sequential(
            nn.Linear(int(input_dim), int(config.hidden_dim)),
            nn.LayerNorm(int(config.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), 2),
        )

    def forward(
        self,
        features: torch.Tensor,
        msce_anchor: torch.Tensor,
        descriptor_delta: torch.Tensor,
        context_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = self.gate(features)
        trust_gate = torch.sigmoid(logits[:, :1])
        joint_gate = torch.tanh(logits[:, 1:2])
        fused_delta = descriptor_delta + trust_gate * context_delta + 0.5 * joint_gate * (descriptor_delta + context_delta)
        bounded_delta = torch.clamp(fused_delta, min=-self.delta_bound_k, max=self.delta_bound_k)
        pred = msce_anchor + bounded_delta
        return {
            "pred": pred,
            "msce_anchor_pred": msce_anchor,
            "descriptor_delta": descriptor_delta,
            "context_delta": context_delta,
            "bounded_delta": bounded_delta,
            "trust_gate": trust_gate,
            "joint_gate": joint_gate,
            "conflict": torch.abs(descriptor_delta - context_delta),
        }


class RCMFRegressor:
    def __init__(self, config: RCMFConfig, *, train_seed: int) -> None:
        self.config = config
        self.train_seed = int(train_seed)
        self.device = _device()
        self.model: _RCMFGateNet | None = None
        self.descriptor_model: Ridge | None = None
        self.context_model: Ridge | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.blend_alpha: float = 0.0

    def _build_gate_features(
        self,
        *,
        descriptor_delta: np.ndarray,
        context_delta: np.ndarray,
        msce_delta: np.ndarray,
        selection_entropy: np.ndarray,
        dominant_scale_weight: np.ndarray,
        scale_weights: np.ndarray,
    ) -> np.ndarray:
        descriptor_vec = np.asarray(descriptor_delta, dtype=np.float32).reshape(-1, 1)
        context_vec = np.asarray(context_delta, dtype=np.float32).reshape(-1, 1)
        msce_vec = np.asarray(msce_delta, dtype=np.float32).reshape(-1, 1)
        entropy_vec = np.asarray(selection_entropy, dtype=np.float32).reshape(-1, 1)
        dominant_vec = np.asarray(dominant_scale_weight, dtype=np.float32).reshape(-1, 1)
        conflict_vec = np.abs(descriptor_vec - context_vec)
        magnitude_vec = np.abs(msce_vec)
        return np.concatenate(
            [
                descriptor_vec,
                context_vec,
                msce_vec,
                conflict_vec,
                magnitude_vec,
                entropy_vec,
                dominant_vec,
                np.asarray(scale_weights, dtype=np.float32),
            ],
            axis=1,
        )

    def _fit_residual_models(
        self,
        *,
        descriptor_train: np.ndarray,
        context_train: np.ndarray,
        residual_target: np.ndarray,
    ) -> None:
        self.descriptor_model = Ridge(alpha=float(self.config.residual_alpha), solver="svd")
        self.context_model = Ridge(alpha=float(self.config.residual_alpha), solver="svd")
        self.descriptor_model.fit(np.asarray(descriptor_train, dtype=np.float32), residual_target)
        self.context_model.fit(np.asarray(context_train, dtype=np.float32), residual_target)

    def _residual_predictions(
        self,
        *,
        descriptor_matrix: np.ndarray,
        context_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.descriptor_model is None or self.context_model is None:
            raise RuntimeError("residual experts are not fit")
        descriptor_delta = np.asarray(self.descriptor_model.predict(np.asarray(descriptor_matrix, dtype=np.float32)), dtype=np.float32).reshape(-1)
        context_delta = np.asarray(self.context_model.predict(np.asarray(context_matrix, dtype=np.float32)), dtype=np.float32).reshape(-1)
        return descriptor_delta, context_delta

    def _standardize_gate_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("gate feature stats are not available")
        return ((np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_std).astype(np.float32)

    def _prepare_gate_stats(self, features: np.ndarray) -> None:
        self.feature_mean = np.asarray(features, dtype=np.float32).mean(axis=0, keepdims=True)
        self.feature_std = _safe_std(features)

    def _build_loader(
        self,
        *,
        gate_features: np.ndarray,
        msce_anchor: np.ndarray,
        descriptor_delta: np.ndarray,
        context_delta: np.ndarray,
        y_true: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        features_z = self._standardize_gate_features(gate_features)
        tensors = [
            torch.tensor(features_z, dtype=torch.float32),
            torch.tensor(np.asarray(msce_anchor, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
            torch.tensor(np.asarray(descriptor_delta, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
            torch.tensor(np.asarray(context_delta, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
            torch.tensor(np.asarray(y_true, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
        ]
        return DataLoader(TensorDataset(*tensors), batch_size=int(self.config.batch_size), shuffle=bool(shuffle))

    def _loss(self, out: dict[str, torch.Tensor], y_true: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pred_loss = F.smooth_l1_loss(out["pred"], y_true)
        anchor_err = torch.abs(out["msce_anchor_pred"] - y_true)
        pred_err = torch.abs(out["pred"] - y_true)
        trust_margin = torch.relu(pred_err - anchor_err + 0.0015).mean()
        delta_penalty = torch.abs(out["bounded_delta"]).mean()
        total = (
            pred_loss
            + float(self.config.trust_margin_weight) * trust_margin
            + float(self.config.delta_weight) * delta_penalty
        )
        return total, {
            "pred_loss": float(pred_loss.detach().cpu()),
            "trust_margin": float(trust_margin.detach().cpu()),
            "delta_penalty": float(delta_penalty.detach().cpu()),
        }

    def _select_blend_alpha(self, *, anchor_pred: np.ndarray, raw_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
        anchor_vec = np.asarray(anchor_pred, dtype=np.float32).reshape(-1)
        raw_vec = np.asarray(raw_pred, dtype=np.float32).reshape(-1)
        target_vec = np.asarray(y_true, dtype=np.float32).reshape(-1)
        raw_delta = raw_vec - anchor_vec
        best_alpha = 0.0
        best_mae = float(np.mean(np.abs(anchor_vec - target_vec)))
        for alpha in np.linspace(0.0, 1.0, num=max(2, int(self.config.blend_grid_size)), dtype=np.float32):
            pred = anchor_vec + float(alpha) * raw_delta
            mae = float(np.mean(np.abs(pred - target_vec)))
            if mae + 1e-8 < best_mae:
                best_mae = mae
                best_alpha = float(alpha)
        return best_alpha, best_mae

    @torch.no_grad()
    def _predict_internal(
        self,
        *,
        gate_features: np.ndarray,
        msce_anchor: np.ndarray,
        descriptor_delta: np.ndarray,
        context_delta: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("RCMFRegressor must be fit before predict")
        features_z = self._standardize_gate_features(gate_features)
        out = self.model(
            torch.tensor(features_z, dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(msce_anchor, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(descriptor_delta, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(context_delta, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device),
        )
        raw_delta = out["bounded_delta"].detach().cpu().numpy().reshape(-1)
        blended_delta = raw_delta * float(self.blend_alpha)
        return {
            "pred": out["msce_anchor_pred"].detach().cpu().numpy().reshape(-1) + blended_delta,
            "msce_anchor_pred": out["msce_anchor_pred"].detach().cpu().numpy().reshape(-1),
            "bounded_delta": blended_delta,
            "raw_bounded_delta": raw_delta,
            "trust_gate": out["trust_gate"].detach().cpu().numpy().reshape(-1),
            "joint_gate": out["joint_gate"].detach().cpu().numpy().reshape(-1),
            "conflict": out["conflict"].detach().cpu().numpy().reshape(-1),
            "blend_alpha": np.full((features_z.shape[0],), float(self.blend_alpha), dtype=np.float32),
            "backend_name": np.asarray(["rcmf_trusted_fusion"] * features_z.shape[0], dtype=object),
        }

    def fit(
        self,
        *,
        descriptor_train: np.ndarray,
        context_train: np.ndarray,
        msce_train_details: dict[str, np.ndarray],
        y_train: np.ndarray,
        descriptor_val: np.ndarray,
        context_val: np.ndarray,
        msce_val_details: dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "RCMFRegressor":
        _set_seed(self.train_seed)
        residual_target = np.asarray(y_train, dtype=np.float32).reshape(-1) - np.asarray(msce_train_details["pred"], dtype=np.float32).reshape(-1)
        self._fit_residual_models(
            descriptor_train=descriptor_train,
            context_train=context_train,
            residual_target=residual_target,
        )
        train_descriptor_delta, train_context_delta = self._residual_predictions(
            descriptor_matrix=descriptor_train,
            context_matrix=context_train,
        )
        val_descriptor_delta, val_context_delta = self._residual_predictions(
            descriptor_matrix=descriptor_val,
            context_matrix=context_val,
        )
        gate_train = self._build_gate_features(
            descriptor_delta=train_descriptor_delta,
            context_delta=train_context_delta,
            msce_delta=np.asarray(msce_train_details["bounded_delta"], dtype=np.float32),
            selection_entropy=np.asarray(msce_train_details["selection_entropy"], dtype=np.float32),
            dominant_scale_weight=np.asarray(msce_train_details["dominant_scale_weight"], dtype=np.float32),
            scale_weights=np.asarray(msce_train_details["scale_weights"], dtype=np.float32),
        )
        gate_val = self._build_gate_features(
            descriptor_delta=val_descriptor_delta,
            context_delta=val_context_delta,
            msce_delta=np.asarray(msce_val_details["bounded_delta"], dtype=np.float32),
            selection_entropy=np.asarray(msce_val_details["selection_entropy"], dtype=np.float32),
            dominant_scale_weight=np.asarray(msce_val_details["dominant_scale_weight"], dtype=np.float32),
            scale_weights=np.asarray(msce_val_details["scale_weights"], dtype=np.float32),
        )
        self._prepare_gate_stats(gate_train)
        train_loader = self._build_loader(
            gate_features=gate_train,
            msce_anchor=np.asarray(msce_train_details["pred"], dtype=np.float32),
            descriptor_delta=train_descriptor_delta,
            context_delta=train_context_delta,
            y_true=y_train,
            shuffle=True,
        )
        val_loader = self._build_loader(
            gate_features=gate_val,
            msce_anchor=np.asarray(msce_val_details["pred"], dtype=np.float32),
            descriptor_delta=val_descriptor_delta,
            context_delta=val_context_delta,
            y_true=y_val,
            shuffle=False,
        )
        self.model = _RCMFGateNet(input_dim=int(gate_train.shape[1]), config=self.config).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.config.learning_rate), weight_decay=float(self.config.weight_decay))
        best_state: dict[str, torch.Tensor] | None = None
        best_blend_alpha = 0.0
        best_score = float("inf")
        bad_epochs = 0
        for _epoch in range(int(self.config.max_epochs)):
            self.model.train()
            for feature_tensor, msce_anchor_tensor, descriptor_delta_tensor, context_delta_tensor, y_tensor in train_loader:
                out = self.model(
                    feature_tensor.to(self.device),
                    msce_anchor_tensor.to(self.device),
                    descriptor_delta_tensor.to(self.device),
                    context_delta_tensor.to(self.device),
                )
                loss, _ = self._loss(out, y_tensor.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                optimizer.step()

            self.model.eval()
            val_parts: list[np.ndarray] = []
            for feature_tensor, msce_anchor_tensor, descriptor_delta_tensor, context_delta_tensor, _y_tensor in val_loader:
                out = self.model(
                    feature_tensor.to(self.device),
                    msce_anchor_tensor.to(self.device),
                    descriptor_delta_tensor.to(self.device),
                    context_delta_tensor.to(self.device),
                )
                val_parts.append(out["pred"].detach().cpu().numpy().reshape(-1))
            val_raw_pred = np.concatenate(val_parts, axis=0)
            blend_alpha, val_score = self._select_blend_alpha(
                anchor_pred=np.asarray(msce_val_details["pred"], dtype=np.float32),
                raw_pred=val_raw_pred,
                y_true=y_val,
            )
            if val_score < best_score:
                best_score = val_score
                best_blend_alpha = float(blend_alpha)
                best_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.config.patience):
                    break
        if best_state is None or self.model is None:
            raise RuntimeError("failed to store an RCMF checkpoint")
        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        self.blend_alpha = float(best_blend_alpha)
        return self

    def predict_details(
        self,
        *,
        descriptor_matrix: np.ndarray,
        context_matrix: np.ndarray,
        msce_details: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        descriptor_delta, context_delta = self._residual_predictions(
            descriptor_matrix=descriptor_matrix,
            context_matrix=context_matrix,
        )
        gate_features = self._build_gate_features(
            descriptor_delta=descriptor_delta,
            context_delta=context_delta,
            msce_delta=np.asarray(msce_details["bounded_delta"], dtype=np.float32),
            selection_entropy=np.asarray(msce_details["selection_entropy"], dtype=np.float32),
            dominant_scale_weight=np.asarray(msce_details["dominant_scale_weight"], dtype=np.float32),
            scale_weights=np.asarray(msce_details["scale_weights"], dtype=np.float32),
        )
        return self._predict_internal(
            gate_features=gate_features,
            msce_anchor=np.asarray(msce_details["pred"], dtype=np.float32),
            descriptor_delta=descriptor_delta,
            context_delta=context_delta,
        )
