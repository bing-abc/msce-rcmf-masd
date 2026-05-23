from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tg_prediction_pipeline.features import context_scale_layout


@dataclass(frozen=True)
class MSCEConfig:
    anchor_baseline_name: str = "mlp_descriptor_anchor"
    hidden_dim: int = 192
    scale_hidden_dim: int = 96
    top_k_scales: int = 2
    delta_bound_k: float = 80.0
    anchor_tolerance_k: float = 1.5
    dropout: float = 0.10
    learning_rate: float = 0.0005
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 80
    patience: int = 12
    blend_grid_size: int = 21
    linear_residual_alpha: float = 3.0
    anchor_margin_weight: float = 0.15
    residual_target_weight: float = 0.60
    gate_weight: float = 0.0
    delta_weight: float = 0.005
    entropy_weight: float = 0.02


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_training_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float32).std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return std


class _MSCENet(nn.Module):
    def __init__(self, *, descriptor_dim: int, scale_dims: list[int], config: MSCEConfig) -> None:
        super().__init__()
        self.top_k_scales = max(1, min(int(config.top_k_scales), len(scale_dims)))
        self.delta_bound_k = float(config.delta_bound_k)
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, int(config.hidden_dim)),
            nn.LayerNorm(int(config.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(1, int(config.scale_hidden_dim)),
            nn.GELU(),
        )
        self.scale_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(scale_dim), int(config.scale_hidden_dim)),
                    nn.LayerNorm(int(config.scale_hidden_dim)),
                    nn.GELU(),
                )
                for scale_dim in scale_dims
            ]
        )
        gate_input_dim = int(config.hidden_dim) + int(config.scale_hidden_dim)
        self.scale_selector = nn.Sequential(
            nn.Linear(gate_input_dim, int(config.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(config.hidden_dim), len(scale_dims)),
        )
        head_input_dim = int(config.hidden_dim) + 2 * int(config.scale_hidden_dim)
        self.scale_delta_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_input_dim, int(config.hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(config.dropout)),
                    nn.Linear(int(config.hidden_dim), 1),
                )
                for _ in scale_dims
            ]
        )
        self.repair_gate = nn.Sequential(
            nn.Linear(head_input_dim, int(config.scale_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(config.scale_hidden_dim), 1),
        )
        nn.init.constant_(self.repair_gate[-1].bias, 1.0)

    def forward(self, descriptor: torch.Tensor, scale_tensors: list[torch.Tensor], anchor_pred: torch.Tensor) -> dict[str, torch.Tensor]:
        descriptor_hidden = self.descriptor_encoder(descriptor)
        anchor_hidden = self.anchor_encoder(anchor_pred)
        scale_hidden = [encoder(scale_tensor) for encoder, scale_tensor in zip(self.scale_encoders, scale_tensors, strict=True)]

        selector_input = torch.cat([descriptor_hidden, anchor_hidden], dim=1)
        scale_logits = self.scale_selector(selector_input)
        topk_count = min(self.top_k_scales, int(scale_logits.shape[1]))
        topk_indices = scale_logits.topk(k=topk_count, dim=1).indices
        topk_mask = torch.zeros_like(scale_logits, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)
        masked_logits = scale_logits.masked_fill(~topk_mask, float("-inf"))
        scale_weights = torch.softmax(masked_logits, dim=1)

        stacked_hidden = torch.stack(scale_hidden, dim=1)
        context_hidden = (stacked_hidden * scale_weights.unsqueeze(-1)).sum(dim=1)
        scale_delta_values = []
        for scale_hidden_i, delta_head in zip(scale_hidden, self.scale_delta_heads, strict=True):
            scale_head_input = torch.cat([descriptor_hidden, anchor_hidden, scale_hidden_i], dim=1)
            scale_delta_values.append(delta_head(scale_head_input))
        raw_scale_delta = torch.cat(scale_delta_values, dim=1)
        raw_delta = (raw_scale_delta * scale_weights).sum(dim=1, keepdim=True)

        head_input = torch.cat([descriptor_hidden, anchor_hidden, context_hidden], dim=1)
        repair_gate = torch.sigmoid(self.repair_gate(head_input))
        bounded_delta = torch.tanh(raw_delta) * self.delta_bound_k * repair_gate
        pred = anchor_pred + bounded_delta

        normalized_entropy = -(
            scale_weights.clamp_min(1e-8) * torch.log(scale_weights.clamp_min(1e-8))
        ).sum(dim=1, keepdim=True) / math.log(float(scale_weights.shape[1]))
        dominant_weight = scale_weights.max(dim=1, keepdim=True).values
        return {
            "pred": pred,
            "anchor_pred": anchor_pred,
            "bounded_delta": bounded_delta,
            "repair_gate": repair_gate,
            "scale_logits": scale_logits,
            "scale_weights": scale_weights,
            "raw_scale_delta": raw_scale_delta,
            "selection_entropy": normalized_entropy,
            "dominant_scale_weight": dominant_weight,
        }


class MSCERegressor:
    def __init__(self, config: MSCEConfig, *, train_seed: int) -> None:
        self.config = config
        self.train_seed = int(train_seed)
        self.device = _device()
        self.model: _MSCENet | None = None
        self.descriptor_mean: np.ndarray | None = None
        self.descriptor_std: np.ndarray | None = None
        self.context_mean: np.ndarray | None = None
        self.context_std: np.ndarray | None = None
        self.blend_alpha: float = 0.0
        self.selected_backend: str = "neural"
        self.linear_full_context_model: Ridge | None = None
        self.linear_scale_models: list[Ridge] = []
        self.linear_scale_weights: np.ndarray | None = None
        self.linear_scale_alphas: np.ndarray | None = None

    def _standardize_inputs(self, descriptor: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.descriptor_mean is None or self.descriptor_std is None:
            raise RuntimeError("MSCERegressor has not been fit yet")
        if self.context_mean is None or self.context_std is None:
            raise RuntimeError("MSCERegressor has not been fit yet")
        descriptor_z = (np.asarray(descriptor, dtype=np.float32) - self.descriptor_mean) / self.descriptor_std
        context_z = (np.asarray(context, dtype=np.float32) - self.context_mean) / self.context_std
        return descriptor_z.astype(np.float32), context_z.astype(np.float32)

    def _prepare_training_stats(self, descriptor_train: np.ndarray, context_train: np.ndarray) -> None:
        self.descriptor_mean = np.asarray(descriptor_train, dtype=np.float32).mean(axis=0, keepdims=True)
        self.descriptor_std = _safe_std(descriptor_train)
        self.context_mean = np.asarray(context_train, dtype=np.float32).mean(axis=0, keepdims=True)
        self.context_std = _safe_std(context_train)

    def _context_slices(self, context_z: np.ndarray) -> list[np.ndarray]:
        slices: list[np.ndarray] = []
        for item in context_scale_layout():
            slices.append(np.asarray(context_z[:, item["start"] : item["stop"]], dtype=np.float32))
        return slices

    def _build_loader(
        self,
        descriptor: np.ndarray,
        context: np.ndarray,
        anchor_pred: np.ndarray,
        target: np.ndarray,
        *,
        shuffle: bool,
    ) -> DataLoader:
        descriptor_z, context_z = self._standardize_inputs(descriptor, context)
        tensors: list[torch.Tensor] = [
            torch.tensor(descriptor_z, dtype=torch.float32),
            torch.tensor(context_z, dtype=torch.float32),
            torch.tensor(np.asarray(anchor_pred, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
            torch.tensor(np.asarray(target, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
        ]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=int(self.config.batch_size), shuffle=bool(shuffle))

    def _split_context_tensor(self, context_tensor: torch.Tensor) -> list[torch.Tensor]:
        return [
            context_tensor[:, item["start"] : item["stop"]]
            for item in context_scale_layout()
        ]

    def _loss(self, out: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pred_loss = F.smooth_l1_loss(out["pred"], target)
        anchor_err = torch.abs(out["anchor_pred"] - target)
        pred_err = torch.abs(out["pred"] - target)
        residual_target = torch.clamp(target - out["anchor_pred"], min=-self.config.delta_bound_k, max=self.config.delta_bound_k)
        residual_target_loss = F.smooth_l1_loss(out["bounded_delta"], residual_target)
        anchor_margin = torch.relu(pred_err - anchor_err - float(self.config.anchor_tolerance_k)).mean()
        gate_penalty = out["repair_gate"].mean()
        delta_penalty = torch.abs(out["bounded_delta"]).mean()
        entropy_penalty = out["selection_entropy"].mean()
        scale_balance_target = torch.full_like(out["scale_weights"].mean(dim=0), 1.0 / float(out["scale_weights"].shape[1]))
        scale_balance_penalty = F.mse_loss(out["scale_weights"].mean(dim=0), scale_balance_target)
        total = (
            pred_loss
            + float(self.config.anchor_margin_weight) * anchor_margin
            + float(self.config.residual_target_weight) * residual_target_loss
            + float(self.config.gate_weight) * gate_penalty
            + float(self.config.delta_weight) * delta_penalty
            + float(self.config.entropy_weight) * entropy_penalty
            + 0.02 * scale_balance_penalty
        )
        return total, {
            "pred_loss": float(pred_loss.detach().cpu()),
            "residual_target_loss": float(residual_target_loss.detach().cpu()),
            "anchor_margin": float(anchor_margin.detach().cpu()),
            "gate_penalty": float(gate_penalty.detach().cpu()),
            "delta_penalty": float(delta_penalty.detach().cpu()),
            "entropy_penalty": float(entropy_penalty.detach().cpu()),
            "scale_balance_penalty": float(scale_balance_penalty.detach().cpu()),
        }

    def _select_blend_alpha(
        self,
        *,
        anchor_pred: np.ndarray,
        raw_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[float, float]:
        anchor_vec = np.asarray(anchor_pred, dtype=np.float32).reshape(-1)
        raw_vec = np.asarray(raw_pred, dtype=np.float32).reshape(-1)
        target_vec = np.asarray(y_true, dtype=np.float32).reshape(-1)
        raw_delta = raw_vec - anchor_vec
        best_alpha = 0.0
        best_mae = float(np.mean(np.abs(anchor_vec - target_vec)))
        grid_size = max(2, int(self.config.blend_grid_size))
        for alpha in np.linspace(0.0, 1.0, num=grid_size, dtype=np.float32):
            blended = anchor_vec + float(alpha) * raw_delta
            mae = float(np.mean(np.abs(blended - target_vec)))
            if mae + 1e-8 < best_mae:
                best_mae = mae
                best_alpha = float(alpha)
        return best_alpha, best_mae

    def _fit_linear_backend(
        self,
        *,
        descriptor_train: np.ndarray,
        context_train: np.ndarray,
        anchor_train: np.ndarray,
        y_train: np.ndarray,
        descriptor_val: np.ndarray,
        context_val: np.ndarray,
        anchor_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[str, float]:
        residual_train = np.asarray(y_train, dtype=np.float32).reshape(-1) - np.asarray(anchor_train, dtype=np.float32).reshape(-1)
        train_scales = self._context_slices((np.asarray(context_train, dtype=np.float32) - self.context_mean) / self.context_std)
        val_scales = self._context_slices((np.asarray(context_val, dtype=np.float32) - self.context_mean) / self.context_std)

        scale_candidates: list[dict[str, Any]] = []
        anchor_val_vec = np.asarray(anchor_val, dtype=np.float32).reshape(-1)
        y_val_vec = np.asarray(y_val, dtype=np.float32).reshape(-1)
        anchor_val_mae = float(np.mean(np.abs(anchor_val_vec - y_val_vec)))
        self.linear_scale_models = []
        for scale_index, (train_scale, val_scale) in enumerate(zip(train_scales, val_scales, strict=True)):
            model = Ridge(alpha=float(self.config.linear_residual_alpha), solver="svd")
            model.fit(train_scale, residual_train)
            val_delta = np.asarray(model.predict(val_scale), dtype=np.float32).reshape(-1)
            alpha, val_mae = self._select_blend_alpha(
                anchor_pred=anchor_val_vec,
                raw_pred=anchor_val_vec + val_delta,
                y_true=y_val_vec,
            )
            scale_candidates.append(
                {
                    "scale_index": int(scale_index),
                    "model": model,
                    "val_delta": val_delta,
                    "blend_alpha": float(alpha),
                    "val_mae": float(val_mae),
                    "improvement": float(max(0.0, anchor_val_mae - val_mae)),
                }
            )
            self.linear_scale_models.append(model)

        scale_candidates.sort(key=lambda item: item["val_mae"])
        selected_candidates = scale_candidates[: max(1, min(int(self.config.top_k_scales), len(scale_candidates)))]
        improvement_weights = np.asarray([item["improvement"] for item in selected_candidates], dtype=np.float32)
        if float(improvement_weights.sum()) <= 1e-8:
            improvement_weights = np.full((len(selected_candidates),), 1.0 / float(len(selected_candidates)), dtype=np.float32)
        else:
            improvement_weights = improvement_weights / improvement_weights.sum()
        selected_scale_weights = np.zeros((len(context_scale_layout()),), dtype=np.float32)
        selected_scale_alphas = np.zeros((len(context_scale_layout()),), dtype=np.float32)
        ensemble_val_delta = np.zeros_like(anchor_val_vec, dtype=np.float32)
        for weight, candidate in zip(improvement_weights, selected_candidates, strict=True):
            scale_index = int(candidate["scale_index"])
            selected_scale_weights[scale_index] = float(weight)
            selected_scale_alphas[scale_index] = float(candidate["blend_alpha"])
            ensemble_val_delta = ensemble_val_delta + float(weight) * float(candidate["blend_alpha"]) * candidate["val_delta"]
        ensemble_alpha, ensemble_val_mae = self._select_blend_alpha(
            anchor_pred=anchor_val_vec,
            raw_pred=anchor_val_vec + ensemble_val_delta,
            y_true=y_val_vec,
        )

        descriptor_train_z = (np.asarray(descriptor_train, dtype=np.float32) - self.descriptor_mean) / self.descriptor_std
        descriptor_val_z = (np.asarray(descriptor_val, dtype=np.float32) - self.descriptor_mean) / self.descriptor_std
        context_train_z = (np.asarray(context_train, dtype=np.float32) - self.context_mean) / self.context_std
        context_val_z = (np.asarray(context_val, dtype=np.float32) - self.context_mean) / self.context_std
        full_train_z = np.concatenate([descriptor_train_z, context_train_z], axis=1)
        full_val_z = np.concatenate([descriptor_val_z, context_val_z], axis=1)
        full_model = Ridge(alpha=float(self.config.linear_residual_alpha), solver="svd")
        full_model.fit(full_train_z, residual_train)
        full_val_delta = np.asarray(full_model.predict(full_val_z), dtype=np.float32).reshape(-1)
        full_alpha, full_val_mae = self._select_blend_alpha(
            anchor_pred=anchor_val_vec,
            raw_pred=anchor_val_vec + full_val_delta,
            y_true=y_val_vec,
        )

        self.linear_scale_weights = selected_scale_weights
        self.linear_scale_alphas = selected_scale_alphas
        if float(full_val_mae) <= float(ensemble_val_mae):
            self.linear_full_context_model = full_model
            self.blend_alpha = float(full_alpha)
            self.selected_backend = "linear_full_context"
            return self.selected_backend, float(full_val_mae)

        self.linear_full_context_model = None
        self.blend_alpha = float(ensemble_alpha)
        self.selected_backend = "linear_scale_ensemble"
        return self.selected_backend, float(ensemble_val_mae)

    @torch.no_grad()
    def _predict_internal(self, descriptor: np.ndarray, context: np.ndarray, anchor_pred: np.ndarray) -> dict[str, np.ndarray]:
        if self.selected_backend in {"linear_full_context", "linear_scale_ensemble"}:
            if self.context_mean is None or self.context_std is None:
                raise RuntimeError("MSCERegressor has not been fit yet")
            if self.descriptor_mean is None or self.descriptor_std is None:
                raise RuntimeError("MSCERegressor has not been fit yet")
            descriptor_z = (np.asarray(descriptor, dtype=np.float32) - self.descriptor_mean) / self.descriptor_std
            context_z = (np.asarray(context, dtype=np.float32) - self.context_mean) / self.context_std
            anchor_vec = np.asarray(anchor_pred, dtype=np.float32).reshape(-1)
            if self.selected_backend == "linear_full_context":
                if self.linear_full_context_model is None:
                    raise RuntimeError("linear full-context backend is not available")
                full_z = np.concatenate([descriptor_z, context_z], axis=1)
                raw_delta = np.asarray(self.linear_full_context_model.predict(full_z), dtype=np.float32).reshape(-1)
            else:
                if self.linear_scale_weights is None or self.linear_scale_alphas is None or not self.linear_scale_models:
                    raise RuntimeError("linear scale backend is not available")
                raw_delta = np.zeros((context_z.shape[0],), dtype=np.float32)
                for scale_index, scale_matrix in enumerate(self._context_slices(context_z)):
                    if float(self.linear_scale_weights[scale_index]) <= 0.0:
                        continue
                    scale_delta = np.asarray(self.linear_scale_models[scale_index].predict(scale_matrix), dtype=np.float32).reshape(-1)
                    raw_delta = raw_delta + (
                        float(self.linear_scale_weights[scale_index])
                        * float(self.linear_scale_alphas[scale_index])
                        * scale_delta
                    )
            blended_delta = raw_delta * float(self.blend_alpha)
            pred = anchor_vec + blended_delta
            if self.linear_scale_weights is None:
                scale_weight_matrix = np.zeros((context_z.shape[0], len(context_scale_layout())), dtype=np.float32)
            else:
                scale_weight_matrix = np.repeat(self.linear_scale_weights.reshape(1, -1), context_z.shape[0], axis=0)
            if scale_weight_matrix.shape[1] == 0:
                dominant_weight = np.zeros((context_z.shape[0],), dtype=np.float32)
                selection_entropy = np.zeros((context_z.shape[0],), dtype=np.float32)
            else:
                dominant_weight = scale_weight_matrix.max(axis=1)
                safe_weights = np.clip(scale_weight_matrix, 1e-8, 1.0)
                selection_entropy = -np.sum(safe_weights * np.log(safe_weights), axis=1) / math.log(float(scale_weight_matrix.shape[1]))
            return {
                "pred": pred,
                "anchor_pred": anchor_vec,
                "bounded_delta": blended_delta,
                "raw_bounded_delta": raw_delta,
                "repair_gate": np.ones((context_z.shape[0],), dtype=np.float32),
                "scale_weights": scale_weight_matrix,
                "selection_entropy": selection_entropy.astype(np.float32),
                "dominant_scale_weight": dominant_weight.astype(np.float32),
                "blend_alpha": np.full((context_z.shape[0],), float(self.blend_alpha), dtype=np.float32),
                "backend_name": np.asarray([self.selected_backend] * context_z.shape[0], dtype=object),
            }
        if self.model is None:
            raise RuntimeError("MSCERegressor must be fit before predict")
        descriptor_z, context_z = self._standardize_inputs(descriptor, context)
        descriptor_tensor = torch.tensor(descriptor_z, dtype=torch.float32, device=self.device)
        context_tensor = torch.tensor(context_z, dtype=torch.float32, device=self.device)
        anchor_tensor = torch.tensor(np.asarray(anchor_pred, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device)
        out = self.model(descriptor_tensor, self._split_context_tensor(context_tensor), anchor_tensor)
        raw_delta = out["bounded_delta"].detach().cpu().numpy().reshape(-1)
        blended_delta = raw_delta * float(self.blend_alpha)
        pred = out["anchor_pred"].detach().cpu().numpy().reshape(-1) + blended_delta
        return {
            "pred": pred,
            "anchor_pred": out["anchor_pred"].detach().cpu().numpy().reshape(-1),
            "bounded_delta": blended_delta,
            "raw_bounded_delta": raw_delta,
            "repair_gate": out["repair_gate"].detach().cpu().numpy().reshape(-1),
            "scale_weights": out["scale_weights"].detach().cpu().numpy(),
            "selection_entropy": out["selection_entropy"].detach().cpu().numpy().reshape(-1),
            "dominant_scale_weight": out["dominant_scale_weight"].detach().cpu().numpy().reshape(-1),
            "blend_alpha": np.full((descriptor_tensor.shape[0],), float(self.blend_alpha), dtype=np.float32),
            "backend_name": np.asarray([self.selected_backend] * descriptor_tensor.shape[0], dtype=object),
        }

    def fit(
        self,
        *,
        descriptor_train: np.ndarray,
        context_train: np.ndarray,
        anchor_train: np.ndarray,
        y_train: np.ndarray,
        descriptor_val: np.ndarray,
        context_val: np.ndarray,
        anchor_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "MSCERegressor":
        _set_training_seed(self.train_seed)
        self._prepare_training_stats(descriptor_train, context_train)
        train_loader = self._build_loader(descriptor_train, context_train, anchor_train, y_train, shuffle=True)
        val_loader = self._build_loader(descriptor_val, context_val, anchor_val, y_val, shuffle=False)

        scale_dims = [item["width"] for item in context_scale_layout()]
        self.model = _MSCENet(
            descriptor_dim=int(np.asarray(descriptor_train).shape[1]),
            scale_dims=scale_dims,
            config=self.config,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )

        best_score = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_blend_alpha = 0.0
        bad_epochs = 0
        for _epoch in range(int(self.config.max_epochs)):
            self.model.train()
            for descriptor_tensor, context_tensor, anchor_tensor, target_tensor in train_loader:
                descriptor_tensor = descriptor_tensor.to(self.device)
                context_tensor = context_tensor.to(self.device)
                anchor_tensor = anchor_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                out = self.model(descriptor_tensor, self._split_context_tensor(context_tensor), anchor_tensor)
                loss, _ = self._loss(out, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                optimizer.step()

            self.model.eval()
            val_parts: list[dict[str, np.ndarray]] = []
            for descriptor_tensor, context_tensor, anchor_tensor, _target_tensor in val_loader:
                descriptor_tensor = descriptor_tensor.to(self.device)
                context_tensor = context_tensor.to(self.device)
                anchor_tensor = anchor_tensor.to(self.device)
                out = self.model(descriptor_tensor, self._split_context_tensor(context_tensor), anchor_tensor)
                val_parts.append(
                    {
                        "pred": out["pred"].detach().cpu().numpy().reshape(-1),
                        "repair_gate": out["repair_gate"].detach().cpu().numpy().reshape(-1),
                        "bounded_delta": out["bounded_delta"].detach().cpu().numpy().reshape(-1),
                        "selection_entropy": out["selection_entropy"].detach().cpu().numpy().reshape(-1),
                    }
                )
            val_pred = np.concatenate([part["pred"] for part in val_parts], axis=0)
            best_epoch_alpha, val_score = self._select_blend_alpha(
                anchor_pred=np.asarray(anchor_val, dtype=np.float32).reshape(-1),
                raw_pred=val_pred,
                y_true=np.asarray(y_val, dtype=np.float32).reshape(-1),
            )
            if val_score < best_score:
                best_score = val_score
                best_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
                best_blend_alpha = float(best_epoch_alpha)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.config.patience):
                    break

        if best_state is None or self.model is None:
            raise RuntimeError("failed to store an MSCE checkpoint")
        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        neural_blend_alpha = float(best_blend_alpha)
        linear_backend_name, linear_val_mae = self._fit_linear_backend(
            descriptor_train=descriptor_train,
            context_train=context_train,
            anchor_train=anchor_train,
            y_train=y_train,
            descriptor_val=descriptor_val,
            context_val=context_val,
            anchor_val=anchor_val,
            y_val=y_val,
        )
        if float(linear_val_mae) + 1e-8 < float(best_score):
            return self
        self.blend_alpha = neural_blend_alpha
        self.selected_backend = "neural"
        return self

    def predict(self, descriptor: np.ndarray, context: np.ndarray, anchor_pred: np.ndarray) -> np.ndarray:
        return self._predict_internal(descriptor, context, anchor_pred)["pred"]

    def predict_details(self, descriptor: np.ndarray, context: np.ndarray, anchor_pred: np.ndarray) -> dict[str, np.ndarray]:
        return self._predict_internal(descriptor, context, anchor_pred)
