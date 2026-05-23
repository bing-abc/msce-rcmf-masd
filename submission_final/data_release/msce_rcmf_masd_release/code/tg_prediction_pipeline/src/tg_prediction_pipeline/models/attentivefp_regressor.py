from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP


def _set_training_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clone_graphs_with_targets(graphs: list[object], targets: np.ndarray, y_mean: float, y_std: float) -> list[object]:
    dataset: list[object] = []
    scaled_targets = (np.asarray(targets, dtype=np.float32).reshape(-1) - float(y_mean)) / float(y_std)
    for graph, target in zip(graphs, scaled_targets.tolist(), strict=True):
        item = copy.deepcopy(graph)
        item.y = torch.tensor([target], dtype=torch.float32)
        dataset.append(item)
    return dataset


class AttentiveFPRegressor:
    def __init__(
        self,
        *,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        max_epochs: int = 60,
        patience: int = 10,
        train_seed: int = 0,
    ) -> None:
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.num_timesteps = int(num_timesteps)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.train_seed = int(train_seed)
        self.model: AttentiveFP | None = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.device = _device()

    def _build_model(self, sample_graph: object) -> AttentiveFP:
        return AttentiveFP(
            in_channels=int(sample_graph.x.shape[1]),
            hidden_channels=self.hidden_channels,
            out_channels=1,
            edge_dim=int(sample_graph.edge_attr.shape[1]),
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        train_graphs: list[object],
        y_train: np.ndarray,
        val_graphs: list[object],
        y_val: np.ndarray,
    ) -> "AttentiveFPRegressor":
        if not train_graphs:
            raise ValueError("train_graphs must not be empty")
        if not val_graphs:
            raise ValueError("val_graphs must not be empty for attentivefp training")

        _set_training_seed(self.train_seed)
        self.y_mean = float(np.asarray(y_train, dtype=np.float32).mean())
        self.y_std = float(max(np.asarray(y_train, dtype=np.float32).std(), 1e-6))
        train_dataset = _clone_graphs_with_targets(train_graphs, y_train, self.y_mean, self.y_std)
        val_dataset = _clone_graphs_with_targets(val_graphs, y_val, self.y_mean, self.y_std)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = self._build_model(train_graphs[0])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.SmoothL1Loss()

        best_val_mae = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0
        for _epoch in range(self.max_epochs):
            self.model.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).reshape(-1)
                loss = loss_fn(pred, batch.y.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

            val_pred = self.predict(val_graphs)
            val_mae = float(np.mean(np.abs(np.asarray(y_val, dtype=np.float32).reshape(-1) - val_pred)))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is None or self.model is None:
            raise RuntimeError("failed to store an AttentiveFP checkpoint")
        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, graphs: list[object]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("AttentiveFPRegressor must be fit before predict")
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        predictions: list[torch.Tensor] = []
        self.model.eval()
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).reshape(-1)
            predictions.append(pred.detach().cpu())
        if not predictions:
            return np.zeros((0,), dtype=np.float32)
        stacked = torch.cat(predictions, dim=0).numpy().astype(np.float32)
        return stacked * float(self.y_std) + float(self.y_mean)
