from __future__ import annotations

from typing import Any

import numpy as np

from tg_prediction_pipeline.models import AttentiveFPRegressor
from tg_prediction_pipeline.schemas import BaselineSpec, GraphCache, ProtocolSplit


def _select_graphs(graph_cache: GraphCache, indices: tuple[int, ...]) -> list[object]:
    return [graph_cache.graphs[int(index)] for index in indices]


def fit_and_predict_graph_baseline(
    *,
    spec: BaselineSpec,
    graph_cache: GraphCache,
    targets: np.ndarray,
    split: ProtocolSplit,
    train_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.estimator_key != "attentive_fp":
        raise ValueError(f"unsupported graph baseline estimator_key: {spec.estimator_key}")

    hyperparameters: dict[str, Any] = dict(spec.hyperparameters)
    regressor = AttentiveFPRegressor(
        hidden_channels=int(hyperparameters.get("hidden_channels", 128)),
        num_layers=int(hyperparameters.get("num_layers", 3)),
        num_timesteps=int(hyperparameters.get("num_timesteps", 2)),
        dropout=float(hyperparameters.get("dropout", 0.1)),
        learning_rate=float(hyperparameters.get("learning_rate", 0.001)),
        weight_decay=float(hyperparameters.get("weight_decay", 1e-5)),
        batch_size=int(hyperparameters.get("batch_size", 128)),
        max_epochs=int(hyperparameters.get("max_epochs", 60)),
        patience=int(hyperparameters.get("patience", 10)),
        train_seed=int(train_seed),
    )

    y_vector = np.asarray(targets, dtype=np.float32).reshape(-1)
    regressor.fit(
        train_graphs=_select_graphs(graph_cache, split.train_indices),
        y_train=y_vector[list(split.train_indices)],
        val_graphs=_select_graphs(graph_cache, split.val_indices),
        y_val=y_vector[list(split.val_indices)],
    )
    test_pred = regressor.predict(_select_graphs(graph_cache, split.test_indices))
    external_pred = regressor.predict(_select_graphs(graph_cache, split.external_indices))
    return test_pred, external_pred
