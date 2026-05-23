from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from tg_prediction_pipeline.schemas import ProtocolSplit, ProtocolSplitConfig


def _default_protocol_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "protocol.yaml"


def load_protocol_config(config_path: str | Path | None = None) -> ProtocolSplitConfig:
    path = Path(config_path) if config_path is not None else _default_protocol_config_path()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    split_payload = payload.get("split_protocol", {})
    return ProtocolSplitConfig(
        n_repeats=int(split_payload.get("n_repeats", 20)),
        split_id_offset=int(split_payload.get("split_id_offset", 0)),
        target_column=str(split_payload.get("target_column", "tg_k")),
        role_column=str(split_payload.get("role_column", "role")),
        primary_role=str(split_payload.get("primary_role", "primary_pool")),
        supplemental_role=str(split_payload.get("supplemental_role", "supplemental_train")),
        external_role=str(split_payload.get("external_role", "external_holdout")),
        test_fraction=float(split_payload.get("test_fraction", 0.15)),
        val_fraction_within_trainval=float(split_payload.get("val_fraction_within_trainval", 0.1764705882)),
        stratify_bins=int(split_payload.get("stratify_bins", 14)),
    )


def _validate_protocol_config(config: ProtocolSplitConfig) -> None:
    if config.n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if not 0.0 < config.test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0.0 < config.val_fraction_within_trainval < 1.0:
        raise ValueError("val_fraction_within_trainval must be between 0 and 1")
    if config.stratify_bins < 2:
        raise ValueError("stratify_bins must be at least 2")


def _make_stratification_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 3:
        return np.zeros(values.shape[0], dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=True).astype(np.int64)


def _role_indices(dataset: pd.DataFrame, role_column: str, role_name: str) -> np.ndarray:
    return dataset.index[dataset[role_column].astype(str) == str(role_name)].to_numpy(dtype=np.int64)


def build_protocol_split(dataset: pd.DataFrame, split_id: int, config: ProtocolSplitConfig) -> ProtocolSplit:
    _validate_protocol_config(config)

    primary_indices = _role_indices(dataset, config.role_column, config.primary_role)
    supplemental_indices = _role_indices(dataset, config.role_column, config.supplemental_role)
    external_indices = _role_indices(dataset, config.role_column, config.external_role)

    if primary_indices.size == 0:
        raise ValueError("primary pool is empty")

    primary_targets = dataset.loc[primary_indices, config.target_column].to_numpy(dtype=np.float64)
    primary_bins = _make_stratification_bins(primary_targets, config.stratify_bins)

    random_state = int(config.split_id_offset + split_id)
    outer = StratifiedShuffleSplit(n_splits=1, test_size=float(config.test_fraction), random_state=random_state)
    train_val_pos, test_pos = next(outer.split(primary_indices, primary_bins))
    train_val_indices = primary_indices[train_val_pos]
    test_indices = primary_indices[test_pos]

    train_val_bins = primary_bins[train_val_pos]
    inner = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(config.val_fraction_within_trainval),
        random_state=random_state,
    )
    train_pos, val_pos = next(inner.split(train_val_indices, train_val_bins))

    train_indices = np.concatenate([train_val_indices[train_pos], supplemental_indices]).astype(np.int64)
    val_indices = train_val_indices[val_pos].astype(np.int64)
    test_indices = test_indices.astype(np.int64)
    external_indices = external_indices.astype(np.int64)

    metadata: dict[str, Any] = {
        "target_column": config.target_column,
        "role_column": config.role_column,
        "primary_role": config.primary_role,
        "supplemental_role": config.supplemental_role,
        "external_role": config.external_role,
        "test_fraction": float(config.test_fraction),
        "val_fraction_within_trainval": float(config.val_fraction_within_trainval),
        "stratify_bins": int(config.stratify_bins),
        "counts": {
            "n_primary": int(primary_indices.size),
            "n_supplemental": int(supplemental_indices.size),
            "n_external": int(external_indices.size),
            "n_train": int(train_indices.size),
            "n_val": int(val_indices.size),
            "n_test": int(test_indices.size),
        },
    }
    return ProtocolSplit(
        split_id=int(split_id),
        random_state=random_state,
        train_indices=tuple(int(item) for item in train_indices.tolist()),
        val_indices=tuple(int(item) for item in val_indices.tolist()),
        test_indices=tuple(int(item) for item in test_indices.tolist()),
        external_indices=tuple(int(item) for item in external_indices.tolist()),
        metadata=metadata,
    )


def generate_protocol_splits(dataset: pd.DataFrame, config: ProtocolSplitConfig) -> list[ProtocolSplit]:
    return [build_protocol_split(dataset=dataset, split_id=split_id, config=config) for split_id in range(config.n_repeats)]


def export_protocol_splits(splits: list[ProtocolSplit], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"splits": [split.to_dict() for split in splits]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
