from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tg_prediction_pipeline.schemas import HardSubsetConfig, HardSubsetDefinition, HardSubsetSlice


def _default_protocol_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "protocol.yaml"


def load_hard_subset_config(config_path: str | Path | None = None) -> HardSubsetConfig:
    path = Path(config_path) if config_path is not None else _default_protocol_config_path()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    hard_payload = payload.get("hard_subset_protocol", {})
    score_weights = {
        "conflict": float(hard_payload.get("score_weights", {}).get("conflict", 1.0)),
        "uncertainty": float(hard_payload.get("score_weights", {}).get("uncertainty", 1.0)),
        "low_confidence": float(hard_payload.get("score_weights", {}).get("low_confidence", 1.0)),
    }
    return HardSubsetConfig(
        method=str(hard_payload.get("method", "reference_absolute_error")),
        quantile=float(hard_payload.get("quantile", 0.80)),
        minimum_size=int(hard_payload.get("minimum_size", 1)),
        threshold_scope=str(hard_payload.get("threshold_scope", "per_slice")),
        score_weights=score_weights,
    )


def _validate_hard_subset_config(config: HardSubsetConfig) -> None:
    if not 0.0 < config.quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if config.minimum_size <= 0:
        raise ValueError("minimum_size must be positive")
    if config.threshold_scope not in {"per_slice"}:
        raise ValueError(f"unsupported threshold_scope: {config.threshold_scope}")
    if config.method not in {"reference_absolute_error", "weighted_reference_difficulty"}:
        raise ValueError(f"unsupported hard subset method: {config.method}")


def _float_array(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"missing required key: {key}")
    return np.asarray(payload[key], dtype=np.float64).reshape(-1)


def _int_array(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"missing required key: {key}")
    return np.asarray(payload[key], dtype=np.int64).reshape(-1)


def _prediction_array(payload: dict[str, Any]) -> np.ndarray:
    if "prediction" in payload:
        return _float_array(payload, "prediction")
    if "pred" in payload:
        return _float_array(payload, "pred")
    raise KeyError("payload must contain either 'prediction' or 'pred'")


def _reference_error_score(payload: dict[str, Any]) -> np.ndarray:
    y_true = _float_array(payload, "y_true")
    prediction = _prediction_array(payload)
    return np.abs(prediction - y_true)


def _weighted_reference_difficulty_score(payload: dict[str, Any], config: HardSubsetConfig) -> np.ndarray:
    conflict = _float_array(payload, "conflict")
    uncertainty = _float_array(payload, "uncertainty")
    if "confidence" in payload:
        confidence = _float_array(payload, "confidence")
    else:
        confidence = np.ones(conflict.shape[0], dtype=np.float64)
    return (
        float(config.score_weights["conflict"]) * conflict
        + float(config.score_weights["uncertainty"]) * uncertainty
        + float(config.score_weights["low_confidence"]) * (1.0 - confidence)
    )


def _difficulty_score(payload: dict[str, Any], config: HardSubsetConfig) -> np.ndarray:
    if config.method == "reference_absolute_error":
        return _reference_error_score(payload)
    return _weighted_reference_difficulty_score(payload, config)


def _select_top_quantile_mask(scores: np.ndarray, quantile: float, minimum_size: int) -> tuple[float, np.ndarray]:
    if scores.size == 0:
        raise ValueError("difficulty score array is empty")
    threshold = float(np.quantile(scores, quantile))
    mask = scores >= threshold
    minimum_size = min(int(minimum_size), int(scores.size))
    if int(mask.sum()) < minimum_size:
        order = np.argsort(scores, kind="mergesort")
        forced_mask = np.zeros(scores.size, dtype=bool)
        forced_mask[order[-minimum_size:]] = True
        mask = forced_mask
        threshold = float(scores[order[-minimum_size]])
    return threshold, mask


def build_hard_subset_slice(payload: dict[str, Any], config: HardSubsetConfig) -> HardSubsetSlice:
    _validate_hard_subset_config(config)
    sample_index = _int_array(payload, "sample_index")
    difficulty_score = _difficulty_score(payload, config)
    threshold, hard_subset_mask = _select_top_quantile_mask(
        scores=difficulty_score,
        quantile=float(config.quantile),
        minimum_size=int(config.minimum_size),
    )
    metadata = {
        "method": config.method,
        "threshold_scope": config.threshold_scope,
        "quantile": float(config.quantile),
        "minimum_size": int(config.minimum_size),
        "hard_subset_size": int(hard_subset_mask.sum()),
    }
    return HardSubsetSlice(
        sample_index=tuple(int(item) for item in sample_index.tolist()),
        difficulty_score=tuple(float(item) for item in difficulty_score.tolist()),
        hard_subset_mask=tuple(bool(item) for item in hard_subset_mask.tolist()),
        threshold=float(threshold),
        metadata=metadata,
    )


def build_hard_subset_definition(
    *,
    split_id: int,
    reference_model_name: str,
    test_payload: dict[str, Any],
    external_payload: dict[str, Any] | None = None,
    config: HardSubsetConfig | None = None,
) -> HardSubsetDefinition:
    resolved_config = config or HardSubsetConfig()
    test_slice = build_hard_subset_slice(test_payload, resolved_config)
    external_slice = None if external_payload is None else build_hard_subset_slice(external_payload, resolved_config)
    metadata = {
        "threshold_scope": resolved_config.threshold_scope,
        "score_weights": dict(resolved_config.score_weights),
    }
    return HardSubsetDefinition(
        split_id=int(split_id),
        reference_model_name=str(reference_model_name),
        method=str(resolved_config.method),
        test_slice=test_slice,
        external_slice=external_slice,
        metadata=metadata,
    )
