from __future__ import annotations

"""Paper-facing MSCE wrappers over the legacy MSPCE implementation."""

from typing import Any

from train.mspce_repair import (
    collect_repair_metrics as collect_msce_metrics,
    ensure_multiscale_features as ensure_msce_features,
    train_repair_student as _train_legacy_msce_stage,
)


def _resolve_repeat_id(*, repeat_id: int | None, seed: int | None) -> int:
    if repeat_id is not None and seed is not None and int(repeat_id) != int(seed):
        raise ValueError("repeat_id and seed disagree")
    if repeat_id is not None:
        return int(repeat_id)
    if seed is not None:
        return int(seed)
    raise ValueError("either repeat_id or seed must be provided")


def train_msce_stage(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    repeat_id: int | None = None,
    seed: int | None = None,
):
    """Train the legacy MSCE stage using paper-facing naming."""
    resolved_repeat_id = _resolve_repeat_id(repeat_id=repeat_id, seed=seed)
    return _train_legacy_msce_stage(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        seed=resolved_repeat_id,
    )


__all__ = [
    "collect_msce_metrics",
    "ensure_msce_features",
    "train_msce_stage",
]
