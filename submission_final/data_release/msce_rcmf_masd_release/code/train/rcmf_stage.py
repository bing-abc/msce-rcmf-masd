from __future__ import annotations

"""Paper-facing RCMF wrappers over the legacy minimal RCMF implementation."""

from typing import Any

from train.rcmf_min_repair import (
    evaluate_rcmf_external_focus as evaluate_rcmf_stage,
    train_rcmf_external_focus_student as _train_legacy_rcmf_external_focus_stage,
    train_rcmf_student as _train_legacy_rcmf_stage,
)


def _resolve_repeat_id(*, repeat_id: int | None, seed: int | None) -> int:
    if repeat_id is not None and seed is not None and int(repeat_id) != int(seed):
        raise ValueError("repeat_id and seed disagree")
    if repeat_id is not None:
        return int(repeat_id)
    if seed is not None:
        return int(seed)
    raise ValueError("either repeat_id or seed must be provided")


def train_rcmf_stage(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    repair_model: Any,
    repeat_id: int | None = None,
    seed: int | None = None,
):
    """Train the legacy RCMF bridge stage using paper-facing naming."""
    resolved_repeat_id = _resolve_repeat_id(repeat_id=repeat_id, seed=seed)
    return _train_legacy_rcmf_stage(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        repair_model=repair_model,
        seed=resolved_repeat_id,
    )


def train_rcmf_external_focus_stage(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    minimal_rcmf: Any,
    repeat_id: int | None = None,
    seed: int | None = None,
):
    """Train the legacy external-focus RCMF stage using paper-facing naming."""
    resolved_repeat_id = _resolve_repeat_id(repeat_id=repeat_id, seed=seed)
    return _train_legacy_rcmf_external_focus_stage(
        split=split,
        seed_tensors=seed_tensors,
        config=config,
        minimal_rcmf=minimal_rcmf,
        seed=resolved_repeat_id,
    )


__all__ = [
    "evaluate_rcmf_stage",
    "train_rcmf_external_focus_stage",
    "train_rcmf_stage",
]
