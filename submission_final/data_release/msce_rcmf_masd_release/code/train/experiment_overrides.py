from __future__ import annotations

"""Temporary knobs used by fasttrack scans around the locked full model.

The current paper keeps the main chain fixed, so these overrides are only used
by controlled follow-up searches and should not silently leak into default runs.
"""

from contextlib import contextmanager
from typing import Any


_DEFAULTS: dict[str, Any] = {
    # Every override defaults to the locked paper configuration.
    "label": "default",
    "hard_reweight_alpha": 0.0,
    "thresholded_masd_enabled": False,
    "thresholded_masd_bound_k": 0.0,
    "thresholded_masd_tau": 0.70,
    "thresholded_masd_gamma": 8.0,
    "thresholded_masd_mean_lambda": 0.0,
    "thresholded_masd_sparse_lambda": 0.0,
    "thresholded_masd_focus_only": False,
    "rcmf_q_temperature": 1.0,
    "rcmf_entropy_lambda": 0.0,
    "disable_rcmf_anchor": False,
    "mspce_top_k": 3,
}

_STATE: dict[str, Any] = dict(_DEFAULTS)


def clear_experiment_overrides() -> None:
    _STATE.clear()
    _STATE.update(_DEFAULTS)


def set_experiment_overrides(**kwargs: Any) -> None:
    clear_experiment_overrides()
    _STATE.update(kwargs)


def update_experiment_overrides(**kwargs: Any) -> None:
    _STATE.update(kwargs)


def get_experiment_overrides() -> dict[str, Any]:
    return dict(_STATE)


@contextmanager
def temporary_experiment_overrides(**kwargs: Any):
    snapshot = dict(_STATE)
    try:
        clear_experiment_overrides()
        _STATE.update(snapshot)
        _STATE.update(kwargs)
        yield
    finally:
        _STATE.clear()
        _STATE.update(snapshot)


def apply_experiment_overrides(model: Any, *, seed_tensors: dict[str, Any] | None = None) -> None:
    y_std = 1.0
    if seed_tensors is not None and "y_std" in seed_tensors:
        raw = seed_tensors["y_std"]
        if hasattr(raw, "reshape"):
            y_std = float(raw.reshape(-1)[0].item())
        else:
            y_std = float(raw)
    bound_k = float(_STATE.get("thresholded_masd_bound_k", 0.0))
    setattr(model, "pr_experiment_label", str(_STATE.get("label", "default")))
    setattr(model, "pr_hard_reweight_alpha", float(_STATE.get("hard_reweight_alpha", 0.0)))
    setattr(model, "pr_thresholded_masd_enabled", bool(_STATE.get("thresholded_masd_enabled", False)))
    setattr(model, "pr_thresholded_masd_bound_k", bound_k)
    setattr(model, "pr_thresholded_masd_bound", float(bound_k / max(y_std, 1e-6)) if bound_k > 0.0 else 0.0)
    setattr(model, "pr_thresholded_masd_tau", float(_STATE.get("thresholded_masd_tau", 0.70)))
    setattr(model, "pr_thresholded_masd_gamma", float(_STATE.get("thresholded_masd_gamma", 8.0)))
    setattr(model, "pr_thresholded_masd_mean_lambda", float(_STATE.get("thresholded_masd_mean_lambda", 0.0)))
    setattr(model, "pr_thresholded_masd_sparse_lambda", float(_STATE.get("thresholded_masd_sparse_lambda", 0.0)))
    setattr(model, "pr_thresholded_masd_focus_only", bool(_STATE.get("thresholded_masd_focus_only", False)))
    setattr(model, "pr_rcmf_q_temperature", float(_STATE.get("rcmf_q_temperature", 1.0)))
    setattr(model, "pr_rcmf_entropy_lambda", float(_STATE.get("rcmf_entropy_lambda", 0.0)))
    setattr(model, "pr_disable_rcmf_anchor", bool(_STATE.get("disable_rcmf_anchor", False)))
    mspce_top_k = int(_STATE.get("mspce_top_k", 3))
    if hasattr(model, "ctx_encoder") and hasattr(model.ctx_encoder, "top_k_active"):
        model.ctx_encoder.top_k_active = max(1, min(mspce_top_k, len(getattr(model.ctx_encoder, "_scale_slices", []) or [4])))
