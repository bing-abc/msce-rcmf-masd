"""Training and evaluation runners for the standalone pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "run_baseline_benchmark",
    "run_baseline_suite",
    "run_local_baseline_workflow",
    "run_msce_benchmark",
    "run_rcmf_benchmark",
    "run_masd_benchmark",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        if name == "run_baseline_benchmark":
            module = import_module("tg_prediction_pipeline.pipeline.run_baseline_benchmark")
            return getattr(module, name)
        if name == "run_msce_benchmark":
            module = import_module("tg_prediction_pipeline.pipeline.run_msce_benchmark")
            return getattr(module, name)
        if name == "run_rcmf_benchmark":
            module = import_module("tg_prediction_pipeline.pipeline.run_rcmf_benchmark")
            return getattr(module, name)
        if name == "run_masd_benchmark":
            module = import_module("tg_prediction_pipeline.pipeline.run_masd_benchmark")
            return getattr(module, name)
        module = import_module("tg_prediction_pipeline.pipeline.train_baseline")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
