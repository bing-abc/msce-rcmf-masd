from __future__ import annotations

"""Shared numeric helpers for repeated-run paper reporting."""

import math
from typing import Iterable

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        # Re-sample run-level values rather than assuming a parametric shape.
        means[i] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def sign_counts(deltas: Iterable[float], tol: float = 1e-8) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for value in deltas:
        if value < -tol:
            wins += 1
        elif value > tol:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def format_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.6f}, {ci[1]:.6f}]"


def format_sign(counts: tuple[int, int, int]) -> str:
    return f"{counts[0]}/{counts[1]}/{counts[2]}"
