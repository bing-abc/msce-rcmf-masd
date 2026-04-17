from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.seeds import FULL_SEEDS


def _make_bins(y: np.ndarray, n_bins: int = 12) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y, quantiles))
    if len(edges) < 3:
        return np.zeros_like(y, dtype=np.int64)
    bins = np.digitize(y, edges[1:-1], right=True)
    return bins.astype(np.int64)


def generate_splits(dataset: pd.DataFrame) -> dict[str, object]:
    primary_idx = dataset.index[dataset["role"] == "primary_pool"].to_numpy()
    supplemental_idx = dataset.index[dataset["role"] == "supplemental_train"].to_numpy()
    external_idx = dataset.index[dataset["role"] == "external_holdout"].to_numpy()

    primary_y = dataset.loc[primary_idx, "tg_k"].to_numpy(dtype=np.float32)
    primary_bins = _make_bins(primary_y, n_bins=14)

    payload: dict[str, object] = {
        "primary_pool_size": int(len(primary_idx)),
        "supplemental_train_size": int(len(supplemental_idx)),
        "external_size": int(len(external_idx)),
        "seeds": {},
    }

    for seed in FULL_SEEDS:
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_val_pos, test_pos = next(outer.split(primary_idx, primary_bins))
        train_val_idx = primary_idx[train_val_pos]
        test_idx = primary_idx[test_pos]

        train_val_bins = primary_bins[train_val_pos]
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=seed)
        train_pos, val_pos = next(inner.split(train_val_idx, train_val_bins))
        train_idx = np.concatenate([train_val_idx[train_pos], supplemental_idx]).astype(np.int64)
        val_idx = train_val_idx[val_pos].astype(np.int64)
        test_idx = test_idx.astype(np.int64)

        payload["seeds"][str(seed)] = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
            "external": external_idx.astype(np.int64).tolist(),
            "counts": {
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "n_external": int(len(external_idx)),
            },
        }
    return payload


def main() -> int:
    dataset = pd.read_csv(ROOT / "data/dataset.csv")
    splits = generate_splits(dataset)
    path = ROOT / "data/splits.json"
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
