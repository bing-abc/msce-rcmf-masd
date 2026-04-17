from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.full_train import load_artifacts  # noqa: E402

DEFAULT_OUTPUT_PREFIX = "descriptor_tree_baselines"
DEFAULT_REFERENCE_BUNDLE = (
    ROOT / "outputs" / "exp" / "diagnostics" / "protocol_clean_mainline10" / "mainline_bundle.pt"
)


def parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _split_bins(y: np.ndarray, n_bins: int = 14) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y, quantiles))
    if len(edges) < 3:
        return np.zeros_like(y, dtype=np.int64)
    return np.digitize(y, edges[1:-1], right=True).astype(np.int64)


def build_protocol_split(dataset: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    primary_idx = dataset.index[dataset["role"] == "primary_pool"].to_numpy()
    supplemental_idx = dataset.index[dataset["role"] == "supplemental_train"].to_numpy()
    external_idx = dataset.index[dataset["role"] == "external_holdout"].to_numpy()

    primary_y = dataset.loc[primary_idx, "tg_k"].to_numpy(dtype=np.float32)
    primary_bins = _split_bins(primary_y, n_bins=14)

    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=int(seed))
    train_val_pos, test_pos = next(outer.split(primary_idx, primary_bins))
    train_val_idx = primary_idx[train_val_pos]
    test_idx = primary_idx[test_pos]

    train_val_bins = primary_bins[train_val_pos]
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=int(seed))
    train_pos, val_pos = next(inner.split(train_val_idx, train_val_bins))
    train_idx = np.concatenate([train_val_idx[train_pos], supplemental_idx]).astype(np.int64)
    val_idx = train_val_idx[val_pos].astype(np.int64)
    test_idx = test_idx.astype(np.int64)
    external_idx = external_idx.astype(np.int64)
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
        "external": external_idx.tolist(),
    }


def ensure_protocol_split(splits: dict[str, Any], dataset: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    seed_key = str(int(seed))
    seeds_payload = splits.setdefault("seeds", {})
    if seed_key not in seeds_payload:
        seeds_payload[seed_key] = build_protocol_split(dataset, seed=int(seed))
    return seeds_payload[seed_key]


def save_bundle(run_dir: Path, name: str, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / f"{name}.pt"
    tmp_path = run_dir / f".{name}.pt.tmp"
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)


def save_results_csv(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "results.csv"
    tmp_path = run_dir / ".results.csv.tmp"
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(final_path)


def build_mask_lookup(payload: dict[str, Any]) -> dict[int, bool]:
    sample_index = np.asarray(payload["sample_index"], dtype=np.int64).reshape(-1)
    hard_mask = np.asarray(payload["hard_mask"], dtype=bool).reshape(-1)
    return {int(idx): bool(flag) for idx, flag in zip(sample_index.tolist(), hard_mask.tolist(), strict=True)}


def load_reference_masks(reference_bundle_path: Path) -> dict[int, dict[str, dict[int, bool]]]:
    bundle = torch.load(reference_bundle_path, map_location="cpu", weights_only=False)
    masks: dict[int, dict[str, dict[int, bool]]] = {}
    for seed_bundle in bundle.get("seed_bundles", []):
        seed = int(seed_bundle["seed"])
        masks[seed] = {
            "primary": build_mask_lookup(seed_bundle["baseline_primary_clean"]),
            "external": build_mask_lookup(seed_bundle["baseline_external"]),
        }
    return masks


def hard_subgroup_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_index: np.ndarray,
    mask_lookup: dict[int, bool],
) -> float:
    mask = np.asarray([bool(mask_lookup[int(idx)]) for idx in sample_index.tolist()], dtype=bool)
    if not np.any(mask):
        return float(np.mean(np.abs(y_pred - y_true)))
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def x_from_features(features: dict[str, Any], indices: list[int]) -> np.ndarray:
    return np.asarray(features["descriptors"][indices].detach().cpu().numpy(), dtype=np.float32)


def y_from_features(features: dict[str, Any], indices: list[int]) -> np.ndarray:
    return np.asarray(features["targets"][indices].detach().cpu().numpy(), dtype=np.float32).reshape(-1)


def fit_random_forest(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    return model


def run_seed(
    *,
    seed: int,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    reference_masks: dict[int, dict[str, dict[int, bool]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = ensure_protocol_split(splits, dataset, seed=seed)
    if seed not in reference_masks:
        raise RuntimeError(f"seed {seed} missing from reference hard-mask bundle")

    x_train = x_from_features(features, split["train"])
    y_train = y_from_features(features, split["train"])
    x_primary = x_from_features(features, split["test"])
    y_primary = y_from_features(features, split["test"])
    x_external = x_from_features(features, split["external"])
    y_external = y_from_features(features, split["external"])
    primary_index = np.asarray(split["test"], dtype=np.int64)
    external_index = np.asarray(split["external"], dtype=np.int64)

    rf = fit_random_forest(x_train=x_train, y_train=y_train, seed=seed)
    rf_primary_pred = np.asarray(rf.predict(x_primary), dtype=np.float64).reshape(-1)
    rf_external_pred = np.asarray(rf.predict(x_external), dtype=np.float64).reshape(-1)

    rows = [
        {
            "seed": int(seed),
            "model_name": "descriptor_random_forest",
            "primary_clean": float(np.mean(np.abs(rf_primary_pred - y_primary))),
            "primary_noisy": float("nan"),
            "primary_hard_subgroup": hard_subgroup_mae(
                y_primary.astype(np.float64),
                rf_primary_pred,
                primary_index,
                reference_masks[seed]["primary"],
            ),
            "external_holdout": float(np.mean(np.abs(rf_external_pred - y_external))),
            "external_hard_subgroup": hard_subgroup_mae(
                y_external.astype(np.float64),
                rf_external_pred,
                external_index,
                reference_masks[seed]["external"],
            ),
            "result_group": "published_baseline",
            "pass_flag": "",
        }
    ]
    bundle = {
        "seed": int(seed),
        "split": split,
        "reference_mask_source": "baseline_simple_concat_from_protocol_clean_mainline10",
        "models": {
            "descriptor_random_forest": {
                "params": rf.get_params(),
                "primary_pred": rf_primary_pred,
                "primary_true": y_primary.astype(np.float64),
                "primary_index": primary_index,
                "external_pred": rf_external_pred,
                "external_true": y_external.astype(np.float64),
                "external_index": external_index,
            }
        },
    }
    return rows, bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Run descriptor tree baselines on the locked protocol split.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--mainline-seeds", type=str, default="15,16,17,18,19")
    parser.add_argument("--reference-bundle", type=str, default=str(DEFAULT_REFERENCE_BUNDLE))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset, features, splits = load_artifacts()
    reference_bundle_path = Path(args.reference_bundle)
    reference_masks = load_reference_masks(reference_bundle_path)
    mainline_seeds = parse_seed_list(args.mainline_seeds)

    all_rows: list[dict[str, Any]] = []
    seed_bundles: list[dict[str, Any]] = []
    for seed in mainline_seeds:
        rows, bundle = run_seed(
            seed=seed,
            dataset=dataset,
            features=features,
            splits=splits,
            reference_masks=reference_masks,
        )
        all_rows.extend(rows)
        seed_bundles.append(bundle)
        save_results_csv(run_dir, all_rows)
        save_bundle(
            run_dir,
            "mainline_bundle",
            {
                "output_prefix": str(args.output_prefix),
                "reference_bundle": str(reference_bundle_path),
                "rows": all_rows,
                "seed_bundles": seed_bundles,
                "mainline_seeds": mainline_seeds,
                "completed_mainline_seeds": [int(item["seed"]) for item in seed_bundles],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
