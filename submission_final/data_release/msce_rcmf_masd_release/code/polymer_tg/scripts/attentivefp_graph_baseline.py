from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import mae  # noqa: E402
from models.backbone import AttentiveGraphEncoder  # noqa: E402
from models.modules import ExpertHead  # noqa: E402
from train.full_train import (  # noqa: E402
    DEVICE,
    _to_device,
    diagnostic_config,
    load_artifacts,
    make_loader,
    prepare_seed_tensors,
    set_seed,
)

DEFAULT_OUTPUT_PREFIX = "attentivefp_graph_baseline"
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


class AttentiveFPOnlyRegressor(nn.Module):
    def __init__(self, seed_tensors: dict[str, Any], hidden_dim: int = 128) -> None:
        super().__init__()
        graph = seed_tensors["graphs"][0]
        node_dim = int(graph.x.shape[1])
        edge_dim = int(graph.edge_attr.shape[1])
        self.encoder = AttentiveGraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.head = ExpertHead(hidden_dim=hidden_dim)

    def forward(self, graph_batch: Any) -> dict[str, torch.Tensor]:
        graph_emb = self.encoder(graph_batch)
        pred, unc = self.head(graph_emb)
        return {"pred": pred, "unc": unc}


@torch.no_grad()
def collect_predictions(
    model: AttentiveFPOnlyRegressor,
    loader: torch.utils.data.DataLoader,
    seed_tensors: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    y_scaled: list[torch.Tensor] = []
    pred_scaled: list[torch.Tensor] = []
    sample_index: list[torch.Tensor] = []
    for batch in loader:
        batch = _to_device(batch)
        out = model(batch["graph"])
        y_scaled.append(batch["y"].detach().cpu())
        pred_scaled.append(out["pred"].detach().cpu())
        sample_index.append(batch["sample_index"].detach().cpu())
    y_scaled_t = torch.cat(y_scaled, dim=0)
    pred_scaled_t = torch.cat(pred_scaled, dim=0)
    index_t = torch.cat(sample_index, dim=0).reshape(-1)
    y = y_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    pred = pred_scaled_t * seed_tensors["y_std"] + seed_tensors["y_mean"]
    return {
        "y_true": y.numpy().squeeze(1),
        "pred": pred.numpy().squeeze(1),
        "sample_index": index_t.numpy().astype(np.int64),
        "mae_k": float(mae(y.numpy(), pred.numpy())),
    }


def train_model(
    *,
    split: dict[str, Any],
    seed_tensors: dict[str, Any],
    config: Any,
    seed: int,
) -> AttentiveFPOnlyRegressor:
    set_seed(seed)
    model = AttentiveFPOnlyRegressor(seed_tensors, hidden_dim=int(config.hidden_dim)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.comparator_lr), weight_decay=float(config.weight_decay))
    loss_fn = nn.SmoothL1Loss()
    train_loader = make_loader(seed_tensors, split["train"], config.batch_size, shuffle=True)
    val_loader = make_loader(seed_tensors, split["val"], config.batch_size, shuffle=False)

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    for _epoch in range(int(config.comparator_epochs)):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch)
            out = model(batch["graph"])
            loss = loss_fn(out["pred"], batch["y"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        val_pred = collect_predictions(model, val_loader, seed_tensors)
        val_score = float(val_pred["mae_k"])
        if val_score < best_val:
            best_val = val_score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(config.comparator_patience):
                break
    if best_state is None:
        raise RuntimeError("no checkpoint stored for attentivefp_graph_baseline")
    model.load_state_dict(best_state)
    return model


def run_seed(
    *,
    seed: int,
    dataset: pd.DataFrame,
    features: dict[str, Any],
    splits: dict[str, Any],
    config: Any,
    reference_masks: dict[int, dict[str, dict[int, bool]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = ensure_protocol_split(splits, dataset, seed=seed)
    if seed not in reference_masks:
        raise RuntimeError(f"seed {seed} missing from reference hard-mask bundle")
    seed_tensors = prepare_seed_tensors(features, split["train"], dataset)
    model = train_model(split=split, seed_tensors=seed_tensors, config=config, seed=seed)
    primary_loader = make_loader(seed_tensors, split["test"], config.batch_size, shuffle=False)
    external_loader = make_loader(seed_tensors, split["external"], config.batch_size, shuffle=False)
    primary_pred = collect_predictions(model, primary_loader, seed_tensors)
    external_pred = collect_predictions(model, external_loader, seed_tensors)

    rows = [
        {
            "seed": int(seed),
            "model_name": "attentivefp_graph_only",
            "primary_clean": float(primary_pred["mae_k"]),
            "primary_noisy": float("nan"),
            "primary_hard_subgroup": hard_subgroup_mae(
                np.asarray(primary_pred["y_true"], dtype=np.float64),
                np.asarray(primary_pred["pred"], dtype=np.float64),
                np.asarray(primary_pred["sample_index"], dtype=np.int64),
                reference_masks[seed]["primary"],
            ),
            "external_holdout": float(external_pred["mae_k"]),
            "external_hard_subgroup": hard_subgroup_mae(
                np.asarray(external_pred["y_true"], dtype=np.float64),
                np.asarray(external_pred["pred"], dtype=np.float64),
                np.asarray(external_pred["sample_index"], dtype=np.int64),
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
        "model_name": "attentivefp_graph_only",
        "primary_pred": primary_pred,
        "external_pred": external_pred,
    }
    model.cpu()
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return rows, bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an AttentiveFP-only graph baseline on the locked protocol split.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--mainline-seeds", type=str, default="15,16,17,18,19")
    parser.add_argument("--reference-bundle", type=str, default=str(DEFAULT_REFERENCE_BUNDLE))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset, features, splits = load_artifacts()
    config = diagnostic_config()
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
            config=config,
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
                "device": str(DEVICE),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
