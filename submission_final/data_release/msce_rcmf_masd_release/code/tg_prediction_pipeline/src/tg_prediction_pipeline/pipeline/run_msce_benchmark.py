from __future__ import annotations

import json
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tg_prediction_pipeline.baselines import load_baseline_suite, make_regressor
from tg_prediction_pipeline.features import build_feature_cache, context_scale_layout, feature_cache_matches_current_layout, load_feature_cache, save_feature_cache
from tg_prediction_pipeline.models import MSCEConfig, MSCERegressor
from tg_prediction_pipeline.protocol.dataset_protocol import load_local_dataset, summarize_dataset
from tg_prediction_pipeline.protocol.hard_subset_protocol import build_hard_subset_definition, load_hard_subset_config
from tg_prediction_pipeline.protocol.split_protocol import export_protocol_splits, generate_protocol_splits, load_protocol_config


def default_msce_output_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "artifacts" / "msce_benchmark"


def _default_msce_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "msce.yaml"


def load_msce_config(config_path: str | Path | None = None) -> MSCEConfig:
    path = Path(config_path) if config_path is not None else _default_msce_config_path()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    stage_payload = payload.get("msce_stage", {})
    loss_payload = stage_payload.get("loss", {})
    return MSCEConfig(
        anchor_baseline_name=str(stage_payload.get("anchor_baseline_name", "mlp_descriptor_anchor")),
        hidden_dim=int(stage_payload.get("hidden_dim", 192)),
        scale_hidden_dim=int(stage_payload.get("scale_hidden_dim", 96)),
        top_k_scales=int(stage_payload.get("top_k_scales", 2)),
        delta_bound_k=float(stage_payload.get("delta_bound_k", 80.0)),
        anchor_tolerance_k=float(stage_payload.get("anchor_tolerance_k", 1.5)),
        dropout=float(stage_payload.get("dropout", 0.10)),
        learning_rate=float(stage_payload.get("learning_rate", 0.0005)),
        weight_decay=float(stage_payload.get("weight_decay", 1e-5)),
        batch_size=int(stage_payload.get("batch_size", 256)),
        max_epochs=int(stage_payload.get("max_epochs", 80)),
        patience=int(stage_payload.get("patience", 12)),
        blend_grid_size=int(stage_payload.get("blend_grid_size", 21)),
        linear_residual_alpha=float(stage_payload.get("linear_residual_alpha", 3.0)),
        anchor_margin_weight=float(loss_payload.get("anchor_margin_weight", 0.15)),
        residual_target_weight=float(loss_payload.get("residual_target_weight", 0.60)),
        gate_weight=float(loss_payload.get("gate_weight", 0.0)),
        delta_weight=float(loss_payload.get("delta_weight", 0.005)),
        entropy_weight=float(loss_payload.get("entropy_weight", 0.02)),
    )


def _select_descriptor_matrix(descriptor_matrix: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.asarray(descriptor_matrix, dtype=np.float32)[list(indices)]


def _select_context_matrix(context_matrix: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.asarray(context_matrix, dtype=np.float32)[list(indices)]


def _select_target_vector(targets: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.asarray(targets, dtype=np.float32).reshape(-1)[list(indices)]


def _predict_payload(sample_index: tuple[int, ...], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "y_true": np.asarray(y_true, dtype=np.float64).reshape(-1),
        "pred": np.asarray(y_pred, dtype=np.float64).reshape(-1),
    }


def _mask_lookup(sample_index: np.ndarray, mask: np.ndarray) -> dict[int, bool]:
    return {int(index): bool(flag) for index, flag in zip(sample_index.tolist(), mask.tolist(), strict=True)}


def _hard_subset_mae(sample_index: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, mask_lookup: dict[int, bool]) -> float:
    mask = np.asarray([mask_lookup.get(int(index), False) for index in sample_index.tolist()], dtype=bool)
    if not mask.any():
        return float(mean_absolute_error(y_true, y_pred))
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _anchor_spec(suite: Any, anchor_name: str) -> Any:
    for spec in suite.baseline_specs:
        if spec.name == anchor_name:
            return spec
    raise ValueError(f"anchor baseline '{anchor_name}' is not defined in the current suite")


def _fit_anchor(
    *,
    descriptor_matrix: np.ndarray,
    targets: np.ndarray,
    split: Any,
    anchor_spec: Any,
    train_seed: int,
) -> dict[str, np.ndarray]:
    model = make_regressor(anchor_spec, train_seed=train_seed)
    x_train = _select_descriptor_matrix(descriptor_matrix, split.train_indices)
    y_train = _select_target_vector(targets, split.train_indices)
    model.fit(x_train, y_train)
    return {
        "train": np.asarray(model.predict(_select_descriptor_matrix(descriptor_matrix, split.train_indices)), dtype=np.float32).reshape(-1),
        "val": np.asarray(model.predict(_select_descriptor_matrix(descriptor_matrix, split.val_indices)), dtype=np.float32).reshape(-1),
        "test": np.asarray(model.predict(_select_descriptor_matrix(descriptor_matrix, split.test_indices)), dtype=np.float32).reshape(-1),
        "external": np.asarray(model.predict(_select_descriptor_matrix(descriptor_matrix, split.external_indices)), dtype=np.float32).reshape(-1),
    }


def _result_row(
    *,
    split: Any,
    model_name: str,
    is_anchor: bool,
    test_payload: dict[str, Any],
    external_payload: dict[str, Any],
    hard_subset_test_lookup: dict[int, bool],
    hard_subset_external_lookup: dict[int, bool],
    delta_vs_anchor_test_mae: float,
    delta_vs_anchor_test_hard_mae: float,
    delta_vs_anchor_external_mae: float,
    delta_vs_anchor_external_hard_mae: float,
    diagnostics: dict[str, float] | None,
) -> dict[str, Any]:
    test_index = np.asarray(test_payload["sample_index"], dtype=np.int64)
    external_index = np.asarray(external_payload["sample_index"], dtype=np.int64)
    test_true = np.asarray(test_payload["y_true"], dtype=np.float64)
    test_pred = np.asarray(test_payload["pred"], dtype=np.float64)
    external_true = np.asarray(external_payload["y_true"], dtype=np.float64)
    external_pred = np.asarray(external_payload["pred"], dtype=np.float64)
    row: dict[str, Any] = {
        "split_id": int(split.split_id),
        "model_name": model_name,
        "is_anchor": bool(is_anchor),
        "status": "completed",
        "status_detail": "completed",
        "test_mae": float(mean_absolute_error(test_true, test_pred)),
        "test_rmse": _rmse(test_true, test_pred),
        "test_hard_mae": _hard_subset_mae(test_index, test_true, test_pred, hard_subset_test_lookup),
        "external_mae": float(mean_absolute_error(external_true, external_pred)),
        "external_rmse": _rmse(external_true, external_pred),
        "external_hard_mae": _hard_subset_mae(external_index, external_true, external_pred, hard_subset_external_lookup),
        "delta_vs_anchor_test_mae": float(delta_vs_anchor_test_mae),
        "delta_vs_anchor_test_hard_mae": float(delta_vs_anchor_test_hard_mae),
        "delta_vs_anchor_external_mae": float(delta_vs_anchor_external_mae),
        "delta_vs_anchor_external_hard_mae": float(delta_vs_anchor_external_hard_mae),
        "improvement_vs_anchor_test_mae": float(-delta_vs_anchor_test_mae),
        "improvement_vs_anchor_test_hard_mae": float(-delta_vs_anchor_test_hard_mae),
        "improvement_vs_anchor_external_mae": float(-delta_vs_anchor_external_mae),
        "improvement_vs_anchor_external_hard_mae": float(-delta_vs_anchor_external_hard_mae),
    }
    if diagnostics is not None:
        row.update(diagnostics)
    return row


def _msce_diagnostics(details: dict[str, np.ndarray]) -> dict[str, float]:
    diagnostics: dict[str, Any] = {
        "backend_name": str(np.asarray(details["backend_name"], dtype=object).reshape(-1)[0]),
        "blend_alpha_mean": float(np.asarray(details["blend_alpha"], dtype=np.float64).mean()),
        "repair_gate_mean": float(np.asarray(details["repair_gate"], dtype=np.float64).mean()),
        "abs_delta_mean": float(np.abs(np.asarray(details["bounded_delta"], dtype=np.float64)).mean()),
        "raw_abs_delta_mean": float(np.abs(np.asarray(details["raw_bounded_delta"], dtype=np.float64)).mean()),
        "selection_entropy_mean": float(np.asarray(details["selection_entropy"], dtype=np.float64).mean()),
        "dominant_scale_weight_mean": float(np.asarray(details["dominant_scale_weight"], dtype=np.float64).mean()),
    }
    scale_weights = np.asarray(details["scale_weights"], dtype=np.float64)
    for scale_index, scale_info in enumerate(context_scale_layout()):
        diagnostics[f"scale_weight_{scale_info['name']}_mean"] = float(scale_weights[:, scale_index].mean())
    return diagnostics


def _write_summary(results_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    results_frame.to_csv(output_dir / "msce_results.csv", index=False)
    summary_frame = (
        results_frame.groupby(["model_name", "is_anchor"], as_index=False)[
            [
                "test_mae",
                "test_rmse",
                "test_hard_mae",
                "external_mae",
                "external_rmse",
                "external_hard_mae",
                "delta_vs_anchor_test_mae",
                "delta_vs_anchor_test_hard_mae",
                "delta_vs_anchor_external_mae",
                "delta_vs_anchor_external_hard_mae",
                "improvement_vs_anchor_test_mae",
                "improvement_vs_anchor_test_hard_mae",
                "improvement_vs_anchor_external_mae",
                "improvement_vs_anchor_external_hard_mae",
            ]
        ]
        .agg(["mean", "std"])
    )
    summary_frame.columns = ["_".join([part for part in column if part]).strip("_") for column in summary_frame.columns.to_flat_index()]
    summary_frame.to_csv(output_dir / "msce_summary.csv", index=False)
    return summary_frame


def run_msce_benchmark(
    *,
    dataset_path: str | Path | None = None,
    feature_cache_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    protocol_config_path: str | Path | None = None,
    baseline_config_path: str | Path | None = None,
    msce_config_path: str | Path | None = None,
    repeats: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_local_dataset(Path(dataset_path) if dataset_path is not None else None)
    feature_cache = load_feature_cache(feature_cache_path)
    if not feature_cache_matches_current_layout(feature_cache):
        feature_cache = build_feature_cache(dataset)
        save_feature_cache(feature_cache, feature_cache_path)
    output_root = Path(output_dir) if output_dir is not None else default_msce_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_config = load_protocol_config(protocol_config_path)
    if repeats is not None:
        protocol_config = type(protocol_config)(**{**protocol_config.__dict__, "n_repeats": int(repeats)})
    splits = generate_protocol_splits(dataset, protocol_config)
    export_protocol_splits(splits, output_root / "protocol_splits.json")

    suite = load_baseline_suite(baseline_config_path)
    msce_config = load_msce_config(msce_config_path)
    anchor_spec = _anchor_spec(suite, msce_config.anchor_baseline_name)
    hard_subset_config = load_hard_subset_config(protocol_config_path)

    descriptor_matrix = np.asarray(feature_cache.descriptors, dtype=np.float32)
    context_matrix = np.asarray(feature_cache.contexts, dtype=np.float32)
    targets = np.asarray(feature_cache.targets, dtype=np.float32).reshape(-1)

    rows: list[dict[str, Any]] = []
    split_artifacts: list[dict[str, Any]] = []
    for split in splits:
        anchor_train_seed = int(split.random_state * 101 + 17)
        msce_train_seed = int(split.random_state * 101 + 43)
        anchor_predictions = _fit_anchor(
            descriptor_matrix=descriptor_matrix,
            targets=targets,
            split=split,
            anchor_spec=anchor_spec,
            train_seed=anchor_train_seed,
        )
        anchor_test_payload = _predict_payload(
            split.test_indices,
            _select_target_vector(targets, split.test_indices),
            anchor_predictions["test"],
        )
        anchor_external_payload = _predict_payload(
            split.external_indices,
            _select_target_vector(targets, split.external_indices),
            anchor_predictions["external"],
        )
        hard_subset_definition = build_hard_subset_definition(
            split_id=split.split_id,
            reference_model_name=msce_config.anchor_baseline_name,
            test_payload=anchor_test_payload,
            external_payload=anchor_external_payload,
            config=hard_subset_config,
        )
        hard_subset_test_lookup = _mask_lookup(
            np.asarray(hard_subset_definition.test_slice.sample_index, dtype=np.int64),
            np.asarray(hard_subset_definition.test_slice.hard_subset_mask, dtype=bool),
        )
        hard_subset_external_lookup = _mask_lookup(
            np.asarray(hard_subset_definition.external_slice.sample_index, dtype=np.int64),
            np.asarray(hard_subset_definition.external_slice.hard_subset_mask, dtype=bool),
        )

        msce_regressor = MSCERegressor(msce_config, train_seed=msce_train_seed)
        msce_regressor.fit(
            descriptor_train=_select_descriptor_matrix(descriptor_matrix, split.train_indices),
            context_train=_select_context_matrix(context_matrix, split.train_indices),
            anchor_train=anchor_predictions["train"],
            y_train=_select_target_vector(targets, split.train_indices),
            descriptor_val=_select_descriptor_matrix(descriptor_matrix, split.val_indices),
            context_val=_select_context_matrix(context_matrix, split.val_indices),
            anchor_val=anchor_predictions["val"],
            y_val=_select_target_vector(targets, split.val_indices),
        )
        msce_test_details = msce_regressor.predict_details(
            _select_descriptor_matrix(descriptor_matrix, split.test_indices),
            _select_context_matrix(context_matrix, split.test_indices),
            anchor_predictions["test"],
        )
        msce_external_details = msce_regressor.predict_details(
            _select_descriptor_matrix(descriptor_matrix, split.external_indices),
            _select_context_matrix(context_matrix, split.external_indices),
            anchor_predictions["external"],
        )
        msce_test_payload = _predict_payload(
            split.test_indices,
            _select_target_vector(targets, split.test_indices),
            msce_test_details["pred"],
        )
        msce_external_payload = _predict_payload(
            split.external_indices,
            _select_target_vector(targets, split.external_indices),
            msce_external_details["pred"],
        )

        anchor_test_mae = float(mean_absolute_error(anchor_test_payload["y_true"], anchor_test_payload["pred"]))
        anchor_test_hard_mae = _hard_subset_mae(
            np.asarray(anchor_test_payload["sample_index"], dtype=np.int64),
            np.asarray(anchor_test_payload["y_true"], dtype=np.float64),
            np.asarray(anchor_test_payload["pred"], dtype=np.float64),
            hard_subset_test_lookup,
        )
        anchor_external_mae = float(mean_absolute_error(anchor_external_payload["y_true"], anchor_external_payload["pred"]))
        anchor_external_hard_mae = _hard_subset_mae(
            np.asarray(anchor_external_payload["sample_index"], dtype=np.int64),
            np.asarray(anchor_external_payload["y_true"], dtype=np.float64),
            np.asarray(anchor_external_payload["pred"], dtype=np.float64),
            hard_subset_external_lookup,
        )
        msce_test_mae = float(mean_absolute_error(msce_test_payload["y_true"], msce_test_payload["pred"]))
        msce_test_hard_mae = _hard_subset_mae(
            np.asarray(msce_test_payload["sample_index"], dtype=np.int64),
            np.asarray(msce_test_payload["y_true"], dtype=np.float64),
            np.asarray(msce_test_payload["pred"], dtype=np.float64),
            hard_subset_test_lookup,
        )
        msce_external_mae = float(mean_absolute_error(msce_external_payload["y_true"], msce_external_payload["pred"]))
        msce_external_hard_mae = _hard_subset_mae(
            np.asarray(msce_external_payload["sample_index"], dtype=np.int64),
            np.asarray(msce_external_payload["y_true"], dtype=np.float64),
            np.asarray(msce_external_payload["pred"], dtype=np.float64),
            hard_subset_external_lookup,
        )

        rows.append(
            _result_row(
                split=split,
                model_name=msce_config.anchor_baseline_name,
                is_anchor=True,
                test_payload=anchor_test_payload,
                external_payload=anchor_external_payload,
                hard_subset_test_lookup=hard_subset_test_lookup,
                hard_subset_external_lookup=hard_subset_external_lookup,
                delta_vs_anchor_test_mae=0.0,
                delta_vs_anchor_test_hard_mae=0.0,
                delta_vs_anchor_external_mae=0.0,
                delta_vs_anchor_external_hard_mae=0.0,
                diagnostics=None,
            )
        )
        combined_msce_diagnostics = _msce_diagnostics(
            {
                key: np.concatenate([np.asarray(msce_test_details[key]), np.asarray(msce_external_details[key])], axis=0)
                if key != "scale_weights"
                else np.concatenate([np.asarray(msce_test_details[key]), np.asarray(msce_external_details[key])], axis=0)
                for key in msce_test_details
            }
        )
        rows.append(
            _result_row(
                split=split,
                model_name="msce_stage",
                is_anchor=False,
                test_payload=msce_test_payload,
                external_payload=msce_external_payload,
                hard_subset_test_lookup=hard_subset_test_lookup,
                hard_subset_external_lookup=hard_subset_external_lookup,
                delta_vs_anchor_test_mae=float(msce_test_mae - anchor_test_mae),
                delta_vs_anchor_test_hard_mae=float(msce_test_hard_mae - anchor_test_hard_mae),
                delta_vs_anchor_external_mae=float(msce_external_mae - anchor_external_mae),
                delta_vs_anchor_external_hard_mae=float(msce_external_hard_mae - anchor_external_hard_mae),
                diagnostics=combined_msce_diagnostics,
            )
        )

        artifact = {
            "split": split.to_dict(),
            "hard_subset_definition": hard_subset_definition.to_dict(),
            "anchor_reference": {
                "test": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in anchor_test_payload.items()},
                "external": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in anchor_external_payload.items()},
            },
            "msce_stage": {
                "test": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in msce_test_details.items()},
                "external": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in msce_external_details.items()},
            },
        }
        split_artifacts.append(artifact)
        (output_root / f"split_{split.split_id:03d}_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    results_frame = pd.DataFrame(rows)
    summary_frame = _write_summary(results_frame, output_root)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path is not None else "local_processed_dataset",
        "output_dir": str(output_root.resolve()),
        "protocol": dict(protocol_config.__dict__),
        "anchor_baseline_name": msce_config.anchor_baseline_name,
        "msce_config": dict(msce_config.__dict__),
        "dataset_summary": summarize_dataset(dataset).to_dict(orient="records"),
        "artifacts": {
            "results_csv": "msce_results.csv",
            "summary_csv": "msce_summary.csv",
            "splits_json": "protocol_splits.json",
            "manifest_json": "msce_manifest.json",
        },
    }
    (output_root / "msce_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results_frame, summary_frame


def main() -> int:
    parser = ArgumentParser(description="Run the standalone MSCE stage benchmark.")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--feature-cache-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--protocol-config-path", type=str, default="")
    parser.add_argument("--baseline-config-path", type=str, default="")
    parser.add_argument("--msce-config-path", type=str, default="")
    parser.add_argument("--repeats", type=int, default=0)
    args = parser.parse_args()

    run_msce_benchmark(
        dataset_path=args.dataset_path or None,
        feature_cache_path=args.feature_cache_path or None,
        output_dir=args.output_dir or None,
        protocol_config_path=args.protocol_config_path or None,
        baseline_config_path=args.baseline_config_path or None,
        msce_config_path=args.msce_config_path or None,
        repeats=(args.repeats or None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
