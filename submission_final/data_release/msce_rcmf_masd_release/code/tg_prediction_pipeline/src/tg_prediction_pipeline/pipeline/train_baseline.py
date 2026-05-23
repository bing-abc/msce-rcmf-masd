from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tg_prediction_pipeline.baselines import (
    baseline_capability_summary,
    fit_and_predict_graph_baseline,
    baseline_runtime_status,
    load_baseline_suite,
    make_regressor,
)
from tg_prediction_pipeline.features import (
    build_feature_cache,
    build_graph_cache,
    feature_cache_matches_current_layout,
    load_feature_cache,
    load_graph_cache,
    save_feature_cache,
    save_graph_cache,
)
from tg_prediction_pipeline.protocol.dataset_protocol import load_local_dataset
from tg_prediction_pipeline.protocol.hard_subset_protocol import build_hard_subset_definition, load_hard_subset_config
from tg_prediction_pipeline.protocol.split_protocol import generate_protocol_splits, load_protocol_config
from tg_prediction_pipeline.schemas import BaselineSpec, BaselineSuite, FeatureCache, GraphCache, HardSubsetConfig, ProtocolSplit


def default_baseline_output_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "artifacts" / "baselines"


def _select_input_matrix(feature_cache: FeatureCache, input_block: str) -> np.ndarray:
    if input_block == "descriptor":
        return np.asarray(feature_cache.descriptors, dtype=np.float32)
    if input_block == "multimodal":
        descriptor_block = np.asarray(feature_cache.descriptors, dtype=np.float32)
        context_block = np.asarray(feature_cache.contexts, dtype=np.float32)
        return np.concatenate([descriptor_block, context_block], axis=1)
    raise ValueError(f"unsupported input_block: {input_block}")


def _predict_payload(
    *,
    sample_index: tuple[int, ...],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    return {
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "y_true": np.asarray(y_true, dtype=np.float64).reshape(-1),
        "pred": np.asarray(y_pred, dtype=np.float64).reshape(-1),
    }


def _payload_to_jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    jsonable: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            jsonable[key] = value.tolist()
        else:
            jsonable[key] = value
    return jsonable


def _mask_lookup(sample_index: np.ndarray, mask: np.ndarray) -> dict[int, bool]:
    return {int(index): bool(flag) for index, flag in zip(sample_index.tolist(), mask.tolist(), strict=True)}


def _hard_subset_mae(sample_index: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, mask_lookup: dict[int, bool]) -> float:
    mask = np.asarray([mask_lookup.get(int(index), False) for index in sample_index.tolist()], dtype=bool)
    if not mask.any():
        return float(mean_absolute_error(y_true, y_pred))
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _baseline_order(suite: BaselineSuite) -> list[BaselineSpec]:
    ordered = list(suite.baseline_specs)
    if suite.anchor_baseline_name is None:
        return ordered
    ordered.sort(key=lambda spec: (spec.name != suite.anchor_baseline_name, spec.name))
    return ordered


def _fit_and_predict(spec: BaselineSpec, feature_cache: FeatureCache, split: ProtocolSplit, train_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    y_vector = np.asarray(feature_cache.targets, dtype=np.float32).reshape(-1)
    test_index = np.asarray(split.test_indices, dtype=np.int64)
    external_index = np.asarray(split.external_indices, dtype=np.int64)
    x_matrix = _select_input_matrix(feature_cache, spec.input_block)
    x_train = x_matrix[list(split.train_indices)]
    y_train = y_vector[list(split.train_indices)]
    model = make_regressor(spec, train_seed=train_seed)
    model.fit(x_train, y_train)
    test_payload = _predict_payload(
        sample_index=split.test_indices,
        y_true=y_vector[test_index],
        y_pred=np.asarray(model.predict(x_matrix[test_index]), dtype=np.float64),
    )
    external_payload = _predict_payload(
        sample_index=split.external_indices,
        y_true=y_vector[external_index],
        y_pred=np.asarray(model.predict(x_matrix[external_index]), dtype=np.float64),
    )
    return test_payload, external_payload


def _fit_and_predict_graph(
    spec: BaselineSpec,
    feature_cache: FeatureCache,
    graph_cache: GraphCache,
    split: ProtocolSplit,
    train_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_vector = np.asarray(feature_cache.targets, dtype=np.float32).reshape(-1)
    test_index = np.asarray(split.test_indices, dtype=np.int64)
    external_index = np.asarray(split.external_indices, dtype=np.int64)
    test_pred, external_pred = fit_and_predict_graph_baseline(
        spec=spec,
        graph_cache=graph_cache,
        targets=y_vector,
        split=split,
        train_seed=train_seed,
    )
    test_payload = _predict_payload(
        sample_index=split.test_indices,
        y_true=y_vector[test_index],
        y_pred=np.asarray(test_pred, dtype=np.float64),
    )
    external_payload = _predict_payload(
        sample_index=split.external_indices,
        y_true=y_vector[external_index],
        y_pred=np.asarray(external_pred, dtype=np.float64),
    )
    return test_payload, external_payload


def _graph_cache_matches_dataset(graph_cache: GraphCache, feature_cache: FeatureCache) -> bool:
    return tuple(graph_cache.canonical_smiles) == tuple(feature_cache.canonical_smiles)


def _skipped_result_row(split: ProtocolSplit, spec: BaselineSpec, status: str, status_detail: str) -> dict[str, Any]:
    return {
        "split_id": int(split.split_id),
        "baseline_name": spec.name,
        "family": spec.family,
        "input_block": spec.input_block,
        "is_anchor": bool(False),
        "status": status,
        "status_detail": status_detail,
        "test_mae": np.nan,
        "test_rmse": np.nan,
        "test_hard_mae": np.nan,
        "external_mae": np.nan,
        "external_rmse": np.nan,
        "external_hard_mae": np.nan,
    }


def _completed_result_row(
    split: ProtocolSplit,
    spec: BaselineSpec,
    suite: BaselineSuite,
    test_payload: dict[str, Any],
    external_payload: dict[str, Any],
    hard_subset_test_lookup: dict[int, bool],
    hard_subset_external_lookup: dict[int, bool],
) -> dict[str, Any]:
    test_index = np.asarray(test_payload["sample_index"], dtype=np.int64)
    external_index = np.asarray(external_payload["sample_index"], dtype=np.int64)
    test_true = np.asarray(test_payload["y_true"], dtype=np.float64)
    test_pred = np.asarray(test_payload["pred"], dtype=np.float64)
    external_true = np.asarray(external_payload["y_true"], dtype=np.float64)
    external_pred = np.asarray(external_payload["pred"], dtype=np.float64)
    return {
        "split_id": int(split.split_id),
        "baseline_name": spec.name,
        "family": spec.family,
        "input_block": spec.input_block,
        "is_anchor": bool(spec.name == suite.anchor_baseline_name),
        "status": "completed",
        "status_detail": "completed",
        "test_mae": float(mean_absolute_error(test_true, test_pred)),
        "test_rmse": _rmse(test_true, test_pred),
        "test_hard_mae": _hard_subset_mae(test_index, test_true, test_pred, hard_subset_test_lookup),
        "external_mae": float(mean_absolute_error(external_true, external_pred)),
        "external_rmse": _rmse(external_true, external_pred),
        "external_hard_mae": _hard_subset_mae(external_index, external_true, external_pred, hard_subset_external_lookup),
    }


def _write_result_artifacts(results_frame: pd.DataFrame, target_output_dir: Path) -> None:
    results_frame.to_csv(target_output_dir / "baseline_results.csv", index=False)
    results_frame[["split_id", "baseline_name", "is_anchor", "status", "status_detail"]].to_csv(
        target_output_dir / "baseline_status.csv",
        index=False,
    )

    completed_frame = results_frame.loc[results_frame["status"] == "completed"].copy()
    if completed_frame.empty:
        summary_columns = [
            "baseline_name",
            "family",
            "input_block",
            "is_anchor",
            "test_mae_mean",
            "test_mae_std",
            "test_rmse_mean",
            "test_rmse_std",
            "test_hard_mae_mean",
            "test_hard_mae_std",
            "external_mae_mean",
            "external_mae_std",
            "external_rmse_mean",
            "external_rmse_std",
            "external_hard_mae_mean",
            "external_hard_mae_std",
        ]
        pd.DataFrame(columns=summary_columns).to_csv(target_output_dir / "baseline_summary.csv", index=False)
        return

    summary_frame = (
        completed_frame.groupby(["baseline_name", "family", "input_block", "is_anchor"], as_index=False)[
            ["test_mae", "test_rmse", "test_hard_mae", "external_mae", "external_rmse", "external_hard_mae"]
        ]
        .agg(["mean", "std"])
    )
    summary_frame.columns = ["_".join([part for part in column if part]).strip("_") for column in summary_frame.columns.to_flat_index()]
    summary_frame.to_csv(target_output_dir / "baseline_summary.csv", index=False)


def run_baseline_suite(
    *,
    dataset: pd.DataFrame,
    feature_cache: FeatureCache,
    graph_cache: GraphCache | None,
    splits: list[ProtocolSplit],
    suite: BaselineSuite,
    hard_subset_config: HardSubsetConfig,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    split_artifacts: list[dict[str, Any]] = []
    target_output_dir = Path(output_dir) if output_dir is not None else default_baseline_output_dir()
    target_output_dir.mkdir(parents=True, exist_ok=True)
    capability_records = baseline_capability_summary(suite)
    capability_lookup = {record["name"]: record for record in capability_records}
    (target_output_dir / "baseline_capabilities.json").write_text(
        json.dumps(capability_records, indent=2),
        encoding="utf-8",
    )

    if suite.anchor_baseline_name is not None:
        anchor_capability = capability_lookup[suite.anchor_baseline_name]
        if not bool(anchor_capability["available"]):
            raise RuntimeError(
                f"anchor baseline '{suite.anchor_baseline_name}' is not runnable: {anchor_capability['detail']}"
            )

    for split in splits:
        ordered_specs = _baseline_order(suite)
        hard_subset_definition = None
        hard_subset_test_lookup: dict[int, bool] = {}
        hard_subset_external_lookup: dict[int, bool] = {}
        split_payloads: dict[str, Any] = {}
        split_statuses: dict[str, dict[str, Any]] = {}

        for spec in ordered_specs:
            runtime_status = baseline_runtime_status(spec)
            split_statuses[spec.name] = dict(runtime_status)
            if not bool(runtime_status["available"]):
                if spec.name == suite.anchor_baseline_name:
                    raise RuntimeError(
                        f"anchor baseline '{spec.name}' is not runnable: {runtime_status['detail']}"
                    )
                results.append(
                    _skipped_result_row(
                        split=split,
                        spec=spec,
                        status=f"skipped_{runtime_status['status']}",
                        status_detail=str(runtime_status["detail"]),
                    )
                )
                continue

            train_seed = int(split.random_state * 101 + len(results) + 17)
            if spec.input_block == "graph":
                if graph_cache is None:
                    raise RuntimeError("graph baseline requested without a graph cache")
                test_payload, external_payload = _fit_and_predict_graph(spec, feature_cache, graph_cache, split, train_seed)
            else:
                test_payload, external_payload = _fit_and_predict(spec, feature_cache, split, train_seed)
            split_payloads[spec.name] = {
                "test": test_payload,
                "external": external_payload,
            }
            split_statuses[spec.name] = {
                "available": True,
                "status": "completed",
                "detail": "completed",
            }
            if spec.name == suite.anchor_baseline_name:
                hard_subset_definition = build_hard_subset_definition(
                    split_id=split.split_id,
                    reference_model_name=spec.name,
                    test_payload=test_payload,
                    external_payload=external_payload,
                    config=hard_subset_config,
                )
                hard_subset_test_lookup = _mask_lookup(
                    np.asarray(hard_subset_definition.test_slice.sample_index, dtype=np.int64),
                    np.asarray(hard_subset_definition.test_slice.hard_subset_mask, dtype=bool),
                )
                if hard_subset_definition.external_slice is not None:
                    hard_subset_external_lookup = _mask_lookup(
                        np.asarray(hard_subset_definition.external_slice.sample_index, dtype=np.int64),
                        np.asarray(hard_subset_definition.external_slice.hard_subset_mask, dtype=bool),
                    )

        if hard_subset_definition is None:
            raise RuntimeError("anchor baseline payload was not produced; cannot define hard subset")

        for spec in ordered_specs:
            if spec.name not in split_payloads:
                continue
            test_payload = split_payloads[spec.name]["test"]
            external_payload = split_payloads[spec.name]["external"]
            results.append(
                _completed_result_row(
                    split=split,
                    spec=spec,
                    suite=suite,
                    test_payload=test_payload,
                    external_payload=external_payload,
                    hard_subset_test_lookup=hard_subset_test_lookup,
                    hard_subset_external_lookup=hard_subset_external_lookup,
                )
            )

        artifact = {
            "split": split.to_dict(),
            "hard_subset_definition": hard_subset_definition.to_dict(),
            "baseline_status": split_statuses,
            "baseline_payloads": {
                baseline_name: {
                    "test": _payload_to_jsonable(payloads["test"]),
                    "external": _payload_to_jsonable(payloads["external"]),
                }
                for baseline_name, payloads in split_payloads.items()
            },
        }
        split_artifacts.append(artifact)
        (target_output_dir / f"split_{split.split_id:03d}_artifact.json").write_text(
            json.dumps(artifact, indent=2),
            encoding="utf-8",
        )

    results_frame = pd.DataFrame(results)
    _write_result_artifacts(results_frame, target_output_dir)
    return results_frame, split_artifacts


def run_local_baseline_workflow(
    *,
    dataset_path: str | Path | None = None,
    feature_cache_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    split_count: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    dataset = load_local_dataset(Path(dataset_path) if dataset_path is not None else None)
    try:
        feature_cache = load_feature_cache(feature_cache_path)
    except FileNotFoundError:
        feature_cache = build_feature_cache(dataset)
        save_feature_cache(feature_cache, feature_cache_path)
    if not feature_cache_matches_current_layout(feature_cache):
        feature_cache = build_feature_cache(dataset)
        save_feature_cache(feature_cache, feature_cache_path)

    suite = load_baseline_suite()
    graph_cache = None
    if any(spec.input_block == "graph" for spec in suite.baseline_specs):
        try:
            graph_cache = load_graph_cache()
        except FileNotFoundError:
            graph_cache = build_graph_cache(feature_cache.canonical_smiles)
            save_graph_cache(graph_cache)
        if not _graph_cache_matches_dataset(graph_cache, feature_cache):
            graph_cache = build_graph_cache(feature_cache.canonical_smiles)
            save_graph_cache(graph_cache)

    protocol_config = load_protocol_config()
    if split_count is not None:
        protocol_config = type(protocol_config)(**{**protocol_config.__dict__, "n_repeats": int(split_count)})
    splits = generate_protocol_splits(dataset, protocol_config)
    hard_subset_config = load_hard_subset_config()
    return run_baseline_suite(
        dataset=dataset,
        feature_cache=feature_cache,
        graph_cache=graph_cache,
        splits=splits,
        suite=suite,
        hard_subset_config=hard_subset_config,
        output_dir=output_dir,
    )


def main() -> int:
    parser = ArgumentParser(description="Run the standalone baseline suite.")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--feature-cache-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--split-count", type=int, default=0)
    args = parser.parse_args()

    run_local_baseline_workflow(
        dataset_path=args.dataset_path or None,
        feature_cache_path=args.feature_cache_path or None,
        output_dir=args.output_dir or None,
        split_count=(args.split_count or None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
