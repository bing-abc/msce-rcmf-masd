from __future__ import annotations

import json
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from tg_prediction_pipeline.baselines import load_baseline_suite
from tg_prediction_pipeline.features import build_feature_cache, feature_cache_matches_current_layout, load_feature_cache, save_feature_cache
from tg_prediction_pipeline.models import MASDConfig, MASDRegressor, MSCERegressor, RCMFRegressor
from tg_prediction_pipeline.pipeline.run_msce_benchmark import load_msce_config
from tg_prediction_pipeline.pipeline.run_rcmf_benchmark import (
    _anchor_spec,
    _fit_anchor,
    _hard_subset_mae,
    _mask_lookup,
    _predict_payload,
    _result_row,
    _select_rows,
    _select_targets,
    _stage_diagnostics,
    load_rcmf_config,
)
from tg_prediction_pipeline.protocol.dataset_protocol import load_local_dataset, summarize_dataset
from tg_prediction_pipeline.protocol.hard_subset_protocol import (
    build_hard_subset_definition,
    build_hard_subset_slice,
    load_hard_subset_config,
)
from tg_prediction_pipeline.protocol.split_protocol import export_protocol_splits, generate_protocol_splits, load_protocol_config


def default_masd_output_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "artifacts" / "masd_benchmark"


def _default_masd_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "masd.yaml"


def load_masd_config(config_path: str | Path | None = None) -> MASDConfig:
    path = Path(config_path) if config_path is not None else _default_masd_config_path()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    stage_payload = payload.get("masd_stage", {})
    loss_payload = stage_payload.get("loss", {})
    return MASDConfig(
        slot_count=int(stage_payload.get("slot_count", 4)),
        hidden_dim=int(stage_payload.get("hidden_dim", 128)),
        slot_hidden_dim=int(stage_payload.get("slot_hidden_dim", 64)),
        dropout=float(stage_payload.get("dropout", 0.10)),
        learning_rate=float(stage_payload.get("learning_rate", 0.001)),
        weight_decay=float(stage_payload.get("weight_decay", 1e-5)),
        batch_size=int(stage_payload.get("batch_size", 128)),
        max_epochs=int(stage_payload.get("max_epochs", 120)),
        patience=int(stage_payload.get("patience", 16)),
        late_epoch_candidate_count=int(stage_payload.get("late_epoch_candidate_count", 6)),
        late_epoch_top_soup_count=int(stage_payload.get("late_epoch_top_soup_count", 4)),
        tail_focus_epochs=int(stage_payload.get("tail_focus_epochs", 4)),
        tail_focus_lr_scale=float(stage_payload.get("tail_focus_lr_scale", 0.35)),
        tail_focus_weight_scale=float(stage_payload.get("tail_focus_weight_scale", 2.0)),
        enable_split_b=bool(stage_payload.get("enable_split_b", False)),
        split_b_fraction=float(stage_payload.get("split_b_fraction", 0.20)),
        split_b_min_size=int(stage_payload.get("split_b_min_size", 32)),
        split_b_head_epochs=int(stage_payload.get("split_b_head_epochs", 6)),
        split_b_patience=int(stage_payload.get("split_b_patience", 2)),
        split_b_lr_scale=float(stage_payload.get("split_b_lr_scale", 0.12)),
        split_b_weight_decay_scale=float(stage_payload.get("split_b_weight_decay_scale", 3.0)),
        split_b_primary_epsilon=float(stage_payload.get("split_b_primary_epsilon", 0.02)),
        blend_grid_size=int(stage_payload.get("blend_grid_size", 21)),
        blend_alpha_max=float(stage_payload.get("blend_alpha_max", 3.00)),
        delta_bound_k=float(stage_payload.get("delta_bound_k", 20.0)),
        slot_min_magnitude_k=float(stage_payload.get("slot_min_magnitude_k", 0.5)),
        slot_max_magnitude_k=float(stage_payload.get("slot_max_magnitude_k", 10.0)),
        residual_cap_k=float(stage_payload.get("residual_cap_k", 3.0)),
        alpha_temperature=float(stage_payload.get("alpha_temperature", 0.70)),
        gate_low=float(stage_payload.get("gate_low", 0.02)),
        gate_high=float(stage_payload.get("gate_high", 0.55)),
        hard_threshold_tau=float(stage_payload.get("hard_threshold_tau", 0.70)),
        hard_threshold_gamma=float(stage_payload.get("hard_threshold_gamma", 8.0)),
        calibration_tau_candidates=tuple(float(value) for value in stage_payload.get("calibration_tau_candidates", [0.55, 0.60, 0.65, 0.70])),
        calibration_gate_floor_candidates=tuple(float(value) for value in stage_payload.get("calibration_gate_floor_candidates", [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0])),
        calibration_focus_gain_candidates=tuple(float(value) for value in stage_payload.get("calibration_focus_gain_candidates", [1.0, 2.0, 4.0, 6.0, 8.0, 12.0])),
        calibration_cap_candidates=tuple(float(value) for value in stage_payload.get("calibration_cap_candidates", [2.0, 4.0, 6.0, 8.0, 12.0])),
        activation_floor=float(stage_payload.get("activation_floor", 0.05)),
        activation_ceiling=float(stage_payload.get("activation_ceiling", 0.45)),
        activation_delta_threshold_k=float(stage_payload.get("activation_delta_threshold_k", 0.25)),
        anchor_margin_weight=float(loss_payload.get("anchor_margin_weight", 0.84)),
        proxy_alignment_weight=float(loss_payload.get("proxy_alignment_weight", 0.14)),
        sign_weight=float(loss_payload.get("sign_weight", 0.22)),
        sparse_weight=float(loss_payload.get("sparse_weight", 0.10)),
        calibrator_weight=float(loss_payload.get("calibrator_weight", 0.10)),
        diversity_weight=float(loss_payload.get("diversity_weight", 0.10)),
        gate_high_penalty_weight=float(loss_payload.get("gate_high_penalty_weight", 0.14)),
        gate_low_penalty_weight=float(loss_payload.get("gate_low_penalty_weight", 0.05)),
        delta_weight=float(loss_payload.get("delta_weight", 0.04)),
        hard_focus_weight=float(loss_payload.get("hard_focus_weight", 0.08)),
        residual_target_weight=float(loss_payload.get("residual_target_weight", 0.18)),
    )


def _masd_stage_diagnostics(details: dict[str, np.ndarray]) -> dict[str, float | str]:
    diagnostics: dict[str, float | str] = dict(_stage_diagnostics(details, "masd"))
    diagnostics["masd_aggregation_name"] = str(np.asarray(details["aggregation_name"], dtype=object).reshape(-1)[0])
    diagnostics["masd_gate_mean"] = float(np.asarray(details["gate"], dtype=np.float64).mean())
    diagnostics["masd_thresholded_gate_mean"] = float(np.asarray(details["thresholded_gate"], dtype=np.float64).mean())
    diagnostics["masd_hard_proxy_mean"] = float(np.asarray(details["hard_proxy"], dtype=np.float64).mean())
    diagnostics["masd_calibrated_tau_mean"] = float(np.asarray(details["calibrated_tau"], dtype=np.float64).mean())
    diagnostics["masd_gate_floor_mean"] = float(np.asarray(details["gate_floor"], dtype=np.float64).mean())
    diagnostics["masd_focus_gain_mean"] = float(np.asarray(details["focus_gain"], dtype=np.float64).mean())
    diagnostics["masd_cap_k_mean"] = float(np.asarray(details["cap_k"], dtype=np.float64).mean())
    diagnostics["masd_activation_rate"] = float(np.asarray(details["activation_flag"], dtype=np.float64).mean())
    diagnostics["masd_split_b_fraction"] = float(np.asarray(details["split_b_fraction"], dtype=np.float64).mean())
    diagnostics["masd_entropy_mean"] = float(np.asarray(details["entropy"], dtype=np.float64).mean())
    diagnostics["masd_alpha_max_mean"] = float(np.asarray(details["alpha_max"], dtype=np.float64).mean())
    diagnostics["masd_alpha_margin_mean"] = float(np.asarray(details["alpha_margin"], dtype=np.float64).mean())
    diagnostics["masd_mechanism_disagreement_mean"] = float(np.asarray(details["mechanism_disagreement"], dtype=np.float64).mean())
    alpha = np.asarray(details["alpha"], dtype=np.float64)
    for slot_index in range(alpha.shape[1]):
        diagnostics[f"masd_slot_weight_{slot_index}_mean"] = float(alpha[:, slot_index].mean())
    return diagnostics


def _write_summary(results_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    results_frame.to_csv(output_dir / "masd_results.csv", index=False)
    metric_columns = [
        column
        for column in results_frame.columns
        if column not in {"split_id", "stage_name", "model_name", "status", "status_detail"}
        and pd.api.types.is_numeric_dtype(results_frame[column])
    ]
    summary_frame = results_frame.groupby(["stage_name", "model_name"], as_index=False)[metric_columns].agg(["mean", "std"])
    summary_frame.columns = ["_".join([part for part in column if part]).strip("_") for column in summary_frame.columns.to_flat_index()]
    summary_frame.to_csv(output_dir / "masd_summary.csv", index=False)
    return summary_frame


def run_masd_benchmark(
    *,
    dataset_path: str | Path | None = None,
    feature_cache_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    protocol_config_path: str | Path | None = None,
    baseline_config_path: str | Path | None = None,
    msce_config_path: str | Path | None = None,
    rcmf_config_path: str | Path | None = None,
    masd_config_path: str | Path | None = None,
    repeats: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_local_dataset(Path(dataset_path) if dataset_path is not None else None)
    feature_cache = load_feature_cache(feature_cache_path)
    if not feature_cache_matches_current_layout(feature_cache):
        feature_cache = build_feature_cache(dataset)
        save_feature_cache(feature_cache, feature_cache_path)
    output_root = Path(output_dir) if output_dir is not None else default_masd_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_config = load_protocol_config(protocol_config_path)
    if repeats is not None:
        protocol_config = type(protocol_config)(**{**protocol_config.__dict__, "n_repeats": int(repeats)})
    splits = generate_protocol_splits(dataset, protocol_config)
    export_protocol_splits(splits, output_root / "protocol_splits.json")

    suite = load_baseline_suite(baseline_config_path)
    msce_config = load_msce_config(msce_config_path)
    rcmf_config = load_rcmf_config(rcmf_config_path)
    masd_config = load_masd_config(masd_config_path)
    anchor_spec = _anchor_spec(suite, msce_config.anchor_baseline_name)
    hard_subset_config = load_hard_subset_config(protocol_config_path)

    descriptor_matrix = np.asarray(feature_cache.descriptors, dtype=np.float32)
    context_matrix = np.asarray(feature_cache.contexts, dtype=np.float32)
    targets = np.asarray(feature_cache.targets, dtype=np.float32).reshape(-1)

    rows: list[dict[str, object]] = []
    for split in splits:
        anchor_predictions = _fit_anchor(
            descriptor_matrix=descriptor_matrix,
            targets=targets,
            split=split,
            anchor_spec=anchor_spec,
            train_seed=int(split.random_state * 101 + 17),
        )
        anchor_test_payload = _predict_payload(split.test_indices, _select_targets(targets, split.test_indices), anchor_predictions["test"])
        anchor_external_payload = _predict_payload(split.external_indices, _select_targets(targets, split.external_indices), anchor_predictions["external"])
        anchor_train_payload = _predict_payload(split.train_indices, _select_targets(targets, split.train_indices), anchor_predictions["train"])
        anchor_val_payload = _predict_payload(split.val_indices, _select_targets(targets, split.val_indices), anchor_predictions["val"])
        hard_subset_definition = build_hard_subset_definition(
            split_id=split.split_id,
            reference_model_name=msce_config.anchor_baseline_name,
            test_payload=anchor_test_payload,
            external_payload=anchor_external_payload,
            config=hard_subset_config,
        )
        train_hard_slice = build_hard_subset_slice(anchor_train_payload, hard_subset_config)
        val_hard_slice = build_hard_subset_slice(anchor_val_payload, hard_subset_config)
        hard_subset_test_lookup = _mask_lookup(
            np.asarray(hard_subset_definition.test_slice.sample_index, dtype=np.int64),
            np.asarray(hard_subset_definition.test_slice.hard_subset_mask, dtype=bool),
        )
        hard_subset_external_lookup = _mask_lookup(
            np.asarray(hard_subset_definition.external_slice.sample_index, dtype=np.int64),
            np.asarray(hard_subset_definition.external_slice.hard_subset_mask, dtype=bool),
        )

        msce_regressor = MSCERegressor(msce_config, train_seed=int(split.random_state * 101 + 43))
        msce_regressor.fit(
            descriptor_train=_select_rows(descriptor_matrix, split.train_indices),
            context_train=_select_rows(context_matrix, split.train_indices),
            anchor_train=anchor_predictions["train"],
            y_train=_select_targets(targets, split.train_indices),
            descriptor_val=_select_rows(descriptor_matrix, split.val_indices),
            context_val=_select_rows(context_matrix, split.val_indices),
            anchor_val=anchor_predictions["val"],
            y_val=_select_targets(targets, split.val_indices),
        )
        msce_train_details = msce_regressor.predict_details(_select_rows(descriptor_matrix, split.train_indices), _select_rows(context_matrix, split.train_indices), anchor_predictions["train"])
        msce_val_details = msce_regressor.predict_details(_select_rows(descriptor_matrix, split.val_indices), _select_rows(context_matrix, split.val_indices), anchor_predictions["val"])
        msce_test_details = msce_regressor.predict_details(_select_rows(descriptor_matrix, split.test_indices), _select_rows(context_matrix, split.test_indices), anchor_predictions["test"])
        msce_external_details = msce_regressor.predict_details(_select_rows(descriptor_matrix, split.external_indices), _select_rows(context_matrix, split.external_indices), anchor_predictions["external"])

        rcmf_regressor = RCMFRegressor(rcmf_config, train_seed=int(split.random_state * 101 + 71))
        rcmf_regressor.fit(
            descriptor_train=_select_rows(descriptor_matrix, split.train_indices),
            context_train=_select_rows(context_matrix, split.train_indices),
            msce_train_details=msce_train_details,
            y_train=_select_targets(targets, split.train_indices),
            descriptor_val=_select_rows(descriptor_matrix, split.val_indices),
            context_val=_select_rows(context_matrix, split.val_indices),
            msce_val_details=msce_val_details,
            y_val=_select_targets(targets, split.val_indices),
        )
        rcmf_test_details = rcmf_regressor.predict_details(
            descriptor_matrix=_select_rows(descriptor_matrix, split.test_indices),
            context_matrix=_select_rows(context_matrix, split.test_indices),
            msce_details=msce_test_details,
        )
        rcmf_external_details = rcmf_regressor.predict_details(
            descriptor_matrix=_select_rows(descriptor_matrix, split.external_indices),
            context_matrix=_select_rows(context_matrix, split.external_indices),
            msce_details=msce_external_details,
        )

        masd_regressor = MASDRegressor(masd_config, train_seed=int(split.random_state * 101 + 97))
        masd_regressor.fit(
            descriptor_train=_select_rows(descriptor_matrix, split.train_indices),
            context_train=_select_rows(context_matrix, split.train_indices),
            msce_train_details=msce_train_details,
            rcmf_train_details=rcmf_regressor.predict_details(
                descriptor_matrix=_select_rows(descriptor_matrix, split.train_indices),
                context_matrix=_select_rows(context_matrix, split.train_indices),
                msce_details=msce_train_details,
            ),
            y_train=_select_targets(targets, split.train_indices),
            descriptor_val=_select_rows(descriptor_matrix, split.val_indices),
            context_val=_select_rows(context_matrix, split.val_indices),
            msce_val_details=msce_val_details,
            rcmf_val_details=rcmf_regressor.predict_details(
                descriptor_matrix=_select_rows(descriptor_matrix, split.val_indices),
                context_matrix=_select_rows(context_matrix, split.val_indices),
                msce_details=msce_val_details,
            ),
            y_val=_select_targets(targets, split.val_indices),
            benchmark_hard_train_mask=np.asarray(train_hard_slice.hard_subset_mask, dtype=bool),
            benchmark_hard_val_mask=np.asarray(val_hard_slice.hard_subset_mask, dtype=bool),
        )
        masd_test_details = masd_regressor.predict_details(
            descriptor_matrix=_select_rows(descriptor_matrix, split.test_indices),
            context_matrix=_select_rows(context_matrix, split.test_indices),
            msce_details=msce_test_details,
            rcmf_details=rcmf_test_details,
        )
        masd_external_details = masd_regressor.predict_details(
            descriptor_matrix=_select_rows(descriptor_matrix, split.external_indices),
            context_matrix=_select_rows(context_matrix, split.external_indices),
            msce_details=msce_external_details,
            rcmf_details=rcmf_external_details,
        )

        msce_test_payload = _predict_payload(split.test_indices, _select_targets(targets, split.test_indices), msce_test_details["pred"])
        msce_external_payload = _predict_payload(split.external_indices, _select_targets(targets, split.external_indices), msce_external_details["pred"])
        rcmf_test_payload = _predict_payload(split.test_indices, _select_targets(targets, split.test_indices), rcmf_test_details["pred"])
        rcmf_external_payload = _predict_payload(split.external_indices, _select_targets(targets, split.external_indices), rcmf_external_details["pred"])
        masd_test_payload = _predict_payload(split.test_indices, _select_targets(targets, split.test_indices), masd_test_details["pred"])
        masd_external_payload = _predict_payload(split.external_indices, _select_targets(targets, split.external_indices), masd_external_details["pred"])

        anchor_test_mae = float(mean_absolute_error(anchor_test_payload["y_true"], anchor_test_payload["pred"]))
        anchor_test_hard_mae = _hard_subset_mae(np.asarray(anchor_test_payload["sample_index"], dtype=np.int64), np.asarray(anchor_test_payload["y_true"], dtype=np.float64), np.asarray(anchor_test_payload["pred"], dtype=np.float64), hard_subset_test_lookup)
        anchor_external_mae = float(mean_absolute_error(anchor_external_payload["y_true"], anchor_external_payload["pred"]))
        anchor_external_hard_mae = _hard_subset_mae(np.asarray(anchor_external_payload["sample_index"], dtype=np.int64), np.asarray(anchor_external_payload["y_true"], dtype=np.float64), np.asarray(anchor_external_payload["pred"], dtype=np.float64), hard_subset_external_lookup)
        msce_test_mae = float(mean_absolute_error(msce_test_payload["y_true"], msce_test_payload["pred"]))
        msce_test_hard_mae = _hard_subset_mae(np.asarray(msce_test_payload["sample_index"], dtype=np.int64), np.asarray(msce_test_payload["y_true"], dtype=np.float64), np.asarray(msce_test_payload["pred"], dtype=np.float64), hard_subset_test_lookup)
        msce_external_mae = float(mean_absolute_error(msce_external_payload["y_true"], msce_external_payload["pred"]))
        msce_external_hard_mae = _hard_subset_mae(np.asarray(msce_external_payload["sample_index"], dtype=np.int64), np.asarray(msce_external_payload["y_true"], dtype=np.float64), np.asarray(msce_external_payload["pred"], dtype=np.float64), hard_subset_external_lookup)
        rcmf_test_mae = float(mean_absolute_error(rcmf_test_payload["y_true"], rcmf_test_payload["pred"]))
        rcmf_test_hard_mae = _hard_subset_mae(np.asarray(rcmf_test_payload["sample_index"], dtype=np.int64), np.asarray(rcmf_test_payload["y_true"], dtype=np.float64), np.asarray(rcmf_test_payload["pred"], dtype=np.float64), hard_subset_test_lookup)
        rcmf_external_mae = float(mean_absolute_error(rcmf_external_payload["y_true"], rcmf_external_payload["pred"]))
        rcmf_external_hard_mae = _hard_subset_mae(np.asarray(rcmf_external_payload["sample_index"], dtype=np.int64), np.asarray(rcmf_external_payload["y_true"], dtype=np.float64), np.asarray(rcmf_external_payload["pred"], dtype=np.float64), hard_subset_external_lookup)
        masd_test_mae = float(mean_absolute_error(masd_test_payload["y_true"], masd_test_payload["pred"]))
        masd_test_hard_mae = _hard_subset_mae(np.asarray(masd_test_payload["sample_index"], dtype=np.int64), np.asarray(masd_test_payload["y_true"], dtype=np.float64), np.asarray(masd_test_payload["pred"], dtype=np.float64), hard_subset_test_lookup)
        masd_external_mae = float(mean_absolute_error(masd_external_payload["y_true"], masd_external_payload["pred"]))
        masd_external_hard_mae = _hard_subset_mae(np.asarray(masd_external_payload["sample_index"], dtype=np.int64), np.asarray(masd_external_payload["y_true"], dtype=np.float64), np.asarray(masd_external_payload["pred"], dtype=np.float64), hard_subset_external_lookup)

        rows.append(
            _result_row(
                split=split,
                stage_name="anchor_baseline",
                model_name=msce_config.anchor_baseline_name,
                test_payload=anchor_test_payload,
                external_payload=anchor_external_payload,
                hard_subset_test_lookup=hard_subset_test_lookup,
                hard_subset_external_lookup=hard_subset_external_lookup,
                delta_vs_anchor_test_mae=0.0,
                delta_vs_anchor_test_hard_mae=0.0,
                delta_vs_anchor_external_mae=0.0,
                delta_vs_anchor_external_hard_mae=0.0,
                delta_vs_previous_test_mae=0.0,
                delta_vs_previous_test_hard_mae=0.0,
                delta_vs_previous_external_mae=0.0,
                delta_vs_previous_external_hard_mae=0.0,
                diagnostics=None,
            )
        )
        rows.append(
            _result_row(
                split=split,
                stage_name="msce_stage",
                model_name="msce_stage",
                test_payload=msce_test_payload,
                external_payload=msce_external_payload,
                hard_subset_test_lookup=hard_subset_test_lookup,
                hard_subset_external_lookup=hard_subset_external_lookup,
                delta_vs_anchor_test_mae=float(msce_test_mae - anchor_test_mae),
                delta_vs_anchor_test_hard_mae=float(msce_test_hard_mae - anchor_test_hard_mae),
                delta_vs_anchor_external_mae=float(msce_external_mae - anchor_external_mae),
                delta_vs_anchor_external_hard_mae=float(msce_external_hard_mae - anchor_external_hard_mae),
                delta_vs_previous_test_mae=float(msce_test_mae - anchor_test_mae),
                delta_vs_previous_test_hard_mae=float(msce_test_hard_mae - anchor_test_hard_mae),
                delta_vs_previous_external_mae=float(msce_external_mae - anchor_external_mae),
                delta_vs_previous_external_hard_mae=float(msce_external_hard_mae - anchor_external_hard_mae),
                diagnostics=_stage_diagnostics(msce_test_details, "msce"),
            )
        )
        rows.append(
            _result_row(
                split=split,
                stage_name="rcmf_stage",
                model_name="msce_plus_rcmf",
                test_payload=rcmf_test_payload,
                external_payload=rcmf_external_payload,
                hard_subset_test_lookup=hard_subset_test_lookup,
                hard_subset_external_lookup=hard_subset_external_lookup,
                delta_vs_anchor_test_mae=float(rcmf_test_mae - anchor_test_mae),
                delta_vs_anchor_test_hard_mae=float(rcmf_test_hard_mae - anchor_test_hard_mae),
                delta_vs_anchor_external_mae=float(rcmf_external_mae - anchor_external_mae),
                delta_vs_anchor_external_hard_mae=float(rcmf_external_hard_mae - anchor_external_hard_mae),
                delta_vs_previous_test_mae=float(rcmf_test_mae - msce_test_mae),
                delta_vs_previous_test_hard_mae=float(rcmf_test_hard_mae - msce_test_hard_mae),
                delta_vs_previous_external_mae=float(rcmf_external_mae - msce_external_mae),
                delta_vs_previous_external_hard_mae=float(rcmf_external_hard_mae - msce_external_hard_mae),
                diagnostics=_stage_diagnostics(rcmf_test_details, "rcmf"),
            )
        )
        rows.append(
            _result_row(
                split=split,
                stage_name="masd_stage",
                model_name="msce_plus_rcmf_plus_masd",
                test_payload=masd_test_payload,
                external_payload=masd_external_payload,
                hard_subset_test_lookup=hard_subset_test_lookup,
                hard_subset_external_lookup=hard_subset_external_lookup,
                delta_vs_anchor_test_mae=float(masd_test_mae - anchor_test_mae),
                delta_vs_anchor_test_hard_mae=float(masd_test_hard_mae - anchor_test_hard_mae),
                delta_vs_anchor_external_mae=float(masd_external_mae - anchor_external_mae),
                delta_vs_anchor_external_hard_mae=float(masd_external_hard_mae - anchor_external_hard_mae),
                delta_vs_previous_test_mae=float(masd_test_mae - rcmf_test_mae),
                delta_vs_previous_test_hard_mae=float(masd_test_hard_mae - rcmf_test_hard_mae),
                delta_vs_previous_external_mae=float(masd_external_mae - rcmf_external_mae),
                delta_vs_previous_external_hard_mae=float(masd_external_hard_mae - rcmf_external_hard_mae),
                diagnostics=_masd_stage_diagnostics(masd_test_details),
            )
        )

    results_frame = pd.DataFrame(rows)
    summary_frame = _write_summary(results_frame, output_root)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path is not None else "local_processed_dataset",
        "output_dir": str(output_root.resolve()),
        "protocol": dict(protocol_config.__dict__),
        "msce_config": dict(msce_config.__dict__),
        "rcmf_config": dict(rcmf_config.__dict__),
        "masd_config": dict(masd_config.__dict__),
        "dataset_summary": summarize_dataset(dataset).to_dict(orient="records"),
        "artifacts": {
            "results_csv": "masd_results.csv",
            "summary_csv": "masd_summary.csv",
            "splits_json": "protocol_splits.json",
            "manifest_json": "masd_manifest.json",
        },
    }
    (output_root / "masd_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results_frame, summary_frame


def main() -> int:
    parser = ArgumentParser(description="Run the standalone MASD stage benchmark.")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--feature-cache-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--protocol-config-path", type=str, default="")
    parser.add_argument("--baseline-config-path", type=str, default="")
    parser.add_argument("--msce-config-path", type=str, default="")
    parser.add_argument("--rcmf-config-path", type=str, default="")
    parser.add_argument("--masd-config-path", type=str, default="")
    parser.add_argument("--repeats", type=int, default=0)
    args = parser.parse_args()
    run_masd_benchmark(
        dataset_path=args.dataset_path or None,
        feature_cache_path=args.feature_cache_path or None,
        output_dir=args.output_dir or None,
        protocol_config_path=args.protocol_config_path or None,
        baseline_config_path=args.baseline_config_path or None,
        msce_config_path=args.msce_config_path or None,
        rcmf_config_path=args.rcmf_config_path or None,
        masd_config_path=args.masd_config_path or None,
        repeats=(args.repeats or None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
