from __future__ import annotations

import json
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tg_prediction_pipeline.baselines import baseline_capability_summary, load_baseline_suite
from tg_prediction_pipeline.features import (
    build_feature_cache,
    build_graph_cache,
    feature_cache_matches_current_layout,
    load_feature_cache,
    load_graph_cache,
    save_feature_cache,
    save_graph_cache,
)
from tg_prediction_pipeline.pipeline.train_baseline import run_baseline_suite
from tg_prediction_pipeline.protocol.dataset_protocol import load_local_dataset, summarize_dataset
from tg_prediction_pipeline.protocol.hard_subset_protocol import load_hard_subset_config
from tg_prediction_pipeline.protocol.split_protocol import export_protocol_splits, generate_protocol_splits, load_protocol_config
from tg_prediction_pipeline.schemas import FeatureCache, GraphCache


def default_benchmark_output_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "artifacts" / "baseline_benchmark"


def _load_or_build_feature_cache(dataset: pd.DataFrame, feature_cache_path: str | Path | None) -> FeatureCache:
    try:
        feature_cache = load_feature_cache(feature_cache_path)
    except FileNotFoundError:
        feature_cache = build_feature_cache(dataset)
        save_feature_cache(feature_cache, feature_cache_path)
        return feature_cache
    if not feature_cache_matches_current_layout(feature_cache):
        feature_cache = build_feature_cache(dataset)
        save_feature_cache(feature_cache, feature_cache_path)
    return feature_cache


def _graph_cache_matches_feature_cache(graph_cache: GraphCache, feature_cache: FeatureCache) -> bool:
    return tuple(graph_cache.canonical_smiles) == tuple(feature_cache.canonical_smiles)


def _load_or_build_graph_cache(
    feature_cache: FeatureCache,
    graph_cache_path: str | Path | None,
    require_graph_cache: bool,
) -> GraphCache | None:
    if not require_graph_cache:
        return None
    try:
        graph_cache = load_graph_cache(graph_cache_path)
    except FileNotFoundError:
        graph_cache = build_graph_cache(feature_cache.canonical_smiles)
        save_graph_cache(graph_cache, graph_cache_path)
        return graph_cache

    if not _graph_cache_matches_feature_cache(graph_cache, feature_cache):
        graph_cache = build_graph_cache(feature_cache.canonical_smiles)
        save_graph_cache(graph_cache, graph_cache_path)
    return graph_cache


def _ranked_summary(results_frame: pd.DataFrame) -> pd.DataFrame:
    completed = results_frame.loc[results_frame["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(
            columns=[
                "baseline_name",
                "family",
                "input_block",
                "is_anchor",
                "n_completed_splits",
                "test_mae_mean",
                "test_mae_std",
                "external_mae_mean",
                "external_mae_std",
                "test_hard_mae_mean",
                "test_hard_mae_std",
                "external_hard_mae_mean",
                "external_hard_mae_std",
                "rank_test_mae",
                "rank_external_mae",
                "rank_test_hard_mae",
                "rank_external_hard_mae",
            ]
        )

    summary = (
        completed.groupby(["baseline_name", "family", "input_block", "is_anchor"], as_index=False)
        .agg(
            n_completed_splits=("split_id", "count"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            external_mae_mean=("external_mae", "mean"),
            external_mae_std=("external_mae", "std"),
            test_hard_mae_mean=("test_hard_mae", "mean"),
            test_hard_mae_std=("test_hard_mae", "std"),
            external_hard_mae_mean=("external_hard_mae", "mean"),
            external_hard_mae_std=("external_hard_mae", "std"),
        )
        .sort_values(["external_mae_mean", "test_mae_mean", "baseline_name"], ignore_index=True)
    )
    summary["rank_test_mae"] = summary["test_mae_mean"].rank(method="min", ascending=True).astype(int)
    summary["rank_external_mae"] = summary["external_mae_mean"].rank(method="min", ascending=True).astype(int)
    summary["rank_test_hard_mae"] = summary["test_hard_mae_mean"].rank(method="min", ascending=True).astype(int)
    summary["rank_external_hard_mae"] = summary["external_hard_mae_mean"].rank(method="min", ascending=True).astype(int)
    return summary


def _manifest_payload(
    *,
    dataset_path: str | Path | None,
    output_dir: Path,
    protocol_config: Any,
    hard_subset_config: Any,
    suite: Any,
    capability_rows: list[dict[str, Any]],
    dataset_summary: pd.DataFrame,
    ranked_summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path is not None else "local_processed_dataset",
        "output_dir": str(output_dir.resolve()),
        "protocol": dict(protocol_config.__dict__),
        "hard_subset_protocol": dict(hard_subset_config.__dict__),
        "baseline_suite": suite.to_dict(),
        "baseline_capabilities": capability_rows,
        "dataset_summary": dataset_summary.to_dict(orient="records"),
        "ranked_baselines": ranked_summary[
            [
                "baseline_name",
                "family",
                "input_block",
                "is_anchor",
                "n_completed_splits",
                "test_mae_mean",
                "external_mae_mean",
                "rank_test_mae",
                "rank_external_mae",
            ]
        ].to_dict(orient="records"),
        "artifacts": {
            "results_csv": "baseline_results.csv",
            "summary_csv": "baseline_summary.csv",
            "status_csv": "baseline_status.csv",
            "capabilities_json": "baseline_capabilities.json",
            "ranked_summary_csv": "baseline_ranked_summary.csv",
            "manifest_json": "benchmark_manifest.json",
            "splits_json": "protocol_splits.json",
        },
    }


def run_baseline_benchmark(
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    feature_cache_path: str | Path | None = None,
    graph_cache_path: str | Path | None = None,
    protocol_config_path: str | Path | None = None,
    baseline_config_path: str | Path | None = None,
    repeats: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_local_dataset(Path(dataset_path) if dataset_path is not None else None)
    output_root = Path(output_dir) if output_dir is not None else default_benchmark_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    feature_cache = _load_or_build_feature_cache(dataset, feature_cache_path)
    suite = load_baseline_suite(baseline_config_path)
    require_graph_cache = any(spec.input_block == "graph" for spec in suite.baseline_specs)
    graph_cache = _load_or_build_graph_cache(feature_cache, graph_cache_path, require_graph_cache=require_graph_cache)

    protocol_config = load_protocol_config(protocol_config_path)
    if repeats is not None:
        protocol_config = type(protocol_config)(**{**protocol_config.__dict__, "n_repeats": int(repeats)})
    splits = generate_protocol_splits(dataset, protocol_config)
    export_protocol_splits(splits, output_root / "protocol_splits.json")

    hard_subset_config = load_hard_subset_config(protocol_config_path)
    results_frame, _ = run_baseline_suite(
        dataset=dataset,
        feature_cache=feature_cache,
        graph_cache=graph_cache,
        splits=splits,
        suite=suite,
        hard_subset_config=hard_subset_config,
        output_dir=output_root,
    )
    ranked_summary = _ranked_summary(results_frame)
    ranked_summary.to_csv(output_root / "baseline_ranked_summary.csv", index=False)

    capability_rows = baseline_capability_summary(suite)
    dataset_summary = summarize_dataset(dataset)
    manifest = _manifest_payload(
        dataset_path=dataset_path,
        output_dir=output_root,
        protocol_config=protocol_config,
        hard_subset_config=hard_subset_config,
        suite=suite,
        capability_rows=capability_rows,
        dataset_summary=dataset_summary,
        ranked_summary=ranked_summary,
    )
    (output_root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset_summary.to_csv(output_root / "dataset_summary.csv", index=False)
    return results_frame, ranked_summary


def main() -> int:
    parser = ArgumentParser(description="Run the full repeated-split baseline benchmark.")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--feature-cache-path", type=str, default="")
    parser.add_argument("--graph-cache-path", type=str, default="")
    parser.add_argument("--protocol-config-path", type=str, default="")
    parser.add_argument("--baseline-config-path", type=str, default="")
    parser.add_argument("--repeats", type=int, default=0)
    args = parser.parse_args()

    run_baseline_benchmark(
        dataset_path=args.dataset_path or None,
        output_dir=args.output_dir or None,
        feature_cache_path=args.feature_cache_path or None,
        graph_cache_path=args.graph_cache_path or None,
        protocol_config_path=args.protocol_config_path or None,
        baseline_config_path=args.baseline_config_path or None,
        repeats=(args.repeats or None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
