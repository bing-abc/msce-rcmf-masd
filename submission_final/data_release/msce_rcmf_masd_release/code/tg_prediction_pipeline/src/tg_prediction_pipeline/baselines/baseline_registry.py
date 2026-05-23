from __future__ import annotations

from pathlib import Path

import yaml

from tg_prediction_pipeline.baselines.backend_support import baseline_runtime_status
from tg_prediction_pipeline.schemas import BaselineSpec, BaselineSuite

_ALLOWED_FAMILIES = {
    "linear",
    "tree",
    "boosted_tree",
    "kernel",
    "neural",
    "graph",
    "transformer",
}
_ALLOWED_INPUT_BLOCKS = {"descriptor", "graph", "multimodal"}
_ALLOWED_ESTIMATORS = {
    "linear": {"ridge", "elastic_net"},
    "tree": {"random_forest"},
    "boosted_tree": {"xgboost", "lightgbm"},
    "kernel": {"svr"},
    "neural": {"mlp"},
    "graph": {"attentive_fp", "mpnn", "gcn"},
    "transformer": {"polybert"},
}
_BANNED_NAME_TOKENS = ("mspce", "current", "locked", "seed", "bundle")


def _default_baseline_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "baselines.yaml"


def _build_baseline_spec(raw_spec: dict[str, object]) -> BaselineSpec:
    return BaselineSpec(
        name=str(raw_spec["name"]),
        family=str(raw_spec["family"]),
        estimator_key=str(raw_spec["estimator_key"]),
        input_block=str(raw_spec["input_block"]),
        description=str(raw_spec.get("description", "")),
        hyperparameters=dict(raw_spec.get("hyperparameters", {})),
    )


def validate_baseline_suite(suite: BaselineSuite) -> None:
    seen_names: set[str] = set()
    for spec in suite.baseline_specs:
        if spec.name in seen_names:
            raise ValueError(f"duplicate baseline name: {spec.name}")
        seen_names.add(spec.name)
        if spec.family not in _ALLOWED_FAMILIES:
            raise ValueError(f"unsupported baseline family: {spec.family}")
        if spec.input_block not in _ALLOWED_INPUT_BLOCKS:
            raise ValueError(f"unsupported input block: {spec.input_block}")
        allowed_estimators = _ALLOWED_ESTIMATORS.get(spec.family, set())
        if spec.estimator_key not in allowed_estimators:
            raise ValueError(f"unsupported estimator '{spec.estimator_key}' for family '{spec.family}'")
        for banned_token in _BANNED_NAME_TOKENS:
            if banned_token in spec.name:
                raise ValueError(f"baseline name contains banned token '{banned_token}': {spec.name}")
        if " " in spec.name:
            raise ValueError(f"baseline name must be snake_case without spaces: {spec.name}")
    if suite.anchor_baseline_name is not None and suite.anchor_baseline_name not in seen_names:
        raise ValueError(f"anchor baseline '{suite.anchor_baseline_name}' is not defined in the suite")


def load_baseline_suite(config_path: str | Path | None = None) -> BaselineSuite:
    path = Path(config_path) if config_path is not None else _default_baseline_config_path()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_baselines = payload.get("baselines", [])
    anchor_policy = payload.get("anchor_policy", {})
    suite = BaselineSuite(
        baseline_specs=tuple(_build_baseline_spec(raw_spec) for raw_spec in raw_baselines),
        anchor_baseline_name=anchor_policy.get("anchor_baseline_name"),
        metadata={"anchor_mode": str(anchor_policy.get("mode", "fixed_name"))},
    )
    validate_baseline_suite(suite)
    return suite


def baseline_suite_summary(suite: BaselineSuite) -> list[dict[str, object]]:
    validate_baseline_suite(suite)
    return [
        {
            "name": spec.name,
            "family": spec.family,
            "estimator_key": spec.estimator_key,
            "input_block": spec.input_block,
            "is_anchor": bool(spec.name == suite.anchor_baseline_name),
        }
        for spec in suite.baseline_specs
    ]


def baseline_capability_summary(suite: BaselineSuite) -> list[dict[str, object]]:
    validate_baseline_suite(suite)
    summary: list[dict[str, object]] = []
    for spec in suite.baseline_specs:
        runtime_status = baseline_runtime_status(spec)
        summary.append(
            {
                "name": spec.name,
                "family": spec.family,
                "estimator_key": spec.estimator_key,
                "input_block": spec.input_block,
                "is_anchor": bool(spec.name == suite.anchor_baseline_name),
                "available": bool(runtime_status["available"]),
                "status": str(runtime_status["status"]),
                "detail": str(runtime_status["detail"]),
            }
        )
    return summary
