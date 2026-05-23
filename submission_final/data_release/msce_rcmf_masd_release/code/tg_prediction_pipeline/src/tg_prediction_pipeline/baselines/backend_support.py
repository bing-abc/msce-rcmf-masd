from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from tg_prediction_pipeline.schemas import BaselineSpec

_OPTIONAL_BACKEND_MODULES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}
_GRAPH_BACKEND_MODULES = ("torch", "torch_geometric")
_SUPPORTED_INPUT_BLOCKS = {"descriptor", "multimodal", "graph"}
_SUPPORTED_MATRIX_ESTIMATORS = {
    "ridge",
    "elastic_net",
    "random_forest",
    "svr",
    "mlp",
    "xgboost",
    "lightgbm",
}
_SUPPORTED_GRAPH_ESTIMATORS = {"attentive_fp"}


def _wrapped_regressor(regressor: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def baseline_runtime_status(spec: BaselineSpec) -> dict[str, Any]:
    if spec.input_block not in _SUPPORTED_INPUT_BLOCKS:
        return {
            "available": False,
            "status": "unsupported_input_block",
            "detail": f"input_block '{spec.input_block}' is not implemented in the standalone baseline trainer",
        }

    if spec.input_block == "graph":
        if spec.estimator_key not in _SUPPORTED_GRAPH_ESTIMATORS:
            return {
                "available": False,
                "status": "unsupported_estimator_backend",
                "detail": f"graph estimator '{spec.estimator_key}' is not implemented in the standalone baseline trainer",
            }
        for module_name in _GRAPH_BACKEND_MODULES:
            if find_spec(module_name) is None:
                return {
                    "available": False,
                    "status": "missing_dependency",
                    "detail": f"optional dependency '{module_name}' is not installed",
                }
        return {
            "available": True,
            "status": "ready",
            "detail": "ready",
        }

    if spec.estimator_key not in _SUPPORTED_MATRIX_ESTIMATORS:
        return {
            "available": False,
            "status": "unsupported_estimator_backend",
            "detail": f"estimator '{spec.estimator_key}' is not implemented in the standalone baseline trainer",
        }

    if spec.estimator_key in _OPTIONAL_BACKEND_MODULES:
        module_name = _OPTIONAL_BACKEND_MODULES[spec.estimator_key]
        if find_spec(module_name) is None:
            return {
                "available": False,
                "status": "missing_dependency",
                "detail": f"optional dependency '{module_name}' is not installed",
            }

    return {
        "available": True,
        "status": "ready",
        "detail": "ready",
    }


def make_regressor(spec: BaselineSpec, train_seed: int) -> Any:
    availability = baseline_runtime_status(spec)
    if not availability["available"]:
        raise RuntimeError(str(availability["detail"]))
    if spec.input_block == "graph":
        raise RuntimeError("graph baselines are trained through the dedicated graph runner")

    hyperparameters = dict(spec.hyperparameters)
    if spec.estimator_key == "ridge":
        return _wrapped_regressor(make_pipeline(StandardScaler(), Ridge(alpha=float(hyperparameters.get("alpha", 1.0)))))
    if spec.estimator_key == "elastic_net":
        return _wrapped_regressor(
            make_pipeline(
                StandardScaler(),
                ElasticNet(
                    alpha=float(hyperparameters.get("alpha", 0.001)),
                    l1_ratio=float(hyperparameters.get("l1_ratio", 0.5)),
                    random_state=int(train_seed),
                    max_iter=5000,
                ),
            )
        )
    if spec.estimator_key == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(hyperparameters.get("n_estimators", 600)),
            max_features=hyperparameters.get("max_features", "sqrt"),
            min_samples_leaf=int(hyperparameters.get("min_samples_leaf", 2)),
            random_state=int(train_seed),
            n_jobs=1,
        )
    if spec.estimator_key == "svr":
        return _wrapped_regressor(
            make_pipeline(
                StandardScaler(),
                SVR(
                    kernel=str(hyperparameters.get("kernel", "rbf")),
                    C=float(hyperparameters.get("c", 10.0)),
                    epsilon=float(hyperparameters.get("epsilon", 0.1)),
                ),
            )
        )
    if spec.estimator_key == "mlp":
        hidden_dims = tuple(int(item) for item in hyperparameters.get("hidden_dims", [256, 128]))
        return _wrapped_regressor(
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=hidden_dims,
                    learning_rate_init=float(hyperparameters.get("learning_rate", 0.001)),
                    random_state=int(train_seed),
                    max_iter=600,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            )
        )
    if spec.estimator_key == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=int(hyperparameters.get("n_estimators", 600)),
            max_depth=int(hyperparameters.get("max_depth", 6)),
            learning_rate=float(hyperparameters.get("learning_rate", 0.05)),
            subsample=float(hyperparameters.get("subsample", 0.9)),
            colsample_bytree=float(hyperparameters.get("colsample_bytree", 0.9)),
            reg_lambda=float(hyperparameters.get("reg_lambda", 1.0)),
            random_state=int(train_seed),
            n_jobs=1,
            objective="reg:squarederror",
        )
    if spec.estimator_key == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=int(hyperparameters.get("n_estimators", 600)),
            num_leaves=int(hyperparameters.get("num_leaves", 31)),
            learning_rate=float(hyperparameters.get("learning_rate", 0.05)),
            subsample=float(hyperparameters.get("subsample", 0.9)),
            colsample_bytree=float(hyperparameters.get("colsample_bytree", 0.9)),
            reg_lambda=float(hyperparameters.get("reg_lambda", 0.0)),
            random_state=int(train_seed),
            n_jobs=1,
            verbosity=-1,
        )
    raise ValueError(f"unsupported estimator_key: {spec.estimator_key}")
