from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProtocolSplitConfig:
    n_repeats: int = 20
    split_id_offset: int = 0
    target_column: str = "tg_k"
    role_column: str = "role"
    primary_role: str = "primary_pool"
    supplemental_role: str = "supplemental_train"
    external_role: str = "external_holdout"
    test_fraction: float = 0.15
    val_fraction_within_trainval: float = 0.1764705882
    stratify_bins: int = 14


@dataclass(frozen=True)
class ProtocolSplit:
    split_id: int
    random_state: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    external_indices: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": int(self.split_id),
            "random_state": int(self.random_state),
            "train": list(self.train_indices),
            "val": list(self.val_indices),
            "test": list(self.test_indices),
            "external": list(self.external_indices),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HardSubsetConfig:
    method: str = "reference_absolute_error"
    quantile: float = 0.80
    minimum_size: int = 1
    threshold_scope: str = "per_slice"
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "conflict": 1.0,
            "uncertainty": 1.0,
            "low_confidence": 1.0,
        }
    )


@dataclass(frozen=True)
class HardSubsetSlice:
    sample_index: tuple[int, ...]
    difficulty_score: tuple[float, ...]
    hard_subset_mask: tuple[bool, ...]
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": list(self.sample_index),
            "difficulty_score": list(self.difficulty_score),
            "hard_subset_mask": list(self.hard_subset_mask),
            "threshold": float(self.threshold),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HardSubsetDefinition:
    split_id: int
    reference_model_name: str
    method: str
    test_slice: HardSubsetSlice
    external_slice: HardSubsetSlice | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": int(self.split_id),
            "reference_model_name": str(self.reference_model_name),
            "method": str(self.method),
            "test_slice": self.test_slice.to_dict(),
            "external_slice": None if self.external_slice is None else self.external_slice.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    estimator_key: str
    input_block: str
    description: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "family": str(self.family),
            "estimator_key": str(self.estimator_key),
            "input_block": str(self.input_block),
            "description": str(self.description),
            "hyperparameters": dict(self.hyperparameters),
        }


@dataclass(frozen=True)
class BaselineSuite:
    baseline_specs: tuple[BaselineSpec, ...]
    anchor_baseline_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.baseline_specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_baseline_name": self.anchor_baseline_name,
            "metadata": dict(self.metadata),
            "baselines": [spec.to_dict() for spec in self.baseline_specs],
        }


@dataclass(frozen=True)
class FeatureCache:
    descriptors: Any
    contexts: Any
    targets: Any
    canonical_smiles: tuple[str, ...]
    roles: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptors": self.descriptors,
            "contexts": self.contexts,
            "targets": self.targets,
            "canonical_smiles": list(self.canonical_smiles),
            "roles": list(self.roles),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphCache:
    graphs: tuple[Any, ...]
    canonical_smiles: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graphs": list(self.graphs),
            "canonical_smiles": list(self.canonical_smiles),
            "metadata": dict(self.metadata),
        }
