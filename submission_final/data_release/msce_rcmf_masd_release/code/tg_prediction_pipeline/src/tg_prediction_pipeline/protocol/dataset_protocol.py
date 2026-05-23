from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REQUIRED_DATASET_COLUMNS = (
    "record_id",
    "source_name",
    "role",
    "polymer_smiles",
    "canonical_smiles",
    "canonical_hash",
    "tg_k",
    "unit",
    "source_rows",
    "tg_std_k",
    "experimental_only",
    "overlap_with_external",
    "provenance_note",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    filename: str
    smiles_column: str
    target_column: str
    unit_column: str
    experimental_only: bool
    role: str


def pipeline_root() -> Path:
    return Path(__file__).resolve().parents[3]


def local_data_dir() -> Path:
    return pipeline_root() / "data"


def local_raw_dir() -> Path:
    return local_data_dir() / "raw"


def local_processed_dir() -> Path:
    return local_data_dir() / "processed"


def local_dataset_path() -> Path:
    return local_processed_dir() / "dataset.csv"


def local_dataset_report_path() -> Path:
    return local_processed_dir() / "dataset_report.csv"


def default_source_specs() -> tuple[SourceSpec, ...]:
    return (
        SourceSpec(
            name="polymetrix_tg",
            filename="polymetrix_tg.csv",
            smiles_column="polymer_smiles",
            target_column="Tg",
            unit_column="units",
            experimental_only=True,
            role="primary_pool",
        ),
        SourceSpec(
            name="mendeley_non_grea_tg383",
            filename="mendeley_non_grea_tg383.csv",
            smiles_column="polymer_smiles",
            target_column="y_target",
            unit_column="y_unit",
            experimental_only=True,
            role="supplemental_train",
        ),
        SourceSpec(
            name="bicerano_bigsmiles_tg304",
            filename="Bicerano_bigsmiles.csv",
            smiles_column="smiles",
            target_column="y_target",
            unit_column="units",
            experimental_only=True,
            role="external_holdout",
        ),
    )


def normalize_polymer_smiles(smiles: str) -> str:
    text = str(smiles).strip()
    placeholder = "__STAR__"
    text = text.replace("[*]", placeholder)
    text = text.replace("*", "[*]")
    return text.replace(placeholder, "[*]")


def canonicalize_polymer_smiles(smiles: str) -> str:
    normalized = normalize_polymer_smiles(smiles)
    if not normalized:
        raise ValueError("empty smiles")
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        raise ValueError(f"failed to parse smiles: {smiles}")
    return normalize_polymer_smiles(Chem.MolToSmiles(mol, canonical=True))


def _canonical_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _load_source(spec: SourceSpec, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"missing raw source file: {path}")
    frame = pd.read_csv(path)
    required = {spec.smiles_column, spec.target_column, spec.unit_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[[spec.smiles_column, spec.target_column, spec.unit_column]].copy()
    frame.columns = ["polymer_smiles", "tg_k", "unit"]
    frame["source_name"] = spec.name
    frame["experimental_only"] = bool(spec.experimental_only)
    frame["role"] = spec.role
    frame["canonical_smiles"] = frame["polymer_smiles"].map(canonicalize_polymer_smiles)
    frame["canonical_hash"] = frame["canonical_smiles"].map(_canonical_hash)
    return frame


def _aggregate_source(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["source_name", "experimental_only", "role", "canonical_smiles", "canonical_hash"])
        .agg(
            polymer_smiles=("polymer_smiles", "first"),
            tg_k=("tg_k", "mean"),
            unit=("unit", "first"),
            source_rows=("tg_k", "size"),
            tg_std_k=("tg_k", "std"),
        )
        .reset_index()
    )
    grouped["tg_std_k"] = grouped["tg_std_k"].fillna(0.0)
    return grouped


def validate_dataset_schema(dataset: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_DATASET_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")


def summarize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    validate_dataset_schema(dataset)
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "row_type": "summary",
            "source_name": "all",
            "n_rows": int(dataset.shape[0]),
            "n_primary": int((dataset["role"] == "primary_pool").sum()),
            "n_supplemental": int((dataset["role"] == "supplemental_train").sum()),
            "n_external": int((dataset["role"] == "external_holdout").sum()),
            "n_overlap_flagged": int(dataset["overlap_with_external"].sum()),
        }
    )
    for source_name, frame in dataset.groupby("source_name", sort=True):
        rows.append(
            {
                "row_type": "source",
                "source_name": str(source_name),
                "n_rows": int(frame.shape[0]),
                "n_primary": int((frame["role"] == "primary_pool").sum()),
                "n_supplemental": int((frame["role"] == "supplemental_train").sum()),
                "n_external": int((frame["role"] == "external_holdout").sum()),
                "n_overlap_flagged": int(frame["overlap_with_external"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_clean_dataset(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_root = raw_dir or local_raw_dir()
    raw_frames = {spec.name: _load_source(spec, source_root) for spec in default_source_specs()}
    aggregated = {name: _aggregate_source(frame) for name, frame in raw_frames.items()}

    primary = aggregated["polymetrix_tg"].copy()
    supplemental = aggregated["mendeley_non_grea_tg383"].copy()
    external = aggregated["bicerano_bigsmiles_tg304"].copy()

    primary_overlap = set(primary["canonical_smiles"]) & set(external["canonical_smiles"])
    supplemental_overlap = set(supplemental["canonical_smiles"]) & set(external["canonical_smiles"])

    primary["overlap_with_external"] = primary["canonical_smiles"].isin(primary_overlap)
    supplemental["overlap_with_external"] = supplemental["canonical_smiles"].isin(supplemental_overlap)
    external["overlap_with_external"] = False

    primary = primary.loc[~primary["overlap_with_external"]].copy()
    dataset = pd.concat([primary, supplemental, external], ignore_index=True)
    dataset["record_id"] = [f"tgp_{index:05d}" for index in range(dataset.shape[0])]
    dataset["provenance_note"] = dataset["source_rows"].map(
        lambda count: "aggregated_mean_from_duplicate_rows" if int(count) > 1 else "single_source_row"
    )
    dataset = dataset[list(REQUIRED_DATASET_COLUMNS)].sort_values(
        ["role", "source_name", "record_id"],
        ignore_index=True,
    )
    report = summarize_dataset(dataset)
    return dataset, report


def write_dataset_outputs(dataset: pd.DataFrame, report: pd.DataFrame, output_dir: Path | None = None) -> tuple[Path, Path]:
    target_dir = output_dir or local_processed_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = target_dir / "dataset.csv"
    report_path = target_dir / "dataset_report.csv"
    dataset.to_csv(dataset_path, index=False)
    report.to_csv(report_path, index=False)
    return dataset_path, report_path


def load_local_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    path = dataset_path or local_dataset_path()
    if not path.exists():
        raise FileNotFoundError(f"local dataset not found: {path}")
    dataset = pd.read_csv(path)
    validate_dataset_schema(dataset)
    return dataset


def import_existing_clean_dataset(source_dataset_path: str | Path, destination_path: Path | None = None) -> tuple[Path, Path]:
    source_path = Path(source_dataset_path)
    if not source_path.exists():
        raise FileNotFoundError(f"source dataset does not exist: {source_path}")
    dataset = pd.read_csv(source_path)
    validate_dataset_schema(dataset)
    report = summarize_dataset(dataset)
    target_dir = (destination_path.parent if destination_path is not None else local_processed_dir())
    dataset_path = destination_path or (target_dir / "dataset.csv")
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)
    report_path = target_dir / "dataset_report.csv"
    report.to_csv(report_path, index=False)
    return dataset_path, report_path
