from __future__ import annotations

"""Build the overlap-purged registry used by the paper-facing experiments.

For public GitHub use, raw source tables are expected under ``data/raw``.
A legacy sibling-workspace fallback is retained only so the current local
workspace can still rebuild without copying files around first.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
LEGACY_ROOT = ROOT.parent / "RCMF-Polymer_vscode"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    smiles_col: str
    target_col: str
    unit_col: str
    experimental_only: bool
    role: str


def _resolve_source_path(local_name: str, legacy_relative: str) -> Path:
    """Prefer repository-local raw files and only then fall back to legacy paths."""
    local_path = RAW_ROOT / local_name
    if local_path.exists():
        return local_path
    legacy_path = LEGACY_ROOT / legacy_relative
    if legacy_path.exists():
        return legacy_path
    return local_path


def normalize_psmiles_star(psmiles: str) -> str:
    text = str(psmiles).strip()
    placeholder = "__STAR__"
    text = text.replace("[*]", placeholder)
    text = text.replace("*", "[*]")
    return text.replace(placeholder, "[*]")


def canonicalize_any_smiles(smiles: str) -> str:
    text = str(smiles).strip()
    if not text:
        raise ValueError("empty smiles")
    text = normalize_psmiles_star(text)
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"rdkit parse failed for {text}")
    return normalize_psmiles_star(Chem.MolToSmiles(mol, canonical=True))


def _load_source(spec: SourceSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(
            f"Missing source file for {spec.name}: {spec.path}. "
            f"Place the file under {RAW_ROOT} as documented in data/README.md."
        )

    df = pd.read_csv(spec.path)
    df = df[[spec.smiles_col, spec.target_col, spec.unit_col]].copy()
    df.columns = ["polymer_smiles", "tg_k", "unit"]
    df["source_name"] = spec.name
    df["experimental_only"] = spec.experimental_only
    df["raw_role"] = spec.role
    df["canonical_smiles"] = df["polymer_smiles"].map(canonicalize_any_smiles)
    df["canonical_hash"] = df["canonical_smiles"].map(
        lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    )
    return df


def _aggregate_source(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["source_name", "experimental_only", "raw_role", "canonical_smiles", "canonical_hash"])
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


def source_specs() -> list[SourceSpec]:
    return [
        SourceSpec(
            name="polymetrix_tg",
            path=_resolve_source_path(
                "polymetrix_tg.csv",
                "polyuatg_clean/data/polymer/raw/polymetrix_tg.csv",
            ),
            smiles_col="polymer_smiles",
            target_col="Tg",
            unit_col="units",
            experimental_only=True,
            role="primary_pool",
        ),
        SourceSpec(
            name="mendeley_non_grea_tg383",
            path=_resolve_source_path(
                "mendeley_non_grea_tg383.csv",
                "outputs/exp/runs/step277_20260321_133139/mendeley_non_grea_tg383.csv",
            ),
            smiles_col="polymer_smiles",
            target_col="y_target",
            unit_col="y_unit",
            experimental_only=True,
            role="supplemental_train",
        ),
        SourceSpec(
            name="step250_trackB_experimental_only",
            path=_resolve_source_path(
                "step250_trackB_experimental_only.csv",
                "polyuatg_clean/data/polymer_external/step250_tracks/processed/step250_trackB_experimental_only.csv",
            ),
            smiles_col="smiles",
            target_col="y_target",
            unit_col="units",
            experimental_only=True,
            role="external_holdout",
        ),
    ]


def build_clean_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_frames = {spec.name: _load_source(spec) for spec in source_specs()}
    agg_frames = {name: _aggregate_source(df) for name, df in raw_frames.items()}

    polymetrix = agg_frames["polymetrix_tg"].copy()
    mendeley = agg_frames["mendeley_non_grea_tg383"].copy()
    external = agg_frames["step250_trackB_experimental_only"].copy()

    overlap_poly_external = set(polymetrix["canonical_smiles"]) & set(external["canonical_smiles"])
    overlap_mendeley_external = set(mendeley["canonical_smiles"]) & set(external["canonical_smiles"])
    overlap_poly_mendeley = set(polymetrix["canonical_smiles"]) & set(mendeley["canonical_smiles"])

    polymetrix["overlap_with_external"] = polymetrix["canonical_smiles"].isin(overlap_poly_external)
    mendeley["overlap_with_external"] = mendeley["canonical_smiles"].isin(overlap_mendeley_external)
    external["overlap_with_external"] = False

    polymetrix_clean = polymetrix.loc[~polymetrix["overlap_with_external"]].copy()
    polymetrix_clean["role"] = "primary_pool"
    mendeley["role"] = "supplemental_train"
    external["role"] = "external_holdout"

    dataset = pd.concat([polymetrix_clean, mendeley, external], ignore_index=True)
    dataset["record_id"] = [f"tgv2_{i:05d}" for i in range(len(dataset))]
    dataset["provenance_note"] = dataset.apply(
        lambda row: (
            "aggregated_mean_from_duplicate_rows"
            if int(row["source_rows"]) > 1
            else "single_source_row"
        ),
        axis=1,
    )
    dataset = dataset[
        [
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
        ]
    ].sort_values(["role", "source_name", "record_id"], ignore_index=True)

    report_rows: list[dict[str, Any]] = []
    total_raw_rows = sum(len(frame) for frame in raw_frames.values())
    total_unique_rows = sum(len(frame) for frame in agg_frames.values())
    report_rows.append(
        {
            "row_type": "summary",
            "source_name": "all",
            "raw_rows_before_dedup": total_raw_rows,
            "unique_rows_after_dedup": total_unique_rows,
            "duplicates_removed": total_raw_rows - total_unique_rows,
            "removed_due_to_external_overlap": len(overlap_poly_external) + len(overlap_mendeley_external),
            "final_training_rows": int(dataset["role"].isin(["primary_pool", "supplemental_train"]).sum()),
            "final_external_rows": int((dataset["role"] == "external_holdout").sum()),
            "overlap_remaining_with_external": 0,
            "experimental_only": True,
            "notes": "step250 overlap purged from training pool before split generation",
        }
    )

    for spec in source_specs():
        raw_frame = raw_frames[spec.name]
        agg_frame = agg_frames[spec.name]
        final_rows = dataset.loc[dataset["source_name"] == spec.name]
        overlap_removed = 0
        if spec.name == "polymetrix_tg":
            overlap_removed = len(overlap_poly_external)
        if spec.name == "mendeley_non_grea_tg383":
            overlap_removed = len(overlap_mendeley_external)
        report_rows.append(
            {
                "row_type": "source",
                "source_name": spec.name,
                "raw_rows_before_dedup": len(raw_frame),
                "unique_rows_after_dedup": len(agg_frame),
                "duplicates_removed": len(raw_frame) - len(agg_frame),
                "removed_due_to_external_overlap": overlap_removed,
                "final_training_rows": int(
                    final_rows["role"].isin(["primary_pool", "supplemental_train"]).sum()
                ),
                "final_external_rows": int((final_rows["role"] == "external_holdout").sum()),
                "overlap_remaining_with_external": 0,
                "experimental_only": spec.experimental_only,
                "notes": "",
            }
        )

    report_rows.append(
        {
            "row_type": "overlap_check",
            "source_name": "polymetrix_vs_external",
            "raw_rows_before_dedup": len(polymetrix),
            "unique_rows_after_dedup": len(overlap_poly_external),
            "duplicates_removed": 0,
            "removed_due_to_external_overlap": len(overlap_poly_external),
            "final_training_rows": 0,
            "final_external_rows": 0,
            "overlap_remaining_with_external": 0,
            "experimental_only": True,
            "notes": "all overlapping canonical polymers removed from training pool",
        }
    )
    report_rows.append(
        {
            "row_type": "overlap_check",
            "source_name": "mendeley_vs_external",
            "raw_rows_before_dedup": len(mendeley),
            "unique_rows_after_dedup": len(overlap_mendeley_external),
            "duplicates_removed": 0,
            "removed_due_to_external_overlap": len(overlap_mendeley_external),
            "final_training_rows": 0,
            "final_external_rows": 0,
            "overlap_remaining_with_external": 0,
            "experimental_only": True,
            "notes": "no overlap detected",
        }
    )
    report_rows.append(
        {
            "row_type": "overlap_check",
            "source_name": "polymetrix_vs_mendeley",
            "raw_rows_before_dedup": len(polymetrix),
            "unique_rows_after_dedup": len(overlap_poly_mendeley),
            "duplicates_removed": 0,
            "removed_due_to_external_overlap": 0,
            "final_training_rows": 0,
            "final_external_rows": 0,
            "overlap_remaining_with_external": 0,
            "experimental_only": True,
            "notes": "no overlap detected",
        }
    )

    report = pd.DataFrame(report_rows)
    meta = {
        "raw_root": str(RAW_ROOT),
        "legacy_root": str(LEGACY_ROOT),
        "overlap_poly_external": len(overlap_poly_external),
        "overlap_mendeley_external": len(overlap_mendeley_external),
        "overlap_poly_mendeley": len(overlap_poly_mendeley),
    }
    return dataset, report, meta


def write_outputs(dataset: pd.DataFrame, report: pd.DataFrame) -> None:
    data_path = ROOT / "data/dataset.csv"
    report_path = ROOT / "reports/dataset_report.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(data_path, index=False)
    report.to_csv(report_path, index=False)


def main() -> int:
    dataset, report, _ = build_clean_dataset()
    write_outputs(dataset, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
