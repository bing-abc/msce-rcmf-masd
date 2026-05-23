from __future__ import annotations

import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import torch

from _submission_utils import ROOT, write_text


SUBMISSION_DIR = ROOT / "submission_final"
RELEASE_DIR = SUBMISSION_DIR / "data_release"
STAGE_DIR = RELEASE_DIR / "msce_rcmf_masd_release"
BUNDLE_PATH = ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run_merged_raw" / "mainline_bundle.pt"


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pt", "*.pdf", "*.png", "*.jpg", "*.jpeg", "*.svg"),
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: object) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def as_float(value: object) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def extract_split_and_mask_artifacts() -> None:
    bundle = torch.load(BUNDLE_PATH, map_location="cpu")
    split_dir = STAGE_DIR / "reproducibility" / "processed_split_definitions"
    mask_dir = STAGE_DIR / "reproducibility" / "fixed_hard_masks"

    primary_split_rows: list[dict[str, object]] = []
    external_split_rows: list[dict[str, object]] = []
    primary_mask_rows: list[dict[str, object]] = []
    external_mask_rows: list[dict[str, object]] = []
    seed_manifest_rows: list[dict[str, object]] = []
    primary_summary_rows: list[dict[str, object]] = []
    external_freq_counter: Counter[int] = Counter()

    first_external_written = False
    for seed_bundle in bundle["seed_bundles"]:
        seed = as_int(seed_bundle["seed"])
        primary = seed_bundle["baseline_primary_clean"]
        external = seed_bundle["baseline_external"]

        primary_hard_n = sum(as_int(flag) for flag in primary["hard_mask"])
        external_hard_n = sum(as_int(flag) for flag in external["hard_mask"])
        seed_manifest_rows.append(
            {
                "seed": seed,
                "primary_test_n": len(primary["sample_index"]),
                "primary_hard_n": primary_hard_n,
                "external_n": len(external["sample_index"]),
                "external_hard_n": external_hard_n,
            }
        )
        primary_summary_rows.append(
            {
                "seed": seed,
                "primary_test_n": len(primary["sample_index"]),
                "primary_hard_n": primary_hard_n,
                "primary_hard_fraction": f"{primary_hard_n / len(primary['sample_index']):.6f}",
            }
        )

        for position, (sample_index, hard_mask, hard_score, error) in enumerate(
            zip(primary["sample_index"], primary["hard_mask"], primary["hard_score"], primary["error"]),
            start=1,
        ):
            primary_split_rows.append(
                {
                    "seed": seed,
                    "position": position,
                    "sample_index": as_int(sample_index),
                }
            )
            primary_mask_rows.append(
                {
                    "seed": seed,
                    "position": position,
                    "sample_index": as_int(sample_index),
                    "hard_mask": as_int(hard_mask),
                    "baseline_abs_error_K": f"{as_float(error):.6f}",
                    "baseline_hard_score": f"{as_float(hard_score):.6f}",
                }
            )

        for position, (sample_index, hard_mask, hard_score, error) in enumerate(
            zip(external["sample_index"], external["hard_mask"], external["hard_score"], external["error"]),
            start=1,
        ):
            sample_idx = as_int(sample_index)
            external_freq_counter[sample_idx] += as_int(hard_mask)
            if not first_external_written:
                external_split_rows.append(
                    {
                        "position": position,
                        "sample_index": sample_idx,
                    }
                )
            external_mask_rows.append(
                {
                    "seed": seed,
                    "position": position,
                    "sample_index": sample_idx,
                    "hard_mask": as_int(hard_mask),
                    "baseline_abs_error_K": f"{as_float(error):.6f}",
                    "baseline_hard_score": f"{as_float(hard_score):.6f}",
                }
            )
        first_external_written = True

    external_freq_rows = [
        {
            "sample_index": sample_index,
            "hard_count_across_100_runs": hard_count,
            "hard_frequency": f"{hard_count / len(bundle['seed_bundles']):.6f}",
        }
        for sample_index, hard_count in sorted(external_freq_counter.items())
    ]

    write_csv(
        split_dir / "primary_clean_sample_indices_seedwise.csv",
        primary_split_rows,
        ["seed", "position", "sample_index"],
    )
    write_csv(
        split_dir / "external_sample_indices.csv",
        external_split_rows,
        ["position", "sample_index"],
    )
    write_csv(
        split_dir / "seed_manifest.csv",
        seed_manifest_rows,
        ["seed", "primary_test_n", "primary_hard_n", "external_n", "external_hard_n"],
    )
    write_csv(
        mask_dir / "primary_clean_seedwise_masks.csv",
        primary_mask_rows,
        ["seed", "position", "sample_index", "hard_mask", "baseline_abs_error_K", "baseline_hard_score"],
    )
    write_csv(
        mask_dir / "external_seedwise_masks.csv",
        external_mask_rows,
        ["seed", "position", "sample_index", "hard_mask", "baseline_abs_error_K", "baseline_hard_score"],
    )
    write_csv(
        mask_dir / "primary_clean_hard_mask_summary.csv",
        primary_summary_rows,
        ["seed", "primary_test_n", "primary_hard_n", "primary_hard_fraction"],
    )
    write_csv(
        mask_dir / "external_hard_mask_frequency.csv",
        external_freq_rows,
        ["sample_index", "hard_count_across_100_runs", "hard_frequency"],
    )


def write_release_docs() -> None:
    accessions_rows = [
        {
            "source_name": "PolyMetriX curated polymer Tg ecosystem",
            "local_expected_filename": "polymetrix_tg.csv",
            "accession_or_doi": "10.5281/zenodo.14980914",
            "role_in_study": "primary in-domain registry source",
            "redistribution_note": "not redistributed in full; use upstream accession",
        },
        {
            "source_name": "Liu Mendeley polymer Tg supplement",
            "local_expected_filename": "mendeley_non_grea_tg383.csv",
            "accession_or_doi": "10.17632/385nsgjvkm.1",
            "role_in_study": "supplemental in-domain registry source",
            "redistribution_note": "not redistributed in full; use upstream accession",
        },
        {
            "source_name": "Choi BigSMILES external benchmark package",
            "local_expected_filename": "step250_trackB_experimental_only.csv",
            "accession_or_doi": "10.6084/m9.figshare.c.6858337.v1",
            "role_in_study": "fixed external holdout source",
            "redistribution_note": "not redistributed in full; use upstream accession",
        },
    ]
    write_csv(
        STAGE_DIR / "docs" / "upstream_data_accessions.csv",
        accessions_rows,
        ["source_name", "local_expected_filename", "accession_or_doi", "role_in_study", "redistribution_note"],
    )

    package_readme = """# MSCE-RCMF-MASD archival release

This archive accompanies the manuscript:

> Multimodal Molecular Representation Learning for Polymer Glass Transition Temperature Prediction

It contains the code, processed split definitions, fixed hard-subgroup masks, figure source data, statistical-test outputs, and result exports used to support the manuscript submission. The archive is intentionally focused on reproducibility assets and does not re-host every third-party raw polymer source table.

## Archive contents

- `code/`: model, training, evaluation, and paper-support scripts
- `paper_sources/`: current manuscript and supplementary LaTeX sources
- `reproducibility/processed_split_definitions/`: seed-wise primary test indices and fixed external indices
- `reproducibility/fixed_hard_masks/`: seed-wise hard masks extracted from the authoritative 100-run baseline package
- `reproducibility/statistical_tests/`: paired t-test, Wilcoxon, bootstrap CI, and sign-rate outputs
- `reproducibility/result_exports/`: headline tables and supporting CSV/JSON exports
- `reproducibility/figure_source_data/`: source data for main and supplementary figures
- `docs/`: provenance notes, upstream accessions, and package manifest

## What is not redistributed here

- full third-party raw polymer source tables
- large checkpoint files
- local workspace diagnostics unrelated to the submitted manuscript

Use `docs/upstream_data_accessions.csv` together with `code/data/README.md` to reacquire the upstream public sources before rebuilding the processed registry.
"""
    write_text(STAGE_DIR / "README.md", package_readme)

    provenance = """# Data provenance

The manuscript integrates multiple public polymer-data sources. This release archives the processing scripts, split definitions, fixed hard-subgroup masks, result exports, and figure source data used in the submission-ready package.

The raw third-party source tables are not redistributed here in full because the local repository is prepared as a reproducibility release rather than a mirror of every upstream dataset. Upstream accessions and expected local filenames are listed in `upstream_data_accessions.csv`.

The authoritative headline metrics in the manuscript come from the frozen 100-run package `masd_final_trisoup_unionmask_clean_100run`, with merged seed-level evidence recovered from `mainline_bundle.pt` and `results.csv`. The fixed hard-subgroup masks released here are extracted directly from that bundle rather than recreated manually.
"""
    write_text(STAGE_DIR / "docs" / "data_provenance.md", provenance)

    zenodo_readme = """# Zenodo release note

This directory contains the archival release package prepared for a DOI-backed Zenodo or Mendeley Data deposition before manuscript submission or upon acceptance.

Included assets:

- source code needed to rerun the protocol
- processed split definitions
- fixed hard-subgroup masks
- result exports and statistical-test outputs
- figure source data
- manuscript-support scripts
- provenance and upstream-accession notes

Excluded assets:

- full third-party raw data tables when redistribution is uncertain
- heavyweight checkpoint bundles not needed for review-facing reproducibility

Before submission, mint a DOI for `msce_rcmf_masd_release.zip` if timing allows and update the manuscript Data Availability statement accordingly.

## Recommended Zenodo metadata

Title:
Multimodal Polymer Tg Prediction Reproducibility Package

Description:
This archive contains the code, fixed split definitions, hard-subgroup masks, result exports, statistical-test outputs, and figure source data supporting the Materials & Design submission.

Keywords:
polymer informatics; glass transition temperature; multimodal learning; molecular representation; reproducibility

Authors:
Songjiang Li; Bing Zhu; Yujie Feng; Yapeng Diao; Wenzhuo Jia; Xuan Fang; XiaoWan Gu; Peng Wang
"""
    write_text(RELEASE_DIR / "README_for_Zenodo.md", zenodo_readme)


def copy_release_payload() -> None:
    copy_targets = [
        (ROOT / "LICENSE", STAGE_DIR / "LICENSE"),
        (ROOT / "AUTHORS.md", STAGE_DIR / "AUTHORS.md"),
        (ROOT / "CITATION.cff", STAGE_DIR / "CITATION.cff"),
        (ROOT / "requirements.txt", STAGE_DIR / "requirements.txt"),
        (ROOT / "RELEASE_CHECKLIST.md", STAGE_DIR / "RELEASE_CHECKLIST.md"),
        (ROOT / "README.md", STAGE_DIR / "docs" / "local_repository_README.md"),
        (ROOT / "data" / "README.md", STAGE_DIR / "code" / "data" / "README.md"),
        (ROOT / "data" / "build_dataset.py", STAGE_DIR / "code" / "data" / "build_dataset.py"),
        (ROOT / "data" / "featurize.py", STAGE_DIR / "code" / "data" / "featurize.py"),
        (ROOT / "data" / "split_dataset.py", STAGE_DIR / "code" / "data" / "split_dataset.py"),
        (ROOT / "data" / "dedup.py", STAGE_DIR / "code" / "data" / "dedup.py"),
        (ROOT / "data" / "splits.json", STAGE_DIR / "reproducibility" / "processed_split_definitions" / "legacy_reference_splits.json"),
        (ROOT / "submission_final" / "source" / "main.tex", STAGE_DIR / "paper_sources" / "main.tex"),
        (ROOT / "submission_final" / "source" / "main.bbl", STAGE_DIR / "paper_sources" / "main.bbl"),
        (ROOT / "submission_final" / "source" / "references.bib", STAGE_DIR / "paper_sources" / "references.bib"),
        (ROOT / "submission_final" / "source" / "extra.bib", STAGE_DIR / "paper_sources" / "extra.bib"),
        (ROOT / "submission_final" / "supplementary" / "supplementary_information.tex", STAGE_DIR / "paper_sources" / "supplementary_information.tex"),
        (ROOT / "submission_final" / "supplementary" / "table_s4_statistical_tests.csv", STAGE_DIR / "reproducibility" / "statistical_tests" / "table_s4_statistical_tests.csv"),
        (ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run" / "stats.json", STAGE_DIR / "reproducibility" / "result_exports" / "stats.json"),
        (ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run" / "main_results_table.csv", STAGE_DIR / "reproducibility" / "result_exports" / "main_results_table.csv"),
        (ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run" / "improvement_table.csv", STAGE_DIR / "reproducibility" / "result_exports" / "improvement_table.csv"),
        (ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run" / "cluster_results_table.csv", STAGE_DIR / "reproducibility" / "result_exports" / "cluster_results_table.csv"),
        (ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run" / "subgroup_results_table.csv", STAGE_DIR / "reproducibility" / "result_exports" / "subgroup_results_table.csv"),
        (ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run" / "final_summary.md", STAGE_DIR / "reproducibility" / "result_exports" / "final_summary.md"),
        (ROOT / "outputs" / "exp" / "diagnostics" / "masd_final_trisoup_unionmask_clean_100run_merged_raw" / "results.csv", STAGE_DIR / "reproducibility" / "result_exports" / "merged_raw_results.csv"),
        (ROOT / "results_md_revision" / "polybert_multiseed_results.csv", STAGE_DIR / "reproducibility" / "result_exports" / "polybert_multiseed_results.csv"),
        (ROOT / "results_md_revision" / "rcmf_masd_ablation_results.csv", STAGE_DIR / "reproducibility" / "result_exports" / "rcmf_masd_ablation_results.csv"),
        (ROOT / "tables_md_revision" / "statistical_tests.csv", STAGE_DIR / "reproducibility" / "statistical_tests" / "statistical_tests.csv"),
        (ROOT / "tables_md_revision" / "final_metric_manifest.csv", STAGE_DIR / "reproducibility" / "statistical_tests" / "final_metric_manifest.csv"),
        (ROOT / "tables_md_revision" / "external_stratified_performance.csv", STAGE_DIR / "reproducibility" / "result_exports" / "external_stratified_performance.csv"),
        (ROOT / "tables_md_revision" / "cluster_failure_diagnosis.csv", STAGE_DIR / "reproducibility" / "result_exports" / "cluster_failure_diagnosis.csv"),
        (ROOT / "tables_md_revision" / "cluster_design_relevance.csv", STAGE_DIR / "reproducibility" / "result_exports" / "cluster_design_relevance.csv"),
        (ROOT / "scripts_md_revision" / "rebuild_main_manuscript_figures.py", STAGE_DIR / "reproducibility" / "figure_scripts" / "rebuild_main_manuscript_figures.py"),
        (ROOT / "scripts_final" / "rebuild_supplementary_figures.py", STAGE_DIR / "reproducibility" / "figure_scripts" / "rebuild_supplementary_figures.py"),
        (ROOT / "scripts_final" / "rebuild_graphical_abstract.py", STAGE_DIR / "reproducibility" / "figure_scripts" / "rebuild_graphical_abstract.py"),
        (ROOT / "outputs" / "paper" / "generate_figures.py", STAGE_DIR / "reproducibility" / "figure_scripts" / "generate_figures.py"),
    ]
    for src, dst in copy_targets:
        if src.exists():
            copy_file(src, dst)

    tree_targets = [
        (ROOT / "models", STAGE_DIR / "code" / "models"),
        (ROOT / "train", STAGE_DIR / "code" / "train"),
        (ROOT / "eval", STAGE_DIR / "code" / "eval"),
        (ROOT / "polymer_tg" / "scripts", STAGE_DIR / "code" / "polymer_tg" / "scripts"),
        (ROOT / "tg_prediction_pipeline" / "src", STAGE_DIR / "code" / "tg_prediction_pipeline" / "src"),
        (ROOT / "scripts_final", STAGE_DIR / "code" / "scripts_final"),
        (ROOT / "submission_final" / "source_data", STAGE_DIR / "reproducibility" / "figure_source_data"),
    ]
    for src, dst in tree_targets:
        if src.exists():
            copy_tree(src, dst)


def write_manifest() -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(STAGE_DIR.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(STAGE_DIR).as_posix(),
                    "size_bytes": str(path.stat().st_size),
                }
            )
    write_csv(STAGE_DIR / "docs" / "release_manifest.csv", rows, ["relative_path", "size_bytes"])


def build_zip() -> None:
    zip_path = RELEASE_DIR / "msce_rcmf_masd_release.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(STAGE_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(RELEASE_DIR))


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    copy_release_payload()
    extract_split_and_mask_artifacts()
    write_release_docs()
    write_manifest()
    build_zip()

    summary = {
        "package_dir": str(STAGE_DIR.relative_to(ROOT)),
        "package_zip": str((RELEASE_DIR / "msce_rcmf_masd_release.zip").relative_to(ROOT)),
        "bundle_source": str(BUNDLE_PATH.relative_to(ROOT)),
    }
    write_text(RELEASE_DIR / "package_summary.json", json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
