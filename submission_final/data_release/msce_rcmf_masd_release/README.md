# MSCE-RCMF-MASD archival release

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
