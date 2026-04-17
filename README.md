# msce-rcmf-masd

Public repository name: `msce-rcmf-masd`

This repository contains the code for the paper:

> **Selective Multimodal Regression with Multiscale Context Encoding, Reliability-Conditioned Multimodal Fusion, and Signed Decomposition**

The method is a three-stage selective multimodal regression chain:

- `MSCE`: multiscale context encoding
- `RCMF`: reliability-conditioned multimodal fusion
- `MASD`: bounded signed correction focused on difficult samples

The evaluation domain is polymer glass-transition temperature (`T_g`) prediction, but the paper is positioned as a pattern-recognition methodology study about uneven branch reliability in multimodal regression.

## Authors

The repository metadata uses the full author names listed in [AUTHORS.md](./AUTHORS.md). The corresponding author for the code release is **Wang Peng**.

## What this public repository is for

This repository is prepared as a **code release**, not as a dump of the local research workspace.

It is intended to provide:

- the model code
- the dataset-construction and feature-construction scripts
- the split-generation logic
- the experiment entry points used by the paper
- the instructions needed to reproduce the released protocol

It is **not** intended to blindly re-host every merged upstream source table. For public release, the code repository should publish the processing pipeline and source-specific access instructions, while large artifacts and archival snapshots should go to Zenodo or another repository archive.

## Repository layout

- [data](./data): dataset building, canonicalization, feature construction, and split generation
- [models](./models): backbones, fusion logic, and reusable network modules
- [train](./train): training harness, calibration helpers, and controlled overrides
- [polyuatg_clean/scripts](./polyuatg_clean/scripts): paper-facing runners, ablations, and evaluation scripts
- [eval](./eval): metrics and comparator summaries

The local workspace also contains manuscript and packaging assets under `outputs/paper`, but those are kept out of the public code release by default.

## Important naming note

The manuscript now uses the term `MSCE` ("Multiscale Context Encoding"). Some code files still contain the older `mspce` token in filenames for continuity with the internal experiment history. The paper-facing method definition is the same chain described in the manuscript.

## Environment

The paper-facing runs were developed with Python 3.11 and the following package set:

- `numpy==1.26.4`
- `pandas==2.1.4`
- `scipy==1.11.4`
- `scikit-learn==1.2.2`
- `matplotlib==3.8.4`
- `torch==2.5.1`
- `torch-geometric==2.7.0`
- `rdkit==2022.09.5`

Install the base environment with:

```powershell
pip install -r requirements.txt
```

Notes:

- For CUDA-enabled PyTorch, use the official PyTorch selector to install the matching wheel for your machine.
- RDKit is often easiest to install through conda-forge on a clean environment.

## Data preparation

Place the raw public source files under [data/raw](./data/raw) as described in [data/README.md](./data/README.md).

Then run:

```powershell
python data/build_dataset.py
python data/featurize.py
python data/split_dataset.py
```

These steps create:

- `data/dataset.csv`
- `data/features.pt`
- `data/splits.json`

Those files are rebuildable and are ignored by default for public GitHub upload.

## Main entry points

### End-to-end mainline

- [polyuatg_clean/scripts/masd_v3_run.py](./polyuatg_clean/scripts/masd_v3_run.py): main paper-facing training and repeated-run driver
- [polyuatg_clean/scripts/masd_v3_eval.py](./polyuatg_clean/scripts/masd_v3_eval.py): evaluation packager for tables, diagnostics, and figure support artifacts

### Baselines and ablations

- [polyuatg_clean/scripts/descriptor_tree_baselines.py](./polyuatg_clean/scripts/descriptor_tree_baselines.py)
- [polyuatg_clean/scripts/attentivefp_graph_baseline.py](./polyuatg_clean/scripts/attentivefp_graph_baseline.py)
- [polyuatg_clean/scripts/masd_slot_ablation.py](./polyuatg_clean/scripts/masd_slot_ablation.py)
- [polyuatg_clean/scripts/rcmf_anchor_ablation.py](./polyuatg_clean/scripts/rcmf_anchor_ablation.py)
- [polyuatg_clean/scripts/mspce_k_ablation.py](./polyuatg_clean/scripts/mspce_k_ablation.py)

## Canonical reproduction path

Minimal smoke test:

```powershell
python polyuatg_clean/scripts/masd_v3_run.py --run-dir outputs/exp/diagnostics/repro_probe_one_seed --output-prefix repro_probe_one_seed --mainline-seeds 18 --external-supporting-seeds 18 --ablation-seeds=
python polyuatg_clean/scripts/masd_v3_eval.py --run-dir outputs/exp/diagnostics/repro_probe_one_seed --output-prefix repro_probe_one_seed
```

Current mainline run:

```powershell
python polyuatg_clean/scripts/masd_v3_run.py --run-dir outputs/exp/diagnostics/masd_final --output-prefix masd_final
python polyuatg_clean/scripts/masd_v3_eval.py --run-dir outputs/exp/diagnostics/masd_final --output-prefix masd_final
```

## Public release recommendations

For each public release:

- review [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md)
- create a GitHub release tag
- connect the repository to Zenodo and archive the release
- update [CITATION.cff](./CITATION.cff) if the release version, DOI, or repository metadata changes

## Citation

GitHub understands [CITATION.cff](./CITATION.cff). After the DOI is available, update that file so the citation shown on GitHub matches the public release.
