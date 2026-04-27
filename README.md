# msce-rcmf-masd

Public repository name: `msce-rcmf-masd`

This repository contains the code for the paper:

> **Selective Multimodal Regression under Heterogeneous Branch Reliability for Polymer Glass-Transition Temperature Prediction**

The method is a three-stage selective multimodal regression chain:

- `MSCE`: multiscale context encoding
- `RCMF`: reliability-conditioned multimodal fusion
- `MASD`: capped signed correction focused on the hard subgroup

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
- the protocol-matched baseline and audit scripts referenced by the paper
- the instructions needed to reproduce the released protocol

It is **not** intended to blindly re-host every merged upstream source table. For public release, the code repository should publish the processing pipeline and source-specific access instructions, while large artifacts and archival snapshots should go to Zenodo or another repository archive.

For public GitHub release, keep the repository focused on reproducible code and
paper-cited controls. Do not use the repository as a dump for manuscript
packaging, presentation drafts, or large generated experiment artifacts.

## Repository layout

- [data](./data): dataset building, canonicalization, feature construction, and split generation
- [models](./models): backbones, fusion logic, and reusable network modules
- [train](./train): training harness, calibration helpers, and controlled overrides
- [polymer_tg/scripts](./polymer_tg/scripts): paper-facing runners, ablations, and evaluation scripts
- [eval](./eval): metrics and comparator summaries

The local workspace also contains manuscript and packaging assets under `outputs/paper`, but those are kept out of the public code release by default.

## Important naming note

The manuscript now uses the term `MSCE` ("Multiscale Context Encoding"). Some code files still contain the older `mspce` token in filenames for continuity with the internal experiment history. The paper-facing method definition is the same chain described in the manuscript.

For paper-facing entry points, prefer the clean wrapper interfaces:

- [train/msce_stage.py](./train/msce_stage.py)
- [train/rcmf_stage.py](./train/rcmf_stage.py)
- [train/seed_semantics.py](./train/seed_semantics.py)

These wrappers preserve legacy behavior while exposing `MSCE`, `RCMF`, and
repeat-id semantics explicitly. The older `mspce_*` and `teacher/student`
tokens remain in the legacy internals for checkpoint compatibility and should
not be treated as the manuscript terminology.

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

- [polymer_tg/scripts/mainline_run.py](./polymer_tg/scripts/mainline_run.py): main paper-facing training and repeated-run driver
- [polymer_tg/scripts/mainline_eval.py](./polymer_tg/scripts/mainline_eval.py): evaluation packager for tables, diagnostics, and figure support artifacts

### Baselines and ablations

- [polymer_tg/scripts/descriptor_tree_baselines.py](./polymer_tg/scripts/descriptor_tree_baselines.py)
- [polymer_tg/scripts/attentivefp_graph_baseline.py](./polymer_tg/scripts/attentivefp_graph_baseline.py)
- [polymer_tg/scripts/masd_slot_ablation.py](./polymer_tg/scripts/masd_slot_ablation.py)
- [polymer_tg/scripts/rcmf_anchor_ablation.py](./polymer_tg/scripts/rcmf_anchor_ablation.py)
- [polymer_tg/scripts/msce_topk_ablation.py](./polymer_tg/scripts/msce_topk_ablation.py)

### Paper-supporting controls

- [train/curriculum_controlled_baseline.py](./train/curriculum_controlled_baseline.py): tests whether hard-subgroup curriculum alone can reproduce the hard-subgroup gain
- [train/polybert_baseline.py](./train/polybert_baseline.py): same-protocol polyBERT baseline probe

## Canonical reproduction path

Minimal smoke test:

```powershell
python polymer_tg/scripts/mainline_run.py --run-dir outputs/exp/diagnostics/repro_probe_one_seed --output-prefix repro_probe_one_seed --mainline-seeds 18 --external-supporting-seeds 18 --ablation-seeds=
python polymer_tg/scripts/mainline_eval.py --run-dir outputs/exp/diagnostics/repro_probe_one_seed --output-prefix repro_probe_one_seed
```

Current mainline run:

```powershell
python polymer_tg/scripts/mainline_run.py --run-dir outputs/exp/diagnostics/masd_final --output-prefix masd_final
python polymer_tg/scripts/mainline_eval.py --run-dir outputs/exp/diagnostics/masd_final --output-prefix masd_final
```

Paper-supporting controls:

```powershell
python train/curriculum_controlled_baseline.py --seeds 10,11,12,13,14,15,16,17,18,19
python train/polybert_baseline.py --seeds 10,11,12,13,14,15,16,17,18,19 --epochs 30
```

If `polyBERT` is not already cached locally, pass a Hugging Face model
identifier or local checkpoint path explicitly:

```powershell
python train/polybert_baseline.py --model-name-or-path kuelumbus/polyBERT
```

## Public release recommendations

For each public release:

- review [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md)
- create a GitHub release tag
- connect the repository to Zenodo and archive the release
- update [CITATION.cff](./CITATION.cff) if the release version, DOI, or repository metadata changes

## Citation

GitHub understands [CITATION.cff](./CITATION.cff). After the DOI is available, update that file so the citation shown on GitHub matches the public release.
