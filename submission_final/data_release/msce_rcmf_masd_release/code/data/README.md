# Data layout

This repository is prepared for a public code release. By default, the repository should contain:

- the dataset construction scripts
- the feature-building scripts
- split-generation logic
- instructions for obtaining the upstream public source files

It should not blindly re-host merged upstream source tables unless redistribution has been checked.

## Expected raw files

Place the raw public source files under `data/raw/` with the following filenames:

- `polymetrix_tg.csv`
- `mendeley_non_grea_tg383.csv`
- `step250_trackB_experimental_only.csv`

In the manuscript, these correspond to:

- the PolyMetriX-derived main source
- the Mendeley supplement
- the `Bicerano_bigsmiles.csv` external benchmark source used as the fixed external holdout

The local raw filename `step250_trackB_experimental_only.csv` is retained here for script compatibility; it is the repository-side filename expected by the current dataset builder for the external benchmark source.

The builder in [dedup.py](./dedup.py) looks for these local files first. For backward compatibility it can still fall back to the older sibling-workspace paths if they exist locally, but public GitHub use should rely on `data/raw/`.

## Generated artifacts

The following files are generated and are ignored by default in `.gitignore`:

- `dataset.csv`
- `features.pt`
- `splits.json`

Generate them with:

```powershell
python data/build_dataset.py
python data/featurize.py
python data/split_dataset.py
```

If you later decide to publish processed data or split files, review redistribution constraints first and then remove the corresponding `.gitignore` entries deliberately.
