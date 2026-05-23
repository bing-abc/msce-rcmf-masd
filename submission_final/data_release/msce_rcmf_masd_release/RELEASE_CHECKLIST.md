# GitHub release checklist

Use this checklist before each public release.

## 1. Repository metadata

- Use `msce-rcmf-masd` as the public repository name unless you deliberately choose a broader project scope.
- Set the repository name, description, and topics on GitHub.
- Confirm the repository URL, author list, and release version in `CITATION.cff`.
- Keep the `LICENSE` file in sync with the intended public release.

## 2. Data review

- Confirm which processed files can be redistributed.
- If redistribution is limited, keep merged tables out of the public repo and publish only:
  - code
  - split definitions
  - metadata
  - instructions for obtaining upstream public sources
- If a file is too large for normal Git history, prefer Zenodo, Mendeley Data, or Git LFS.

## 3. Reproducibility

- Create a fresh environment from `requirements.txt`.
- Run the smoke test commands from `README.md`.
- Confirm the code no longer depends on local sibling directories.

## 4. GitHub presentation

- Keep the root `README.md` readable on its own.
- Add a release tag after the first public push.
- Link the repository to Zenodo so the release gets a DOI.
- Copy the DOI back into the manuscript and `CITATION.cff` after the archive is minted.

## 5. Final safety pass

- Confirm `.gitignore` excludes generated artifacts and local work directories.
- Search for private paths, machine-specific usernames, or unpublished notes before pushing:

```powershell
rg -n "C:\\\\|Users\\\\|tg_clean_v7|RCMF-Polymer_vscode|TODO|FIXME"
```
