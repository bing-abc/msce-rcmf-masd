from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski

from tg_prediction_pipeline.schemas import FeatureCache

CONTEXT_SCALE_LAYOUT = (
    ("smiles_ngram_2", 128),
    ("smiles_ngram_3", 128),
    ("smiles_window_4", 96),
    ("graph_neighborhood_hash", 96),
    ("interpretable_context", 12),
)


def local_feature_cache_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "features.npz"


def context_scale_layout() -> tuple[dict[str, int], ...]:
    layout: list[dict[str, int]] = []
    cursor = 0
    for name, width in CONTEXT_SCALE_LAYOUT:
        start = cursor
        stop = cursor + int(width)
        layout.append({"name": str(name), "start": int(start), "stop": int(stop), "width": int(width)})
        cursor = stop
    return tuple(layout)


def expected_context_dim() -> int:
    return int(sum(item["width"] for item in context_scale_layout()))


def feature_cache_matches_current_layout(feature_cache: FeatureCache) -> bool:
    context_matrix = np.asarray(feature_cache.contexts, dtype=np.float32)
    metadata_layout = list(feature_cache.metadata.get("context_layout", [])) if isinstance(feature_cache.metadata, dict) else []
    return bool(
        context_matrix.ndim == 2
        and int(context_matrix.shape[1]) == expected_context_dim()
        and metadata_layout
        and len(metadata_layout) == len(context_scale_layout())
    )


def _base_molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"invalid canonical_smiles: {smiles}")
    return molecule


def tokenize_polymer_smiles(smiles: str) -> list[str]:
    text = str(smiles)
    tokens: list[str] = []
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char == "[":
            end = text.find("]", cursor)
            if end == -1:
                tokens.append(text[cursor:])
                break
            tokens.append(text[cursor : end + 1])
            cursor = end + 1
            continue
        if cursor + 1 < len(text) and text[cursor : cursor + 2] in {"Cl", "Br"}:
            tokens.append(text[cursor : cursor + 2])
            cursor += 2
            continue
        tokens.append(char)
        cursor += 1
    return tokens


def _stable_bucket(text: str, dim: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(dim)


def _hashed_ngram_counts(tokens: Iterable[str], ngram: int, dim: int) -> np.ndarray:
    token_list = list(tokens)
    vector = np.zeros((dim,), dtype=np.float32)
    if len(token_list) < ngram:
        return vector
    for start in range(len(token_list) - ngram + 1):
        key = "|".join(token_list[start : start + ngram])
        vector[_stable_bucket(key, dim)] += 1.0
    norm = float(max(vector.sum(), 1.0))
    return vector / norm


def _interpretable_context_features(smiles: str, tokens: list[str], molecule: Chem.Mol) -> np.ndarray:
    text = str(smiles)
    atom_count = float(molecule.GetNumAtoms())
    bond_count = float(molecule.GetNumBonds())
    ring_count = float(molecule.GetRingInfo().NumRings())
    aromatic_atoms = float(sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic()))
    hetero_atoms = float(sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() not in {1, 6}))
    return np.asarray(
        [
            float(len(tokens)),
            float(tokens.count("[*]")),
            float(text.count("(")),
            float(text.count(")")),
            float(text.count("=")),
            float(text.count("#")),
            atom_count,
            bond_count,
            ring_count,
            aromatic_atoms / max(atom_count, 1.0),
            hetero_atoms / max(atom_count, 1.0),
            float(Lipinski.FractionCSP3(molecule)),
        ],
        dtype=np.float32,
    )


def _graph_neighborhood_context(molecule: Chem.Mol, dim: int = 96) -> np.ndarray:
    vector = np.zeros((dim,), dtype=np.float32)
    for atom in molecule.GetAtoms():
        neighbor_atomic_numbers = sorted(int(neighbor.GetAtomicNum()) for neighbor in atom.GetNeighbors())
        bond_types = sorted(str(bond.GetBondType()) for bond in atom.GetBonds())
        key = "|".join(
            [
                f"a{int(atom.GetAtomicNum())}",
                f"deg{int(atom.GetDegree())}",
                f"ring{int(atom.IsInRing())}",
                f"aro{int(atom.GetIsAromatic())}",
                "nbr:" + ",".join(str(item) for item in neighbor_atomic_numbers),
                "bond:" + ",".join(bond_types),
            ]
        )
        vector[_stable_bucket(key, dim)] += 1.0
    norm = float(max(vector.sum(), 1.0))
    return vector / norm


def build_context_vector(smiles: str) -> np.ndarray:
    tokens = tokenize_polymer_smiles(smiles)
    molecule = _base_molecule(smiles)
    grams2 = _hashed_ngram_counts(tokens, ngram=2, dim=128)
    grams3 = _hashed_ngram_counts(tokens, ngram=3, dim=128)
    window4 = _hashed_ngram_counts(tokens, ngram=4, dim=96)
    graph_context = _graph_neighborhood_context(molecule, dim=96)
    interpretable = _interpretable_context_features(smiles, tokens, molecule)
    return np.concatenate([grams2, grams3, window4, graph_context, interpretable], axis=0)


def split_context_by_scale(context_matrix: np.ndarray) -> list[np.ndarray]:
    matrix = np.asarray(context_matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    expected_dim = expected_context_dim()
    if int(matrix.shape[1]) != int(expected_dim):
        raise ValueError(f"context matrix has dim {matrix.shape[1]}, expected {expected_dim}")
    return [matrix[:, item["start"] : item["stop"]] for item in context_scale_layout()]


def build_descriptor_vector(smiles: str, fp_bits: int = 256) -> np.ndarray:
    molecule = _base_molecule(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=int(fp_bits))
    fingerprint_array = np.zeros((fp_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    descriptor_array = np.asarray(
        [
            float(Descriptors.MolWt(molecule)),
            float(Crippen.MolLogP(molecule)),
            float(Descriptors.TPSA(molecule)),
            float(Lipinski.HeavyAtomCount(molecule)),
            float(Lipinski.NumHAcceptors(molecule)),
            float(Lipinski.NumHDonors(molecule)),
            float(Lipinski.NumRotatableBonds(molecule)),
            float(Lipinski.RingCount(molecule)),
            float(Lipinski.NumAromaticRings(molecule)),
            float(Lipinski.FractionCSP3(molecule)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([descriptor_array, fingerprint_array], axis=0)


def build_feature_cache(dataset: pd.DataFrame) -> FeatureCache:
    if "canonical_smiles" not in dataset.columns or "tg_k" not in dataset.columns or "role" not in dataset.columns:
        raise ValueError("dataset must contain canonical_smiles, tg_k, and role columns")
    descriptor_rows: list[np.ndarray] = []
    context_rows: list[np.ndarray] = []
    for smiles in dataset["canonical_smiles"].astype(str):
        descriptor_rows.append(build_descriptor_vector(smiles))
        context_rows.append(build_context_vector(smiles))
    return FeatureCache(
        descriptors=np.asarray(descriptor_rows, dtype=np.float32),
        contexts=np.asarray(context_rows, dtype=np.float32),
        targets=dataset["tg_k"].to_numpy(dtype=np.float32).reshape(-1, 1),
        canonical_smiles=tuple(dataset["canonical_smiles"].astype(str).tolist()),
        roles=tuple(dataset["role"].astype(str).tolist()),
        metadata={
            "descriptor_dim": int(np.asarray(descriptor_rows[0]).shape[0]) if descriptor_rows else 0,
            "context_dim": int(np.asarray(context_rows[0]).shape[0]) if context_rows else 0,
            "context_layout": list(context_scale_layout()),
            "n_samples": int(dataset.shape[0]),
        },
    )


def save_feature_cache(feature_cache: FeatureCache, output_path: str | Path | None = None) -> Path:
    path = Path(output_path) if output_path is not None else local_feature_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        descriptors=np.asarray(feature_cache.descriptors, dtype=np.float32),
        contexts=np.asarray(feature_cache.contexts, dtype=np.float32),
        targets=np.asarray(feature_cache.targets, dtype=np.float32),
        canonical_smiles=np.asarray(feature_cache.canonical_smiles, dtype=object),
        roles=np.asarray(feature_cache.roles, dtype=object),
        metadata=np.asarray([feature_cache.metadata], dtype=object),
    )
    return path


def load_feature_cache(feature_path: str | Path | None = None) -> FeatureCache:
    path = Path(feature_path) if feature_path is not None else local_feature_cache_path()
    if not path.exists():
        raise FileNotFoundError(f"feature cache not found: {path}")
    payload = np.load(path, allow_pickle=True)
    metadata_array = payload["metadata"]
    metadata = dict(metadata_array[0]) if metadata_array.size else {}
    return FeatureCache(
        descriptors=np.asarray(payload["descriptors"], dtype=np.float32),
        contexts=np.asarray(payload["contexts"], dtype=np.float32),
        targets=np.asarray(payload["targets"], dtype=np.float32),
        canonical_smiles=tuple(str(item) for item in payload["canonical_smiles"].tolist()),
        roles=tuple(str(item) for item in payload["roles"].tolist()),
        metadata=metadata,
    )
