from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|Sn|@@?|=|#|\(|\)|\.|\/|\\\\|\+|-|\d+|[A-Za-z*])")
MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=512)


def _hash_bucket(text: str, dim: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % dim


def tokenize_polymer(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(str(text))
    return tokens if tokens else list(str(text))


def hashed_ngram_counts(tokens: list[str], dim: int, ngram: int) -> np.ndarray:
    seq = ["<bos>"] + tokens + ["<eos>"]
    vec = np.zeros((dim,), dtype=np.float32)
    if len(seq) < ngram:
        vec[_hash_bucket(" ".join(seq), dim)] += 1.0
        return vec
    for i in range(len(seq) - ngram + 1):
        key = " ".join(seq[i : i + ngram])
        vec[_hash_bucket(key, dim)] += 1.0
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def hashed_segment_windows(tokens: list[str], dim: int, window: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    if len(tokens) == 0:
        return vec
    if len(tokens) <= window:
        vec[_hash_bucket("".join(tokens), dim)] += 1.0
    else:
        for i in range(len(tokens) - window + 1):
            segment = "".join(tokens[i : i + window])
            vec[_hash_bucket(segment, dim)] += 1.0
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def hashed_neighborhood_context(mol: Chem.Mol, dim: int, radius: int) -> np.ndarray:
    dist = Chem.GetDistanceMatrix(mol)
    vec = np.zeros((dim,), dtype=np.float32)
    atom_count = mol.GetNumAtoms()
    for i in range(atom_count):
        center = mol.GetAtomWithIdx(i).GetSymbol()
        within = [
            mol.GetAtomWithIdx(j).GetSymbol()
            for j in range(atom_count)
            if j != i and dist[i, j] > 0 and dist[i, j] <= radius
        ]
        within.sort()
        key = f"r{radius}|{center}|n{len(within)}|{'-'.join(within[:10])}"
        vec[_hash_bucket(key, dim)] += 1.0
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def context_interpretable_features(smiles: str, tokens: list[str], mol: Chem.Mol) -> tuple[np.ndarray, list[str]]:
    token_count = float(len(tokens))
    atom_tokens = [tok for tok in tokens if tok and tok[0].isalpha()]
    branch_tokens = [tok for tok in tokens if tok in {"(", ")"}]
    ring_tokens = [tok for tok in tokens if tok.isdigit()]
    aromatic_tokens = [tok for tok in tokens if tok.isalpha() and tok.islower()]
    star_tokens = [tok for tok in tokens if "*" in tok]
    unique_ratio = float(len(set(tokens)) / max(len(tokens), 1))
    n_atoms = float(mol.GetNumAtoms())
    n_bonds = float(mol.GetNumBonds())
    degrees = np.array([atom.GetDegree() for atom in mol.GetAtoms()], dtype=np.float32)
    aromatic_atoms = np.array([float(atom.GetIsAromatic()) for atom in mol.GetAtoms()], dtype=np.float32)
    hetero_atoms = np.array(
        [float(atom.GetAtomicNum() not in {1, 6}) for atom in mol.GetAtoms()],
        dtype=np.float32,
    )
    halogen_atoms = np.array(
        [float(atom.GetAtomicNum() in {9, 17, 35, 53}) for atom in mol.GetAtoms()],
        dtype=np.float32,
    )
    ring_count = float(rdMolDescriptors.CalcNumRings(mol))
    features = np.array(
        [
            token_count,
            float(len(atom_tokens) / max(len(tokens), 1)),
            float(len(branch_tokens) / max(len(tokens), 1)),
            float(len(ring_tokens) / max(len(tokens), 1)),
            float(len(aromatic_tokens) / max(len(tokens), 1)),
            float(len(star_tokens) / max(len(tokens), 1)),
            unique_ratio,
            n_atoms,
            n_bonds,
            float(degrees.mean()) if degrees.size else 0.0,
            float(degrees.max()) if degrees.size else 0.0,
            ring_count,
            float(aromatic_atoms.mean()) if aromatic_atoms.size else 0.0,
            float(hetero_atoms.mean()) if hetero_atoms.size else 0.0,
            float(halogen_atoms.mean()) if halogen_atoms.size else 0.0,
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(Descriptors.TPSA(mol) / 200.0),
            float(Descriptors.MolWt(mol) / 1000.0),
        ],
        dtype=np.float32,
    )
    names = [
        "token_count",
        "atom_token_ratio",
        "branch_token_ratio",
        "ring_marker_ratio",
        "aromatic_token_ratio",
        "star_token_ratio",
        "unique_token_ratio",
        "n_atoms",
        "n_bonds",
        "mean_degree",
        "max_degree",
        "ring_count",
        "aromatic_atom_ratio",
        "hetero_atom_ratio",
        "halogen_atom_ratio",
        "fraction_csp3",
        "hbond_acceptor_count",
        "hbond_donor_count",
        "tpsa_scaled",
        "molwt_scaled",
    ]
    return features, names


def build_mspce_multiscale_context(smiles: str) -> tuple[np.ndarray, dict[str, object]]:
    tokens = tokenize_polymer(smiles)
    mol = _base_mol(smiles)
    segments: list[np.ndarray] = []
    scales: list[dict[str, object]] = []
    feature_names: list[str] = []
    cursor = 0

    def _push(name: str, kind: str, values: np.ndarray, names: list[str] | None = None) -> None:
        nonlocal cursor
        start = cursor
        end = start + int(values.shape[0])
        segments.append(values.astype(np.float32))
        scales.append({"name": name, "kind": kind, "start": start, "end": end})
        cursor = end
        if names is None:
            feature_names.extend([f"{name}_{i}" for i in range(start, end)])
        else:
            feature_names.extend(names)

    _push("smiles_ngram2", "smiles_chain", hashed_ngram_counts(tokens, dim=192, ngram=2))
    _push("smiles_ngram3", "smiles_chain", hashed_ngram_counts(tokens, dim=192, ngram=3))
    _push("smiles_ngram4", "smiles_chain", hashed_ngram_counts(tokens, dim=192, ngram=4))
    _push("smiles_segment_w5", "smiles_segment", hashed_segment_windows(tokens, dim=128, window=5))
    _push("graph_radius1", "graph_neighborhood", hashed_neighborhood_context(mol, dim=128, radius=1))
    _push("graph_radius2", "graph_neighborhood", hashed_neighborhood_context(mol, dim=128, radius=2))
    interpretable_values, interpretable_names = context_interpretable_features(smiles, tokens, mol)
    _push("interpretable", "interpretable", interpretable_values, names=interpretable_names)

    return np.concatenate(segments, axis=0), {"scales": scales, "feature_names": feature_names}


def _base_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"invalid smiles: {smiles}")
    return mol


def _geometry_ready_mol(smiles: str) -> Chem.Mol:
    mol = Chem.RWMol(_base_mol(smiles))
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
    geom = mol.GetMol()
    Chem.SanitizeMol(geom)
    return geom


def descriptor_vector(smiles: str, fp_bits: int = 512) -> np.ndarray:
    mol = _base_mol(smiles)
    if int(fp_bits) == 512:
        fp = MORGAN_GENERATOR.GetFingerprint(mol)
    else:
        fp = GetMorganGenerator(radius=2, fpSize=int(fp_bits)).GetFingerprint(mol)
    fp_arr = np.zeros((fp_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    desc = np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.HeavyAtomCount(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.FractionCSP3(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
        ],
        dtype=np.float32,
    )
    return np.concatenate([desc, fp_arr], axis=0)


def led_descriptor_vector(smiles: str) -> tuple[np.ndarray, float]:
    try:
        mol = _geometry_ready_mol(smiles)
        if mol.GetNumAtoms() > 64:
            return np.zeros((11,), dtype=np.float32), 0.0
        params = AllChem.ETKDGv3()
        params.randomSeed = 0
        params.maxAttempts = 1
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            raise RuntimeError("embed failed")
        coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
        center = coords.mean(axis=0, keepdims=True)
        centered = coords - center
        radial = np.linalg.norm(centered, axis=1)
        bbox = coords.max(axis=0) - coords.min(axis=0)
        vec = np.array(
            [
                rdMolDescriptors.CalcRadiusOfGyration(mol),
                rdMolDescriptors.CalcAsphericity(mol),
                rdMolDescriptors.CalcEccentricity(mol),
                rdMolDescriptors.CalcInertialShapeFactor(mol),
                rdMolDescriptors.CalcNPR1(mol),
                rdMolDescriptors.CalcNPR2(mol),
                rdMolDescriptors.CalcPMI1(mol),
                rdMolDescriptors.CalcPMI2(mol),
                rdMolDescriptors.CalcPMI3(mol),
                rdMolDescriptors.CalcSpherocityIndex(mol),
                float(radial.mean() + bbox.mean()),
            ],
            dtype=np.float32,
        )
        return vec, 1.0
    except Exception:
        return np.zeros((11,), dtype=np.float32), 0.0


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs(includeNeighbors=True)),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(int(atom.GetHybridization())),
        float(atom.GetMass() / 100.0),
        float(atom.GetTotalValence()),
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(int(bond.GetStereo())),
    ]


def graph_from_smiles(smiles: str) -> Data:
    mol = _base_mol(smiles)
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edge_pairs: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_pairs.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, 7), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)


def build_feature_cache(dataset: pd.DataFrame) -> dict[str, object]:
    graphs: list[Data] = []
    descriptors: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    context_layout: dict[str, object] | None = None
    led_desc: list[np.ndarray] = []
    led_mask: list[float] = []
    for smiles in dataset["canonical_smiles"].astype(str):
        graphs.append(graph_from_smiles(smiles))
        descriptors.append(descriptor_vector(smiles))
        context_vec, meta = build_mspce_multiscale_context(smiles)
        contexts.append(context_vec)
        if context_layout is None:
            context_layout = meta
        led_vec, led_ok = led_descriptor_vector(smiles)
        led_desc.append(led_vec)
        led_mask.append(led_ok)
    if context_layout is None:
        raise RuntimeError("context layout is empty")
    return {
        "graphs": graphs,
        "descriptors": torch.tensor(np.stack(descriptors), dtype=torch.float32),
        "contexts": torch.tensor(np.stack(contexts), dtype=torch.float32),
        "context_layout": context_layout,
        "led": torch.tensor(np.stack(led_desc), dtype=torch.float32),
        "led_mask": torch.tensor(np.array(led_mask), dtype=torch.float32).unsqueeze(1),
        "targets": torch.tensor(dataset["tg_k"].to_numpy(dtype=np.float32)).unsqueeze(1),
        "roles": dataset["role"].astype(str).tolist(),
        "sources": dataset["source_name"].astype(str).tolist(),
        "record_ids": dataset["record_id"].astype(str).tolist(),
        "canonical_smiles": dataset["canonical_smiles"].astype(str).tolist(),
    }


def main() -> int:
    dataset = pd.read_csv(ROOT / "data/dataset.csv")
    cache = build_feature_cache(dataset)
    torch.save(cache, ROOT / "data/features.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

