from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

from tg_prediction_pipeline.schemas import GraphCache

_ATOM_NUMBER_BUCKETS = (0, 1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
_DEGREE_BUCKETS = (0, 1, 2, 3, 4, 5)
_FORMAL_CHARGE_BUCKETS = (-2, -1, 0, 1, 2)
_HYDROGEN_BUCKETS = (0, 1, 2, 3, 4)
_HYBRIDIZATION_BUCKETS = (
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
)
_BOND_TYPE_BUCKETS = (
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
)
_BOND_STEREO_BUCKETS = (
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
)


def local_graph_cache_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "graph_cache.pt"


def _one_hot_with_other(value: object, buckets: tuple[object, ...]) -> list[float]:
    vector = [0.0] * (len(buckets) + 1)
    try:
        vector[buckets.index(value)] = 1.0
    except ValueError:
        vector[-1] = 1.0
    return vector


def _atom_features(atom: Chem.Atom) -> list[float]:
    return [
        *_one_hot_with_other(atom.GetAtomicNum(), _ATOM_NUMBER_BUCKETS),
        *_one_hot_with_other(atom.GetDegree(), _DEGREE_BUCKETS),
        *_one_hot_with_other(atom.GetFormalCharge(), _FORMAL_CHARGE_BUCKETS),
        *_one_hot_with_other(atom.GetTotalNumHs(), _HYDROGEN_BUCKETS),
        *_one_hot_with_other(atom.GetHybridization(), _HYBRIDIZATION_BUCKETS),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetMass() / 200.0),
    ]


def _bond_features(bond: Chem.Bond) -> list[float]:
    return [
        *_one_hot_with_other(bond.GetBondType(), _BOND_TYPE_BUCKETS),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *_one_hot_with_other(bond.GetStereo(), _BOND_STEREO_BUCKETS),
    ]


def smiles_to_graph(smiles: str) -> Data:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"invalid canonical_smiles for graph featurization: {smiles}")

    node_features = [_atom_features(atom) for atom in molecule.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float32)

    edge_index_rows: list[list[int]] = []
    edge_features: list[list[float]] = []
    for bond in molecule.GetBonds():
        start = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        features = _bond_features(bond)
        edge_index_rows.append([start, end])
        edge_index_rows.append([end, start])
        edge_features.append(features)
        edge_features.append(features)

    edge_attr_dim = len(_bond_features(next(iter(molecule.GetBonds())))) if molecule.GetNumBonds() > 0 else len(_bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))
    if edge_index_rows:
        edge_index = torch.tensor(edge_index_rows, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, edge_attr_dim), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_graph_cache(canonical_smiles: list[str] | tuple[str, ...]) -> GraphCache:
    smiles_list = [str(item) for item in canonical_smiles]
    graphs = tuple(smiles_to_graph(smiles) for smiles in smiles_list)
    if graphs:
        metadata = {
            "n_samples": int(len(graphs)),
            "node_feature_dim": int(graphs[0].x.shape[1]),
            "edge_feature_dim": int(graphs[0].edge_attr.shape[1]),
        }
    else:
        metadata = {
            "n_samples": 0,
            "node_feature_dim": 0,
            "edge_feature_dim": 0,
        }
    return GraphCache(
        graphs=graphs,
        canonical_smiles=tuple(smiles_list),
        metadata=metadata,
    )


def save_graph_cache(graph_cache: GraphCache, output_path: str | Path | None = None) -> Path:
    path = Path(output_path) if output_path is not None else local_graph_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graphs": list(graph_cache.graphs),
            "canonical_smiles": list(graph_cache.canonical_smiles),
            "metadata": dict(graph_cache.metadata),
        },
        path,
    )
    return path


def load_graph_cache(graph_path: str | Path | None = None) -> GraphCache:
    path = Path(graph_path) if graph_path is not None else local_graph_cache_path()
    if not path.exists():
        raise FileNotFoundError(f"graph cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return GraphCache(
        graphs=tuple(payload["graphs"]),
        canonical_smiles=tuple(str(item) for item in payload["canonical_smiles"]),
        metadata=dict(payload.get("metadata", {})),
    )
