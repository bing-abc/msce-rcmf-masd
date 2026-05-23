from __future__ import annotations

"""Backbone encoders used by the multimodal experiment stack."""

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import AttentiveFP, GINEConv, global_max_pool, global_mean_pool


def build_mlp(dims: list[int], dropout: float = 0.1) -> nn.Sequential:
    """Small shared MLP builder for descriptor and graph branches."""
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DescriptorBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            build_mlp([in_dim, 256, hidden_dim], dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = build_mlp([hidden_dim, hidden_dim, hidden_dim], dropout=0.1)
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = torch.relu(self.node_proj(batch.x))
        for conv in self.convs:
            x = torch.relu(conv(x, batch.edge_index, batch.edge_attr))
        pooled = torch.cat([global_mean_pool(x, batch.batch), global_max_pool(x, batch.batch)], dim=1)
        return self.out_proj(pooled)


class AttentiveGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.model = AttentiveFP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            num_layers=3,
            num_timesteps=2,
            dropout=0.1,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


class BackboneBase(nn.Module):
    def __init__(self, desc_dim: int, node_dim: int, edge_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.descriptor_branch = DescriptorBranch(desc_dim, hidden_dim=hidden_dim, dropout=0.1)

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: Batch, descriptors: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "graph_emb": self.encode_graph(batch),
            "desc_emb": self.descriptor_branch(descriptors),
        }


class HybridBackbone(BackboneBase):
    def __init__(self, desc_dim: int, node_dim: int, edge_dim: int, hidden_dim: int = 128) -> None:
        super().__init__(desc_dim=desc_dim, node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.graph_encoder = HybridGraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        return self.graph_encoder(batch)


class AttentiveBackbone(BackboneBase):
    def __init__(self, desc_dim: int, node_dim: int, edge_dim: int, hidden_dim: int = 128) -> None:
        super().__init__(desc_dim=desc_dim, node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.graph_encoder = AttentiveGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
        )

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        return self.graph_encoder(batch)


def build_backbone(
    name: str,
    desc_dim: int,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int = 128,
) -> BackboneBase:
    if name == "hybrid_gine":
        return HybridBackbone(desc_dim=desc_dim, node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    if name == "attentive_fp":
        return AttentiveBackbone(desc_dim=desc_dim, node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    raise ValueError(f"unsupported backbone: {name}")
