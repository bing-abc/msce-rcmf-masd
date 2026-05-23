from __future__ import annotations

"""Reusable building blocks for MSPCE, RCMF-style fusion, and MASD auxiliaries."""

import math

import torch
from torch import nn


def zero_last_linear(module: nn.Module) -> None:
    """Zero the last linear layer when a residual head should start conservatively."""
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class MSPCEEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        context_layout: dict[str, object] | None = None,
        top_k_active: int = 3,
    ) -> None:
        super().__init__()
        scales = []
        if context_layout is not None:
            scales = list(context_layout.get("scales", []))
        self._scale_slices: list[tuple[str, int, int]] = []
        self._scale_kinds: list[str] = []
        for scale in scales:
            name = str(scale.get("name", ""))
            kind = str(scale.get("kind", ""))
            start = int(scale.get("start", 0))
            end = int(scale.get("end", 0))
            if name and 0 <= start < end <= in_dim:
                self._scale_slices.append((name, start, end))
                self._scale_kinds.append(kind)
        self.use_multiscale = len(self._scale_slices) >= 2
        self.top_k_active = max(1, min(int(top_k_active), len(self._scale_slices))) if self._scale_slices else 1
        if self.use_multiscale:
            self.scale_encoders = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.LayerNorm(end - start),
                        nn.Linear(end - start, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for name, start, end in self._scale_slices
                }
            )
            self.scale_score = nn.Linear(hidden_dim, 1)
            self.scale_prior = nn.Parameter(torch.zeros(len(self._scale_slices)))
            self.fuse = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, hidden_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(x)["embedding"]

    def forward_with_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.use_multiscale:
            emb = self.net(x)
            weights = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
            scale_embeddings = emb.unsqueeze(1)
            return {
                "embedding": emb,
                "scale_weights": weights,
                "scale_embeddings": scale_embeddings,
                "chain_embedding": emb,
                "segment_embedding": emb,
                "graph_embedding": emb,
                "interpretable_embedding": emb,
            }
        embs = []
        for name, start, end in self._scale_slices:
            embs.append(self.scale_encoders[name](x[:, start:end]))
        stacked = torch.stack(embs, dim=1)
        logits = self.scale_score(stacked).squeeze(-1) + self.scale_prior.reshape(1, -1)
        dense_weights = torch.softmax(logits, dim=1)
        topk = torch.topk(logits, k=self.top_k_active, dim=1)
        active_mask = torch.zeros_like(logits, dtype=stacked.dtype)
        active_mask.scatter_(1, topk.indices, 1.0)
        sparse_weights = dense_weights * active_mask
        sparse_weights = sparse_weights / sparse_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        inactive_weights = dense_weights * (1.0 - active_mask)
        inactive_mass = inactive_weights.sum(dim=1, keepdim=True)
        inactive_weights = inactive_weights / inactive_mass.clamp_min(1e-6)
        active_embedding = (sparse_weights.unsqueeze(-1) * stacked).sum(dim=1)
        inactive_embedding = (inactive_weights.unsqueeze(-1) * stacked).sum(dim=1)
        pooled_active = (active_mask.unsqueeze(-1) * stacked).sum(dim=1) / active_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = self.fuse(torch.cat([active_embedding, pooled_active, 0.20 * inactive_embedding], dim=1))
        scale_entropy = -(dense_weights * torch.log(dense_weights.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        scale_entropy = scale_entropy / math.log(float(len(self._scale_slices)))

        def _pool_by_kind(accepted_kinds: set[str]) -> torch.Tensor:
            indices = [idx for idx, kind in enumerate(self._scale_kinds) if kind in accepted_kinds]
            if not indices:
                return active_embedding
            chosen = stacked[:, indices, :]
            chosen_weights = sparse_weights[:, indices]
            if torch.allclose(chosen_weights.sum(dim=1, keepdim=True), torch.zeros_like(chosen_weights.sum(dim=1, keepdim=True))):
                chosen_weights = dense_weights[:, indices]
            chosen_weights = chosen_weights / chosen_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return (chosen_weights.unsqueeze(-1) * chosen).sum(dim=1)

        return {
            "embedding": fused,
            "scale_weights": sparse_weights,
            "scale_dense_weights": dense_weights,
            "scale_embeddings": stacked,
            "active_embedding": active_embedding,
            "inactive_embedding": inactive_embedding,
            "active_scale_mask": active_mask,
            "active_scale_count": active_mask.sum(dim=1, keepdim=True),
            "inactive_scale_mass": inactive_mass,
            "scale_entropy": scale_entropy,
            "chain_embedding": _pool_by_kind({"smiles_chain", "smiles_segment"}),
            "segment_embedding": _pool_by_kind({"smiles_segment"}),
            "graph_embedding": _pool_by_kind({"graph_neighborhood"}),
            "interpretable_embedding": _pool_by_kind({"interpretable"}),
        }


class ExpertHead(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.unc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        unc = torch.nn.functional.softplus(self.unc_head(x)) + 1e-4
        return mean, unc


class TwoWayGate(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, zero_init: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        if zero_init:
            zero_last_linear(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


class ScalarGate(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        max_value: float = 1.0,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.max_value = float(max_value)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        if zero_init:
            zero_last_linear(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_value * torch.sigmoid(self.net(x))


class ResidualHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, zero_init: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        if zero_init:
            zero_last_linear(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MSPCEContextInjector(nn.Module):
    def __init__(self, hidden_dim: int, zero_init: bool = False, rank_dim: int = 16) -> None:
        super().__init__()
        self.rank_dim = int(rank_dim)
        self.concat_anchor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.chain_down = nn.Linear(hidden_dim, self.rank_dim, bias=False)
        self.graph_down = nn.Linear(hidden_dim, self.rank_dim, bias=False)
        self.segment_down = nn.Linear(hidden_dim, self.rank_dim, bias=False)
        self.active_down = nn.Linear(hidden_dim, self.rank_dim, bias=False)
        self.desc_scale_up = nn.Linear(self.rank_dim, hidden_dim, bias=False)
        self.desc_shift_up = nn.Linear(self.rank_dim, hidden_dim, bias=False)
        self.graph_scale_up = nn.Linear(self.rank_dim, hidden_dim, bias=False)
        self.graph_shift_up = nn.Linear(self.rank_dim, hidden_dim, bias=False)
        self.bottleneck_down = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.rank_dim),
        )
        self.bottleneck_up = nn.Linear(self.rank_dim, hidden_dim, bias=False)
        self.mod_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5),
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adapter_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5),
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.desc_scale_up.weight)
        nn.init.zeros_(self.desc_shift_up.weight)
        nn.init.zeros_(self.graph_scale_up.weight)
        nn.init.zeros_(self.graph_shift_up.weight)
        nn.init.zeros_(self.bottleneck_up.weight)
        if zero_init:
            zero_last_linear(self.mod_gate)
            zero_last_linear(self.adapter_gate)
        else:
            nn.init.zeros_(self.mod_gate[-1].weight)
            nn.init.zeros_(self.mod_gate[-1].bias)
            nn.init.zeros_(self.adapter_gate[-1].weight)
            nn.init.zeros_(self.adapter_gate[-1].bias)

    def forward(
        self,
        *,
        desc_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        ctx_emb: torch.Tensor,
        active_ctx: torch.Tensor,
        inactive_ctx: torch.Tensor,
        chain_ctx: torch.Tensor,
        graph_ctx: torch.Tensor,
        segment_ctx: torch.Tensor,
        baseline_rel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        concat_hidden = self.concat_anchor(torch.cat([desc_emb, graph_emb, ctx_emb], dim=1))
        rank_state = torch.tanh(
            self.chain_down(chain_ctx)
            + self.graph_down(graph_ctx)
            + 0.50 * self.segment_down(segment_ctx)
            + 0.75 * self.active_down(active_ctx)
        )
        mod_gate_input = torch.cat([concat_hidden, active_ctx, baseline_rel], dim=1)
        modulation_gate = 0.12 * torch.sigmoid(self.mod_gate(mod_gate_input) - 4.0)
        desc_scale = 1.0 + modulation_gate * 0.08 * torch.tanh(self.desc_scale_up(rank_state))
        graph_scale = 1.0 + modulation_gate * 0.08 * torch.tanh(self.graph_scale_up(rank_state))
        desc_shift = modulation_gate * 0.04 * torch.tanh(self.desc_shift_up(rank_state))
        graph_shift = modulation_gate * 0.04 * torch.tanh(self.graph_shift_up(rank_state))
        desc_mod = desc_emb * desc_scale + desc_shift
        graph_mod = graph_emb * graph_scale + graph_shift
        bottleneck_rank = torch.tanh(self.bottleneck_down(torch.cat([active_ctx, segment_ctx, graph_ctx], dim=1)))
        bottleneck = torch.tanh(self.bottleneck_up(bottleneck_rank))
        gate_input = torch.cat([concat_hidden, active_ctx, baseline_rel], dim=1)
        adapter_gate = 0.08 * torch.sigmoid(self.adapter_gate(gate_input) - 4.0)
        injected_hidden = concat_hidden + adapter_gate * bottleneck
        return {
            "concat_hidden": concat_hidden,
            "desc_mod": desc_mod,
            "graph_mod": graph_mod,
            "active_ctx": active_ctx,
            "inactive_ctx": inactive_ctx,
            "rank_state": rank_state,
            "bottleneck": bottleneck,
            "modulation_gate": modulation_gate,
            "adapter_gate": adapter_gate,
            "injected_hidden": injected_hidden,
            "hidden_shift": injected_hidden - concat_hidden,
        }


class RCMFTrustedFusion(nn.Module):
    def __init__(self, hidden_dim: int, rank_dim: int = 16, zero_init: bool = True) -> None:
        super().__init__()
        self.rank_dim = int(rank_dim)
        self.state_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5 + 3),
            nn.Linear(hidden_dim * 2 + 5 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.trust_head = nn.Linear(hidden_dim, 3)
        self.desc_rank = nn.Linear(hidden_dim, self.rank_dim, bias=False)
        self.graph_rank = nn.Linear(hidden_dim, self.rank_dim, bias=False)
        self.ctx_rank = nn.Linear(hidden_dim, self.rank_dim, bias=False)
        self.mix_up = nn.Sequential(
            nn.LayerNorm(self.rank_dim * 3),
            nn.Linear(self.rank_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3),
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5),
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if zero_init:
            zero_last_linear(self.mix_up)
            zero_last_linear(self.confidence_head)
            zero_last_linear(self.residual_gate)

    def forward(
        self,
        *,
        desc_hidden: torch.Tensor,
        graph_hidden: torch.Tensor,
        ctx_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        baseline_rel: torch.Tensor,
        conflict: torch.Tensor,
        uncertainty: torch.Tensor,
        anchor_stability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        signal_state = torch.cat(
            [anchor_hidden, ctx_hidden, baseline_rel, conflict, uncertainty, anchor_stability],
            dim=1,
        )
        controller_state = self.state_proj(signal_state)
        trust_weights = torch.softmax(self.trust_head(controller_state), dim=1)
        desc_rank = torch.tanh(self.desc_rank(desc_hidden))
        graph_rank = torch.tanh(self.graph_rank(graph_hidden))
        ctx_rank = torch.tanh(self.ctx_rank(ctx_hidden))
        pair_desc_ctx = trust_weights[:, 0:1] * desc_rank * ctx_rank
        pair_graph_ctx = trust_weights[:, 1:2] * graph_rank * ctx_rank
        pair_desc_graph = trust_weights[:, 2:3] * desc_rank * graph_rank
        mix_hidden = torch.tanh(self.mix_up(torch.cat([pair_desc_ctx, pair_graph_ctx, pair_desc_graph], dim=1)))
        confidence = torch.sigmoid(
            self.confidence_head(torch.cat([controller_state, conflict, uncertainty, 1.0 - anchor_stability], dim=1)))
        risk = torch.sigmoid(2.6 * (uncertainty + 0.8 * conflict + 0.6 * anchor_stability - 0.35))
        residual_gate = 0.06 * torch.sigmoid(
            self.residual_gate(torch.cat([controller_state, mix_hidden, baseline_rel], dim=1)) - 4.0
        )
        residual_gate = residual_gate * confidence * torch.clamp(1.0 - 0.65 * risk, min=0.10, max=1.0)
        trust_entropy = -(trust_weights * torch.log(trust_weights.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        trust_entropy = trust_entropy / math.log(3.0)
        selector_score = trust_weights.max(dim=1, keepdim=True).values
        trusted_hidden = anchor_hidden + residual_gate * mix_hidden
        return {
            "controller_state": controller_state,
            "trust_weights": trust_weights,
            "mix_hidden": mix_hidden,
            "confidence": confidence,
            "risk": risk,
            "residual_gate": residual_gate,
            "trust_entropy": trust_entropy,
            "selector_score": selector_score,
            "trusted_hidden": trusted_hidden,
        }


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LEDPrior(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pred_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        pred = self.pred_head(latent)
        return latent, pred


def reliability_features(preds: torch.Tensor, unc: torch.Tensor) -> torch.Tensor:
    parts = [preds, unc]
    diffs = []
    for i in range(preds.size(1)):
        for j in range(i + 1, preds.size(1)):
            diffs.append(torch.abs(preds[:, i : i + 1] - preds[:, j : j + 1]))
    if diffs:
        parts.append(torch.cat(diffs, dim=1))
    return torch.cat(parts, dim=1)

