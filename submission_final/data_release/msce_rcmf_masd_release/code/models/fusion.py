from __future__ import annotations

"""FusionModel and controllers for current and legacy experiment modes.

The current paper uses the MSPCE-RCMF-MASD path, but older ladder stages stay
here so ablations and archived diagnostics can be reproduced from one model.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.backbone import build_backbone
from models.modules import (
    LEDPrior,
    ExpertHead,
    MLPRegressor,
    MSPCEEncoder,
    MSPCEContextInjector,
    RCMFTrustedFusion,
    ResidualHead,
    ScalarGate,
    TwoWayGate,
    reliability_features,
)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Sparsemax is used where we want slot/controller sparsity instead of softmax spread.
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(shifted, dim=dim, descending=True).values
    range_shape = [1] * shifted.dim()
    range_shape[dim] = shifted.size(dim)
    steps = torch.arange(1, shifted.size(dim) + 1, device=shifted.device, dtype=shifted.dtype).view(range_shape)
    bound = 1.0 + steps * zs
    cumulative = torch.cumsum(zs, dim=dim)
    support = (bound > cumulative).to(shifted.dtype)
    k = support.sum(dim=dim, keepdim=True).clamp_min(1.0)
    tau = ((cumulative.gather(dim, (k.long() - 1).clamp_min(0))) - 1.0) / k
    output = torch.clamp(shifted - tau, min=0.0)
    return output / output.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class MSPCEFusionController(nn.Module):
    def __init__(self, hidden_dim: int, zero_init: bool = False) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.modality_head = nn.Linear(hidden_dim, 3)
        self.interaction_head = nn.Linear(hidden_dim, 3)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        if zero_init:
            nn.init.zeros_(self.modality_head.weight)
            nn.init.zeros_(self.modality_head.bias)
            nn.init.zeros_(self.interaction_head.weight)
            nn.init.zeros_(self.interaction_head.bias)
            nn.init.zeros_(self.alpha_head.weight)
            nn.init.zeros_(self.alpha_head.bias)

    def forward(self, h_mspce: torch.Tensor) -> dict[str, torch.Tensor]:
        core = self.backbone(h_mspce)
        modality_trust = torch.softmax(self.modality_head(core), dim=1)
        interaction_strength = torch.softmax(self.interaction_head(core), dim=1)
        alpha_controller = torch.sigmoid(self.alpha_head(core))
        return {
            "controller_core": core,
            "modality_trust": modality_trust,
            "interaction_strength": interaction_strength,
            "alpha_controller": alpha_controller,
        }


class AnchorMixController(nn.Module):
    def __init__(self, hidden_dim: int, zero_init: bool = False) -> None:
        super().__init__()
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5),
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
        )
        self.beta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.beta_head[-1].weight)
            nn.init.zeros_(self.beta_head[-1].bias)

    def forward(
        self,
        *,
        h_mspce: torch.Tensor,
        h_desc: torch.Tensor,
        h_graph: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        ctx_state = self.ctx_proj(h_mspce)
        desc_graph_state = torch.cat([h_desc, h_graph, reliability], dim=1)
        fusion_state = self.state_proj(desc_graph_state)
        beta_input = torch.cat([ctx_state, fusion_state], dim=1)
        beta = torch.sigmoid(self.beta_head(beta_input))
        return beta


class FusionModel(nn.Module):
    SUPPORTED_MASD_SLOT_COUNTS = (2, 3, 4, 6)

    def __init__(
        self,
        backbone_name: str,
        mode: str,
        desc_dim: int,
        ctx_dim: int,
        led_dim: int,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        safe_residual: bool = False,
        innovation_limit: float = 0.30,
        certification_rule: dict[str, object] | None = None,
        context_layout: dict[str, object] | None = None,
        masd_slot_count: int = 4,
    ) -> None:
        super().__init__()
        # `mode` keeps the historical ladder and the current paper chain on one code path.
        self.mode = mode
        self.safe_residual = bool(safe_residual)
        self.innovation_limit = float(innovation_limit)
        self.has_certification = certification_rule is not None

        self.backbone = build_backbone(backbone_name, desc_dim, node_dim, edge_dim, hidden_dim)
        self.ctx_encoder = MSPCEEncoder(
            ctx_dim,
            hidden_dim=hidden_dim,
            context_layout=context_layout,
        )

        self.desc_head = ExpertHead(hidden_dim=hidden_dim)
        self.graph_head = ExpertHead(hidden_dim=hidden_dim)
        self.baseline_head = ExpertHead(hidden_dim=hidden_dim)

        self.conflict_gate = TwoWayGate(
            in_dim=5,
            hidden_dim=hidden_dim,
            zero_init=self.safe_residual,
        )
        self.no_context_head = MLPRegressor(in_dim=hidden_dim * 2, hidden_dim=hidden_dim * 2)
        self.concat_head = MLPRegressor(in_dim=hidden_dim * 3, hidden_dim=hidden_dim * 2)
        self.static_head = MLPRegressor(in_dim=hidden_dim, hidden_dim=hidden_dim)
        self.static_logits = nn.Parameter(torch.zeros(3))
        self.mspce_context_injector = MSPCEContextInjector(hidden_dim=hidden_dim, zero_init=self.safe_residual)
        self.mspce_repair_delta_head = ResidualHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.mspce_repair_gate = ScalarGate(
            in_dim=hidden_dim + 5,
            hidden_dim=hidden_dim,
            max_value=0.10,
            zero_init=False,
        )
        self.rcmf_min_fusion = RCMFTrustedFusion(hidden_dim=hidden_dim, rank_dim=16, zero_init=True)
        self.rcmf_min_delta_head = ResidualHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.led_min_bridge = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 5),
            nn.Linear(hidden_dim * 3 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.led_min_down = nn.Linear(hidden_dim, 16, bias=False)
        self.led_min_up = nn.Linear(16, hidden_dim, bias=False)
        self.led_min_confidence = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.led_min_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5),
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.led_min_delta_head = ResidualHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.led_teacher_refine = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 5),
            nn.Linear(hidden_dim * 3 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.led_teacher_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.led_teacher_source_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.led_teacher_source_score = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.led_foundation_bank = nn.Embedding(4, hidden_dim)
        self.led_foundation_selector = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5),
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.led_foundation_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.led_unimol2_task_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.led_unimol2_task_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.led_unimol2_task_score = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.led_unimol2_aux_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.led_unimol2_aux_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.led_unimol2_aux_score = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.regime_summary = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.regime_posterior = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )
        self.regime_token_bank = nn.Embedding(4, hidden_dim)
        self.regime_teacher_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.regime_rcmf_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.regime_led_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.led_proto_bank = nn.Embedding(6, hidden_dim)
        self.led_proto_selector = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5),
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        self.led_proto_teacher_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.led_proto_relation = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.full_min_bridge = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 5),
            nn.Linear(hidden_dim * 4 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.full_min_down = nn.Linear(hidden_dim, 16, bias=False)
        self.full_min_up = nn.Linear(16, hidden_dim, bias=False)
        self.full_min_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 5),
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.full_min_delta_head = ResidualHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.full_min_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.full_shared_z_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 8),
            nn.Linear(hidden_dim * 2 + 8, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 12),
        )
        self.full_shared_hidden = nn.Linear(12, hidden_dim, bias=False)
        self.full_shared_rcmf_gate = nn.Linear(12, 1)
        self.full_shared_led_gate = nn.Linear(12, 1)
        self.full_shared_gate = nn.Linear(12, 1)
        self.full_shared_delta_head = ResidualHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.full_factor_ctx_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 8),
        )
        self.full_factor_rel_proj = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 8),
        )
        self.full_factor_ctx_hidden = nn.Linear(8, hidden_dim, bias=False)
        self.full_factor_rel_hidden = nn.Linear(8, hidden_dim, bias=False)
        self.full_factor_rcmf_gate = nn.Linear(16, 1)
        self.full_factor_led_gate = nn.Linear(8, 1)
        self.full_factor_gate = nn.Linear(16, 1)
        self.full_factor_delta_head = ResidualHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.full_stage_u_trigger = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.full_stage_led_trigger = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.full_stage_u_mag = nn.Linear(16, 1)
        self.full_stage_led_mag = nn.Linear(8, 1)
        self.full_stage_gate = nn.Linear(16, 1)
        self.full_bucket_embed = nn.Embedding(4, 8)
        self.full_bucket_hidden = nn.Linear(8, hidden_dim, bias=False)
        self.full_bucket_rcmf_gate = nn.Linear(8, 1)
        self.full_bucket_led_gate = nn.Linear(8, 1)
        self.full_bucket_gate = nn.Linear(8, 1)
        nn.init.zeros_(self.led_min_up.weight)
        nn.init.zeros_(self.led_min_gate[-1].weight)
        nn.init.zeros_(self.led_min_gate[-1].bias)
        nn.init.zeros_(self.led_min_confidence[-1].weight)
        nn.init.zeros_(self.led_min_confidence[-1].bias)
        nn.init.zeros_(self.full_min_up.weight)
        nn.init.zeros_(self.full_min_gate[-1].weight)
        nn.init.zeros_(self.full_min_gate[-1].bias)
        nn.init.zeros_(self.full_min_mix[-1].weight)
        nn.init.zeros_(self.full_min_mix[-1].bias)
        nn.init.zeros_(self.full_shared_hidden.weight)
        nn.init.zeros_(self.full_shared_rcmf_gate.weight)
        nn.init.zeros_(self.full_shared_rcmf_gate.bias)
        nn.init.zeros_(self.full_shared_led_gate.weight)
        nn.init.zeros_(self.full_shared_led_gate.bias)
        nn.init.zeros_(self.full_shared_gate.weight)
        nn.init.zeros_(self.full_shared_gate.bias)
        nn.init.zeros_(self.full_factor_ctx_hidden.weight)
        nn.init.zeros_(self.full_factor_rel_hidden.weight)
        nn.init.zeros_(self.full_factor_rcmf_gate.weight)
        nn.init.zeros_(self.full_factor_rcmf_gate.bias)
        nn.init.zeros_(self.full_factor_led_gate.weight)
        nn.init.zeros_(self.full_factor_led_gate.bias)
        nn.init.zeros_(self.full_factor_gate.weight)
        nn.init.zeros_(self.full_factor_gate.bias)
        nn.init.zeros_(self.full_stage_u_trigger[-1].weight)
        nn.init.zeros_(self.full_stage_u_trigger[-1].bias)
        nn.init.zeros_(self.full_stage_led_trigger[-1].weight)
        nn.init.zeros_(self.full_stage_led_trigger[-1].bias)
        nn.init.zeros_(self.full_stage_u_mag.weight)
        nn.init.zeros_(self.full_stage_u_mag.bias)
        nn.init.zeros_(self.full_stage_led_mag.weight)
        nn.init.zeros_(self.full_stage_led_mag.bias)
        nn.init.zeros_(self.full_stage_gate.weight)
        nn.init.zeros_(self.full_stage_gate.bias)
        nn.init.zeros_(self.full_bucket_hidden.weight)
        nn.init.zeros_(self.full_bucket_rcmf_gate.weight)
        nn.init.zeros_(self.full_bucket_rcmf_gate.bias)
        nn.init.zeros_(self.full_bucket_led_gate.weight)
        nn.init.zeros_(self.full_bucket_led_gate.bias)
        nn.init.zeros_(self.full_bucket_gate.weight)
        nn.init.zeros_(self.full_bucket_gate.bias)

        self.ctx_delta_head = ResidualHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zero_init=self.safe_residual,
        )
        self.rcmf_gate = ScalarGate(
            in_dim=5 + hidden_dim,
            hidden_dim=hidden_dim,
            max_value=1.0,
            zero_init=self.safe_residual,
        )
        self.ctx_scale_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dynamic_fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, 4),
        )
        self.fusion_temperature = 1.6
        self.fusion_uniform_mix = 0.20
        self.fuse_desc_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fuse_graph_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fuse_ctx_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fuse_led_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.ctx_to_desc_film = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.ctx_to_graph_film = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.anchor_mix_controller = AnchorMixController(hidden_dim=hidden_dim, zero_init=self.safe_residual)
        self.mspce_fusion_controller = MSPCEFusionController(hidden_dim=hidden_dim, zero_init=self.safe_residual)
        self.controller_pair_dg = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.controller_pair_dl = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.controller_pair_gl = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ctx_query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.interaction_key_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.interaction_value_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.interaction_gate = ScalarGate(
            in_dim=hidden_dim * 2 + 5,
            hidden_dim=hidden_dim,
            max_value=1.0,
            zero_init=self.safe_residual,
        )
        self.led_proxy = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_delta_head = ResidualHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zero_init=self.safe_residual,
        )
        self.fusion_alpha_gate = ScalarGate(
            in_dim=hidden_dim + 5,
            hidden_dim=hidden_dim,
            max_value=0.65,
            zero_init=self.safe_residual,
        )
        self.dynamic_control = ScalarGate(
            in_dim=hidden_dim * 4 + 5,
            hidden_dim=hidden_dim,
            max_value=1.0,
            zero_init=self.safe_residual,
        )
        self.dynamic_head = MLPRegressor(in_dim=hidden_dim, hidden_dim=hidden_dim * 2)
        self.masd_slot_count = self._resolve_masd_slot_count(masd_slot_count)
        self.masd_slot_temperature = 0.70
        # The slot bank gives MASD a small, bounded set of signed correction channels.
        self.masd_core_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 5),
            nn.Linear(hidden_dim * 4 + 5, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.masd_slot_bank = nn.Embedding(self.masd_slot_count, hidden_dim)
        self.masd_slot_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.masd_alpha_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.masd_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.masd_base_term_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.masd_mag_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.masd_res_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.masd_calib_context_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.masd_slot_count),
        )
        self.masd_calib_proxy_scale = nn.Parameter(torch.ones(self.masd_slot_count))
        self.masd_calib_proxy_bias = nn.Parameter(torch.zeros(self.masd_slot_count))
        self.masd_safety_gate = ScalarGate(
            in_dim=hidden_dim + 11,
            hidden_dim=hidden_dim,
            max_value=1.0,
            zero_init=True,
        )
        self.masd_safety_gate_v3 = ScalarGate(
            in_dim=hidden_dim + 12,
            hidden_dim=hidden_dim,
            max_value=1.0,
            zero_init=True,
        )
        self.masd_alpha_risk_weights = nn.Parameter(torch.tensor([1.10, 1.00, 0.70, 0.80], dtype=torch.float32))
        self.masd_alpha_risk_bias = nn.Parameter(torch.tensor(-2.10, dtype=torch.float32))
        self.masd_gate_context_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 5),
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.masd_gate_risk_weights = nn.Parameter(torch.tensor([1.20, 1.05, 0.85, 0.90, 0.65], dtype=torch.float32))
        self.masd_gate_bias = nn.Parameter(torch.tensor(1.05, dtype=torch.float32))

        self.masd_gate_low = 0.02
        self.masd_gate_high = 0.55
        self.register_buffer(
            "masd_sign_prior",
            self._build_masd_sign_prior(self.masd_slot_count),
        )
        self.masd_main_mag_floor = 0.09 * self.innovation_limit
        self.masd_main_mag_cap = 0.26 * self.innovation_limit
        self.masd_residual_cap = 0.045 * self.innovation_limit
        self.led_prior = LEDPrior(led_dim, hidden_dim=hidden_dim)
        if certification_rule is None:
            self.register_buffer("cert_mean", torch.zeros(5))
            self.register_buffer("cert_std", torch.ones(5))
            self.register_buffer("cert_weights", torch.zeros(5))
            self.cert_bias = 0.0
            self.cert_positive_threshold = 0.0
            self.cert_negative_threshold = 0.0
        else:
            self.register_buffer(
                "cert_mean",
                torch.tensor(certification_rule["feature_mean"], dtype=torch.float32),
            )
            self.register_buffer(
                "cert_std",
                torch.tensor(certification_rule["feature_std"], dtype=torch.float32),
            )
            self.register_buffer(
                "cert_weights",
                torch.tensor(certification_rule["weights"], dtype=torch.float32),
            )
            self.cert_bias = float(certification_rule["bias"])
            self.cert_positive_threshold = float(certification_rule["positive_threshold"])
            self.cert_negative_threshold = float(certification_rule["negative_threshold"])

    @classmethod
    def _resolve_masd_slot_count(cls, masd_slot_count: int) -> int:
        slot_count = int(masd_slot_count)
        if slot_count not in cls.SUPPORTED_MASD_SLOT_COUNTS:
            raise ValueError(
                f"unsupported MASD slot count {slot_count}; expected one of {cls.SUPPORTED_MASD_SLOT_COUNTS}"
            )
        return slot_count

    @staticmethod
    def _build_masd_sign_prior(slot_count: int) -> torch.Tensor:
        sign_prior_map = {
            2: [1.0, -1.0],
            3: [1.0, 1.0, -1.0],
            4: [1.0, 0.65, -1.0, -1.0],
            6: [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
        }
        return torch.tensor(sign_prior_map[int(slot_count)], dtype=torch.float32)

    def _expand_masd_proxy_scores(self, base_scores: torch.Tensor) -> torch.Tensor:
        if self.masd_slot_count == 4:
            return base_scores
        pos_1 = base_scores[:, 0:1]
        pos_2 = base_scores[:, 1:2]
        neg_1 = base_scores[:, 2:3]
        neg_2 = base_scores[:, 3:4]
        pos_mean = 0.5 * (pos_1 + pos_2)
        neg_mean = 0.5 * (neg_1 + neg_2)
        if self.masd_slot_count == 2:
            return torch.cat([pos_mean, neg_mean], dim=1)
        if self.masd_slot_count == 3:
            return torch.cat([pos_1, pos_2, neg_mean], dim=1)
        if self.masd_slot_count == 6:
            return torch.cat([pos_1, pos_2, pos_mean, neg_1, neg_2, neg_mean], dim=1)
        raise RuntimeError(f"unexpected MASD slot count {self.masd_slot_count}")

    @staticmethod
    def _normalized_pr_hard_score(
        *,
        desc_pred: torch.Tensor,
        graph_pred: torch.Tensor,
        desc_unc: torch.Tensor,
        graph_unc: torch.Tensor,
        conflict_score: torch.Tensor,
    ) -> torch.Tensor:
        raw = torch.abs(desc_pred - graph_pred) + desc_unc + graph_unc + conflict_score
        raw_detached = raw.detach()
        raw_min = raw_detached.min()
        raw_span = (raw_detached.max() - raw_min).clamp_min(1e-6)
        return torch.clamp((raw_detached - raw_min) / raw_span, min=0.0, max=1.0)

    def certification_features(
        self,
        *,
        desc_pred: torch.Tensor,
        graph_pred: torch.Tensor,
        baseline_unc: torch.Tensor,
        ctx_delta: torch.Tensor,
        student_baseline_pred: torch.Tensor,
        teacher_anchor: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                torch.abs(desc_pred - graph_pred),
                baseline_unc,
                torch.abs(ctx_delta),
                torch.abs(student_baseline_pred - teacher_anchor),
                torch.abs(student_baseline_pred + ctx_delta - teacher_anchor),
            ],
            dim=1,
        )

    def certification_score(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.cert_mean.reshape(1, -1)) / self.cert_std.reshape(1, -1)
        score = normalized @ self.cert_weights.reshape(-1, 1) + self.cert_bias
        return score

    def _masd_proxy_scores(
        self,
        descriptors: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        batch = descriptors.shape[0]
        zeros = torch.zeros((batch, 1), dtype=descriptors.dtype, device=descriptors.device)

        def _take(tensor: torch.Tensor, index: int) -> torch.Tensor:
            if index < tensor.shape[1]:
                return tensor[:, index : index + 1]
            return zeros

        interpretable = contexts[:, 960:980] if contexts.shape[1] >= 980 else torch.zeros((batch, 20), dtype=contexts.dtype, device=contexts.device)

        molwt = _take(descriptors, 0)
        logp = _take(descriptors, 1)
        tpsa = _take(descriptors, 2)
        hba = _take(descriptors, 4)
        hbd = _take(descriptors, 5)
        rot = _take(descriptors, 6)
        ring = _take(descriptors, 7)
        aromatic_ring = _take(descriptors, 8)
        kappa1 = _take(descriptors, 14)
        branch_ratio = _take(interpretable, 2)
        aromatic_ratio = _take(interpretable, 12)
        hetero_ratio = _take(interpretable, 13)
        fraction_csp3 = _take(interpretable, 15)
        hba_ctx = _take(interpretable, 16)
        hbd_ctx = _take(interpretable, 17)
        tpsa_ctx = _take(interpretable, 18)
        molwt_ctx = _take(interpretable, 19)
        mean_degree = _take(interpretable, 9)
        ring_count = _take(interpretable, 11)
        star_ratio = _take(interpretable, 5)

        rigidity_rotation = (
            0.42 * aromatic_ring
            + 0.28 * ring
            + 0.22 * kappa1
            + 0.24 * aromatic_ratio
            + 0.18 * mean_degree
            - 0.52 * rot
            - 0.18 * fraction_csp3
        )
        intermolecular_polarity = (
            0.46 * tpsa
            + 0.30 * hba
            + 0.26 * hbd
            + 0.28 * hetero_ratio
            + 0.18 * hba_ctx
            + 0.18 * hbd_ctx
            + 0.20 * tpsa_ctx
            - 0.10 * logp
        )
        freevolume_packing = (
            0.32 * molwt
            + 0.24 * molwt_ctx
            + 0.26 * ring_count
            + 0.16 * mean_degree
            + 0.16 * aromatic_ratio
            - 0.26 * branch_ratio
            - 0.20 * fraction_csp3
        )
        sidechain_internalplasticization = (
            0.52 * rot
            + 0.34 * branch_ratio
            + 0.28 * fraction_csp3
            + 0.18 * star_ratio
            + 0.12 * logp
            - 0.28 * ring
            - 0.22 * aromatic_ratio
        )
        base_scores = torch.cat(
            [
                rigidity_rotation,
                intermolecular_polarity,
                freevolume_packing,
                sidechain_internalplasticization,
            ],
            dim=1,
        )
        scores = self._expand_masd_proxy_scores(base_scores)
        scores = scores - scores.mean(dim=1, keepdim=True)
        scores = scores / scores.std(dim=1, keepdim=True).clamp_min(1e-6)
        return scores

    def _masd_forward(
        self,
        parts: dict[str, torch.Tensor],
        descriptors: torch.Tensor,
        contexts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor_pred = parts["rcmf_min_pred"]
        anchor_hidden = parts["rcmf_min_hidden"]
        proxy_scores = self._masd_proxy_scores(descriptors, contexts)
        proxy_target = torch.softmax(proxy_scores / 0.80, dim=1)
        core_input = torch.cat(
            [
                anchor_hidden,
                parts["ctx_chain_emb"],
                parts["ctx_graph_emb"],
                parts["ctx_segment_emb"],
                parts["baseline_rel"],
            ],
            dim=1,
        )
        mechanism_core = self.masd_core_proj(core_input)
        slot_bank = self.masd_slot_bank.weight.unsqueeze(0).expand(mechanism_core.shape[0], -1, -1)
        slot_hidden = torch.tanh(
            self.masd_slot_proj(
                torch.cat([mechanism_core.unsqueeze(1).expand(-1, self.masd_slot_count, -1), slot_bank], dim=2)
            )
        )
        alpha_logits = self.masd_alpha_head(slot_hidden).squeeze(-1)
        if self.mode == "main_core_sci2_masd_no_competition":
            alpha = torch.full_like(alpha_logits, 1.0 / float(self.masd_slot_count))
        else:
            alpha = torch.softmax((alpha_logits + 0.30 * proxy_scores) / self.masd_slot_temperature, dim=1)
        delta_raw = self.masd_delta_head(slot_hidden).squeeze(-1)
        delta = 0.55 * self.innovation_limit * torch.tanh(delta_raw / max(self.innovation_limit, 1e-6))
        base_delta = 0.16 * self.innovation_limit * torch.tanh(
            self.masd_base_term_head(torch.cat([mechanism_core, parts["baseline_rel"]], dim=1))
            / max(self.innovation_limit, 1e-6)
        )
        contribution = alpha * delta
        pred = anchor_pred + base_delta + contribution.sum(dim=1, keepdim=True)
        entropy = -(alpha * torch.log(alpha.clamp_min(1e-8))).sum(dim=1, keepdim=True) / math.log(float(self.masd_slot_count))
        slot_norm = F.normalize(slot_hidden, dim=2)
        pairwise = torch.matmul(slot_norm, slot_norm.transpose(1, 2))
        eye = torch.eye(self.masd_slot_count, dtype=slot_hidden.dtype, device=slot_hidden.device).unsqueeze(0)
        offdiag = pairwise * (1.0 - eye)
        diversity = 1.0 - offdiag.abs().sum(dim=(1, 2), keepdim=True) / float(self.masd_slot_count * (self.masd_slot_count - 1))
        dominant_mechanism = alpha.argmax(dim=1)
        payload = dict(parts)
        payload.update(
            {
                "pred": pred,
                "baseline_pred": anchor_pred,
                "teacher_anchor_pred": anchor_pred,
                "innovation_pred": pred,
                "innovation_score": torch.max(alpha, dim=1, keepdim=True).values * contribution.abs().sum(dim=1, keepdim=True),
                "fused_latent": mechanism_core,
                "baseline_unc": parts["baseline_unc"],
                "rcmf_gate": parts["rcmf_min_gate"],
                "ctx_delta": pred - anchor_pred,
                "mspce_anchor_pred": parts["mspce_repair_pred"],
                "masd_anchor_pred": anchor_pred,
                "masd_base_delta": base_delta,
                "masd_alpha_logits": alpha_logits,
                "masd_alpha": alpha,
                "masd_delta": delta,
                "masd_contribution": contribution,
                "masd_proxy_scores": proxy_scores,
                "masd_proxy_target": proxy_target,
                "masd_slot_hidden": slot_hidden,
                "masd_entropy": entropy,
                "masd_diversity": diversity,
                "masd_dominant_mechanism": dominant_mechanism,
            }
        )
        return payload

    def _masd_forward_v2(
        self,
        parts: dict[str, torch.Tensor],
        descriptors: torch.Tensor,
        contexts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor_pred = parts["rcmf_min_pred"]
        anchor_hidden = parts["rcmf_min_hidden"]
        proxy_scores = self._masd_proxy_scores(descriptors, contexts)
        proxy_target = torch.softmax(proxy_scores / 0.80, dim=1)
        sign_prior = self.masd_sign_prior.reshape(1, -1).to(proxy_scores)
        signed_proxy_target = proxy_target * sign_prior
        core_input = torch.cat(
            [
                anchor_hidden,
                parts["ctx_chain_emb"],
                parts["ctx_graph_emb"],
                parts["ctx_segment_emb"],
                parts["baseline_rel"],
            ],
            dim=1,
        )
        mechanism_core = self.masd_core_proj(core_input)
        slot_bank = self.masd_slot_bank.weight.unsqueeze(0).expand(mechanism_core.shape[0], -1, -1)
        slot_hidden = torch.tanh(
            self.masd_slot_proj(
                torch.cat([mechanism_core.unsqueeze(1).expand(-1, self.masd_slot_count, -1), slot_bank], dim=2)
            )
        )
        alpha_logits = self.masd_alpha_head(slot_hidden).squeeze(-1)
        alpha = torch.softmax((alpha_logits + 0.34 * proxy_scores) / self.masd_slot_temperature, dim=1)
        mag_raw = self.masd_mag_head(slot_hidden).squeeze(-1)
        res_raw = self.masd_res_head(slot_hidden).squeeze(-1)
        main_mag = self.masd_main_mag_floor + torch.clamp(F.softplus(mag_raw), max=self.masd_main_mag_cap)
        residual = self.masd_residual_cap * torch.tanh(res_raw)
        if self.mode == "main_core_sci2_masd_v2_no_sign_prior":
            delta = torch.clamp(main_mag * torch.tanh(mag_raw) + residual, min=-self.masd_main_mag_cap, max=self.masd_main_mag_cap)
        else:
            delta = sign_prior * main_mag + residual
        contribution = alpha * delta
        entropy = -(alpha * torch.log(alpha.clamp_min(1e-8))).sum(dim=1, keepdim=True) / math.log(float(self.masd_slot_count))
        alpha_max = alpha.max(dim=1, keepdim=True).values
        risk_features = torch.cat(
            [
                mechanism_core,
                parts["baseline_rel"],
                parts["conflict_level"],
                parts["uncertainty_level"],
                parts["rcmf_min_risk"],
                parts["rcmf_min_confidence"],
                entropy,
                1.0 - alpha_max,
            ],
            dim=1,
        )
        gate_learned = self.masd_safety_gate(risk_features)
        gate_heuristic = torch.sigmoid(
            2.2
            - 1.05 * parts["conflict_level"]
            - 1.15 * parts["uncertainty_level"]
            - 0.95 * entropy
            - 1.10 * (1.0 - alpha_max)
            - 0.80 * parts["rcmf_min_risk"]
        )
        if self.mode == "main_core_sci2_masd_v2_no_safety_gate":
            gate = torch.ones_like(gate_learned)
        else:
            gate = gate_learned * gate_heuristic
        contribution_sum = contribution.sum(dim=1, keepdim=True)
        pred = anchor_pred + gate * contribution_sum
        slot_norm = F.normalize(slot_hidden, dim=2)
        pairwise = torch.matmul(slot_norm, slot_norm.transpose(1, 2))
        eye = torch.eye(self.masd_slot_count, dtype=slot_hidden.dtype, device=slot_hidden.device).unsqueeze(0)
        offdiag = pairwise * (1.0 - eye)
        diversity = 1.0 - offdiag.abs().sum(dim=(1, 2), keepdim=True) / float(self.masd_slot_count * (self.masd_slot_count - 1))
        dominant_mechanism = alpha.argmax(dim=1)
        payload = dict(parts)
        payload.update(
            {
                "pred": pred,
                "baseline_pred": anchor_pred,
                "teacher_anchor_pred": anchor_pred,
                "innovation_pred": pred,
                "innovation_score": gate * contribution.abs().sum(dim=1, keepdim=True),
                "fused_latent": mechanism_core,
                "baseline_unc": parts["baseline_unc"],
                "rcmf_gate": parts["rcmf_min_gate"],
                "ctx_delta": pred - anchor_pred,
                "mspce_anchor_pred": parts["mspce_repair_pred"],
                "masd_anchor_pred": anchor_pred,
                "masd_base_delta": torch.zeros_like(anchor_pred),
                "masd_alpha_logits": alpha_logits,
                "masd_alpha": alpha,
                "masd_delta": delta,
                "masd_contribution": contribution,
                "masd_proxy_scores": proxy_scores,
                "masd_proxy_target": proxy_target,
                "masd_signed_proxy_target": signed_proxy_target,
                "masd_slot_hidden": slot_hidden,
                "masd_entropy": entropy,
                "masd_diversity": diversity,
                "masd_dominant_mechanism": dominant_mechanism,
                "masd_gate": gate,
                "masd_gate_learned": gate_learned,
                "masd_gate_heuristic": gate_heuristic,
                "masd_alpha_max": alpha_max,
                "masd_residual_delta": residual,
                "masd_main_mag": main_mag,
                "masd_sign_prior": sign_prior.expand_as(alpha),
            }
        )
        return payload

    def _masd_forward_v3(
        self,
        parts: dict[str, torch.Tensor],
        descriptors: torch.Tensor,
        contexts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor_pred = parts["rcmf_min_pred"]
        anchor_hidden = parts["rcmf_min_hidden"]
        proxy_scores = self._masd_proxy_scores(descriptors, contexts)
        proxy_target = torch.softmax(proxy_scores / 0.80, dim=1)
        sign_prior = self.masd_sign_prior.reshape(1, -1).to(proxy_scores)
        signed_proxy_target = proxy_target * sign_prior
        core_input = torch.cat(
            [
                anchor_hidden,
                parts["ctx_chain_emb"],
                parts["ctx_graph_emb"],
                parts["ctx_segment_emb"],
                parts["baseline_rel"],
            ],
            dim=1,
        )
        mechanism_core = self.masd_core_proj(core_input)
        slot_bank = self.masd_slot_bank.weight.unsqueeze(0).expand(mechanism_core.shape[0], -1, -1)
        slot_hidden = torch.tanh(
            self.masd_slot_proj(
                torch.cat([mechanism_core.unsqueeze(1).expand(-1, self.masd_slot_count, -1), slot_bank], dim=2)
            )
        )
        alpha_logits = self.masd_alpha_head(slot_hidden).squeeze(-1) + 0.38 * proxy_scores
        if self.mode == "main_core_sci2_masd_v3_no_sparse_alpha":
            alpha = torch.softmax(alpha_logits / self.masd_slot_temperature, dim=1)
        else:
            alpha = sparsemax(alpha_logits / self.masd_slot_temperature, dim=1)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        alpha_sorted, _ = torch.sort(alpha, dim=1, descending=True)
        alpha_max = alpha_sorted[:, :1]
        alpha_margin = alpha_sorted[:, :1] - alpha_sorted[:, 1:2]
        entropy = -(alpha * torch.log(alpha.clamp_min(1e-8))).sum(dim=1, keepdim=True) / math.log(float(self.masd_slot_count))

        res_raw = self.masd_res_head(slot_hidden).squeeze(-1)
        residual = self.masd_residual_cap * torch.tanh(res_raw)
        if self.mode == "main_core_sci2_masd_v3_no_monotonic_calibrator":
            mag_raw = self.masd_mag_head(slot_hidden).squeeze(-1)
            main_mag = self.masd_main_mag_floor + torch.clamp(F.softplus(mag_raw), max=self.masd_main_mag_cap)
            delta = sign_prior * main_mag + residual
        else:
            calib_context = 0.18 * torch.tanh(self.masd_calib_context_head(mechanism_core))
            proxy_scale = F.softplus(self.masd_calib_proxy_scale).reshape(1, -1)
            calib_logits = proxy_scale * proxy_scores + calib_context + self.masd_calib_proxy_bias.reshape(1, -1)
            calib = self.masd_main_mag_floor + (self.masd_main_mag_cap - self.masd_main_mag_floor) * torch.sigmoid(calib_logits)
            main_mag = calib
            delta = sign_prior * calib + residual

        contribution = alpha * delta
        contribution_sum = contribution.sum(dim=1, keepdim=True)
        mechanism_disagreement = torch.std(contribution, dim=1, keepdim=True) / max(self.innovation_limit, 1e-6)
        risk_features = torch.cat(
            [
                mechanism_core,
                parts["baseline_rel"],
                parts["conflict_level"],
                parts["uncertainty_level"],
                parts["rcmf_min_risk"],
                parts["rcmf_min_confidence"],
                entropy,
                1.0 - alpha_max,
                mechanism_disagreement,
            ],
            dim=1,
        )
        gate_learned = self.masd_safety_gate_v3(risk_features)
        gate_heuristic = torch.sigmoid(
            2.55
            - 1.30 * parts["uncertainty_level"]
            - 1.15 * parts["conflict_level"]
            - 1.00 * entropy
            - 0.85 * mechanism_disagreement
            - 1.10 * (1.0 - alpha_max)
            - 0.75 * parts["rcmf_min_risk"]
        )
        if self.mode == "main_core_sci2_masd_v3_no_uncertainty_gate":
            gate = torch.ones_like(gate_learned)
        else:
            gate = torch.clamp(gate_learned * gate_heuristic, min=0.0, max=1.0)
        pred = anchor_pred + gate * contribution_sum

        slot_norm = F.normalize(slot_hidden, dim=2)
        pairwise = torch.matmul(slot_norm, slot_norm.transpose(1, 2))
        eye = torch.eye(self.masd_slot_count, dtype=slot_hidden.dtype, device=slot_hidden.device).unsqueeze(0)
        offdiag = pairwise * (1.0 - eye)
        diversity = 1.0 - offdiag.abs().sum(dim=(1, 2), keepdim=True) / float(self.masd_slot_count * (self.masd_slot_count - 1))
        dominant_mechanism = alpha.argmax(dim=1)
        payload = dict(parts)
        payload.update(
            {
                "pred": pred,
                "baseline_pred": anchor_pred,
                "teacher_anchor_pred": anchor_pred,
                "innovation_pred": pred,
                "innovation_score": gate * contribution.abs().sum(dim=1, keepdim=True),
                "fused_latent": mechanism_core,
                "baseline_unc": parts["baseline_unc"],
                "rcmf_gate": parts["rcmf_min_gate"],
                "ctx_delta": pred - anchor_pred,
                "mspce_anchor_pred": parts["mspce_repair_pred"],
                "masd_anchor_pred": anchor_pred,
                "masd_base_delta": torch.zeros_like(anchor_pred),
                "masd_alpha_logits": alpha_logits,
                "masd_alpha": alpha,
                "masd_delta": delta,
                "masd_contribution": contribution,
                "masd_proxy_scores": proxy_scores,
                "masd_proxy_target": proxy_target,
                "masd_signed_proxy_target": signed_proxy_target,
                "masd_slot_hidden": slot_hidden,
                "masd_entropy": entropy,
                "masd_diversity": diversity,
                "masd_dominant_mechanism": dominant_mechanism,
                "masd_gate": gate,
                "masd_gate_learned": gate_learned,
                "masd_gate_heuristic": gate_heuristic,
                "masd_alpha_max": alpha_max,
                "masd_alpha_margin": alpha_margin,
                "masd_residual_delta": residual,
                "masd_main_mag": main_mag,
                "masd_sign_prior": sign_prior.expand_as(alpha),
                "masd_mechanism_disagreement": mechanism_disagreement,
            }
        )
        return payload

    def _masd_forward_current(
        self,
        parts: dict[str, torch.Tensor],
        descriptors: torch.Tensor,
        contexts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Current MASD path: anchor on the fusion bridge, then apply bounded signed correction.
        disable_rcmf_anchor = bool(getattr(self, "pr_disable_rcmf_anchor", False))
        if disable_rcmf_anchor:
            anchor_pred = parts["mspce_repair_pred"]
            anchor_hidden = parts["mspce_repair_hidden"]
            anchor_risk = torch.sigmoid(
                2.2 * (parts["uncertainty_level"] + 0.80 * parts["conflict_level"] - 0.72)
            )
            anchor_confidence = torch.clamp(1.0 - 0.55 * anchor_risk, min=0.10, max=1.0)
        else:
            anchor_pred = parts["rcmf_min_pred"]
            anchor_hidden = parts["rcmf_min_hidden"]
            anchor_risk = parts["rcmf_min_risk"]
            anchor_confidence = parts["rcmf_min_confidence"]
        proxy_scores = self._masd_proxy_scores(descriptors, contexts)
        proxy_target = torch.softmax(proxy_scores / 0.80, dim=1)
        sign_prior = self.masd_sign_prior.reshape(1, -1).to(proxy_scores)
        signed_proxy_target = proxy_target * sign_prior
        core_input = torch.cat(
            [
                anchor_hidden,
                parts["ctx_chain_emb"],
                parts["ctx_graph_emb"],
                parts["ctx_segment_emb"],
                parts["baseline_rel"],
            ],
            dim=1,
        )
        mechanism_core = self.masd_core_proj(core_input)
        slot_bank = self.masd_slot_bank.weight.unsqueeze(0).expand(mechanism_core.shape[0], -1, -1)
        slot_hidden = torch.tanh(
            self.masd_slot_proj(
                torch.cat([mechanism_core.unsqueeze(1).expand(-1, self.masd_slot_count, -1), slot_bank], dim=2)
            )
        )
        alpha_logits = self.masd_alpha_head(slot_hidden).squeeze(-1) + 0.38 * proxy_scores

        slot_norm = F.normalize(slot_hidden, dim=2)
        pairwise = torch.matmul(slot_norm, slot_norm.transpose(1, 2))
        eye = torch.eye(self.masd_slot_count, dtype=slot_hidden.dtype, device=slot_hidden.device).unsqueeze(0)
        offdiag = pairwise * (1.0 - eye)
        diversity = 1.0 - offdiag.abs().sum(dim=(1, 2), keepdim=True) / float(self.masd_slot_count * (self.masd_slot_count - 1))

        alpha_sparse = sparsemax(alpha_logits / self.masd_slot_temperature, dim=1)
        alpha_sparse = alpha_sparse / alpha_sparse.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy_sparse = -(alpha_sparse * torch.log(alpha_sparse.clamp_min(1e-8))).sum(dim=1, keepdim=True) / math.log(float(self.masd_slot_count))
        alpha_max_sparse = alpha_sparse.max(dim=1, keepdim=True).values
        disagreement_seed = (1.0 - diversity).reshape(-1, 1)
        risk_alpha_linear = (
            F.softplus(self.masd_alpha_risk_weights[0]) * parts["uncertainty_level"]
            + F.softplus(self.masd_alpha_risk_weights[1]) * parts["conflict_level"]
            + F.softplus(self.masd_alpha_risk_weights[2]) * entropy_sparse
            + F.softplus(self.masd_alpha_risk_weights[3]) * disagreement_seed
            + self.masd_alpha_risk_bias
        )
        risk_alpha = torch.sigmoid(risk_alpha_linear)
        if self.mode == "main_core_sci2_masd_current_no_risk_adaptive_alpha":
            alpha = alpha_sparse
        else:
            temp = self.masd_slot_temperature + 0.42 * risk_alpha
            alpha = sparsemax(alpha_logits / temp, dim=1)
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        alpha_sorted, _ = torch.sort(alpha, dim=1, descending=True)
        alpha_max = alpha_sorted[:, :1]
        alpha_margin = alpha_sorted[:, :1] - alpha_sorted[:, 1:2]
        entropy = -(alpha * torch.log(alpha.clamp_min(1e-8))).sum(dim=1, keepdim=True) / math.log(float(self.masd_slot_count))

        res_raw = self.masd_res_head(slot_hidden).squeeze(-1)
        residual = self.masd_residual_cap * torch.tanh(res_raw)
        mag_raw = self.masd_mag_head(slot_hidden).squeeze(-1)
        if self.mode == "main_core_sci2_masd_current_no_monotonic_calibrator":
            main_mag = self.masd_main_mag_floor + torch.clamp(F.softplus(mag_raw), max=self.masd_main_mag_cap)
            delta = sign_prior * main_mag + residual
        else:
            calib_context = 0.18 * torch.tanh(self.masd_calib_context_head(mechanism_core))
            proxy_scale = F.softplus(self.masd_calib_proxy_scale).reshape(1, -1)
            calib_logits = proxy_scale * proxy_scores + calib_context + self.masd_calib_proxy_bias.reshape(1, -1)
            calib = self.masd_main_mag_floor + (self.masd_main_mag_cap - self.masd_main_mag_floor) * torch.sigmoid(calib_logits)
            main_mag = calib
            delta = sign_prior * calib + residual

        contribution = alpha * delta
        contribution_sum = contribution.sum(dim=1, keepdim=True)
        mechanism_disagreement = torch.std(contribution, dim=1, keepdim=True) / max(self.innovation_limit, 1e-6)

        if self.mode == "main_core_sci2_masd_current_no_monotone_risk_gate":
            risk_features = torch.cat(
                [
                    mechanism_core,
                    parts["baseline_rel"],
                    parts["conflict_level"],
                    parts["uncertainty_level"],
                    anchor_risk,
                    anchor_confidence,
                    entropy,
                    1.0 - alpha_max,
                    mechanism_disagreement,
                ],
                dim=1,
            )
            gate_learned = self.masd_safety_gate_v3(risk_features)
            gate_heuristic = torch.sigmoid(
                2.55
                - 1.30 * parts["uncertainty_level"]
                - 1.15 * parts["conflict_level"]
                - 1.00 * entropy
                - 0.85 * mechanism_disagreement
                - 1.10 * (1.0 - alpha_max)
                - 0.75 * anchor_risk
            )
            gate = torch.clamp(gate_learned * gate_heuristic, min=0.0, max=1.0)
            gate_learned_out = gate_learned
            gate_heuristic_out = gate_heuristic
        else:
            context_shift = 0.18 * torch.tanh(self.masd_gate_context_head(torch.cat([mechanism_core, parts["baseline_rel"]], dim=1)))
            risk_linear = (
                F.softplus(self.masd_gate_risk_weights[0]) * parts["uncertainty_level"]
                + F.softplus(self.masd_gate_risk_weights[1]) * parts["conflict_level"]
                + F.softplus(self.masd_gate_risk_weights[2]) * entropy
                + F.softplus(self.masd_gate_risk_weights[3]) * mechanism_disagreement
                + F.softplus(self.masd_gate_risk_weights[4]) * (1.0 - alpha_max)
            )
            gate = self.masd_gate_low + (self.masd_gate_high - self.masd_gate_low) * torch.sigmoid(self.masd_gate_bias + context_shift - risk_linear)
            gate_learned_out = gate
            gate_heuristic_out = 1.0 - risk_alpha

        hard_score = parts.get("pr_hard_score", torch.zeros_like(anchor_pred))
        thresholded_gate = torch.ones_like(gate)
        thresholded_delta = contribution_sum
        if bool(getattr(self, "pr_thresholded_masd_enabled", False)):
            tau = float(getattr(self, "pr_thresholded_masd_tau", 0.70))
            gamma = float(getattr(self, "pr_thresholded_masd_gamma", 8.0))
            bound = float(getattr(self, "pr_thresholded_masd_bound", 0.0))
            thresholded_gate = torch.sigmoid(gamma * (hard_score - tau))
            if bound > 0.0:
                thresholded_delta = bound * torch.tanh(contribution_sum / max(bound, 1e-6))
            pred = anchor_pred + gate * thresholded_gate * thresholded_delta
        else:
            pred = anchor_pred + gate * contribution_sum
        applied_delta = pred - anchor_pred
        dominant_mechanism = alpha.argmax(dim=1)
        payload = dict(parts)
        payload.update(
            {
                "pred": pred,
                "baseline_pred": anchor_pred,
                "teacher_anchor_pred": anchor_pred,
                "innovation_pred": pred,
                "innovation_score": gate * contribution.abs().sum(dim=1, keepdim=True),
                "fused_latent": mechanism_core,
                "baseline_unc": parts["baseline_unc"],
                "rcmf_gate": parts["rcmf_min_gate"],
                "ctx_delta": pred - anchor_pred,
                "mspce_anchor_pred": parts["mspce_repair_pred"],
                "masd_anchor_pred": anchor_pred,
                "masd_base_delta": torch.zeros_like(anchor_pred),
                "masd_alpha_logits": alpha_logits,
                "masd_alpha": alpha,
                "masd_delta": delta,
                "masd_contribution": contribution,
                "masd_proxy_scores": proxy_scores,
                "masd_proxy_target": proxy_target,
                "masd_signed_proxy_target": signed_proxy_target,
                "masd_slot_hidden": slot_hidden,
                "masd_entropy": entropy,
                "masd_diversity": diversity,
                "masd_dominant_mechanism": dominant_mechanism,
                "masd_gate": gate,
                "masd_gate_learned": gate_learned_out,
                "masd_gate_heuristic": gate_heuristic_out,
                "masd_alpha_max": alpha_max,
                "masd_alpha_margin": alpha_margin,
                "masd_residual_delta": residual,
                "masd_main_mag": main_mag,
                "masd_sign_prior": sign_prior.expand_as(alpha),
                "masd_mechanism_disagreement": mechanism_disagreement,
                "masd_alpha_risk": risk_alpha,
                "masd_thresholded_gate": thresholded_gate,
                "masd_thresholded_delta": thresholded_delta,
                "masd_applied_delta": applied_delta,
                "masd_disable_rcmf_anchor": torch.full_like(anchor_pred, 1.0 if disable_rcmf_anchor else 0.0),
            }
        )
        return payload

    def _branches(
        self,
        graph_batch: Batch,
        descriptors: torch.Tensor,
        contexts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        backbone_out = self.backbone(graph_batch, descriptors)
        desc_emb = backbone_out["desc_emb"]
        graph_emb = backbone_out["graph_emb"]
        ctx_pack = self.ctx_encoder.forward_with_details(contexts)
        ctx_emb = ctx_pack["embedding"]
        active_ctx = ctx_pack["active_embedding"]
        inactive_ctx = ctx_pack["inactive_embedding"]
        chain_ctx = ctx_pack["chain_embedding"]
        segment_ctx = ctx_pack["segment_embedding"]
        graph_ctx = ctx_pack["graph_embedding"]
        scale_embs = ctx_pack["scale_embeddings"]
        pooled_scale = self.ctx_scale_pool(scale_embs.mean(dim=1))

        desc_pred, desc_unc = self.desc_head(desc_emb)
        graph_pred, graph_unc = self.graph_head(graph_emb)
        baseline_rel = reliability_features(
            torch.cat([desc_pred, graph_pred], dim=1),
            torch.cat([desc_unc, graph_unc], dim=1),
        )
        conflict_weights = self.conflict_gate(baseline_rel)
        baseline_latent = (
            conflict_weights[:, 0:1] * desc_emb + conflict_weights[:, 1:2] * graph_emb
        )
        student_baseline_pred, baseline_unc = self.baseline_head(baseline_latent)
        concat_latent = torch.cat([desc_emb, graph_emb, ctx_emb], dim=1)
        concat_pred = self.concat_head(concat_latent)
        repair_pack = self.mspce_context_injector(
            desc_emb=desc_emb,
            graph_emb=graph_emb,
            ctx_emb=ctx_emb,
            active_ctx=active_ctx,
            inactive_ctx=inactive_ctx,
            chain_ctx=chain_ctx,
            graph_ctx=graph_ctx,
            segment_ctx=segment_ctx,
            baseline_rel=baseline_rel,
        )
        repair_latent = torch.cat(
            [
                repair_pack["desc_mod"],
                repair_pack["graph_mod"],
                0.85 * ctx_emb + 0.15 * active_ctx,
            ],
            dim=1,
        )
        repair_candidate_pred = self.concat_head(repair_latent)
        repair_delta = 0.65 * self.innovation_limit * torch.tanh(
            (repair_candidate_pred - concat_pred) / max(self.innovation_limit, 1e-6)
        )
        repair_gate = torch.minimum(
            self.mspce_repair_gate(torch.cat([repair_pack["injected_hidden"], baseline_rel], dim=1)),
            repair_pack["modulation_gate"] + repair_pack["adapter_gate"],
        )
        conflict_level = torch.abs(desc_pred - graph_pred)
        uncertainty_level = 0.5 * (desc_unc + graph_unc)
        pr_hard_score = self._normalized_pr_hard_score(
            desc_pred=desc_pred,
            graph_pred=graph_pred,
            desc_unc=desc_unc,
            graph_unc=graph_unc,
            conflict_score=conflict_level,
        )
        scale_entropy = ctx_pack["scale_entropy"]
        inactive_scale_mass = ctx_pack["inactive_scale_mass"]
        repair_stability_proxy = torch.sigmoid(
            2.4
            * (
                baseline_unc
                + 0.35 * conflict_level
                + 0.30 * scale_entropy
                + 0.20 * inactive_scale_mass
                - 0.10 * ctx_pack["active_scale_count"] / max(float(self.ctx_encoder.top_k_active), 1.0)
                - 0.15
            )
        )
        repair_pred = concat_pred + repair_gate * repair_delta
        scale_consistency = torch.mean((active_ctx + 0.20 * inactive_ctx - ctx_emb.detach()) ** 2, dim=1, keepdim=True)
        hidden_anchor_distance = torch.mean((repair_pack["injected_hidden"] - repair_pack["concat_hidden"]) ** 2, dim=1, keepdim=True)
        q_temperature = max(float(getattr(self, "pr_rcmf_q_temperature", 1.0)), 1e-6)
        baseline_rel_for_rcmf = baseline_rel / q_temperature
        rcmf_pack = self.rcmf_min_fusion(
            desc_hidden=repair_pack["desc_mod"],
            graph_hidden=repair_pack["graph_mod"],
            ctx_hidden=active_ctx,
            anchor_hidden=repair_pack["injected_hidden"],
            baseline_rel=baseline_rel_for_rcmf,
            conflict=conflict_level,
            uncertainty=uncertainty_level,
            anchor_stability=repair_stability_proxy,
        )
        rcmf_delta_raw = self.rcmf_min_delta_head(rcmf_pack["trusted_hidden"])
        rcmf_delta = 0.50 * self.innovation_limit * torch.tanh(
            rcmf_delta_raw / max(self.innovation_limit, 1e-6)
        )
        rcmf_pred = repair_pred + rcmf_pack["residual_gate"] * rcmf_delta
        rcmf_hidden_consistency = torch.mean(
            (rcmf_pack["trusted_hidden"] - repair_pack["injected_hidden"]) ** 2,
            dim=1,
            keepdim=True,
        )
        raw_ctx_delta = self.ctx_delta_head(ctx_emb)
        bounded_ctx_delta = self.innovation_limit * torch.tanh(
            raw_ctx_delta / max(self.innovation_limit, 1e-6)
        )
        innovation_features = torch.cat([baseline_rel, ctx_emb], dim=1)
        rcmf_gate = self.rcmf_gate(innovation_features)
        innovation_score = rcmf_gate * torch.abs(bounded_ctx_delta)
        led_proxy_input = torch.cat([desc_emb, graph_emb, ctx_emb, baseline_rel], dim=1)
        led_proxy = self.led_proxy(led_proxy_input)
        desc_f = self.fuse_desc_proj(desc_emb)
        graph_f = self.fuse_graph_proj(graph_emb)
        ctx_f = self.fuse_ctx_proj(ctx_emb)
        led_f = self.fuse_led_proj(led_proxy)
        desc_gamma, desc_beta = torch.chunk(self.ctx_to_desc_film(ctx_f), chunks=2, dim=1)
        graph_gamma, graph_beta = torch.chunk(self.ctx_to_graph_film(ctx_f), chunks=2, dim=1)
        desc_mod = (1.0 + 0.10 * torch.tanh(desc_gamma)) * desc_f + 0.10 * torch.tanh(desc_beta)
        graph_mod = (1.0 + 0.10 * torch.tanh(graph_gamma)) * graph_f + 0.10 * torch.tanh(graph_beta)
        controller_pack = self.mspce_fusion_controller(ctx_f)
        modality_trust = controller_pack["modality_trust"]
        interaction_strength = controller_pack["interaction_strength"]
        alpha_controller = controller_pack["alpha_controller"]
        if self.mode in {"rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
            desc_ctrl = desc_f * modality_trust[:, 0:1]
            graph_ctrl = graph_f * modality_trust[:, 1:2]
            led_ctrl = led_f * modality_trust[:, 2:3]
            pair_dg = self.controller_pair_dg(torch.cat([desc_ctrl * graph_ctrl, torch.abs(desc_ctrl - graph_ctrl)], dim=1))
            pair_dl = self.controller_pair_dl(torch.cat([desc_ctrl * led_ctrl, torch.abs(desc_ctrl - led_ctrl)], dim=1))
            pair_gl = self.controller_pair_gl(torch.cat([graph_ctrl * led_ctrl, torch.abs(graph_ctrl - led_ctrl)], dim=1))
            interaction_latent = (
                interaction_strength[:, 0:1] * pair_dg
                + interaction_strength[:, 1:2] * pair_dl
                + interaction_strength[:, 2:3] * pair_gl
            )
            dynamic_condition = torch.cat([desc_ctrl, graph_ctrl, interaction_latent, ctx_f, baseline_rel], dim=1)
            dynamic_logits = self.dynamic_fusion_gate(dynamic_condition) / self.fusion_temperature
            dynamic_weights = torch.softmax(dynamic_logits, dim=1)
            dynamic_weights = (1.0 - self.fusion_uniform_mix) * dynamic_weights + self.fusion_uniform_mix * 0.25
            dynamic_latent = (
                dynamic_weights[:, 0:1] * desc_ctrl
                + dynamic_weights[:, 1:2] * graph_ctrl
                + dynamic_weights[:, 2:3] * interaction_latent
                + dynamic_weights[:, 3:4] * ctx_f
            )
            attn_weights = torch.cat([modality_trust, 1.0 - modality_trust[:, 2:3]], dim=1)
            interaction_gate = alpha_controller
        else:
            interaction_tokens = torch.stack([desc_mod, graph_mod, led_f], dim=1)
            ctx_query = self.ctx_query_proj(ctx_f).unsqueeze(1)
            attn_key = self.interaction_key_proj(interaction_tokens)
            attn_value = self.interaction_value_proj(interaction_tokens)
            attn_logits = (ctx_query * attn_key).sum(dim=2) / math.sqrt(float(desc_mod.shape[1]))
            attn_weights = torch.softmax(attn_logits, dim=1)
            cross_latent = (attn_weights.unsqueeze(-1) * attn_value).sum(dim=1)
            interaction_gate_input = torch.cat([ctx_f, cross_latent, baseline_rel], dim=1)
            interaction_gate = self.interaction_gate(interaction_gate_input)
            interaction_latent = interaction_gate * cross_latent + (1.0 - interaction_gate) * ctx_f
            dynamic_condition = torch.cat(
                [desc_mod, graph_mod, interaction_latent, led_f, baseline_rel],
                dim=1,
            )
            dynamic_logits = self.dynamic_fusion_gate(dynamic_condition) / self.fusion_temperature
            dynamic_weights = torch.softmax(dynamic_logits, dim=1)
            dynamic_weights = (1.0 - self.fusion_uniform_mix) * dynamic_weights + self.fusion_uniform_mix * 0.25
            dynamic_latent = (
                dynamic_weights[:, 0:1] * desc_mod
                + dynamic_weights[:, 1:2] * graph_mod
                + dynamic_weights[:, 2:3] * interaction_latent
                + dynamic_weights[:, 3:4] * led_f
            )
            dynamic_latent = dynamic_latent + 0.15 * ctx_f
        dynamic_control = self.dynamic_control(dynamic_condition)
        dynamic_pred = self.dynamic_head(dynamic_latent)
        dynamic_delta = self.innovation_limit * torch.tanh(
            (dynamic_pred - student_baseline_pred) / max(self.innovation_limit, 1e-6)
        )
        mspce_pred = student_baseline_pred + bounded_ctx_delta
        fusion_delta_raw = self.fusion_delta_head(dynamic_latent)
        fusion_delta = 0.60 * self.innovation_limit * torch.tanh(
            fusion_delta_raw / max(self.innovation_limit, 1e-6)
        )
        alpha_input = torch.cat([dynamic_latent, baseline_rel], dim=1)
        alpha_raw = self.fusion_alpha_gate(alpha_input)
        ctx_strength = torch.abs(bounded_ctx_delta)
        led_confidence = torch.sigmoid(2.4 - torch.mean(torch.abs(led_f - ctx_f), dim=1, keepdim=True))
        primary_confidence = torch.sigmoid(2.2 * (ctx_strength - 0.70 * uncertainty_level)) * torch.sigmoid(
            1.8 * (conflict_level + 0.15)
        )
        risk_signal = (
            0.55 * uncertainty_level
            + 0.35 * conflict_level
            + 0.45 * (1.0 - led_confidence)
            - 0.30 * ctx_strength
        )
        external_risk = torch.sigmoid(3.0 * (risk_signal - 0.25))
        external_shrink = torch.clamp(1.0 - 0.90 * external_risk, min=0.08, max=1.0)
        confidence = primary_confidence * external_shrink
        alpha = 0.15 * alpha_raw * confidence
        if self.mode in {"rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
            alpha = torch.clamp(0.32 * alpha_controller * alpha_raw * torch.sigmoid(2.2 * (conflict_level - uncertainty_level + 0.15)), max=0.45)
        anchor_beta = self.anchor_mix_controller(
            h_mspce=ctx_f,
            h_desc=desc_f,
            h_graph=graph_f,
            reliability=baseline_rel,
        )
        dual_anchor_pred = anchor_beta * concat_pred + (1.0 - anchor_beta) * mspce_pred
        fusion_pred = mspce_pred + alpha * fusion_delta
        dual_fusion_pred = dual_anchor_pred + alpha * fusion_delta

        return {
            "desc_emb": desc_emb,
            "graph_emb": graph_emb,
            "ctx_emb": ctx_emb,
            "ctx_active_emb": active_ctx,
            "ctx_inactive_emb": inactive_ctx,
            "ctx_chain_emb": chain_ctx,
            "ctx_graph_emb": graph_ctx,
            "ctx_segment_emb": segment_ctx,
            "mspce_scale_weights": ctx_pack["scale_weights"],
            "mspce_scale_dense_weights": ctx_pack["scale_dense_weights"],
            "mspce_scale_embeddings": scale_embs,
            "mspce_active_scale_mask": ctx_pack["active_scale_mask"],
            "mspce_active_scale_count": ctx_pack["active_scale_count"],
            "mspce_inactive_scale_mass": inactive_scale_mass,
            "mspce_scale_entropy": scale_entropy,
            "pooled_scale": pooled_scale,
            "baseline_latent": baseline_latent,
            "student_baseline_pred": student_baseline_pred,
            "baseline_unc": baseline_unc,
            "conflict_weights": conflict_weights,
            "baseline_rel": baseline_rel,
            "pr_hard_score": pr_hard_score,
            "desc_pred": desc_pred,
            "graph_pred": graph_pred,
            "desc_unc": desc_unc,
            "graph_unc": graph_unc,
            "conflict_level": conflict_level,
            "uncertainty_level": uncertainty_level,
            "raw_ctx_delta": raw_ctx_delta,
            "ctx_delta": bounded_ctx_delta,
            "rcmf_gate": rcmf_gate,
            "innovation_score": innovation_score,
            "controller_core": controller_pack["controller_core"],
            "controller_modality_trust_desc": modality_trust[:, 0:1],
            "controller_modality_trust_graph": modality_trust[:, 1:2],
            "controller_modality_trust_led": modality_trust[:, 2:3],
            "controller_interaction_dg": interaction_strength[:, 0:1],
            "controller_interaction_dl": interaction_strength[:, 1:2],
            "controller_interaction_gl": interaction_strength[:, 2:3],
            "controller_alpha": alpha_controller,
            "anchor_beta": anchor_beta,
            "dual_anchor_pred": dual_anchor_pred,
            "dynamic_condition": dynamic_condition,
            "dynamic_weights": dynamic_weights,
            "dynamic_control": dynamic_control,
            "dynamic_latent": dynamic_latent,
            "interaction_latent": interaction_latent,
            "interaction_gate": interaction_gate,
            "interaction_attn_desc": attn_weights[:, 0:1],
            "interaction_attn_graph": attn_weights[:, 1:2],
            "interaction_attn_led": attn_weights[:, 2:3],
            "desc_fuse_proj": desc_f,
            "graph_fuse_proj": graph_f,
            "desc_fuse_mod": desc_mod,
            "graph_fuse_mod": graph_mod,
            "ctx_fuse_proj": ctx_f,
            "led_fuse_proj": led_f,
            "dynamic_pred": dynamic_pred,
            "dynamic_delta": dynamic_delta,
            "led_proxy": led_proxy,
            "concat_pred": concat_pred,
            "concat_hidden": repair_pack["concat_hidden"],
            "mspce_repair_latent": repair_latent,
            "mspce_repair_candidate_pred": repair_candidate_pred,
            "mspce_repair_pred": repair_pred,
            "mspce_repair_hidden": repair_pack["injected_hidden"],
            "mspce_repair_delta": repair_delta,
            "mspce_repair_gate": repair_gate,
            "mspce_repair_modulation_gate": repair_pack["modulation_gate"],
            "mspce_repair_adapter_gate": repair_pack["adapter_gate"],
            "mspce_repair_rank_state": repair_pack["rank_state"],
            "mspce_repair_bottleneck": repair_pack["bottleneck"],
            "mspce_repair_hidden_shift": repair_pack["hidden_shift"],
            "mspce_repair_scale_consistency": scale_consistency,
            "mspce_repair_hidden_consistency": hidden_anchor_distance,
            "mspce_repair_stability_proxy": repair_stability_proxy,
            "mspce_repair_desc_mod": repair_pack["desc_mod"],
            "mspce_repair_graph_mod": repair_pack["graph_mod"],
            "rcmf_min_pred": rcmf_pred,
            "rcmf_min_hidden": rcmf_pack["trusted_hidden"],
            "rcmf_min_delta": rcmf_delta,
            "rcmf_min_gate": rcmf_pack["residual_gate"],
            "rcmf_min_confidence": rcmf_pack["confidence"],
            "rcmf_min_risk": rcmf_pack["risk"],
            "rcmf_min_selector_score": rcmf_pack["selector_score"],
            "rcmf_min_trust_entropy": rcmf_pack["trust_entropy"],
            "rcmf_min_trust_desc": rcmf_pack["trust_weights"][:, 0:1],
            "rcmf_min_trust_graph": rcmf_pack["trust_weights"][:, 1:2],
            "rcmf_min_trust_ctx": rcmf_pack["trust_weights"][:, 2:3],
            "rcmf_min_controller": rcmf_pack["controller_state"],
            "rcmf_min_mix_hidden": rcmf_pack["mix_hidden"],
            "rcmf_min_hidden_consistency": rcmf_hidden_consistency,
            "mspce_pred": mspce_pred,
            "fusion_delta": fusion_delta,
            "fusion_alpha_raw": alpha_raw,
            "fusion_confidence": confidence,
            "primary_confidence": primary_confidence,
            "led_confidence": led_confidence,
            "external_risk": external_risk,
            "external_shrink": external_shrink,
            "fusion_alpha": alpha,
            "fusion_pred": fusion_pred,
            "dual_fusion_pred": dual_fusion_pred,
        }

    def forward(
        self,
        graph_batch: Batch,
        descriptors: torch.Tensor,
        contexts: torch.Tensor,
        teacher_pred: torch.Tensor | None = None,
        led: torch.Tensor | None = None,
        led_mask: torch.Tensor | None = None,
        full_control_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        # Forward always returns the prediction plus diagnostics used by ablations and reports.
        parts = self._branches(graph_batch, descriptors, contexts)
        student_baseline_pred = parts["student_baseline_pred"]
        baseline_unc = parts["baseline_unc"]
        baseline_latent = parts["baseline_latent"]
        ctx_emb = parts["ctx_emb"]
        ctx_delta = parts["ctx_delta"]
        innovation_score = parts["innovation_score"]
        rcmf_gate = parts["rcmf_gate"]

        if self.mode == "conflict_only":
            payload = dict(parts)
            payload.update(
                {
                "pred": student_baseline_pred,
                "baseline_pred": student_baseline_pred,
                "teacher_anchor_pred": student_baseline_pred,
                "innovation_pred": student_baseline_pred,
                "innovation_score": torch.zeros_like(student_baseline_pred),
                "fused_latent": baseline_latent,
                "baseline_unc": baseline_unc,
                "rcmf_gate": torch.zeros_like(student_baseline_pred),
                "ctx_delta": torch.zeros_like(student_baseline_pred),
                }
            )
            return payload

        if self.mode == "simple_concat":
            fused = torch.cat([parts["desc_emb"], parts["graph_emb"], ctx_emb], dim=1)
            pred = parts["concat_pred"]
            payload = dict(parts)
            payload.update(
                {
                "pred": pred,
                "baseline_pred": student_baseline_pred,
                "teacher_anchor_pred": student_baseline_pred,
                "innovation_pred": pred,
                "innovation_score": torch.ones_like(pred),
                "fused_latent": fused,
                "baseline_unc": baseline_unc,
                "rcmf_gate": torch.ones_like(pred),
                "ctx_delta": pred - student_baseline_pred,
                }
            )
            return payload

        if self.mode == "mspce_context_injection":
            pred = parts["mspce_repair_pred"]
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": parts["concat_pred"],
                    "teacher_anchor_pred": parts["concat_pred"],
                    "innovation_pred": pred,
                    "innovation_score": parts["mspce_repair_gate"] * torch.abs(parts["mspce_repair_delta"]),
                    "fused_latent": parts["mspce_repair_hidden"],
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["mspce_repair_gate"],
                    "ctx_delta": pred - parts["concat_pred"],
                    "mspce_anchor_pred": pred,
                }
            )
            return payload

        if self.mode == "rcmf_min_trusted_fusion":
            conflict_level = torch.abs(parts["desc_pred"] - parts["graph_pred"])
            uncertainty_level = 0.5 * (parts["desc_unc"] + parts["graph_unc"])
            hard_external_focus = torch.sigmoid(
                6.3
                * (
                    conflict_level
                    + 1.36 * uncertainty_level
                    + 0.42 * (1.0 - parts["rcmf_min_confidence"])
                    - 0.70
                )
            )
            external_temperature = torch.clamp(
                0.93 + 0.18 * hard_external_focus + 0.05 * (1.0 - parts["rcmf_min_risk"]),
                min=0.91,
                max=1.08,
            )
            pred = parts["mspce_repair_pred"] + external_temperature * (
                parts["rcmf_min_pred"] - parts["mspce_repair_pred"]
            )
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": parts["mspce_repair_pred"],
                    "teacher_anchor_pred": parts["mspce_repair_pred"],
                    "innovation_pred": pred,
                    "innovation_score": parts["rcmf_min_gate"] * torch.abs(parts["rcmf_min_delta"]),
                    "fused_latent": parts["rcmf_min_hidden"],
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["rcmf_min_gate"],
                    "ctx_delta": pred - parts["mspce_repair_pred"],
                    "mspce_anchor_pred": parts["mspce_repair_pred"],
                    "rcmf_min_external_focus": hard_external_focus,
                    "rcmf_min_external_temperature": external_temperature,
                }
            )
            return payload

        if self.mode in {
            "main_core_sci2_masd_v1",
            "main_core_sci2_masd_no_competition",
            "main_core_sci2_masd_no_anchor",
            "main_core_sci2_masd_no_diversity",
        }:
            return self._masd_forward(parts, descriptors, contexts)

        if self.mode in {
            "main_core_sci2_masd_v2",
            "main_core_sci2_masd_v2_no_sign_prior",
            "main_core_sci2_masd_v2_no_contribution_anchor",
            "main_core_sci2_masd_v2_no_safety_gate",
        }:
            return self._masd_forward_v2(parts, descriptors, contexts)

        if self.mode in {
            "main_core_sci2_masd_v3",
            "main_core_sci2_masd_v3_no_sparse_alpha",
            "main_core_sci2_masd_v3_no_monotonic_calibrator",
            "main_core_sci2_masd_v3_no_uncertainty_gate",
        }:
            return self._masd_forward_v3(parts, descriptors, contexts)

        if self.mode in {
            "main_core_sci2_masd_current",
            "main_core_sci2_masd_current_locked",
            "main_core_sci2_masd_final",
            "main_core_sci2_masd_current_no_risk_adaptive_alpha",
            "main_core_sci2_masd_current_no_monotone_risk_gate",
            "main_core_sci2_masd_current_no_group_dro_lite",
            "main_core_sci2_masd_current_no_monotonic_calibrator",
        }:
            return self._masd_forward_current(parts, descriptors, contexts)

        if self.mode == "led_min_conditional_distill":
            if led is None:
                led = torch.zeros_like(parts["ctx_emb"])
            if led_mask is None:
                led_mask = torch.zeros_like(parts["rcmf_min_pred"])
            led_teacher_latent, led_teacher_pred = self.led_prior(led)
            conflict_level = torch.abs(parts["desc_pred"] - parts["graph_pred"])
            uncertainty_level = 0.5 * (parts["desc_unc"] + parts["graph_unc"])
            control_mode = full_control_mode or ""
            pair_mspce = control_mode.startswith("pair_mspce_")
            base_control_mode = control_mode[len("pair_mspce_"):] if pair_mspce else control_mode
            teacher_audit_modes = {
                "led_teacher_audit_raw": "raw",
                "led_teacher_audit_ensemble": "ensemble",
                "led_teacher_audit_student": "student_informed",
                "led_teacher_audit_hybrid": "hybrid",
                "led_teacher_audit_ensemble_topk": "ensemble_topk",
                "led_teacher_audit_student_topk": "student_topk",
                "led_teacher_audit_unimol_bank": "unimol_bank",
                "led_teacher_audit_unimol2_bank": "unimol2_bank",
                "led_teacher_audit_unimol2_headspec": "unimol2_headspec",
                "led_teacher_audit_unimol2_ensemble": "unimol2_ensemble",
                "led_teacher_audit_polymer_unimol_bank": "polymer_unimol_bank",
                "led_teacher_audit_mmpolymer_bank": "mmpolymer_bank",
            }
            distill_teacher_modes = {
                "latent_only_led_teacher_ensemble": "ensemble",
                "led_profile_teacher_ensemble": "ensemble",
                "latent_only_led_teacher_student": "student_informed",
                "led_profile_teacher_student": "student_informed",
                "latent_only_led_teacher_hybrid": "hybrid",
                "led_profile_teacher_hybrid": "hybrid",
                "latent_only_led_teacher_ensemble_topk": "ensemble_topk",
                "led_profile_teacher_ensemble_topk": "ensemble_topk",
                "latent_only_led_teacher_student_topk": "student_topk",
                "led_profile_teacher_student_topk": "student_topk",
                "latent_only_led_teacher_unimol_bank": "unimol_bank",
                "led_profile_teacher_unimol_bank": "unimol_bank",
                "latent_only_led_teacher_unimol2_bank": "unimol2_bank",
                "led_profile_teacher_unimol2_bank": "unimol2_bank",
                "latent_only_led_teacher_unimol2_headspec": "unimol2_headspec",
                "led_profile_teacher_unimol2_headspec": "unimol2_headspec",
                "latent_only_led_teacher_unimol2_ensemble": "unimol2_ensemble",
                "led_profile_teacher_unimol2_ensemble": "unimol2_ensemble",
                "latent_only_led_teacher_polymer_unimol_bank": "polymer_unimol_bank",
                "led_profile_teacher_polymer_unimol_bank": "polymer_unimol_bank",
                "latent_only_led_teacher_mmpolymer_bank": "mmpolymer_bank",
                "led_profile_teacher_mmpolymer_bank": "mmpolymer_bank",
            }
            simple_led_controls = {
                "led_off",
                "latent_only_led_curriculum",
                "led_profile_curriculum",
                "latent_only_led_softgate",
                "led_profile_softgate",
                "latent_only_led_curriculum_softgate",
                "led_profile_curriculum_softgate",
                *teacher_audit_modes.keys(),
                *distill_teacher_modes.keys(),
            }
            regime_features = torch.cat(
                [
                    conflict_level,
                    uncertainty_level,
                    parts["rcmf_min_risk"],
                    parts["rcmf_min_confidence"],
                    parts["mspce_repair_gate"],
                ],
                dim=1,
            )
            if base_control_mode in simple_led_controls or (pair_mspce and base_control_mode == "led_off"):
                anchor_hidden = parts["mspce_repair_hidden"] if pair_mspce else parts["rcmf_min_hidden"]
                anchor_pred = parts["mspce_repair_pred"] if pair_mspce else parts["rcmf_min_pred"]
                regime_summary = self.regime_summary(torch.cat([parts["mspce_repair_hidden"], regime_features], dim=1))
                regime_posterior = torch.softmax(
                    self.regime_posterior(torch.cat([regime_summary, regime_features], dim=1)),
                    dim=1,
                )
                regime_token = regime_posterior @ self.regime_token_bank.weight
                regime_teacher_latent = torch.tanh(
                    self.regime_teacher_proj(torch.cat([led_teacher_latent, regime_token], dim=1))
                )
                teacher_source_latent = torch.tanh(
                    self.led_teacher_refine(
                        torch.cat(
                            [
                                led_teacher_latent,
                                parts["mspce_repair_hidden"],
                                anchor_hidden,
                                parts["baseline_rel"],
                            ],
                            dim=1,
                        )
                    )
                )
                teacher_source_delta = 0.10 * self.innovation_limit * torch.tanh(
                    self.led_teacher_delta_head(teacher_source_latent) / max(self.innovation_limit, 1e-6)
                )
                teacher_source_gate = torch.sigmoid(
                    self.led_teacher_source_gate(torch.cat([teacher_source_latent, parts["baseline_rel"]], dim=1))
                )
                teacher_source_score = torch.sigmoid(
                    self.led_teacher_source_score(torch.cat([teacher_source_latent, parts["baseline_rel"]], dim=1))
                )
                foundation_bank_logits = self.led_foundation_selector(
                    torch.cat([teacher_source_latent, regime_teacher_latent, parts["baseline_rel"]], dim=1)
                )
                foundation_bank_weights = torch.softmax(foundation_bank_logits, dim=1)
                foundation_bank_token = foundation_bank_weights @ self.led_foundation_bank.weight
                foundation_bank_score = foundation_bank_weights.max(dim=1, keepdim=True).values
                foundation_teacher_latent = torch.tanh(
                    self.led_foundation_proj(
                        torch.cat(
                            [
                                led_teacher_latent,
                                teacher_source_latent,
                                regime_teacher_latent,
                                foundation_bank_token,
                            ],
                            dim=1,
                        )
                    )
                )
                foundation_teacher_delta = 0.12 * self.innovation_limit * torch.tanh(
                    self.led_teacher_delta_head(foundation_teacher_latent) / max(self.innovation_limit, 1e-6)
                )
                unimol2_task_latent = torch.tanh(
                    self.led_unimol2_task_proj(
                        torch.cat(
                            [
                                foundation_teacher_latent,
                                teacher_source_latent,
                                parts["mspce_repair_hidden"],
                                anchor_hidden,
                            ],
                            dim=1,
                        )
                    )
                )
                unimol2_task_delta = 0.12 * self.innovation_limit * torch.tanh(
                    self.led_unimol2_task_delta(unimol2_task_latent) / max(self.innovation_limit, 1e-6)
                )
                unimol2_task_score = torch.sigmoid(
                    self.led_unimol2_task_score(torch.cat([unimol2_task_latent, parts["baseline_rel"]], dim=1))
                )
                unimol2_aux_latent = torch.tanh(
                    self.led_unimol2_aux_proj(
                        torch.cat(
                            [
                                foundation_teacher_latent,
                                regime_teacher_latent,
                                parts["mspce_repair_hidden"],
                                anchor_hidden,
                            ],
                            dim=1,
                        )
                    )
                )
                unimol2_aux_delta = 0.12 * self.innovation_limit * torch.tanh(
                    self.led_unimol2_aux_delta(unimol2_aux_latent) / max(self.innovation_limit, 1e-6)
                )
                unimol2_aux_score = torch.sigmoid(
                    self.led_unimol2_aux_score(torch.cat([unimol2_aux_latent, parts["baseline_rel"]], dim=1))
                )
                raw_teacher_reliability = led_mask * teacher_source_score * torch.sigmoid(
                    2.1 - 1.1 * uncertainty_level - 0.9 * conflict_level + 0.30 * (1.0 - parts["rcmf_min_risk"])
                )
                teacher_kind = teacher_audit_modes.get(base_control_mode, distill_teacher_modes.get(base_control_mode, "raw"))
                teacher_head_disagreement = torch.zeros_like(anchor_pred)
                if teacher_kind == "ensemble":
                    selected_teacher_latent = 0.55 * regime_teacher_latent + 0.45 * teacher_source_latent
                    selected_teacher_pred = anchor_pred + 0.35 * teacher_source_gate * (
                        led_teacher_pred.detach() - anchor_pred.detach()
                    ) + 0.65 * teacher_source_gate * teacher_source_delta
                    selected_teacher_reliability = raw_teacher_reliability * torch.sigmoid(10.0 * (teacher_source_score - 0.52))
                elif teacher_kind == "student_informed":
                    selected_teacher_latent = 0.25 * led_teacher_latent + 0.75 * teacher_source_latent
                    selected_teacher_pred = anchor_pred + teacher_source_gate * teacher_source_delta
                    selected_teacher_reliability = raw_teacher_reliability * torch.sigmoid(12.0 * (teacher_source_score - 0.58))
                elif teacher_kind == "hybrid":
                    selected_teacher_latent = 0.30 * led_teacher_latent + 0.35 * regime_teacher_latent + 0.35 * teacher_source_latent
                    selected_teacher_pred = anchor_pred + 0.50 * teacher_source_gate * teacher_source_delta + 0.18 * (
                        led_teacher_pred.detach() - anchor_pred.detach()
                    )
                    selected_teacher_reliability = raw_teacher_reliability * torch.sigmoid(14.0 * (teacher_source_score - 0.62))
                elif teacher_kind == "ensemble_topk":
                    selected_teacher_latent = 0.62 * teacher_source_latent + 0.38 * regime_teacher_latent
                    selected_teacher_pred = anchor_pred + 0.82 * teacher_source_gate * teacher_source_delta + 0.12 * (
                        led_teacher_pred.detach() - anchor_pred.detach()
                    )
                    selected_teacher_reliability = raw_teacher_reliability * torch.sigmoid(20.0 * (teacher_source_score - 0.70))
                elif teacher_kind == "student_topk":
                    selected_teacher_latent = 0.78 * teacher_source_latent + 0.22 * led_teacher_latent
                    selected_teacher_pred = anchor_pred + 0.95 * teacher_source_gate * teacher_source_delta
                    selected_teacher_reliability = raw_teacher_reliability * torch.sigmoid(22.0 * (teacher_source_score - 0.74))
                elif teacher_kind == "unimol_bank":
                    selected_teacher_latent = 0.50 * foundation_teacher_latent + 0.30 * teacher_source_latent + 0.20 * led_teacher_latent
                    selected_teacher_pred = anchor_pred + 0.88 * teacher_source_gate * foundation_teacher_delta + 0.10 * (
                        led_teacher_pred.detach() - anchor_pred.detach()
                    )
                    selected_teacher_reliability = raw_teacher_reliability * foundation_bank_score * torch.sigmoid(18.0 * (teacher_source_score - 0.66))
                elif teacher_kind == "unimol2_bank":
                    selected_teacher_latent = 0.68 * foundation_teacher_latent + 0.20 * regime_teacher_latent + 0.12 * teacher_source_latent
                    selected_teacher_pred = anchor_pred + teacher_source_gate * (
                        0.92 * foundation_teacher_delta + 0.10 * teacher_source_delta
                    )
                    selected_teacher_reliability = raw_teacher_reliability * foundation_bank_score * torch.sigmoid(22.0 * (teacher_source_score - 0.70))
                elif teacher_kind == "unimol2_headspec":
                    selected_teacher_latent = 0.72 * unimol2_task_latent + 0.18 * foundation_teacher_latent + 0.10 * regime_teacher_latent
                    selected_teacher_pred = anchor_pred + teacher_source_gate * (
                        1.05 * unimol2_task_delta + 0.06 * foundation_teacher_delta
                    )
                    task_margin_hint = torch.abs(unimol2_task_delta - teacher_source_delta)
                    selected_teacher_reliability = raw_teacher_reliability * foundation_bank_score * torch.sigmoid(24.0 * (unimol2_task_score - 0.60)) * torch.sigmoid(26.0 * (0.020 - task_margin_hint))
                elif teacher_kind == "unimol2_ensemble":
                    selected_teacher_latent = 0.45 * unimol2_task_latent + 0.45 * unimol2_aux_latent + 0.10 * foundation_teacher_latent
                    ensemble_delta = 0.58 * unimol2_task_delta + 0.42 * unimol2_aux_delta
                    selected_teacher_pred = anchor_pred + teacher_source_gate * ensemble_delta
                    teacher_head_disagreement = torch.abs(unimol2_task_delta - unimol2_aux_delta)
                    ensemble_score = 0.5 * unimol2_task_score + 0.5 * unimol2_aux_score
                    selected_teacher_reliability = raw_teacher_reliability * foundation_bank_score * torch.sigmoid(24.0 * (ensemble_score - 0.60)) * torch.sigmoid(32.0 * (0.015 - teacher_head_disagreement))
                elif teacher_kind == "polymer_unimol_bank":
                    selected_teacher_latent = 0.55 * foundation_teacher_latent + 0.25 * parts["mspce_repair_hidden"] + 0.20 * regime_teacher_latent
                    selected_teacher_pred = anchor_pred + teacher_source_gate * (
                        0.80 * foundation_teacher_delta + 0.16 * teacher_source_delta
                    )
                    selected_teacher_reliability = raw_teacher_reliability * torch.sigmoid(16.0 * (foundation_bank_score - 0.42)) * torch.sigmoid(18.0 * (teacher_source_score - 0.64))
                elif teacher_kind == "mmpolymer_bank":
                    selected_teacher_latent = 0.42 * foundation_teacher_latent + 0.30 * teacher_source_latent + 0.28 * parts["mspce_repair_hidden"]
                    selected_teacher_pred = anchor_pred + teacher_source_gate * (
                        0.72 * foundation_teacher_delta + 0.24 * teacher_source_delta + 0.06 * (led_teacher_pred.detach() - anchor_pred.detach())
                    )
                    selected_teacher_reliability = raw_teacher_reliability * foundation_bank_score * torch.sigmoid(20.0 * (teacher_source_score - 0.68))
                else:
                    selected_teacher_latent = regime_teacher_latent
                    selected_teacher_pred = led_teacher_pred
                    selected_teacher_reliability = raw_teacher_reliability
                bridge_input = torch.cat(
                    [
                        anchor_hidden,
                        selected_teacher_latent,
                        parts["mspce_repair_hidden"],
                        parts["baseline_rel"],
                    ],
                    dim=1,
                )
                led_bridge = self.led_min_bridge(bridge_input)
                led_rank = torch.tanh(self.led_min_down(led_bridge))
                led_delta_hidden = torch.tanh(self.led_min_up(led_rank))
                led_confidence = torch.sigmoid(
                    self.led_min_confidence(torch.cat([led_bridge, parts["baseline_rel"]], dim=1))
                )
                teacher_available = led_mask
                reliability_context = torch.sigmoid(
                    2.2
                    - 1.12 * uncertainty_level
                    - 0.86 * conflict_level
                    + 0.38 * parts["mspce_repair_gate"]
                    + 0.22 * (1.0 - parts["rcmf_min_risk"])
                )
                base_teacher_reliability = teacher_available * led_confidence * reliability_context * torch.clamp(
                    0.25 + 0.75 * selected_teacher_reliability,
                    min=0.0,
                    max=1.0,
                )
                teacher_delta_target = torch.clamp(
                    selected_teacher_pred.detach() - anchor_pred.detach(),
                    min=-0.28 * self.innovation_limit,
                    max=0.28 * self.innovation_limit,
                )
                if base_control_mode in {
                    "latent_only_led_curriculum",
                    "led_profile_curriculum",
                }:
                    reliability_shift = 0.84
                    reliability_scale = 24.0
                    delta_scale = 18.0
                    delta_shift = 0.010
                    gate_mix = 0.28
                    residual_mix = 0.18
                elif base_control_mode in {
                    "latent_only_led_curriculum_softgate",
                    "led_profile_curriculum_softgate",
                }:
                    reliability_shift = 0.80
                    reliability_scale = 22.0
                    delta_scale = 18.0
                    delta_shift = 0.010
                    gate_mix = 0.38
                    residual_mix = 0.22
                elif base_control_mode in {
                    "latent_only_led_softgate",
                    "led_profile_softgate",
                }:
                    reliability_shift = 0.74
                    reliability_scale = 18.0
                    delta_scale = 14.0
                    delta_shift = 0.008
                    gate_mix = 0.52
                    residual_mix = 0.24
                elif base_control_mode in teacher_audit_modes:
                    reliability_shift = 0.76
                    reliability_scale = 20.0
                    delta_scale = 16.0
                    delta_shift = 0.006
                    gate_mix = 0.0
                    residual_mix = 0.0
                else:
                    reliability_shift = 0.74
                    reliability_scale = 16.0
                    delta_scale = 12.0
                    delta_shift = 0.006
                    gate_mix = 0.0
                    residual_mix = 0.0
                top_reliability_gate = torch.sigmoid(
                    reliability_scale * (base_teacher_reliability - reliability_shift)
                )
                teacher_delta_gate = torch.sigmoid(
                    delta_scale * (torch.abs(teacher_delta_target) - delta_shift)
                )
                hard_teacher_score = base_teacher_reliability * top_reliability_gate * teacher_delta_gate
                gate_logits = self.led_min_gate(
                    torch.cat([anchor_hidden, led_bridge, parts["baseline_rel"]], dim=1)
                )
                soft_gate = torch.sigmoid(gate_logits - 3.8)
                if base_control_mode in {"led_profile_softgate", "latent_only_led_softgate", "led_profile_curriculum_softgate", "latent_only_led_curriculum_softgate"}:
                    led_weight = teacher_available * torch.clamp(
                        hard_teacher_score * (gate_mix + (1.0 - gate_mix) * soft_gate),
                        min=0.0,
                        max=1.0,
                    )
                else:
                    activation_mask = torch.sigmoid(26.0 * (hard_teacher_score - 0.60))
                    led_weight = teacher_available * activation_mask * torch.clamp(
                        gate_mix + (1.0 - gate_mix) * soft_gate,
                        min=0.0,
                        max=1.0,
                    )
                led_hidden = anchor_hidden + led_weight * led_delta_hidden
                led_delta = 0.08 * self.innovation_limit * torch.tanh(
                    self.led_min_delta_head(led_hidden) / max(self.innovation_limit, 1e-6)
                )
                blended_led_delta = 0.86 * teacher_delta_target + 0.14 * led_delta
                if base_control_mode in teacher_audit_modes:
                    pred = selected_teacher_pred
                    innovation_score = hard_teacher_score * torch.abs(teacher_delta_target)
                    ctx_delta = pred - anchor_pred
                elif base_control_mode in {
                    "latent_only_led_curriculum",
                    "latent_only_led_softgate",
                    "latent_only_led_curriculum_softgate",
                    "latent_only_led_teacher_ensemble",
                    "latent_only_led_teacher_student",
                    "latent_only_led_teacher_hybrid",
                    "latent_only_led_teacher_ensemble_topk",
                    "latent_only_led_teacher_student_topk",
                    "latent_only_led_teacher_unimol_bank",
                    "latent_only_led_teacher_unimol2_bank",
                    "latent_only_led_teacher_unimol2_headspec",
                    "latent_only_led_teacher_unimol2_ensemble",
                    "latent_only_led_teacher_polymer_unimol_bank",
                    "latent_only_led_teacher_mmpolymer_bank",
                }:
                    pred = anchor_pred
                    innovation_score = hard_teacher_score * torch.abs(led_delta_hidden).mean(dim=1, keepdim=True)
                    ctx_delta = torch.zeros_like(anchor_pred)
                elif base_control_mode == "led_off":
                    pred = anchor_pred
                    led_weight = torch.zeros_like(led_weight)
                    innovation_score = torch.zeros_like(anchor_pred)
                    ctx_delta = torch.zeros_like(anchor_pred)
                else:
                    pred = anchor_pred + led_weight * (residual_mix + (1.0 - residual_mix) * hard_teacher_score) * blended_led_delta
                    innovation_score = led_weight * (
                        0.55 * torch.abs(blended_led_delta)
                        + 0.45 * torch.abs(led_delta_hidden).mean(dim=1, keepdim=True)
                    )
                    ctx_delta = pred - anchor_pred
                payload = dict(parts)
                payload.update(
                    {
                        "pred": pred,
                        "baseline_pred": anchor_pred,
                        "teacher_anchor_pred": anchor_pred,
                        "innovation_pred": pred,
                        "innovation_score": innovation_score,
                        "fused_latent": led_hidden,
                        "baseline_unc": baseline_unc,
                        "rcmf_gate": parts["rcmf_min_gate"],
                        "ctx_delta": ctx_delta,
                        "mspce_anchor_pred": anchor_pred,
                        "led_min_teacher_latent": selected_teacher_latent,
                        "led_min_teacher_pred": selected_teacher_pred,
                        "led_min_bridge": led_bridge,
                        "led_min_delta_hidden": led_delta_hidden,
                        "led_min_gate": soft_gate,
                        "led_min_soft_gate": soft_gate,
                        "led_min_confidence": led_confidence,
                        "led_min_teacher_mask": teacher_available,
                        "led_min_teacher_reliability": base_teacher_reliability,
                        "led_min_top_reliability_gate": top_reliability_gate,
                        "led_min_hard_teacher_score": hard_teacher_score,
                        "led_min_teacher_delta_target": teacher_delta_target,
                        "led_min_weight": led_weight,
                        "led_min_delta": blended_led_delta,
                        "led_teacher_head_disagreement": teacher_head_disagreement,
                        "led_min_hidden_consistency": torch.mean(
                            (led_hidden - anchor_hidden) ** 2,
                            dim=1,
                            keepdim=True,
                        ),
                        "led_min_teacher_alignment": torch.mean(
                            (led_hidden - selected_teacher_latent.detach()) ** 2,
                            dim=1,
                            keepdim=True,
                        ),
                    }
                )
                return payload
            regime_summary = self.regime_summary(torch.cat([parts["mspce_repair_hidden"], regime_features], dim=1))
            regime_posterior = torch.softmax(
                self.regime_posterior(torch.cat([regime_summary, regime_features], dim=1)),
                dim=1,
            )
            regime_token = regime_posterior @ self.regime_token_bank.weight
            regime_teacher_latent = torch.tanh(
                self.regime_teacher_proj(torch.cat([led_teacher_latent, regime_token], dim=1))
            )
            regime_led_focus = torch.sigmoid(self.regime_led_gate(torch.cat([regime_token, regime_features], dim=1)))
            prototype_profile = full_control_mode in {"led_profile_proto", "latent_only_led_proto"}
            prototype_ranked_profile = full_control_mode in {"led_profile_proto_ranked", "latent_only_led_proto_ranked"}
            prototype_topk_profile = full_control_mode in {"led_profile_proto_topk", "latent_only_led_proto_topk"}
            regime_profile = full_control_mode in {"led_profile_regime", "latent_only_led_regime"}
            regime_topk_profile = full_control_mode in {"led_profile_regime_topk", "latent_only_led_regime_topk"}
            consensus_profile = full_control_mode in {"led_profile_consensus", "latent_only_led_consensus"}
            consensus_ranked_profile = full_control_mode in {
                "led_profile_consensus_ranked",
                "latent_only_led_consensus_ranked",
            }
            consensus_topk_profile = full_control_mode in {
                "led_profile_consensus_topk",
                "latent_only_led_consensus_topk",
            }
            teacher_struct_latent = 0.62 * led_teacher_latent + 0.38 * parts["mspce_repair_hidden"]
            teacher_relation_latent = led_teacher_latent - parts["rcmf_min_hidden"]
            teacher_local_norm = nn.functional.normalize(led_teacher_latent, dim=1)
            teacher_struct_norm = nn.functional.normalize(teacher_struct_latent, dim=1)
            teacher_relation_norm = nn.functional.normalize(teacher_relation_latent, dim=1)
            teacher_local_struct = (teacher_local_norm * teacher_struct_norm).sum(dim=1, keepdim=True)
            teacher_local_relation = (teacher_local_norm * teacher_relation_norm).sum(dim=1, keepdim=True)
            teacher_struct_relation = (teacher_struct_norm * teacher_relation_norm).sum(dim=1, keepdim=True)
            teacher_consensus_agreement = torch.clamp(
                (teacher_local_struct + teacher_local_relation + teacher_struct_relation + 3.0) / 6.0,
                min=0.0,
                max=1.0,
            )
            consensus_logits = torch.cat(
                [
                    teacher_local_struct + teacher_local_relation,
                    teacher_local_struct + teacher_struct_relation,
                    teacher_local_relation + teacher_struct_relation,
                ],
                dim=1,
            )
            consensus_weights = torch.softmax(3.8 * consensus_logits, dim=1)
            consensus_teacher_latent = (
                consensus_weights[:, 0:1] * led_teacher_latent
                + consensus_weights[:, 1:2] * teacher_struct_latent
                + consensus_weights[:, 2:3] * teacher_relation_latent
            )
            local_global_gap = torch.mean(torch.abs(teacher_struct_latent - teacher_relation_latent), dim=1, keepdim=True)
            prototype_teacher_anchor = torch.tanh(
                self.led_proto_teacher_proj(
                    torch.cat([led_teacher_latent, teacher_struct_latent, regime_teacher_latent], dim=1)
                )
            )
            proto_bank = nn.functional.normalize(self.led_proto_bank.weight, dim=1)
            student_proto_logits = self.led_proto_selector(
                torch.cat([regime_token, parts["mspce_repair_hidden"], parts["baseline_rel"]], dim=1)
            )
            teacher_proto_logits = 6.0 * (nn.functional.normalize(prototype_teacher_anchor, dim=1) @ proto_bank.transpose(0, 1))
            if prototype_ranked_profile:
                student_proto_logits = 1.18 * student_proto_logits
                teacher_proto_logits = 1.12 * teacher_proto_logits
            elif prototype_topk_profile:
                student_proto_logits = 1.34 * student_proto_logits
                teacher_proto_logits = 1.26 * teacher_proto_logits
            if prototype_topk_profile:
                teacher_topk = torch.topk(teacher_proto_logits, k=2, dim=1).indices
                teacher_mask = torch.zeros_like(teacher_proto_logits).scatter_(1, teacher_topk, 1.0)
                teacher_proto_logits = teacher_proto_logits.masked_fill(teacher_mask < 0.5, -1e4)
                student_topk = torch.topk(student_proto_logits, k=2, dim=1).indices
                student_mask = torch.zeros_like(student_proto_logits).scatter_(1, student_topk, 1.0)
                student_proto_logits = student_proto_logits.masked_fill(student_mask < 0.5, -1e4)
            student_proto_weights = torch.softmax(student_proto_logits, dim=1)
            teacher_proto_weights = torch.softmax(teacher_proto_logits, dim=1)
            prototype_student_latent = student_proto_weights @ self.led_proto_bank.weight
            prototype_teacher_latent = teacher_proto_weights @ self.led_proto_bank.weight
            prototype_agreement = torch.clamp(
                0.5 * (
                    (nn.functional.normalize(prototype_student_latent, dim=1) * nn.functional.normalize(prototype_teacher_latent, dim=1)).sum(dim=1, keepdim=True)
                    + 1.0
                ),
                min=0.0,
                max=1.0,
            )
            prototype_gap = torch.mean(torch.abs(student_proto_weights - teacher_proto_weights), dim=1, keepdim=True)
            if prototype_topk_profile:
                bridge_teacher_latent = 0.82 * prototype_teacher_latent + 0.18 * prototype_teacher_anchor
            elif prototype_ranked_profile:
                bridge_teacher_latent = 0.72 * prototype_teacher_latent + 0.28 * prototype_teacher_anchor
            elif prototype_profile:
                bridge_teacher_latent = 0.62 * prototype_teacher_latent + 0.38 * prototype_teacher_anchor
            elif regime_topk_profile:
                bridge_teacher_latent = 0.20 * led_teacher_latent + 0.80 * regime_teacher_latent
            elif regime_profile:
                bridge_teacher_latent = 0.35 * led_teacher_latent + 0.65 * regime_teacher_latent
            else:
                bridge_teacher_latent = consensus_teacher_latent if (consensus_profile or consensus_ranked_profile or consensus_topk_profile) else led_teacher_latent
            bridge_input = torch.cat(
                [
                    parts["rcmf_min_hidden"],
                    bridge_teacher_latent,
                    parts["mspce_repair_hidden"],
                    parts["baseline_rel"],
                ],
                dim=1,
            )
            led_bridge = self.led_min_bridge(bridge_input)
            led_rank = torch.tanh(self.led_min_down(led_bridge))
            led_delta_hidden = torch.tanh(self.led_min_up(led_rank))
            confidence_input = torch.cat([led_bridge, parts["baseline_rel"]], dim=1)
            teacher_available = led_mask
            led_confidence = torch.sigmoid(self.led_min_confidence(confidence_input))
            led_confidence = led_confidence * torch.sigmoid(1.8 - 1.2 * uncertainty_level - 0.9 * conflict_level)
            harddelta_profile = full_control_mode in {"led_profile_harddelta", "latent_only_led_harddelta"}
            toptranche_profile = full_control_mode in {"led_profile_toptranche", "latent_only_led_toptranche"}
            relprior_profile = full_control_mode in {"led_profile_relprior", "latent_only_led_relprior"}
            teacher_delta_clip = 0.35
            if prototype_topk_profile:
                teacher_delta_clip = 0.14
            elif prototype_ranked_profile:
                teacher_delta_clip = 0.16
            elif prototype_profile:
                teacher_delta_clip = 0.20
            elif regime_topk_profile:
                teacher_delta_clip = 0.14
            elif regime_profile:
                teacher_delta_clip = 0.18
            elif consensus_topk_profile:
                teacher_delta_clip = 0.16
            elif consensus_ranked_profile:
                teacher_delta_clip = 0.18
            elif consensus_profile:
                teacher_delta_clip = 0.22
            elif harddelta_profile:
                teacher_delta_clip = 0.30
            elif toptranche_profile:
                teacher_delta_clip = 0.20
            elif relprior_profile:
                teacher_delta_clip = 0.18
            teacher_delta_target = torch.clamp(
                led_teacher_pred.detach() - parts["rcmf_min_pred"].detach(),
                min=-teacher_delta_clip * self.innovation_limit,
                max=teacher_delta_clip * self.innovation_limit,
            )
            base_teacher_reliability = teacher_available * led_confidence * torch.sigmoid(
                2.9 * (led_confidence - parts["rcmf_min_risk"] - 0.16)
            )
            if prototype_topk_profile:
                base_teacher_reliability = base_teacher_reliability * torch.sigmoid(18.0 * (prototype_agreement - 0.78))
            elif prototype_ranked_profile:
                base_teacher_reliability = base_teacher_reliability * torch.sigmoid(16.0 * (prototype_agreement - 0.70))
            elif prototype_profile:
                base_teacher_reliability = base_teacher_reliability * torch.sigmoid(14.0 * (prototype_agreement - 0.64))
            elif regime_topk_profile:
                base_teacher_reliability = base_teacher_reliability * torch.sigmoid(18.0 * (regime_led_focus - 0.72))
            elif regime_profile:
                base_teacher_reliability = base_teacher_reliability * torch.sigmoid(14.0 * (regime_led_focus - 0.62))
            consensus_gate = torch.ones_like(base_teacher_reliability)
            if consensus_profile or consensus_ranked_profile or consensus_topk_profile:
                consensus_shift = 0.74
                consensus_scale = 15.0
                if consensus_ranked_profile:
                    consensus_shift = 0.80
                    consensus_scale = 18.0
                elif consensus_topk_profile:
                    consensus_shift = 0.86
                    consensus_scale = 22.0
                consensus_gate = torch.sigmoid(
                    consensus_scale * (teacher_consensus_agreement - consensus_shift)
                )
            teacher_reliability = base_teacher_reliability * consensus_gate
            strict_led_profile = full_control_mode in {
                "led_profile_ranked",
                "latent_only_led_ranked",
                "led_profile_harddelta",
                "latent_only_led_harddelta",
                "led_profile_toptranche",
                "latent_only_led_toptranche",
                "led_profile_relprior",
                "latent_only_led_relprior",
                "led_profile_consensus",
                "latent_only_led_consensus",
                "led_profile_consensus_ranked",
                "latent_only_led_consensus_ranked",
                "led_profile_consensus_topk",
                "latent_only_led_consensus_topk",
                "led_profile_regime",
                "latent_only_led_regime",
                "led_profile_regime_topk",
                "latent_only_led_regime_topk",
                "led_profile_proto",
                "latent_only_led_proto",
                "led_profile_proto_ranked",
                "latent_only_led_proto_ranked",
                "led_profile_proto_topk",
                "latent_only_led_proto_topk",
            }
            if prototype_topk_profile:
                top_gate_shift = 0.74
                top_gate_scale = 22.0
                delta_gate_shift = 0.008
                delta_gate_scale = 18.0
                blend_teacher = 1.00
                activation_floor = 0.82
                gate_scale = 0.11
                gate_bias = 5.2
                gate_mix = 1.00
            elif prototype_ranked_profile:
                top_gate_shift = 0.68
                top_gate_scale = 18.0
                delta_gate_shift = 0.010
                delta_gate_scale = 16.0
                blend_teacher = 0.98
                activation_floor = 0.74
                gate_scale = 0.09
                gate_bias = 4.8
                gate_mix = 0.98
            elif prototype_profile:
                top_gate_shift = 0.62
                top_gate_scale = 15.0
                delta_gate_shift = 0.012
                delta_gate_scale = 14.0
                blend_teacher = 0.95
                activation_floor = 0.66
                gate_scale = 0.08
                gate_bias = 4.5
                gate_mix = 0.96
            elif regime_topk_profile:
                top_gate_shift = 0.76
                top_gate_scale = 20.0
                delta_gate_shift = 0.008
                delta_gate_scale = 18.0
                blend_teacher = 1.00
                activation_floor = 0.84
                gate_scale = 0.10
                gate_bias = 5.2
                gate_mix = 1.00
            elif regime_profile:
                top_gate_shift = 0.68
                top_gate_scale = 16.0
                delta_gate_shift = 0.010
                delta_gate_scale = 16.0
                blend_teacher = 0.96
                activation_floor = 0.72
                gate_scale = 0.08
                gate_bias = 4.8
                gate_mix = 0.96
            elif consensus_topk_profile:
                top_gate_shift = 0.82
                top_gate_scale = 24.0
                delta_gate_shift = 0.010
                delta_gate_scale = 18.0
                blend_teacher = 0.995
                activation_floor = 0.90
                gate_scale = 0.14
                gate_bias = 5.8
                gate_mix = 1.00
            elif consensus_ranked_profile:
                top_gate_shift = 0.74
                top_gate_scale = 20.0
                delta_gate_shift = 0.012
                delta_gate_scale = 16.0
                blend_teacher = 0.97
                activation_floor = 0.76
                gate_scale = 0.12
                gate_bias = 5.0
                gate_mix = 0.98
            elif consensus_profile:
                top_gate_shift = 0.68
                top_gate_scale = 16.0
                delta_gate_shift = 0.014
                delta_gate_scale = 14.0
                blend_teacher = 0.94
                activation_floor = 0.68
                gate_scale = 0.10
                gate_bias = 4.7
                gate_mix = 0.96
            elif relprior_profile:
                top_gate_shift = 0.91
                top_gate_scale = 26.0
                delta_gate_shift = 0.010
                delta_gate_scale = 18.0
                blend_teacher = 0.98
                activation_floor = 0.95
                gate_scale = 0.06
                gate_bias = 5.6
                gate_mix = 1.00
            elif toptranche_profile:
                top_gate_shift = 0.88
                top_gate_scale = 24.0
                delta_gate_shift = 0.012
                delta_gate_scale = 16.0
                blend_teacher = 1.00
                activation_floor = 0.92
                gate_scale = 0.10
                gate_bias = 5.2
                gate_mix = 1.00
            elif harddelta_profile:
                top_gate_shift = 0.74
                top_gate_scale = 15.0
                delta_gate_shift = 0.020
                delta_gate_scale = 12.0
                blend_teacher = 0.95
                activation_floor = 0.68
                gate_scale = 0.08
                gate_bias = 4.3
                gate_mix = 0.92
            else:
                top_gate_shift = 0.66 if not strict_led_profile else 0.72
                top_gate_scale = 11.0 if not strict_led_profile else 14.0
                delta_gate_shift = 0.016 if not strict_led_profile else 0.022
                delta_gate_scale = 8.0 if not strict_led_profile else 10.0
                blend_teacher = 0.82 if not strict_led_profile else 0.90
                activation_floor = 0.56 if not strict_led_profile else 0.66
                gate_scale = 0.05
                gate_bias = 4.0
                gate_mix = 0.80
            top_reliability_gate = torch.sigmoid(top_gate_scale * (teacher_reliability - top_gate_shift))
            teacher_delta_gate = torch.sigmoid(delta_gate_scale * (torch.abs(teacher_delta_target) - delta_gate_shift))
            if prototype_profile or prototype_ranked_profile or prototype_topk_profile:
                top_reliability_gate = top_reliability_gate * torch.sigmoid(
                    16.0 * (prototype_agreement - (0.60 if prototype_profile else (0.68 if prototype_ranked_profile else 0.78)))
                )
            if regime_profile or regime_topk_profile:
                top_reliability_gate = top_reliability_gate * torch.sigmoid(
                    18.0 * (regime_led_focus - (0.66 if regime_profile else 0.78))
                )
            if consensus_profile or consensus_ranked_profile or consensus_topk_profile:
                consensus_quantile = 0.88
                if consensus_ranked_profile:
                    consensus_quantile = 0.91
                elif consensus_topk_profile:
                    consensus_quantile = 0.95
                active_teacher = (teacher_available.squeeze(1) > 0).detach()
                if bool(active_teacher.any()):
                    consensus_cut = torch.quantile(
                        teacher_reliability.detach().squeeze(1)[active_teacher],
                        consensus_quantile,
                    )
                else:
                    consensus_cut = teacher_reliability.new_tensor(1.0)
                top_reliability_gate = top_reliability_gate * torch.sigmoid(
                    22.0 * (teacher_reliability - consensus_cut)
                )
            hard_teacher_score = teacher_reliability * top_reliability_gate * teacher_delta_gate
            activation_mask = (hard_teacher_score >= activation_floor).float()
            led_gate_raw = gate_scale * torch.sigmoid(
                self.led_min_gate(
                    torch.cat([parts["rcmf_min_hidden"], led_bridge, parts["baseline_rel"]], dim=1)
                )
                - gate_bias
            )
            led_gate_raw = led_gate_raw * activation_mask * torch.clamp(0.30 + 0.70 * hard_teacher_score, max=1.0)
            led_weight = torch.clamp(
                activation_mask
                * (
                    (0.05 if relprior_profile else (0.08 if toptranche_profile else (0.12 if harddelta_profile else 0.20)))
                    + (0.95 if relprior_profile else (0.92 if toptranche_profile else (0.88 if harddelta_profile else 0.80))) * hard_teacher_score
                ),
                min=0.0,
                max=1.0,
            )
            if full_control_mode == "led_off":
                led_weight = torch.zeros_like(led_weight)
                led_gate_raw = torch.zeros_like(led_gate_raw)
            effective_led_scale = led_weight * ((1.0 - gate_mix) * led_gate_raw + gate_mix * activation_mask)
            base_teacher_hidden = led_teacher_latent
            if prototype_profile or prototype_ranked_profile or prototype_topk_profile:
                base_teacher_hidden = prototype_teacher_latent
            elif regime_profile or regime_topk_profile:
                base_teacher_hidden = regime_teacher_latent
            elif consensus_profile or consensus_ranked_profile or consensus_topk_profile:
                base_teacher_hidden = consensus_teacher_latent
            if prototype_profile or prototype_ranked_profile or prototype_topk_profile:
                teacher_relation_hidden = torch.tanh(
                    self.led_proto_relation(torch.cat([base_teacher_hidden, prototype_student_latent], dim=1))
                    - parts["rcmf_min_hidden"]
                )
            else:
                teacher_relation_hidden = torch.tanh(base_teacher_hidden - parts["rcmf_min_hidden"])
            if relprior_profile:
                led_hidden = parts["rcmf_min_hidden"] + effective_led_scale * teacher_relation_hidden
            else:
                led_hidden = parts["rcmf_min_hidden"] + effective_led_scale * led_delta_hidden
            led_delta = 0.12 * self.innovation_limit * torch.tanh(
                self.led_min_delta_head(led_hidden) / max(self.innovation_limit, 1e-6)
            )
            consensus_delta_target = teacher_delta_target * torch.clamp(
                0.42 + 0.58 * teacher_consensus_agreement,
                min=0.0,
                max=1.0,
            )
            regime_delta_target = teacher_delta_target * torch.clamp(
                0.46 + 0.54 * regime_led_focus,
                min=0.0,
                max=1.0,
            )
            prototype_delta_target = teacher_delta_target * torch.clamp(
                0.34 + 0.66 * prototype_agreement,
                min=0.0,
                max=1.0,
            )
            if prototype_topk_profile:
                blended_led_delta = prototype_delta_target
            elif prototype_ranked_profile:
                blended_led_delta = 0.97 * prototype_delta_target + 0.03 * led_delta
            elif prototype_profile:
                blended_led_delta = 0.94 * prototype_delta_target + 0.06 * led_delta
            elif regime_topk_profile:
                blended_led_delta = regime_delta_target
            elif regime_profile:
                blended_led_delta = 0.94 * regime_delta_target + 0.06 * led_delta
            elif consensus_topk_profile:
                blended_led_delta = consensus_delta_target
            elif consensus_ranked_profile:
                blended_led_delta = 0.96 * consensus_delta_target + 0.04 * led_delta
            elif consensus_profile:
                blended_led_delta = 0.92 * consensus_delta_target + 0.08 * led_delta
            elif relprior_profile:
                blended_led_delta = 0.92 * teacher_delta_target + 0.08 * led_delta
            elif toptranche_profile:
                blended_led_delta = teacher_delta_target
            elif harddelta_profile:
                blended_led_delta = teacher_delta_target + 0.08 * led_delta
            else:
                blended_led_delta = blend_teacher * teacher_delta_target + (1.0 - blend_teacher) * led_delta
            if full_control_mode in {
                "latent_only_led",
                "latent_only_led_ranked",
                "latent_only_led_harddelta",
                "latent_only_led_toptranche",
                "latent_only_led_relprior",
                "latent_only_led_consensus",
                "latent_only_led_consensus_ranked",
                "latent_only_led_consensus_topk",
                "latent_only_led_regime",
                "latent_only_led_regime_topk",
                "latent_only_led_proto",
                "latent_only_led_proto_ranked",
                "latent_only_led_proto_topk",
            }:
                pred = parts["rcmf_min_pred"]
                innovation_score = hard_teacher_score * torch.abs(teacher_relation_hidden if relprior_profile else led_delta_hidden).mean(dim=1, keepdim=True)
                ctx_delta = torch.zeros_like(parts["rcmf_min_pred"])
            else:
                if prototype_profile or prototype_ranked_profile or prototype_topk_profile:
                    effective_led_scale = activation_mask * torch.clamp(
                        0.24 + 0.76 * prototype_agreement,
                        min=0.0,
                        max=1.0,
                    )
                elif regime_profile or regime_topk_profile:
                    effective_led_scale = activation_mask * torch.clamp(
                        0.28 + 0.72 * regime_led_focus,
                        min=0.0,
                        max=1.0,
                    )
                elif consensus_profile or consensus_ranked_profile or consensus_topk_profile:
                    effective_led_scale = activation_mask * torch.clamp(
                        0.30 + 0.70 * teacher_consensus_agreement,
                        min=0.0,
                        max=1.0,
                    )
                elif relprior_profile:
                    effective_led_scale = activation_mask
                elif toptranche_profile:
                    effective_led_scale = activation_mask
                elif harddelta_profile:
                    effective_led_scale = activation_mask * torch.clamp(0.35 + 0.65 * hard_teacher_score, max=1.0)
                pred = parts["rcmf_min_pred"] + effective_led_scale * blended_led_delta
                innovation_score = effective_led_scale * (
                    0.5 * torch.abs(blended_led_delta)
                    + 0.5 * torch.abs(led_delta_hidden).mean(dim=1, keepdim=True)
                )
                ctx_delta = pred - parts["rcmf_min_pred"]
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": parts["rcmf_min_pred"],
                    "teacher_anchor_pred": parts["rcmf_min_pred"],
                    "innovation_pred": pred,
                    "innovation_score": innovation_score,
                    "fused_latent": led_hidden,
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["rcmf_min_gate"],
                    "ctx_delta": ctx_delta,
                    "mspce_anchor_pred": parts["rcmf_min_pred"],
                    "led_min_teacher_latent": led_teacher_latent,
                    "led_min_teacher_pred": led_teacher_pred,
                    "led_min_bridge": led_bridge,
                    "led_min_delta_hidden": led_delta_hidden,
                    "led_min_gate": led_gate_raw,
                    "led_min_confidence": led_confidence,
                    "led_min_activation_mask": activation_mask,
                    "led_min_teacher_mask": teacher_available,
                    "led_min_teacher_reliability": teacher_reliability,
                    "led_min_top_reliability_gate": top_reliability_gate,
                    "led_min_hard_teacher_score": hard_teacher_score,
                    "led_min_teacher_delta_target": teacher_delta_target,
                    "led_min_consensus_latent": consensus_teacher_latent,
                    "led_min_consensus_score": teacher_consensus_agreement,
                    "led_min_consensus_delta_target": consensus_delta_target,
                    "led_min_local_global_gap": local_global_gap,
                    "led_min_regime_summary": regime_summary,
                    "led_min_regime_posterior_max": regime_posterior.max(dim=1, keepdim=True).values,
                    "led_min_regime_token": regime_token,
                    "led_min_regime_focus": regime_led_focus,
                    "led_min_regime_teacher_latent": regime_teacher_latent,
                    "led_min_regime_delta_target": regime_delta_target,
                    "led_min_prototype_teacher_latent": prototype_teacher_latent,
                    "led_min_prototype_student_latent": prototype_student_latent,
                    "led_min_prototype_score": prototype_agreement,
                    "led_min_prototype_gap": prototype_gap,
                    "led_min_prototype_delta_target": prototype_delta_target,
                    "led_min_weight": effective_led_scale,
                    "led_min_delta": blended_led_delta,
                    "led_min_hidden_consistency": torch.mean(
                        (led_hidden - parts["rcmf_min_hidden"]) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "led_min_teacher_alignment": torch.mean(
                        (led_hidden - led_teacher_latent.detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                }
            )
            return payload

        if self.mode == "full_min_anchor_preserving":
            if led is None:
                led = torch.zeros_like(parts["ctx_emb"])
            if led_mask is None:
                led_mask = torch.zeros_like(parts["rcmf_min_pred"])
            led_teacher_latent, led_teacher_pred = self.led_prior(led)
            conflict_level = torch.abs(parts["desc_pred"] - parts["graph_pred"])
            uncertainty_level = 0.5 * (parts["desc_unc"] + parts["graph_unc"])
            led_bridge_input = torch.cat(
                [
                    parts["rcmf_min_hidden"],
                    led_teacher_latent,
                    parts["mspce_repair_hidden"],
                    parts["baseline_rel"],
                ],
                dim=1,
            )
            led_bridge = self.led_min_bridge(led_bridge_input)
            led_rank = torch.tanh(self.led_min_down(led_bridge))
            led_delta_hidden = torch.tanh(self.led_min_up(led_rank))
            led_confidence = torch.sigmoid(self.led_min_confidence(torch.cat([led_bridge, parts["baseline_rel"]], dim=1)))
            led_confidence = led_confidence * torch.sigmoid(1.8 - 1.2 * uncertainty_level - 0.9 * conflict_level)
            led_activation_mask = (led_confidence >= 0.52).float()
            teacher_available = led_mask
            led_gate_raw = 0.05 * torch.sigmoid(
                self.led_min_gate(torch.cat([parts["rcmf_min_hidden"], led_bridge, parts["baseline_rel"]], dim=1)) - 4.0
            )
            led_weight = teacher_available * led_activation_mask * led_confidence * torch.clamp(
                1.0 - 0.70 * parts["rcmf_min_risk"],
                min=0.10,
                max=1.0,
            )
            led_hidden = parts["rcmf_min_hidden"] + led_weight * led_gate_raw * led_delta_hidden
            led_delta = 0.12 * self.innovation_limit * torch.tanh(
                self.led_min_delta_head(led_hidden) / max(self.innovation_limit, 1e-6)
            )
            led_pred = parts["rcmf_min_pred"]

            full_mix_input = torch.cat([parts["rcmf_min_hidden"], led_hidden], dim=1)
            full_mix = self.full_min_mix(full_mix_input)
            full_bridge = self.full_min_bridge(
                torch.cat(
                    [
                        parts["mspce_repair_hidden"],
                        parts["rcmf_min_hidden"],
                        led_hidden,
                        full_mix,
                        parts["baseline_rel"],
                    ],
                    dim=1,
                )
            )
            full_rank = torch.tanh(self.full_min_down(full_bridge))
            full_hidden_delta = torch.tanh(self.full_min_up(full_rank))
            high_conflict = torch.sigmoid(3.4 * (conflict_level + 1.10 * uncertainty_level - 0.48))
            low_risk = torch.clamp(1.0 - 0.82 * parts["rcmf_min_risk"], min=0.06, max=1.0)
            full_gate = 0.035 * torch.sigmoid(
                self.full_min_gate(torch.cat([parts["rcmf_min_hidden"], full_bridge, parts["baseline_rel"]], dim=1)) - 4.0
            )
            full_gate = full_gate * high_conflict * low_risk * torch.clamp(0.20 + 0.50 * led_weight, max=0.70)
            full_hidden = led_hidden + full_gate * full_hidden_delta
            full_delta = 0.10 * self.innovation_limit * torch.tanh(
                self.full_min_delta_head(full_hidden) / max(self.innovation_limit, 1e-6)
            )
            pred = led_pred + full_gate * full_delta
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": led_pred,
                    "teacher_anchor_pred": led_pred,
                    "innovation_pred": pred,
                    "innovation_score": full_gate * torch.abs(full_delta),
                    "fused_latent": full_hidden,
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["rcmf_min_gate"],
                    "ctx_delta": pred - led_pred,
                    "mspce_anchor_pred": parts["mspce_repair_pred"],
                    "full_led_anchor_pred": led_pred,
                    "full_led_anchor_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_led_pred": parts["rcmf_min_pred"],
                    "full_drop_rcmf_pred": parts["mspce_repair_pred"],
                    "full_drop_led_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_rcmf_hidden": parts["mspce_repair_hidden"].detach(),
                    "full_min_led_teacher_latent": led_teacher_latent,
                    "full_min_led_teacher_pred": led_teacher_pred,
                    "full_min_led_bridge": led_bridge,
                    "full_min_led_confidence": led_confidence,
                    "full_min_led_activation_mask": led_activation_mask,
                    "full_min_led_teacher_mask": teacher_available,
                    "full_min_led_weight": led_weight,
                    "full_min_led_gate": led_gate_raw,
                    "full_min_led_delta": led_delta,
                    "full_min_led_hidden": led_hidden,
                    "full_min_gate": full_gate,
                    "full_min_bridge": full_bridge,
                    "full_min_mix": full_mix,
                    "full_min_delta": full_delta,
                    "full_min_hidden_consistency": torch.mean(
                        (full_hidden - led_hidden) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_anchor_consistency": torch.mean(
                        (full_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_drop_led_consistency": torch.abs(pred - parts["rcmf_min_pred"].detach()),
                    "full_min_drop_rcmf_consistency": torch.abs(pred - parts["mspce_repair_pred"].detach()),
                    "full_min_high_conflict_gate": high_conflict,
                }
            )
            return payload

        if self.mode == "full_min_shared_conditional":
            if led is None:
                led = torch.zeros_like(parts["ctx_emb"])
            if led_mask is None:
                led_mask = torch.zeros_like(parts["rcmf_min_pred"])
            led_teacher_latent, led_teacher_pred = self.led_prior(led)
            conflict_level = torch.abs(parts["desc_pred"] - parts["graph_pred"])
            uncertainty_level = 0.5 * (parts["desc_unc"] + parts["graph_unc"])
            z_input = torch.cat(
                [
                    parts["ctx_emb"],
                    parts["rcmf_min_hidden"],
                    parts["baseline_rel"],
                    conflict_level,
                    uncertainty_level,
                    parts["rcmf_min_confidence"],
                ],
                dim=1,
            )
            shared_z = torch.tanh(self.full_shared_z_proj(z_input))
            high_conflict = torch.sigmoid(3.2 * (conflict_level + 1.05 * uncertainty_level - 0.46))
            low_risk = torch.clamp(1.0 - 0.82 * parts["rcmf_min_risk"], min=0.06, max=1.0)
            teacher_available = led_mask
            led_confidence = torch.sigmoid(2.0 * parts["rcmf_min_confidence"] - 0.2) * torch.sigmoid(
                1.8 - 1.1 * uncertainty_level - 0.9 * conflict_level
            )
            led_activation_mask = (led_confidence >= 0.52).float()
            rcmf_scale = 0.028 * torch.sigmoid(self.full_shared_rcmf_gate(shared_z) - 4.6) * high_conflict * low_risk
            led_scale = (
                0.024
                * torch.sigmoid(self.full_shared_led_gate(shared_z) - 4.6)
                * teacher_available
                * led_activation_mask
                * led_confidence
                * low_risk
            )
            shared_hidden_delta = torch.tanh(self.full_shared_hidden(shared_z))
            if full_control_mode == "shared_off":
                rcmf_scale = torch.zeros_like(rcmf_scale)
                led_scale = torch.zeros_like(led_scale)
            elif full_control_mode == "rcmf_only":
                led_scale = torch.zeros_like(led_scale)
            elif full_control_mode == "led_only":
                rcmf_scale = torch.zeros_like(rcmf_scale)
            shared_hidden = parts["rcmf_min_hidden"] + rcmf_scale * parts["rcmf_min_mix_hidden"] + led_scale * shared_hidden_delta
            full_gate = 0.020 * torch.sigmoid(self.full_shared_gate(shared_z) - 4.9)
            full_gate = full_gate * high_conflict * low_risk * torch.clamp(0.25 + 0.55 * led_scale / 0.024, max=0.65)
            if full_control_mode == "shared_off":
                full_gate = torch.zeros_like(full_gate)
            full_delta = 0.06 * self.innovation_limit * torch.tanh(
                self.full_shared_delta_head(shared_hidden) / max(self.innovation_limit, 1e-6)
            )
            pred = parts["rcmf_min_pred"] + full_gate * full_delta
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": parts["rcmf_min_pred"],
                    "teacher_anchor_pred": parts["rcmf_min_pred"],
                    "innovation_pred": pred,
                    "innovation_score": full_gate * torch.abs(full_delta),
                    "fused_latent": shared_hidden,
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["rcmf_min_gate"],
                    "ctx_delta": pred - parts["rcmf_min_pred"],
                    "mspce_anchor_pred": parts["mspce_repair_pred"],
                    "full_led_anchor_pred": parts["rcmf_min_pred"],
                    "full_led_anchor_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_led_pred": parts["rcmf_min_pred"],
                    "full_drop_rcmf_pred": parts["mspce_repair_pred"],
                    "full_drop_led_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_rcmf_hidden": parts["mspce_repair_hidden"].detach(),
                    "full_min_led_teacher_latent": led_teacher_latent,
                    "full_min_led_teacher_pred": led_teacher_pred,
                    "full_min_led_bridge": shared_hidden_delta,
                    "full_min_led_confidence": led_confidence,
                    "full_min_led_activation_mask": led_activation_mask,
                    "full_min_led_teacher_mask": teacher_available,
                    "full_min_led_weight": led_scale / 0.024,
                    "full_min_led_gate": led_scale / 0.024,
                    "full_min_led_delta": torch.zeros_like(pred),
                    "full_min_led_hidden": shared_hidden,
                    "full_min_gate": full_gate,
                    "full_min_bridge": shared_hidden_delta,
                    "full_min_mix": parts["rcmf_min_mix_hidden"],
                    "full_min_delta": full_delta,
                    "full_min_hidden_consistency": torch.mean(
                        (shared_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_anchor_consistency": torch.mean(
                        (shared_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_drop_led_consistency": torch.abs(pred - parts["rcmf_min_pred"].detach()),
                    "full_min_drop_rcmf_consistency": torch.abs(pred - parts["mspce_repair_pred"].detach()),
                    "full_min_high_conflict_gate": high_conflict,
                    "full_min_shared_z": shared_z,
                    "full_min_shared_rcmf_scale": rcmf_scale,
                    "full_min_shared_led_scale": led_scale,
                }
            )
            return payload

        if self.mode == "full_min_factorized_conditional":
            if led is None:
                led = torch.zeros_like(parts["ctx_emb"])
            if led_mask is None:
                led_mask = torch.zeros_like(parts["rcmf_min_pred"])
            led_teacher_latent, led_teacher_pred = self.led_prior(led)
            conflict_level = torch.abs(parts["desc_pred"] - parts["graph_pred"])
            uncertainty_level = 0.5 * (parts["desc_unc"] + parts["graph_unc"])
            ctx_input = torch.cat([parts["ctx_chain_emb"], parts["ctx_graph_emb"]], dim=1)
            rel_input = torch.cat(
                [
                    parts["baseline_rel"],
                    conflict_level,
                    uncertainty_level,
                    parts["rcmf_min_confidence"],
                ],
                dim=1,
            )
            z_ctx = torch.tanh(self.full_factor_ctx_proj(ctx_input))
            z_rel = torch.tanh(self.full_factor_rel_proj(rel_input))
            z_rel_control = z_rel
            z_joint_control = torch.cat([z_ctx, z_rel_control], dim=1)
            if full_control_mode == "ctx_rcmf_only":
                z_rel_control = torch.zeros_like(z_rel)
                z_joint_control = torch.cat([z_ctx, z_rel_control], dim=1)
            high_conflict = torch.sigmoid(3.6 * (conflict_level + 1.15 * uncertainty_level - 0.46))
            low_risk = torch.clamp(1.0 - 0.85 * parts["rcmf_min_risk"], min=0.06, max=1.0)
            teacher_available = led_mask
            led_confidence = torch.sigmoid(2.2 * z_rel_control[:, :1]) * torch.sigmoid(
                1.9 - 1.1 * uncertainty_level - 0.8 * conflict_level
            )
            led_activation_mask = (led_confidence >= 0.50).float()
            rcmf_scale = 0.024 * torch.sigmoid(self.full_factor_rcmf_gate(z_joint_control) - 4.8) * high_conflict * low_risk
            led_scale = (
                0.016
                * torch.sigmoid(self.full_factor_led_gate(z_rel_control) - 4.9)
                * teacher_available
                * led_activation_mask
                * led_confidence
                * low_risk
            )
            if full_control_mode in {"shared_off", "factorized_off"}:
                rcmf_scale = torch.zeros_like(rcmf_scale)
                led_scale = torch.zeros_like(led_scale)
            elif full_control_mode == "rcmf_only":
                led_scale = torch.zeros_like(led_scale)
            elif full_control_mode == "led_only":
                rcmf_scale = torch.zeros_like(rcmf_scale)
            elif full_control_mode == "ctx_rcmf_only":
                led_scale = torch.zeros_like(led_scale)
            elif full_control_mode == "rel_led_only":
                rcmf_scale = torch.zeros_like(rcmf_scale)
            ctx_hidden_delta = torch.tanh(self.full_factor_ctx_hidden(z_ctx))
            rel_hidden_delta = torch.tanh(self.full_factor_rel_hidden(z_rel))
            context_rcmf_hidden = 0.72 * ctx_hidden_delta + 0.28 * torch.tanh(parts["rcmf_min_mix_hidden"])
            shared_hidden = parts["rcmf_min_hidden"] + rcmf_scale * context_rcmf_hidden + 0.18 * led_scale * ctx_hidden_delta
            full_gate = 0.014 * torch.sigmoid(self.full_factor_gate(z_joint_control) - 5.1)
            full_gate = full_gate * torch.clamp(0.72 * high_conflict + 0.28 * led_confidence, max=1.0) * low_risk
            if full_control_mode in {"shared_off", "factorized_off"}:
                full_gate = torch.zeros_like(full_gate)
            full_delta = 0.045 * self.innovation_limit * torch.tanh(
                self.full_factor_delta_head(shared_hidden) / max(self.innovation_limit, 1e-6)
            )
            pred = parts["rcmf_min_pred"] + full_gate * full_delta
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": parts["rcmf_min_pred"],
                    "teacher_anchor_pred": parts["rcmf_min_pred"],
                    "innovation_pred": pred,
                    "innovation_score": full_gate * torch.abs(full_delta),
                    "fused_latent": shared_hidden,
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["rcmf_min_gate"],
                    "ctx_delta": pred - parts["rcmf_min_pred"],
                    "mspce_anchor_pred": parts["mspce_repair_pred"],
                    "full_led_anchor_pred": parts["rcmf_min_pred"],
                    "full_led_anchor_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_led_pred": parts["rcmf_min_pred"],
                    "full_drop_rcmf_pred": parts["mspce_repair_pred"],
                    "full_drop_led_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_rcmf_hidden": parts["mspce_repair_hidden"].detach(),
                    "full_min_led_teacher_latent": led_teacher_latent,
                    "full_min_led_teacher_pred": led_teacher_pred,
                    "full_min_led_bridge": ctx_hidden_delta,
                    "full_min_led_confidence": led_confidence,
                    "full_min_led_activation_mask": led_activation_mask,
                    "full_min_led_teacher_mask": teacher_available,
                    "full_min_led_weight": led_scale / 0.016,
                    "full_min_led_gate": led_scale / 0.016,
                    "full_min_led_delta": torch.zeros_like(pred),
                    "full_min_led_hidden": shared_hidden,
                    "full_min_gate": full_gate,
                    "full_min_bridge": rel_hidden_delta,
                    "full_min_mix": parts["rcmf_min_mix_hidden"],
                    "full_min_delta": full_delta,
                    "full_min_hidden_consistency": torch.mean(
                        (shared_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_anchor_consistency": torch.mean(
                        (shared_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_drop_led_consistency": torch.abs(pred - parts["rcmf_min_pred"].detach()),
                    "full_min_drop_rcmf_consistency": torch.abs(pred - parts["mspce_repair_pred"].detach()),
                    "full_min_high_conflict_gate": high_conflict,
                    "full_min_shared_z": z_joint_control,
                    "full_min_shared_rcmf_scale": rcmf_scale,
                    "full_min_shared_led_scale": led_scale,
                    "full_min_stage_rcmf_mask": high_conflict,
                    "full_min_stage_led_mask": teacher_available * led_activation_mask * led_confidence,
                }
            )
            return payload

        if self.mode == "full_min_staged_conditional":
            if led is None:
                led = torch.zeros_like(parts["ctx_emb"])
            if led_mask is None:
                led_mask = torch.zeros_like(parts["rcmf_min_pred"])
            led_teacher_latent, led_teacher_pred = self.led_prior(led)
            conflict_level = torch.abs(parts["desc_pred"] - parts["graph_pred"])
            uncertainty_level = 0.5 * (parts["desc_unc"] + parts["graph_unc"])
            ctx_input = torch.cat([parts["ctx_chain_emb"], parts["ctx_graph_emb"]], dim=1)
            rel_input = torch.cat(
                [
                    parts["baseline_rel"],
                    conflict_level,
                    uncertainty_level,
                    parts["rcmf_min_confidence"],
                ],
                dim=1,
            )
            z_ctx = torch.tanh(self.full_factor_ctx_proj(ctx_input))
            z_rel = torch.tanh(self.full_factor_rel_proj(rel_input))
            z_joint = torch.cat([z_ctx, z_rel], dim=1)
            high_conflict = torch.sigmoid(3.8 * (conflict_level + 1.20 * uncertainty_level - 0.44))
            low_risk = torch.clamp(1.0 - 0.85 * parts["rcmf_min_risk"], min=0.05, max=1.0)
            teacher_available = led_mask
            teacher_reliability = teacher_available * torch.sigmoid(2.3 * z_rel[:, :1]) * torch.sigmoid(
                1.9 - 1.0 * uncertainty_level - 0.7 * conflict_level - 0.6 * parts["rcmf_min_risk"]
            )
            u_trigger_shift = 1.5
            led_trigger_shift = 1.5
            u_trigger_threshold = 0.45
            led_trigger_threshold = 0.40
            u_mag_budget = 0.018
            led_mag_budget = 0.010
            full_gate_budget = 0.010
            monotonic_bias = 0.15
            if full_control_mode == "staged_tuned":
                u_trigger_shift = 1.42
                led_trigger_shift = 1.68
                u_trigger_threshold = 0.44
                led_trigger_threshold = 0.42
                u_mag_budget = 0.0165
                led_mag_budget = 0.0080
                full_gate_budget = 0.0088
                monotonic_bias = 0.11
            u_trigger_features = torch.cat(
                [
                    high_conflict,
                    uncertainty_level,
                    1.0 - parts["rcmf_min_confidence"],
                ],
                dim=1,
            )
            led_trigger_features = torch.cat(
                [
                    teacher_available,
                    teacher_reliability,
                    low_risk,
                ],
                dim=1,
            )
            learned_rcmf_trigger = high_conflict * low_risk * torch.sigmoid(self.full_stage_u_trigger(u_trigger_features) - u_trigger_shift)
            learned_led_trigger = teacher_available * teacher_reliability * torch.sigmoid(
                self.full_stage_led_trigger(led_trigger_features) - led_trigger_shift
            )
            u_trigger_quantile = torch.quantile(learned_rcmf_trigger.detach().squeeze(1), 0.70).reshape(1, 1)
            led_trigger_quantile = torch.quantile(learned_led_trigger.detach().squeeze(1), 0.70).reshape(1, 1)
            hard_rcmf_mask = (learned_rcmf_trigger >= torch.maximum(torch.full_like(learned_rcmf_trigger, 0.034), u_trigger_quantile)).float()
            hard_led_mask = teacher_available * (
                learned_led_trigger >= torch.maximum(torch.full_like(learned_led_trigger, 0.040), led_trigger_quantile)
            ).float()
            hard_union_mask = torch.maximum(hard_rcmf_mask, hard_led_mask)
            stage_rcmf_trigger = learned_rcmf_trigger
            stage_led_trigger = learned_led_trigger
            stage_rcmf_mask = (stage_rcmf_trigger >= u_trigger_threshold).float()
            stage_led_mask = (stage_led_trigger >= led_trigger_threshold).float()
            if full_control_mode == "shared_off":
                stage_rcmf_mask = torch.zeros_like(stage_rcmf_mask)
                stage_led_mask = torch.zeros_like(stage_led_mask)
                stage_rcmf_trigger = torch.zeros_like(stage_rcmf_trigger)
                stage_led_trigger = torch.zeros_like(stage_led_trigger)
            elif full_control_mode == "rcmf_only":
                stage_led_mask = torch.zeros_like(stage_led_mask)
                stage_led_trigger = torch.zeros_like(stage_led_trigger)
            elif full_control_mode == "led_only":
                stage_rcmf_mask = torch.zeros_like(stage_rcmf_mask)
                stage_rcmf_trigger = torch.zeros_like(stage_rcmf_trigger)
            if full_control_mode in {"hard_subgroup_gated", "hard_rcmf_only"}:
                stage_rcmf_mask = stage_rcmf_mask * hard_rcmf_mask
                stage_rcmf_trigger = stage_rcmf_trigger * hard_rcmf_mask
            if full_control_mode in {"hard_subgroup_gated", "hard_led_only"}:
                stage_led_mask = stage_led_mask * hard_led_mask
                stage_led_trigger = stage_led_trigger * hard_led_mask
            ctx_hidden_delta = torch.tanh(self.full_factor_ctx_hidden(z_ctx))
            context_rcmf_hidden = 0.75 * ctx_hidden_delta + 0.25 * torch.tanh(parts["rcmf_min_mix_hidden"])
            learned_rcmf_magnitude = u_mag_budget * torch.sigmoid(self.full_stage_u_mag(z_joint) - 5.0) * low_risk
            learned_led_magnitude = led_mag_budget * torch.sigmoid(self.full_stage_led_mag(z_rel) - 5.1) * teacher_reliability
            rcmf_magnitude = learned_rcmf_magnitude
            led_magnitude = learned_led_magnitude
            if full_control_mode == "trigger_only":
                rcmf_magnitude = u_mag_budget * low_risk
                led_magnitude = led_mag_budget * teacher_reliability
            elif full_control_mode == "magnitude_only":
                stage_rcmf_trigger = high_conflict * low_risk
                stage_led_trigger = teacher_available * teacher_reliability
                stage_rcmf_mask = (stage_rcmf_trigger >= u_trigger_threshold).float()
                stage_led_mask = (stage_led_trigger >= led_trigger_threshold).float()
            rcmf_scale = stage_rcmf_mask * stage_rcmf_trigger * rcmf_magnitude
            led_scale = stage_led_mask * stage_led_trigger * led_magnitude
            shared_hidden = parts["rcmf_min_hidden"] + rcmf_scale * context_rcmf_hidden + 0.10 * led_scale * ctx_hidden_delta
            full_gate = full_gate_budget * torch.sigmoid(self.full_stage_gate(z_joint) - 5.3)
            full_gate = full_gate * torch.maximum(stage_rcmf_mask * stage_rcmf_trigger, stage_led_mask * stage_led_trigger) * low_risk
            if full_control_mode == "shared_off":
                full_gate = torch.zeros_like(full_gate)
            full_delta = 0.030 * self.innovation_limit * torch.tanh(
                self.full_factor_delta_head(shared_hidden) / max(self.innovation_limit, 1e-6)
            )
            pred = parts["rcmf_min_pred"] + full_gate * full_delta
            monotonic_mask = torch.sigmoid(
                4.0 * (parts["rcmf_min_confidence"] - high_conflict - 0.7 * uncertainty_level - monotonic_bias)
            )
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": parts["rcmf_min_pred"],
                    "teacher_anchor_pred": parts["rcmf_min_pred"],
                    "innovation_pred": pred,
                    "innovation_score": full_gate * torch.abs(full_delta),
                    "fused_latent": shared_hidden,
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["rcmf_min_gate"],
                    "ctx_delta": pred - parts["rcmf_min_pred"],
                    "mspce_anchor_pred": parts["mspce_repair_pred"],
                    "full_led_anchor_pred": parts["rcmf_min_pred"],
                    "full_led_anchor_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_led_pred": parts["rcmf_min_pred"],
                    "full_drop_rcmf_pred": parts["mspce_repair_pred"],
                    "full_drop_led_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_rcmf_hidden": parts["mspce_repair_hidden"].detach(),
                    "full_min_led_teacher_latent": led_teacher_latent,
                    "full_min_led_teacher_pred": led_teacher_pred,
                    "full_min_led_bridge": ctx_hidden_delta,
                    "full_min_led_confidence": teacher_reliability,
                    "full_min_led_activation_mask": stage_led_mask,
                    "full_min_led_teacher_mask": teacher_available,
                    "full_min_led_weight": led_scale / 0.010,
                    "full_min_led_gate": stage_led_trigger,
                    "full_min_led_delta": torch.zeros_like(pred),
                    "full_min_led_hidden": shared_hidden,
                    "full_min_gate": full_gate,
                    "full_min_bridge": z_rel,
                    "full_min_mix": parts["rcmf_min_mix_hidden"],
                    "full_min_delta": full_delta,
                    "full_min_hidden_consistency": torch.mean(
                        (shared_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_anchor_consistency": torch.mean(
                        (shared_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_drop_led_consistency": torch.abs(pred - parts["rcmf_min_pred"].detach()),
                    "full_min_drop_rcmf_consistency": torch.abs(pred - parts["mspce_repair_pred"].detach()),
                    "full_min_high_conflict_gate": high_conflict,
                    "full_min_shared_z": z_joint,
                    "full_min_shared_rcmf_scale": rcmf_scale,
                    "full_min_shared_led_scale": led_scale,
                    "full_min_stage_rcmf_mask": stage_rcmf_mask,
                    "full_min_stage_led_mask": stage_led_mask,
                    "full_min_stage_rcmf_trigger": stage_rcmf_trigger,
                    "full_min_stage_led_trigger": stage_led_trigger,
                    "full_min_stage_rcmf_magnitude": rcmf_magnitude,
                    "full_min_stage_led_magnitude": led_magnitude,
                    "full_min_hard_rcmf_mask": hard_rcmf_mask,
                    "full_min_hard_led_mask": hard_led_mask,
                    "full_min_hard_union_mask": hard_union_mask,
                    "full_min_monotonic_mask": monotonic_mask,
                }
            )
            return payload

        if self.mode == "full_min_bucketed_conditional":
            if led is None:
                led = torch.zeros_like(parts["ctx_emb"])
            if led_mask is None:
                led_mask = torch.zeros_like(parts["rcmf_min_pred"])
            led_teacher_latent, led_teacher_pred = self.led_prior(led)
            conflict_level = torch.abs(parts["desc_pred"] - parts["graph_pred"])
            uncertainty_level = 0.5 * (parts["desc_unc"] + parts["graph_unc"])
            ctx_input = torch.cat([parts["ctx_chain_emb"], parts["ctx_graph_emb"]], dim=1)
            rel_input = torch.cat(
                [
                    parts["baseline_rel"],
                    conflict_level,
                    uncertainty_level,
                    parts["rcmf_min_confidence"],
                ],
                dim=1,
            )
            z_ctx = torch.tanh(self.full_factor_ctx_proj(ctx_input))
            z_rel = torch.tanh(self.full_factor_rel_proj(rel_input))
            high_conflict = torch.sigmoid(3.7 * (conflict_level + 1.15 * uncertainty_level - 0.45))
            low_risk = torch.clamp(1.0 - 0.85 * parts["rcmf_min_risk"], min=0.05, max=1.0)
            teacher_available = led_mask
            teacher_reliability = teacher_available * torch.sigmoid(2.3 * z_rel[:, :1]) * torch.sigmoid(
                1.9 - 1.0 * uncertainty_level - 0.7 * conflict_level - 0.6 * parts["rcmf_min_risk"]
            )
            bucket_id = torch.zeros(high_conflict.shape[0], dtype=torch.long, device=high_conflict.device)
            has_teacher = teacher_available.squeeze(1) >= 0.5
            high_conf_mask = high_conflict.squeeze(1) >= 0.55
            high_rel_mask = teacher_reliability.squeeze(1) >= 0.55
            bucket_id = torch.where(~has_teacher, torch.full_like(bucket_id, 3), bucket_id)
            bucket_id = torch.where(has_teacher & high_conf_mask & high_rel_mask, torch.full_like(bucket_id, 1), bucket_id)
            bucket_id = torch.where(has_teacher & (~high_conf_mask) & high_rel_mask, torch.full_like(bucket_id, 0), bucket_id)
            bucket_id = torch.where(has_teacher & (~high_rel_mask) & (~(has_teacher & (~high_conf_mask) & high_rel_mask)) & (~(has_teacher & high_conf_mask & high_rel_mask)), torch.full_like(bucket_id, 2), bucket_id)
            bucket_emb = self.full_bucket_embed(bucket_id)
            bucket_hidden = torch.tanh(self.full_bucket_hidden(bucket_emb))
            rcmf_enable = ((bucket_id == 1) | (bucket_id == 2)).float().unsqueeze(1)
            led_enable = ((bucket_id == 0) | (bucket_id == 1)).float().unsqueeze(1) * teacher_available
            if full_control_mode == "shared_off":
                rcmf_enable = torch.zeros_like(rcmf_enable)
                led_enable = torch.zeros_like(led_enable)
            elif full_control_mode == "rcmf_only":
                led_enable = torch.zeros_like(led_enable)
            elif full_control_mode == "led_only":
                rcmf_enable = torch.zeros_like(rcmf_enable)
            ctx_hidden_delta = torch.tanh(self.full_factor_ctx_hidden(z_ctx))
            context_rcmf_hidden = 0.78 * ctx_hidden_delta + 0.22 * torch.tanh(parts["rcmf_min_mix_hidden"])
            rcmf_scale = 0.016 * torch.sigmoid(self.full_bucket_rcmf_gate(bucket_emb) - 4.8) * rcmf_enable * high_conflict * low_risk
            led_scale = 0.009 * torch.sigmoid(self.full_bucket_led_gate(bucket_emb) - 5.0) * led_enable * teacher_reliability
            shared_hidden = parts["rcmf_min_hidden"] + rcmf_scale * context_rcmf_hidden + 0.08 * led_scale * bucket_hidden
            full_gate = 0.009 * torch.sigmoid(self.full_bucket_gate(bucket_emb) - 5.2)
            full_gate = full_gate * torch.maximum(rcmf_enable * high_conflict, led_enable * teacher_reliability) * low_risk
            if full_control_mode == "shared_off":
                full_gate = torch.zeros_like(full_gate)
            full_delta = 0.026 * self.innovation_limit * torch.tanh(
                self.full_factor_delta_head(shared_hidden) / max(self.innovation_limit, 1e-6)
            )
            pred = parts["rcmf_min_pred"] + full_gate * full_delta
            monotonic_mask = torch.sigmoid(
                4.0 * (parts["rcmf_min_confidence"] - high_conflict - 0.7 * uncertainty_level - 0.15)
            )
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": parts["rcmf_min_pred"],
                    "teacher_anchor_pred": parts["rcmf_min_pred"],
                    "innovation_pred": pred,
                    "innovation_score": full_gate * torch.abs(full_delta),
                    "fused_latent": shared_hidden,
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["rcmf_min_gate"],
                    "ctx_delta": pred - parts["rcmf_min_pred"],
                    "mspce_anchor_pred": parts["mspce_repair_pred"],
                    "full_led_anchor_pred": parts["rcmf_min_pred"],
                    "full_led_anchor_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_led_pred": parts["rcmf_min_pred"],
                    "full_drop_rcmf_pred": parts["mspce_repair_pred"],
                    "full_drop_led_hidden": parts["rcmf_min_hidden"].detach(),
                    "full_drop_rcmf_hidden": parts["mspce_repair_hidden"].detach(),
                    "full_min_led_teacher_latent": led_teacher_latent,
                    "full_min_led_teacher_pred": led_teacher_pred,
                    "full_min_led_bridge": bucket_hidden,
                    "full_min_led_confidence": teacher_reliability,
                    "full_min_led_activation_mask": led_enable,
                    "full_min_led_teacher_mask": teacher_available,
                    "full_min_led_weight": led_scale / 0.009,
                    "full_min_led_gate": led_enable,
                    "full_min_led_delta": torch.zeros_like(pred),
                    "full_min_led_hidden": shared_hidden,
                    "full_min_gate": full_gate,
                    "full_min_bridge": bucket_hidden,
                    "full_min_mix": parts["rcmf_min_mix_hidden"],
                    "full_min_delta": full_delta,
                    "full_min_hidden_consistency": torch.mean(
                        (shared_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_anchor_consistency": torch.mean(
                        (shared_hidden - parts["rcmf_min_hidden"].detach()) ** 2,
                        dim=1,
                        keepdim=True,
                    ),
                    "full_min_drop_led_consistency": torch.abs(pred - parts["rcmf_min_pred"].detach()),
                    "full_min_drop_rcmf_consistency": torch.abs(pred - parts["mspce_repair_pred"].detach()),
                    "full_min_high_conflict_gate": high_conflict,
                    "full_min_shared_z": bucket_emb,
                    "full_min_shared_rcmf_scale": rcmf_scale,
                    "full_min_shared_led_scale": led_scale,
                    "full_min_stage_rcmf_mask": rcmf_enable,
                    "full_min_stage_led_mask": led_enable,
                    "full_min_stage_rcmf_trigger": rcmf_enable * high_conflict,
                    "full_min_stage_led_trigger": led_enable * teacher_reliability,
                    "full_min_stage_rcmf_magnitude": rcmf_scale,
                    "full_min_stage_led_magnitude": led_scale,
                    "full_min_monotonic_mask": monotonic_mask,
                }
            )
            return payload

        if self.mode == "no_context":
            fused = torch.cat([parts["desc_emb"], parts["graph_emb"]], dim=1)
            pred = self.no_context_head(fused)
            payload = dict(parts)
            payload.update(
                {
                "pred": pred,
                "baseline_pred": student_baseline_pred,
                "teacher_anchor_pred": student_baseline_pred,
                "innovation_pred": pred,
                "innovation_score": torch.zeros_like(pred),
                "fused_latent": fused,
                "baseline_unc": baseline_unc,
                "rcmf_gate": torch.zeros_like(pred),
                "ctx_delta": torch.zeros_like(pred),
                }
            )
            return payload

        if self.mode == "static_fusion":
            weights = torch.softmax(self.static_logits, dim=0)
            fused = (
                weights[0] * parts["desc_emb"]
                + weights[1] * parts["graph_emb"]
                + weights[2] * ctx_emb
            )
            pred = self.static_head(fused)
            payload = dict(parts)
            payload.update(
                {
                "pred": pred,
                "baseline_pred": student_baseline_pred,
                "teacher_anchor_pred": student_baseline_pred,
                "innovation_pred": pred,
                "innovation_score": weights[2].reshape(1, 1).expand_as(pred),
                "fused_latent": fused,
                "baseline_unc": baseline_unc,
                "rcmf_gate": weights[2].reshape(1, 1).expand_as(pred),
                "ctx_delta": pred - student_baseline_pred,
                }
            )
            return payload

        if self.mode == "mspce_only":
            pred = student_baseline_pred + ctx_delta
            payload = dict(parts)
            payload.update(
                {
                "pred": pred,
                "baseline_pred": student_baseline_pred,
                "teacher_anchor_pred": student_baseline_pred,
                "innovation_pred": pred,
                "innovation_score": torch.abs(ctx_delta),
                "fused_latent": baseline_latent + ctx_emb,
                "baseline_unc": baseline_unc,
                "rcmf_gate": torch.ones_like(pred),
                "ctx_delta": ctx_delta,
                }
            )
            return payload

        if self.mode == "mspce_only_led":
            pred = parts["mspce_pred"]
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": student_baseline_pred,
                    "teacher_anchor_pred": student_baseline_pred,
                    "innovation_pred": pred,
                    "innovation_score": torch.abs(ctx_delta),
                    "fused_latent": ctx_emb,
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": torch.ones_like(pred),
                    "ctx_delta": ctx_delta,
                    "mspce_anchor_pred": parts["mspce_pred"],
                    "fusion_alpha": torch.zeros_like(pred),
                    "fusion_delta": torch.zeros_like(pred),
                }
            )
            return payload

        if self.mode == "mspce_only_rcmf":
            pred = parts["fusion_pred"]
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": student_baseline_pred,
                    "teacher_anchor_pred": student_baseline_pred,
                    "innovation_pred": pred,
                    "innovation_score": parts["fusion_alpha"] * torch.abs(parts["fusion_delta"]),
                    "fused_latent": parts["dynamic_latent"],
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": parts["fusion_alpha"],
                    "ctx_delta": parts["fusion_delta"],
                    "mspce_anchor_pred": parts["mspce_pred"],
                }
            )
            return payload

        if self.mode == "residual_safe_gating_full":
            pred = student_baseline_pred + rcmf_gate * ctx_delta
            payload = dict(parts)
            payload.update(
                {
                "pred": pred,
                "baseline_pred": student_baseline_pred,
                "teacher_anchor_pred": student_baseline_pred,
                "innovation_pred": pred,
                "innovation_score": innovation_score,
                "fused_latent": baseline_latent + rcmf_gate * ctx_emb,
                "baseline_unc": baseline_unc,
                "rcmf_gate": rcmf_gate,
                "ctx_delta": ctx_delta,
                }
            )
            return payload

        if self.mode == "teacher_anchored_rcmf_full":
            teacher_anchor = teacher_pred if teacher_pred is not None else student_baseline_pred
            effective_gate = torch.clamp(0.15 + 0.85 * rcmf_gate, max=1.0)
            pred = teacher_anchor + effective_gate * ctx_delta
            payload = dict(parts)
            payload.update(
                {
                "pred": pred,
                "baseline_pred": teacher_anchor,
                "teacher_anchor_pred": teacher_anchor,
                "innovation_pred": pred,
                "innovation_score": effective_gate * torch.abs(ctx_delta),
                "fused_latent": baseline_latent + rcmf_gate * ctx_emb,
                "baseline_unc": baseline_unc,
                "rcmf_gate": effective_gate,
                "ctx_delta": ctx_delta,
                }
            )
            return payload

        if self.mode in {"switch_rcmf_multimodal", "rcmf_dynamic_multimodal_full", "rcmf_dynamic_multimodal_full_phasea", "rcmf_dynamic_multimodal_full_dual_anchor"}:
            if self.mode == "rcmf_dynamic_multimodal_full_phasea":
                concat_delta = torch.clamp(
                    parts["concat_pred"] - parts["mspce_pred"],
                    min=-self.innovation_limit,
                    max=self.innovation_limit,
                )
                phasea_delta = 0.40 * parts["fusion_delta"] + 0.20 * parts["dynamic_delta"] + 0.40 * concat_delta
                phasea_alpha = torch.clamp(1.10 * parts["fusion_alpha"], max=0.26)
                concat_bridge = torch.clamp(
                    parts["concat_pred"] - parts["mspce_pred"],
                    min=-self.innovation_limit,
                    max=self.innovation_limit,
                )
                bridge_gain = torch.clamp(
                    0.35 * phasea_alpha * (1.0 - 0.50 * parts["external_risk"]),
                    min=0.0,
                    max=0.18,
                )
                pred = parts["mspce_pred"] + phasea_alpha * phasea_delta + bridge_gain * concat_bridge
            elif self.mode == "rcmf_dynamic_multimodal_full_dual_anchor":
                phasea_delta = 0.70 * parts["fusion_delta"] + 0.30 * parts["dynamic_delta"]
                phasea_alpha = torch.clamp(1.00 * parts["fusion_alpha"], max=0.30)
                pred = parts["dual_anchor_pred"] + phasea_alpha * phasea_delta
            else:
                pred = parts["fusion_pred"]
                phasea_alpha = parts["fusion_alpha"]
                phasea_delta = parts["fusion_delta"]
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": parts["dual_anchor_pred"] if self.mode == "rcmf_dynamic_multimodal_full_dual_anchor" else student_baseline_pred,
                    "teacher_anchor_pred": student_baseline_pred,
                    "candidate_pred": pred,
                    "innovation_pred": pred,
                    "innovation_score": phasea_alpha * torch.abs(phasea_delta) * (1.0 - 0.35 * parts["external_risk"]),
                    "fused_latent": parts["dynamic_latent"] + parts["led_proxy"],
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": phasea_alpha,
                    "gate_probability": phasea_alpha,
                    "switch_mask": torch.ones_like(pred),
                    "ctx_delta": phasea_delta,
                    "arbitration_score": torch.zeros_like(pred),
                    "positive_switch_mask": torch.ones_like(pred),
                    "teacher_region_mask": torch.zeros_like(pred),
                    "uncertain_region_mask": torch.zeros_like(pred),
                    "dynamic_weights": parts["dynamic_weights"],
                    "mspce_anchor_pred": parts["mspce_pred"],
                    "dual_anchor_pred": parts["dual_anchor_pred"],
                    "anchor_beta": parts["anchor_beta"],
                    "fusion_alpha": phasea_alpha,
                    "fusion_delta": phasea_delta,
                }
            )
            return payload

        if self.mode == "mspce_certified_rcmf":
            teacher_anchor = teacher_pred if teacher_pred is not None else student_baseline_pred
            cert_features = self.certification_features(
                desc_pred=parts["desc_pred"],
                graph_pred=parts["graph_pred"],
                baseline_unc=baseline_unc,
                ctx_delta=ctx_delta,
                student_baseline_pred=student_baseline_pred,
                teacher_anchor=teacher_anchor,
            )
            cert_score = self.certification_score(cert_features)
            certified_positive = (cert_score >= self.cert_positive_threshold).float()
            certified_negative = (cert_score <= self.cert_negative_threshold).float()
            effective_gate = certified_positive * torch.clamp(0.10 + 0.90 * rcmf_gate, max=1.0)
            pred = teacher_anchor + effective_gate * ctx_delta
            payload = dict(parts)
            payload.update(
                {
                    "pred": pred,
                    "baseline_pred": teacher_anchor,
                    "teacher_anchor_pred": teacher_anchor,
                    "innovation_pred": pred,
                    "innovation_score": certified_positive * (torch.relu(cert_score - self.cert_positive_threshold) + effective_gate * torch.abs(ctx_delta)),
                    "fused_latent": baseline_latent + effective_gate * ctx_emb,
                    "baseline_unc": baseline_unc,
                    "rcmf_gate": effective_gate,
                    "ctx_delta": ctx_delta,
                    "certification_score": cert_score,
                    "certified_positive_mask": certified_positive,
                    "certified_negative_mask": certified_negative,
                }
            )
            return payload

        raise ValueError(f"unsupported mode: {self.mode}")

