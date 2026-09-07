"""Active PA-STFed model and federated parameter helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .layers import EdgeAwareSpatialGAT, HorizonCrossAttentionDecoder, StaticFunctionalGraph, TemporalEncoder

LOCAL_PARAMETER_PREFIXES = ("functional.embedding_1", "functional.embedding_2")
ALA_PARAMETER_PREFIXES = ("spatial_gate.", "temporal_gate.", "head.")


def local_parameter_prefixes(personalized_head: bool = False) -> tuple[str, ...]:
    return (*LOCAL_PARAMETER_PREFIXES, "head.") if personalized_head else LOCAL_PARAMETER_PREFIXES


def ala_parameter_prefixes() -> tuple[str, ...]:
    return ALA_PARAMETER_PREFIXES


def vanilla_ala_parameter_names(model: nn.Module, layer_idx: int = 2) -> tuple[str, ...]:
    if layer_idx < 1:
        raise ValueError("vanilla FedALA layer_idx must be positive")
    candidates = [name for name, _ in model.named_parameters() if not name.startswith(LOCAL_PARAMETER_PREFIXES)]
    if layer_idx > len(candidates):
        raise ValueError("vanilla FedALA layer_idx exceeds shape-compatible parameter count")
    return tuple(candidates[-layer_idx:])


class PA_STFed(nn.Module):
    """PA-STFed with the active physical/static-functional Transformer path."""

    def __init__(
        self, node_count: int, history: int, horizon: int, input_dim: int = 5,
        hidden_dim: int = 32, functional_dim: int = 16, spatial_heads: int = 4,
        transformer_layers: int = 2, transformer_heads: int = 4, dropout: float = 0.1,
        use_physical: bool = True, use_functional: bool = True,
        use_spatial_gate: bool = True, use_temporal_gate: bool = True,
        use_residual_anchor: bool = False, temporal_architecture: str = "transformer",
        functional_graph_mode: str = "static", use_horizon_decoder: bool = False,
        horizon_decoder_heads: int = 4, horizon_decoder_layers: int = 1,
        horizon_correction_init: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % transformer_heads != 0 or hidden_dim % spatial_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention heads")
        if str(temporal_architecture).lower() != "transformer":
            raise ValueError("Only the active Transformer temporal path is supported")
        self.node_count, self.history, self.horizon = node_count, history, horizon
        self.use_physical, self.use_functional = bool(use_physical), bool(use_functional)
        self.use_spatial_gate, self.use_temporal_gate = bool(use_spatial_gate), bool(use_temporal_gate)
        self.use_residual_anchor = bool(use_residual_anchor)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.physical = EdgeAwareSpatialGAT(hidden_dim, heads=spatial_heads)
        self.functional_graph_mode = str(functional_graph_mode).lower()
        self.functional = StaticFunctionalGraph(node_count, hidden_dim, functional_dim, graph_mode=self.functional_graph_mode)
        self.spatial_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.temporal_architecture = "transformer"
        self.temporal = TemporalEncoder(history, hidden_dim, transformer_layers, transformer_heads, dropout)
        self.temporal_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, horizon))
        if self.use_residual_anchor:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        self.use_horizon_decoder = bool(use_horizon_decoder)
        self.horizon_decoder_heads, self.horizon_decoder_layers = int(horizon_decoder_heads), int(horizon_decoder_layers)
        self.horizon_correction_init = float(horizon_correction_init)
        self.horizon_decoder = None
        if self.use_horizon_decoder:
            self.horizon_decoder = HorizonCrossAttentionDecoder(horizon=horizon, hidden_dim=hidden_dim, heads=self.horizon_decoder_heads, layers=self.horizon_decoder_layers, dropout=dropout, correction_init=self.horizon_correction_init)

    def forward(self, x: Tensor, adjacency: Tensor, edge_features: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4 or x.shape[2] != self.node_count:
            raise ValueError("x must have shape [batch, history, nodes, features]")
        base = self.input_projection(x)
        physical = self.physical(base, adjacency, edge_features) if self.use_physical else torch.zeros_like(base)
        if self.use_functional:
            functional, functional_adjacency = self.functional(base)
        else:
            functional = torch.zeros_like(base)
            functional_adjacency = torch.eye(self.node_count, dtype=base.dtype, device=base.device)
        if self.use_physical and self.use_functional:
            gamma = torch.sigmoid(self.spatial_gate(torch.cat([physical, functional], dim=-1))) if self.use_spatial_gate else torch.full_like(physical, 0.5)
            spatial = gamma * physical + (1.0 - gamma) * functional
        elif self.use_physical:
            gamma, spatial = torch.ones_like(physical), physical
        elif self.use_functional:
            gamma, spatial = torch.zeros_like(functional), functional
        else:
            gamma, spatial = torch.full_like(base, 0.5), base
        temporal_result = self.temporal(self.dropout(spatial), return_memory=self.horizon_decoder is not None)
        if self.horizon_decoder is not None:
            temporal_last, temporal_memory = temporal_result
        else:
            temporal_last, temporal_memory = temporal_result, None
        spatial_last = spatial[:, -1]
        if self.use_temporal_gate:
            gate = torch.sigmoid(self.temporal_gate(torch.cat([spatial_last, temporal_last], dim=-1)))
            fused = gate * spatial_last + (1.0 - gate) * temporal_last
        else:
            gate, fused = torch.full_like(temporal_last, 0.5), temporal_last
        prediction = self.head(fused).permute(0, 2, 1)
        horizon_entropy = horizon_entropy_by_step = None
        if self.horizon_decoder is not None and temporal_memory is not None:
            decoder_hidden, horizon_entropy, horizon_entropy_by_step = self.horizon_decoder(temporal_memory)
            prediction = prediction + self.horizon_decoder.correction_output(decoder_hidden).squeeze(-1).permute(0, 2, 1)
        if self.use_residual_anchor:
            prediction = prediction + x[:, -1, :, 0].unsqueeze(1)
        result = {"prediction": prediction, "gamma": gamma, "temporal_gate": gate, "physical": physical, "functional": functional, "functional_adjacency": functional_adjacency}
        if horizon_entropy is not None:
            result["horizon_cross_attention_entropy"] = horizon_entropy
            result["horizon_cross_attention_entropy_by_step"] = horizon_entropy_by_step
        return result


def horizon_decoder_metadata(model: nn.Module) -> dict | None:
    decoder = getattr(model, "horizon_decoder", None)
    if decoder is None:
        return None
    return {"enabled": True, "heads": int(decoder.heads), "layers": int(decoder.layers), "horizon": int(decoder.horizon), "correction_head_init": float(decoder.correction_init), "causal_source": "historical temporal memory only"}


def check_horizon_decoder_initialization_consistency(seed: int = 2026) -> dict[str, object]:
    torch.manual_seed(int(seed))
    first = HorizonCrossAttentionDecoder(horizon=12, hidden_dim=64, heads=4, layers=1)
    torch.manual_seed(int(seed))
    second = HorizonCrossAttentionDecoder(horizon=12, hidden_dim=64, heads=4, layers=1)
    if first.state_dict().keys() != second.state_dict().keys() or any(not torch.equal(first.state_dict()[n], second.state_dict()[n]) for n in first.state_dict()):
        raise AssertionError("horizon decoder initialization mismatch")
    return {"common_parameter_count": len(first.state_dict()), "added_parameters": []}


def shared_state_dict(model: PA_STFed, personalized_head: bool = False) -> dict[str, Tensor]:
    prefixes = local_parameter_prefixes(personalized_head)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if not name.startswith(prefixes)}


def load_shared_state(model: PA_STFed, state: Mapping[str, Tensor]) -> None:
    current = model.state_dict()
    for name, value in state.items():
        if name in current:
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
