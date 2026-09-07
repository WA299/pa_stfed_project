"""Active graph and temporal layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

class EdgeAwareSpatialGAT(nn.Module):
    """在物理邻域内执行多头加性边属性图注意力。
    对每个注意力头 m，先计算节点投影 u_i^m=W^m h_i，再由源节点项、
    目标节点项和边属性项共同形成加性注意力分数。
    """

    def __init__(self, hidden_dim: int, heads: int = 4, edge_dim: int = 3) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by spatial attention heads")

        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.edge_dim = edge_dim

        # 一次线性映射等价于并行执行 heads 个 W^m，随后再拆分头维度。
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
        )

        # a_m^T[u_i^m || u_j^m] 拆成源节点向量与目标节点向量两项；
        # edge_attention 对应报告中 b_m^T phi(r_ij)。
        self.source_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.target_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.edge_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.activation = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """采用 Xavier 初始化三个加性注意力向量。"""

        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.xavier_uniform_(self.edge_projection[0].weight)
        nn.init.zeros_(self.edge_projection[0].bias)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)
        nn.init.xavier_uniform_(self.edge_attention)

    def forward(self, x: Tensor, adjacency: Tensor, edge_features: Tensor) -> Tensor:
        # x 的形状为 [批量, 历史长度, 节点数, 隐藏维度]。
        if adjacency.shape != (x.shape[2], x.shape[2]):
            raise ValueError("adjacency must have shape [nodes, nodes]")
        if edge_features.shape != (x.shape[2], x.shape[2], self.edge_dim):
            raise ValueError(
                f"edge_features must have shape [nodes, nodes, {self.edge_dim}]"
            )

        batch, history, nodes, _ = x.shape
        projected = self.node_projection(x).reshape(
            batch, history, nodes, self.heads, self.head_dim
        )
        encoded_edges = self.edge_projection(edge_features).reshape(
            nodes, nodes, self.heads, self.head_dim
        )

        # e_ij^m = LeakyReLU(a_src^T u_i^m + a_dst^T u_j^m
        #                    + b_m^T phi(r_ij))。
        source_score = (projected * self.source_attention).sum(dim=-1)
        target_score = (projected * self.target_attention).sum(dim=-1)
        edge_score = (encoded_edges * self.edge_attention).sum(dim=-1)
        logits = self.activation(
            source_score[:, :, :, None, :]
            + target_score[:, :, None, :, :]
            + edge_score[None, None, :, :, :]
        )

        # 只在启发式填充图给定的物理邻域（含自环）中执行 softmax。
        mask = adjacency <= 0
        logits = logits.masked_fill(mask[None, None, :, :, None], -1e4)
        attention = torch.softmax(logits, dim=3)

        # o_i^m = sum_j alpha_ij^m u_j^m，最后拼接全部注意力头。
        attention_by_head = attention.permute(0, 1, 4, 2, 3)
        projected_by_head = projected.permute(0, 1, 3, 2, 4)
        output = torch.matmul(attention_by_head, projected_by_head)
        output = output.permute(0, 1, 3, 2, 4)
        return torch.nn.functional.gelu(output.reshape(batch, history, nodes, -1))


class StaticFunctionalGraph(nn.Module):
    """客户端静态功能关系图。"""

    def __init__(
        self,
        node_count: int,
        hidden_dim: int,
        functional_dim: int,
        graph_mode: str = "static",
    ) -> None:
        super().__init__()
        graph_mode = str(graph_mode).lower()
        if graph_mode != "static":
            raise ValueError(f"Unsupported functional graph mode: {graph_mode!r}; only 'static' is active")
        self.graph_mode = graph_mode
        self.embedding_1 = nn.Parameter(torch.randn(node_count, functional_dim) * 0.02)
        self.embedding_2 = nn.Parameter(torch.randn(node_count, functional_dim) * 0.02)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def static_logits(self) -> Tensor:
        return torch.relu(self.embedding_1 @ self.embedding_2.T)

    def adjacency(self) -> Tensor:
        return torch.softmax(self.static_logits(), dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        relation = torch.softmax(self.static_logits(), dim=-1)
        values = self.value(x)
        output = torch.einsum("ij,bljd->blid", relation, values)
        return torch.nn.functional.gelu(output), relation


class HorizonCrossAttentionDecoder(nn.Module):
    """用完整历史 memory 生成逐 horizon 的残差修正。"""

    def __init__(
        self,
        horizon: int,
        hidden_dim: int,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
        correction_init: float = 0.0,
    ) -> None:
        super().__init__()
        if horizon < 1 or hidden_dim < 1 or heads < 1 or layers != 1:
            raise ValueError("invalid horizon decoder dimensions; layers must be 1")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by horizon decoder heads")
        if correction_init != 0.0:
            raise ValueError("horizon decoder correction head must initialize to zero")
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.layers = int(layers)
        self.correction_init = float(correction_init)
        self.query = nn.Parameter(torch.empty(self.horizon, self.hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.correction = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def correction_output(self, decoded: Tensor) -> Tensor:
        return self.correction(decoded)

    def forward(self, memory: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if memory.ndim != 4:
            raise ValueError("horizon decoder memory must have shape [B, N, T, H]")
        batch, nodes, _, hidden = memory.shape
        if hidden != self.hidden_dim:
            raise ValueError("horizon decoder memory hidden dimension mismatch")
        keys = memory.reshape(batch * nodes, memory.shape[2], hidden)
        queries = self.query.unsqueeze(0).expand(batch * nodes, -1, -1)
        attended, weights = self.cross_attention(
            queries, keys, keys, need_weights=True, average_attn_weights=False
        )
        decoded = self.norm1(queries + attended)
        decoded = self.norm2(decoded + self.ffn(decoded))
        probabilities = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
        entropy_by_horizon = -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=(0, 1))
        entropy = entropy_by_horizon.mean()
        return decoded.reshape(batch, nodes, self.horizon, hidden), entropy, entropy_by_horizon


class TemporalEncoder(nn.Module):
    """因果 Transformer 时间编码器。"""

    def __init__(
        self,
        history: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, history, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: Tensor, return_memory: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        batch, history, nodes, hidden = x.shape
        if history > self.position.shape[1]:
            raise ValueError("Input history is longer than configured positional embedding")
        position = self.position[:, :history, :].unsqueeze(2)
        sequence = (x + position).permute(0, 2, 1, 3).reshape(batch * nodes, history, hidden)
        causal_mask = torch.triu(
            torch.ones(history, history, dtype=torch.bool, device=x.device), diagonal=1
        )
        encoded = self.encoder(sequence, mask=causal_mask)
        last = encoded[:, -1].reshape(batch, nodes, hidden)
        if return_memory:
            return last, encoded.reshape(batch, nodes, history, hidden)
        return last
