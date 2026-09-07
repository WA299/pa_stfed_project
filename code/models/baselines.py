"""Active baseline model implementations."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

class AdaptiveVertexGraphConv(nn.Module):
    """AGCRN 的节点自适应参数池图卷积。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cheb_order: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if cheb_order < 2:
            raise ValueError("AGCRN cheb_order must be at least 2")
        self.cheb_order = cheb_order
        self.weight_pool = nn.Parameter(
            torch.empty(embedding_dim, cheb_order, input_dim, output_dim)
        )
        self.bias_pool = nn.Parameter(torch.empty(embedding_dim, output_dim))
        nn.init.xavier_uniform_(self.weight_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x: Tensor, node_embeddings: Tensor) -> Tensor:
        nodes = node_embeddings.shape[0]
        adaptive = torch.softmax(torch.relu(node_embeddings @ node_embeddings.T), dim=-1)
        supports = [torch.eye(nodes, device=x.device, dtype=x.dtype), adaptive]
        for _ in range(2, self.cheb_order):
            supports.append(2.0 * adaptive @ supports[-1] - supports[-2])
        support_stack = torch.stack(supports, dim=0)
        graph_signal = torch.einsum("knm,bmc->bnkc", support_stack, x)
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weight_pool)
        bias = node_embeddings @ self.bias_pool
        return torch.einsum("bnki,nkio->bno", graph_signal, weights) + bias


class AGCRNCell(nn.Module):
    """使用节点自适应图卷积替代线性映射的 GRU 单元。"""

    def __init__(self, input_dim: int, hidden_dim: int, cheb_order: int, embedding_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = AdaptiveVertexGraphConv(
            input_dim + hidden_dim,
            2 * hidden_dim,
            cheb_order,
            embedding_dim,
        )
        self.candidate = AdaptiveVertexGraphConv(
            input_dim + hidden_dim,
            hidden_dim,
            cheb_order,
            embedding_dim,
        )

    def forward(self, x: Tensor, state: Tensor, node_embeddings: Tensor) -> Tensor:
        update, reset = torch.sigmoid(
            self.gates(torch.cat([x, state], dim=-1), node_embeddings)
        ).chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * state], dim=-1), node_embeddings)
        )
        return update * state + (1.0 - update) * candidate


class AGCRNBaseline(nn.Module):
    """AGCRN 风格的单层 AVWDCRNN 适配基线。

    该实现用于 SmartDS 的统一输入/输出协议，不声称是官方代码的逐行复现；
    论文中应称为 ``AGCRN-adapted``，并明确单层和统一隐藏维度设置。
    """

    def __init__(
        self,
        node_count: int,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        embedding_dim: int = 10,
        cheb_order: int = 2,
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.horizon = horizon
        self.node_embeddings = nn.Parameter(torch.randn(node_count, embedding_dim) * 0.1)
        self.cell = AGCRNCell(input_dim, hidden_dim, cheb_order, embedding_dim)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: Tensor, adjacency: Tensor, edge_features: Tensor) -> dict[str, Tensor]:
        del adjacency, edge_features
        state = torch.zeros(
            x.shape[0], self.node_count, self.cell.hidden_dim,
            device=x.device, dtype=x.dtype,
        )
        for step in range(x.shape[1]):
            state = self.cell(x[:, step], state, self.node_embeddings)
        return {"prediction": self.head(state).permute(0, 2, 1)}


class DiffusionGraphConv(nn.Module):
    """Graph WaveNet 的多支撑扩散图卷积。"""

    def __init__(self, channels: int, support_count: int = 2, order: int = 2) -> None:
        super().__init__()
        self.order = order
        self.projection = nn.Conv2d(
            channels * (1 + support_count * order), channels, kernel_size=1
        )

    def forward(self, x: Tensor, supports: list[Tensor]) -> Tensor:
        features = [x]
        for support in supports:
            propagated = torch.einsum("nm,bcmt->bcnt", support, x)
            features.append(propagated)
            for _ in range(2, self.order + 1):
                propagated = torch.einsum("nm,bcmt->bcnt", support, propagated)
                features.append(propagated)
        return self.projection(torch.cat(features, dim=1))


class GraphWaveNetBaseline(nn.Module):
    """Graph WaveNet 风格的门控扩张卷积适配基线。

    该实现保留扩张时间卷积、物理/自适应双支撑和扩散图卷积，但为统一
    SmartDS 任务采用固定层数与输出头；论文中应标注为适配实现而非官方复现。
    """

    def __init__(
        self,
        node_count: int,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        layers: int = 3,
        embedding_dim: int = 10,
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.input_projection = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.node_embedding_1 = nn.Parameter(torch.randn(node_count, embedding_dim) * 0.1)
        self.node_embedding_2 = nn.Parameter(torch.randn(embedding_dim, node_count) * 0.1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dilations: list[int] = []
        for layer_index in range(layers):
            dilation = 2 ** layer_index
            self.dilations.append(dilation)
            self.filter_convs.append(
                nn.Conv2d(hidden_dim, hidden_dim, (1, 2), dilation=(1, dilation))
            )
            self.gate_convs.append(
                nn.Conv2d(hidden_dim, hidden_dim, (1, 2), dilation=(1, dilation))
            )
            self.graph_convs.append(DiffusionGraphConv(hidden_dim))
            self.skip_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
            self.norms.append(nn.BatchNorm2d(hidden_dim))
        self.output_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, horizon, kernel_size=1),
        )

    @staticmethod
    def _row_normalize(adjacency: Tensor) -> Tensor:
        adjacency = adjacency.to(dtype=torch.float32)
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x: Tensor, adjacency: Tensor, edge_features: Tensor) -> dict[str, Tensor]:
        del edge_features
        hidden = self.input_projection(x.permute(0, 3, 2, 1))
        physical = self._row_normalize(adjacency).to(device=x.device, dtype=x.dtype)
        adaptive = torch.softmax(
            torch.relu(self.node_embedding_1 @ self.node_embedding_2), dim=-1
        )
        supports = [physical, adaptive]
        skip: Tensor | None = None
        for dilation, filter_conv, gate_conv, graph_conv, skip_conv, norm in zip(
            self.dilations,
            self.filter_convs,
            self.gate_convs,
            self.graph_convs,
            self.skip_convs,
            self.norms,
        ):
            residual = hidden
            padded = F.pad(hidden, (dilation, 0, 0, 0))
            temporal = torch.tanh(filter_conv(padded)) * torch.sigmoid(gate_conv(padded))
            skip_value = skip_conv(temporal)
            skip = skip_value if skip is None else skip + skip_value
            hidden = norm(graph_conv(temporal, supports) + residual)
        prediction = self.output_projection(skip)[:, :, :, -1]
        return {"prediction": prediction}


class LSTMBaseline(nn.Module):
    """逐节点 LSTM 负荷预测基线。

    该实现对应短期住宅负荷预测中常见的 LSTM 递归编码器：每个节点独立
    编码自己的历史负荷和已验证日历特征，不使用图拓扑。这样可以把性能差异
    归因于时空建模，而不是额外输入字段。输入输出接口与 PA-STFed 统一。
    """

    def __init__(
        self,
        node_count: int,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("LSTM layers must be positive")
        self.node_count = node_count
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(
        self, x: Tensor, adjacency: Tensor, edge_features: Tensor
    ) -> dict[str, Tensor]:
        del adjacency, edge_features
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, history, nodes, features]")
        if x.shape[2] != self.node_count:
            raise ValueError(f"Expected {self.node_count} nodes, got {x.shape[2]}")
        batch, history, nodes, features = x.shape
        # 将节点并入 batch，避免人为引入节点间信息，严格保持 LSTM 基线的
        # 独立节点建模定义；最终再恢复节点维度。
        sequence = x.permute(0, 2, 1, 3).reshape(batch * nodes, history, features)
        _, (hidden, _) = self.encoder(sequence)
        node_state = hidden[-1].reshape(batch, nodes, -1)
        prediction = self.head(node_state).permute(0, 2, 1)
        return {"prediction": prediction}


class ITransformerBaseline(nn.Module):
    """iTransformer 风格的倒置 Transformer 多变量适配基线。

    iTransformer 将每个变量的历史片段作为一个 token，再沿变量维度执行
    自注意力。本项目把 ``节点 × 输入特征`` 视为变量集合，输出每个节点的
    负荷特征（特征索引 0）未来 horizon 步；不使用邻接矩阵或外部天气字段。
    这是与 SmartDS 输入口径一致的适配基线，不应表述为官方实现的完全复现。
    """

    def __init__(
        self,
        node_count: int,
        input_dim: int,
        history: int,
        horizon: int,
        hidden_dim: int,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("iTransformer hidden_dim must be divisible by heads")
        if layers < 1:
            raise ValueError("iTransformer layers must be positive")
        self.node_count = node_count
        self.input_dim = input_dim
        self.history = history
        self.horizon = horizon
        self.value_embedding = nn.Linear(history, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, horizon)

    def forward(
        self, x: Tensor, adjacency: Tensor, edge_features: Tensor
    ) -> dict[str, Tensor]:
        del adjacency, edge_features
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, history, nodes, features]")
        if x.shape[1] != self.history:
            raise ValueError(f"Expected history={self.history}, got {x.shape[1]}")
        if x.shape[2] != self.node_count or x.shape[3] != self.input_dim:
            raise ValueError(
                f"Expected [nodes, features]=[{self.node_count}, {self.input_dim}], "
                f"got [{x.shape[2]}, {x.shape[3]}]"
            )
        batch, _, nodes, features = x.shape
        # [B,T,N,F] -> [B,N*F,T]：每一个节点-特征组合是一个倒置 token。
        variables = x.permute(0, 2, 3, 1).reshape(batch, nodes * features, self.history)
        tokens = self.value_embedding(variables)
        encoded = self.norm(self.encoder(tokens))
        variable_forecast = self.output_projection(encoded)
        forecast = variable_forecast.reshape(batch, nodes, features, self.horizon)
        # 仅预测负荷通道（索引 0）；周期协变量用于编码，不作为输出目标。
        prediction = forecast[:, :, 0, :].permute(0, 2, 1)
        return {"prediction": prediction}



