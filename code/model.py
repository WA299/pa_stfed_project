"""PA-STFed 神经网络模块。

每个客户端独享功能图节点嵌入；主 FedAvg/FedProx 实验共享预测头，只有
personalized_head 消融才把预测头留在本地。物理图张量在前向传播时传入，
因此 8 个客户端可以具有不同节点数，而共享层的参数形状仍然一致。
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# 功能图节点嵌入因客户端节点数不同而始终本地保留；预测头是否本地化
# 由 federated.personalized_head 控制，主 FedAvg/FedProx 实验默认参与聚合。
LOCAL_PARAMETER_PREFIXES = ("functional.embedding_1", "functional.embedding_2")


def local_parameter_prefixes(personalized_head: bool = False) -> tuple[str, ...]:
    return (*LOCAL_PARAMETER_PREFIXES, "head.") if personalized_head else LOCAL_PARAMETER_PREFIXES


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
    """客户端本地可学习的静态功能关系图。

    两组节点嵌入生成非对称关系分数。该图每个客户端只学习一份，避免在
    仅有 11--12 个节点的小子图上逐时刻重建 TopK 图导致过拟合。
    """

    def __init__(self, node_count: int, hidden_dim: int, functional_dim: int) -> None:
        super().__init__()
        self.embedding_1 = nn.Parameter(torch.randn(node_count, functional_dim) * 0.02)
        self.embedding_2 = nn.Parameter(torch.randn(node_count, functional_dim) * 0.02)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def adjacency(self) -> Tensor:
        # A_k^A = softmax(ReLU(E_{k,1} E_{k,2}^T))。
        score = torch.relu(self.embedding_1 @ self.embedding_2.T)
        return torch.softmax(score, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        relation = self.adjacency()
        values = self.value(x)
        output = torch.einsum("ij,bljd->blid", relation, values)
        return torch.nn.functional.gelu(output), relation


class TemporalEncoder(nn.Module):
    """沿每个节点的历史维度执行带因果掩码的 Transformer 编码。"""

    def __init__(self, history: int, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, history, nodes, hidden = x.shape
        if history > self.position.shape[1]:
            raise ValueError("Input history is longer than configured positional embedding")
        position = self.position[:, :history, :].unsqueeze(2)
        sequence = (x + position).permute(0, 2, 1, 3).reshape(batch * nodes, history, hidden)
        causal_mask = torch.triu(
            torch.ones(history, history, dtype=torch.bool, device=x.device), diagonal=1
        )
        encoded = self.encoder(sequence, mask=causal_mask)
        return encoded[:, -1].reshape(batch, nodes, hidden)


class PA_STFed(nn.Module):
    """单个客户端上的物理--自适应双图时空预测器。"""

    def __init__(
        self,
        node_count: int,
        history: int,
        horizon: int,
        input_dim: int = 8,
        hidden_dim: int = 32,
        functional_dim: int = 16,
        spatial_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        use_physical: bool = True,
        use_functional: bool = True,
        use_spatial_gate: bool = True,
        use_temporal_gate: bool = True,
        use_residual_anchor: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if hidden_dim % spatial_heads != 0:
            raise ValueError("hidden_dim must be divisible by spatial_heads")
        self.node_count = node_count
        self.history = history
        self.horizon = horizon
        self.use_physical = use_physical
        self.use_functional = use_functional
        self.use_spatial_gate = use_spatial_gate
        self.use_temporal_gate = use_temporal_gate
        self.use_residual_anchor = use_residual_anchor
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.physical = EdgeAwareSpatialGAT(hidden_dim, heads=spatial_heads)
        self.functional = StaticFunctionalGraph(node_count, hidden_dim, functional_dim)
        self.spatial_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.temporal = TemporalEncoder(history, hidden_dim, transformer_layers, transformer_heads, dropout)
        self.temporal_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
        if self.use_residual_anchor:
            # 残差头初始输出为 0，使模型从持久性预测开始学习修正量，
            # 避免短期负荷预测在训练初期产生大幅无方向偏移。
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: Tensor, adjacency: Tensor, edge_features: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, history, nodes, features]")
        if x.shape[2] != self.node_count:
            raise ValueError(f"Expected {self.node_count} nodes, got {x.shape[2]}")
        # 1. 将负荷和 7 维已验证周期日历编码投影到统一隐藏空间。
        base = self.input_projection(x)

        # 2. 分别提取物理拓扑表征与客户端本地功能关系表征。
        # 消融实验关闭某一分支时仍保留参数结构，但不执行该分支的前向计算。
        if self.use_physical:
            physical = self.physical(base, adjacency, edge_features)
        else:
            physical = torch.zeros_like(base)
        if self.use_functional:
            functional, functional_adjacency = self.functional(base)
        else:
            functional = torch.zeros_like(base)
            functional_adjacency = torch.eye(
                self.node_count, dtype=base.dtype, device=base.device
            )

        # 3. 第一级门控 gamma 在两个空间来源之间逐元素选择。
        if self.use_physical and self.use_functional:
            if self.use_spatial_gate:
                gamma = torch.sigmoid(
                    self.spatial_gate(torch.cat([physical, functional], dim=-1))
                )
            else:
                # 无空间门控消融：使用固定等权融合，不引入可学习选通。
                gamma = torch.full_like(physical, 0.5)
            spatial = gamma * physical + (1.0 - gamma) * functional
        elif self.use_physical:
            gamma = torch.ones_like(physical)
            spatial = physical
        elif self.use_functional:
            gamma = torch.zeros_like(functional)
            spatial = functional
        else:
            gamma = torch.full_like(base, 0.5)
            spatial = base

        # 4. Transformer 捕捉历史依赖；第二级门控融合即时空间状态与时间状态。
        temporal_last = self.temporal(self.dropout(spatial))
        spatial_last = spatial[:, -1]
        if self.use_temporal_gate:
            gate = torch.sigmoid(
                self.temporal_gate(torch.cat([spatial_last, temporal_last], dim=-1))
            )
            fused = gate * spatial_last + (1.0 - gate) * temporal_last
        else:
            # 无时间门控消融：只使用 Transformer 的最后时刻状态。
            gate = torch.full_like(temporal_last, 0.5)
            fused = temporal_last

        # 5. 每个节点独立输出未来 horizon 个时间步，再转为 [B, H, N]。
        prediction = self.head(fused).permute(0, 2, 1)
        if self.use_residual_anchor:
            # 输入与目标均处于同一节点级归一化空间，最后观测值可直接作为
            # 多步基线；网络只需学习未来相对该基线的残差。
            persistence = x[:, -1, :, 0].unsqueeze(1)
            prediction = prediction + persistence
        return {
            "prediction": prediction,
            "gamma": gamma,
            "temporal_gate": gate,
            "physical": physical,
            "functional": functional,
            "functional_adjacency": functional_adjacency,
        }


def shared_state_dict(model: PA_STFed, personalized_head: bool = False) -> dict[str, Tensor]:
    """提取允许服务器聚合的参数；只有个性化实验才排除预测头。"""

    local_prefixes = local_parameter_prefixes(personalized_head)

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(local_prefixes)
    }


def load_shared_state(model: PA_STFed, state: Mapping[str, Tensor]) -> None:
    """把服务器共享参数回载到客户端，同时保留客户端个性化参数。"""

    current = model.state_dict()
    for name, value in state.items():
        if name in current and current[name].shape == value.shape:
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
