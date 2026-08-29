"""SmartDS 数据读取、启发式拓扑填充和时间窗口构造。

本文件只使用数据中实际存在的负荷、坐标和邻接关系。由于数据没有真实
绝对时间戳，周期特征只能表示序列内部的相对周期，不能解释为具体日期、
星期或节假日。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ``raw`` 和 ``inf`` 保留为历史兼容模式：它们直接在 273 节点图上寻路，
# 其中 ``inf`` 的 MST 端点可能落在零负荷节点上。正式拓扑消融使用下面三种
# 模式：``forest`` 是零负荷中继投影后的原始拓扑图（实验别名，不预设无环），
# ``mst_no_tag`` 在该图上增加候选桥接边但不提供边来源，``mst_tag`` 则显式
# 编码推断边来源。
GraphMode = Literal["raw", "inf", "legacy_inf", "forest", "mst_no_tag", "mst_tag", "projected", "projected_inf"]


@dataclass(frozen=True)
class GraphView:
    """投影到某个客户端有效负荷节点集合上的图张量。"""

    node_indices: np.ndarray
    adjacency: np.ndarray
    edge_features: np.ndarray
    bridge_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    val_end: int
    total: int


@dataclass
class SmartDS:
    """内存中的 SmartDS 图数据及其确定性预处理方法。"""

    node_coords: np.ndarray
    adj: np.ndarray
    edge_index: np.ndarray
    load_ts: np.ndarray
    node_ids: np.ndarray
    source: Path
    _raw_components_cache: tuple[np.ndarray, ...] | None = field(
        default=None, init=False, repr=False
    )
    _imputed_topology_cache: tuple[
        np.ndarray, tuple[tuple[int, int], ...]
    ] | None = field(default=None, init=False, repr=False)
    _projected_topology_cache: tuple[
        np.ndarray, np.ndarray, tuple[np.ndarray, ...]
    ] | None = field(default=None, init=False, repr=False)
    _projected_imputation_cache: tuple[
        np.ndarray, np.ndarray, tuple[tuple[int, int], ...]
    ] | None = field(default=None, init=False, repr=False)
    _stats_cache: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    # 缓存按节点子集和训练统计量构造的完整特征序列，避免 __getitem__ 重复做
    # NumPy 切片、归一化和周期特征拼接。不同节点子集仍保持独立归一化。
    _feature_cache: dict[tuple[int, tuple[int, ...], int, int], np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def load(cls, source: str | Path) -> "SmartDS":
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SmartDS archive not found: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"node_coords", "adj", "edge_index", "load_ts", "node_ids"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Missing NPZ fields: {sorted(missing)}")

            data = cls(
                node_coords=np.asarray(archive["node_coords"], dtype=np.float64),
                adj=np.asarray(archive["adj"], dtype=np.float32),
                edge_index=np.asarray(archive["edge_index"], dtype=np.int64),
                load_ts=np.asarray(archive["load_ts"], dtype=np.float32),
                node_ids=np.asarray(archive["node_ids"]),
                source=path,
            )
        data.validate()
        return data

    def validate(self) -> None:
        n = self.node_coords.shape[0]
        if self.node_coords.shape != (n, 2):
            raise ValueError(f"node_coords must have shape (N, 2), got {self.node_coords.shape}")
        if self.adj.shape != (n, n):
            raise ValueError(f"adj must have shape (N, N), got {self.adj.shape}")
        if self.load_ts.ndim != 2 or self.load_ts.shape[1] != n:
            raise ValueError("load_ts must have shape (T, N) consistent with node_coords")
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if self.node_ids.shape != (n,):
            raise ValueError("node_ids must have shape (N,)")
        if not np.isfinite(self.node_coords).all():
            raise ValueError("node_coords must contain only finite values")
        if not np.isfinite(self.adj).all() or np.any(self.adj < 0):
            raise ValueError("adjacency must be finite and non-negative")
        if not np.isfinite(self.load_ts).all():
            raise ValueError("load_ts must contain only finite values")
        if self.edge_index.size:
            if np.any(self.edge_index < 0) or np.any(self.edge_index >= n):
                raise ValueError("edge_index contains an out-of-range node index")
        if not np.allclose(self.adj, self.adj.T):
            raise ValueError("adjacency must be symmetric for this undirected experiment")

    @property
    def node_count(self) -> int:
        return int(self.load_ts.shape[1])

    @property
    def time_steps(self) -> int:
        return int(self.load_ts.shape[0])

    @property
    def active_mask(self) -> np.ndarray:
        return np.any(self.load_ts != 0.0, axis=0)

    @property
    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active_mask).astype(np.int64)

    @property
    def duplicate_groups(self) -> list[np.ndarray]:
        groups: dict[bytes, list[int]] = {}
        for idx in self.active_indices.tolist():
            key = np.ascontiguousarray(self.load_ts[:, idx]).tobytes()
            groups.setdefault(key, []).append(idx)
        return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]

    def raw_components(self) -> list[np.ndarray]:
        if self._raw_components_cache is not None:
            return [component.copy() for component in self._raw_components_cache]
        graph = nx.from_numpy_array((self.adj > 0).astype(np.uint8))
        components = [np.asarray(sorted(component), dtype=np.int64) for component in nx.connected_components(graph)]
        ordered = tuple(sorted(components, key=lambda component: int(component[0])))
        self._raw_components_cache = ordered
        return [component.copy() for component in ordered]

    def heuristic_topology_imputation(self) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        """通过确定性的连通分量级 MST 执行启发式拓扑填充。

        每两个原始连通分量之间，以节点对的最小欧氏坐标距离作为候选边权；
        再按“距离、分量编号、节点编号”的确定顺序执行 Kruskal 算法。
        57 个连通分量因此只增加 56 条桥接边。

        注意：欧氏坐标距离只用于拓扑启发式排序，不代表导线物理长度，
        更不能解释为线路电阻、电抗或真实电气阻抗。
        """

        if self._imputed_topology_cache is not None:
            adjacency, bridges = self._imputed_topology_cache
            return adjacency.copy(), bridges

        components = self.raw_components()
        count = len(components)
        candidates: list[tuple[float, int, int, int, int]] = []
        for a in range(count):
            for b in range(a + 1, count):
                best: tuple[float, int, int] | None = None
                for i in components[a].tolist():
                    for j in components[b].tolist():
                        distance = float(np.linalg.norm(self.node_coords[i] - self.node_coords[j]))
                        candidate = (distance, int(i), int(j))
                        if best is None or candidate < best:
                            best = candidate
                if best is None:
                    raise RuntimeError("Unable to construct a component bridge candidate")
                candidates.append((best[0], a, b, best[1], best[2]))

        parent = list(range(count))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        bridges: list[tuple[int, int]] = []
        for _, a, b, i, j in sorted(candidates):
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            parent[ra] = rb
            bridges.append((min(i, j), max(i, j)))
            if len(bridges) == count - 1:
                break

        if len(bridges) != count - 1:
            raise RuntimeError("Component MST did not produce count - 1 bridge edges")

        adjacency = (self.adj > 0).astype(np.float32)
        for i, j in bridges:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
        result = (adjacency, tuple(sorted(bridges)))
        self._imputed_topology_cache = result
        return adjacency.copy(), result[1]

    def projected_topology(self) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        """将零负荷节点作为拓扑中继，投影为有效负荷节点图。

        投影不是删除零节点后的诱导子图。对于每个仅由零负荷节点组成的
        连通块，将其相邻的有效节点两两连接；同时保留有效节点之间的原始
        边。这样可以保留 ``有效--零--有效`` 路径，并为每条投影边记录其在
        完整原始图上的最短跳数。返回值依次为：有效节点邻接矩阵、原始跳数
        矩阵和投影图的连通分量。原始全图是森林，但将一个中继块的多个有效
        邻居两两相连可能形成环，因此这里不把投影结果数学上断言为森林。
        """

        if self._projected_topology_cache is not None:
            adjacency, hops, components = self._projected_topology_cache
            return adjacency.copy(), hops.copy(), tuple(c.copy() for c in components)

        active = self.active_indices
        active_set = set(int(i) for i in active.tolist())
        full_graph = nx.from_numpy_array((self.adj > 0).astype(np.uint8))
        local_lookup = {int(node): index for index, node in enumerate(active.tolist())}
        projected = nx.Graph()
        projected.add_nodes_from(range(len(active)))
        hop_matrix = np.zeros((len(active), len(active)), dtype=np.float32)

        # 保留原始有效--有效边。SmartDS 当前文件中该数量为0，但该分支
        # 使投影规则适用于后续替换的图文件。
        for source, target in full_graph.edges():
            if source in active_set and target in active_set:
                left, right = local_lookup[source], local_lookup[target]
                projected.add_edge(left, right)
                hop_matrix[left, right] = hop_matrix[right, left] = 1.0

        # 收缩每个零负荷连通块，并将其有效邻居两两连接。最短路径只经过
        # 该零节点块，因此不会把另一个有效负荷节点误当作中继节点。
        zero_nodes = set(range(self.node_count)) - active_set
        zero_graph = full_graph.subgraph(zero_nodes)
        for zero_component in nx.connected_components(zero_graph):
            neighbors: set[int] = set()
            for zero_node in zero_component:
                neighbors.update(
                    int(neighbor)
                    for neighbor in full_graph.neighbors(zero_node)
                    if int(neighbor) in active_set
                )
            neighbors = sorted(neighbors)
            for position, source in enumerate(neighbors):
                for target in neighbors[position + 1 :]:
                    left, right = local_lookup[source], local_lookup[target]
                    try:
                        distance = nx.shortest_path_length(full_graph, source, target)
                    except nx.NetworkXNoPath:
                        continue
                    projected.add_edge(left, right)
                    previous = hop_matrix[left, right]
                    value = float(distance)
                    if previous == 0.0 or value < previous:
                        hop_matrix[left, right] = hop_matrix[right, left] = value

        adjacency = nx.to_numpy_array(
            projected, nodelist=list(range(len(active))), dtype=np.float32
        )
        np.fill_diagonal(adjacency, 0.0)
        components = tuple(
            np.asarray(sorted(component), dtype=np.int64)
            for component in sorted(nx.connected_components(projected), key=lambda c: min(c))
        )
        self._projected_topology_cache = (adjacency, hop_matrix, components)
        return adjacency.copy(), hop_matrix.copy(), tuple(c.copy() for c in components)

    def heuristic_projected_topology_imputation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
        """在零负荷中继投影图上，以有效节点为端点执行确定性 MST。

        只有投影后仍处于不同连通分量的有效节点才参与候选桥接。候选边权
        为两个分量之间有效节点对的最小坐标欧氏距离，平局按全局节点编号
        字典序打破。该过程是拓扑连通性假设，不代表真实线路。
        """

        if self._projected_imputation_cache is not None:
            adjacency, hops, bridges = self._projected_imputation_cache
            return adjacency.copy(), hops.copy(), bridges

        projected, hop_matrix, components = self.projected_topology()
        count = len(components)
        if count <= 1:
            result = (projected, hop_matrix, ())
            self._projected_imputation_cache = result
            return projected.copy(), hop_matrix.copy(), ()

        active = self.active_indices
        candidates: list[tuple[float, int, int, int, int]] = []
        for left_component in range(count):
            for right_component in range(left_component + 1, count):
                best: tuple[float, int, int] | None = None
                for left_local in components[left_component].tolist():
                    for right_local in components[right_component].tolist():
                        left_global = int(active[left_local])
                        right_global = int(active[right_local])
                        distance = float(
                            np.linalg.norm(
                                self.node_coords[left_global] - self.node_coords[right_global]
                            )
                        )
                        candidate = (
                            distance,
                            min(left_global, right_global),
                            max(left_global, right_global),
                        )
                        if best is None or candidate < best:
                            best = candidate
                if best is None:
                    raise RuntimeError("Unable to construct projected component bridge candidate")
                candidates.append(
                    (best[0], left_component, right_component, best[1], best[2])
                )

        parent = list(range(count))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        bridges: list[tuple[int, int]] = []
        for _, left_component, right_component, left_global, right_global in sorted(candidates):
            root_left, root_right = find(left_component), find(right_component)
            if root_left == root_right:
                continue
            parent[root_left] = root_right
            left_local = int(np.flatnonzero(active == left_global)[0])
            right_local = int(np.flatnonzero(active == right_global)[0])
            projected[left_local, right_local] = 1.0
            projected[right_local, left_local] = 1.0
            # 候选边没有真实原始跳数；用1表示一条候选关系，第三个边
            # 属性会明确其推断来源，不能解释为物理距离。
            hop_matrix[left_local, right_local] = 1.0
            hop_matrix[right_local, left_local] = 1.0
            bridges.append((min(left_global, right_global), max(left_global, right_global)))
            if len(bridges) == count - 1:
                break
        if len(bridges) != count - 1:
            raise RuntimeError("Projected component MST did not produce count - 1 bridge edges")

        result = (projected, hop_matrix, tuple(sorted(bridges)))
        self._projected_imputation_cache = result
        return projected.copy(), hop_matrix.copy(), result[2]

    def graph_view(
        self,
        node_indices: Iterable[int],
        mode: GraphMode = "mst_tag",
        hop_radius: int = 2,
    ) -> GraphView:
        indices = np.asarray(list(node_indices), dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("node_indices must be a non-empty one-dimensional sequence")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("node_indices must not contain duplicates")
        if hop_radius < 1:
            raise ValueError("hop_radius must be >= 1")

        if mode == "raw":
            base_adj = (self.adj > 0).astype(np.float32)
            bridge_edges: tuple[tuple[int, int], ...] = ()
            path_hops = None
            path_nodes = indices
        elif mode in {"inf", "legacy_inf"}:
            base_adj, bridge_edges = self.heuristic_topology_imputation()
            path_hops = None
            path_nodes = indices
        elif mode in {"forest", "projected"}:
            base_adj, path_hops, _projected_components = self.projected_topology()
            bridge_edges = ()
            path_nodes = self.active_indices
        elif mode in {"mst_no_tag", "mst_tag", "projected_inf"}:
            base_adj, path_hops, bridge_edges = self.heuristic_projected_topology_imputation()
            path_nodes = self.active_indices
        else:
            raise ValueError(f"Unsupported graph mode: {mode}")

        base_graph = nx.from_numpy_array(base_adj)
        base_lookup = {int(node): local_index for local_index, node in enumerate(path_nodes.tolist())}
        local_nodes = indices.tolist()
        local_lookup = {global_node: local_index for local_index, global_node in enumerate(local_nodes)}
        n_local = len(local_nodes)
        adjacency = np.eye(n_local, dtype=np.float32)
        features = np.zeros((n_local, n_local, 3), dtype=np.float32)
        bridge_set = {tuple(sorted(edge)) for edge in bridge_edges}

        # legacy 模式在完整的 273 节点网络上寻路，零负荷节点可作为中继；
        # projected 模式使用已经显式收缩中继路径的 92 节点图。两者都避免
        # 直接删除零负荷节点后再寻路所造成的拓扑信息损失。
        legacy_path_mode = mode in {"raw", "inf", "legacy_inf"}
        for local_i, global_i in enumerate(local_nodes):
            source_node = int(global_i) if legacy_path_mode else base_lookup[int(global_i)]
            paths = nx.single_source_shortest_path(base_graph, source_node, cutoff=hop_radius)
            for path_target, path in paths.items():
                global_path = (
                    [int(node) for node in path]
                    if legacy_path_mode
                    else [int(path_nodes[int(node)]) for node in path]
                )
                global_j = int(global_path[-1])
                local_j = local_lookup.get(global_j)
                if local_j is None:
                    continue
                adjacency[local_i, local_j] = 1.0
                if path_hops is None:
                    hop = len(path) - 1
                else:
                    hop = sum(
                        float(path_hops[int(u), int(v)])
                        for u, v in zip(path[:-1], path[1:])
                    )
                path_bridge_count = sum(
                    tuple(sorted((int(u), int(v)))) in bridge_set
                    for u, v in zip(global_path[:-1], global_path[1:])
                )
                # 三维边属性依次是跳数、坐标欧氏距离、路径中的填充边数量。
                # 前两者仅是图关系与位置先验，不是物理线路长度或阻抗。
                features[local_i, local_j, 0] = float(hop)
                features[local_i, local_j, 1] = float(
                    np.linalg.norm(self.node_coords[global_i] - self.node_coords[global_j])
                )
                features[local_i, local_j, 2] = float(
                    path_bridge_count if mode not in {"mst_no_tag"} else 0.0
                )

        # 网络接收的是归一化后的边先验。跳数、坐标距离和填充边计数只用于
        # 表达图拓扑/位置关系，绝不解释为导线物理长度或电气阻抗；归一化可
        # 防止坐标量纲过大时压制节点注意力项。
        connected = (adjacency > 0) & ~np.eye(n_local, dtype=bool)
        if np.any(connected):
            geo_scale = float(np.median(features[:, :, 1][connected]))
            geo_scale = max(geo_scale, 1e-6)
        else:
            geo_scale = 1.0
        features[:, :, 0] /= float(max(hop_radius, 1))
        features[:, :, 1] /= geo_scale
        # 正式 ``mst_tag`` 模式保留推断边计数的离散语义：直接候选边为1，
        # 含两条候选边的路径为2。历史模式沿用旧的 hop_radius 归一化，避免
        # 改写既有结果的特征尺度。
        if mode in {"forest", "projected", "mst_no_tag", "mst_tag", "projected_inf"}:
            features[:, :, 2] /= 1.0
        else:
            features[:, :, 2] /= float(max(hop_radius, 1))

        return GraphView(indices, adjacency, features, bridge_edges)

    def split_bounds(self, train_ratio: float, val_ratio: float) -> SplitBounds:
        if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
            raise ValueError("train_ratio and val_ratio must be in (0, 1)")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1")
        train_end = round(self.time_steps * train_ratio)
        val_end = train_end + round(self.time_steps * val_ratio)
        return SplitBounds(train_end=train_end, val_end=val_end, total=self.time_steps)

    def robust_stats(self, node_indices: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
        node_indices = np.asarray(node_indices, dtype=np.int64)
        key = (int(train_end), tuple(int(index) for index in node_indices.tolist()))
        cached = self._stats_cache.get(key)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()
        if not 0 < train_end <= self.time_steps:
            raise ValueError("train_end must lie within the available time range")
        train = self.load_ts[:train_end, node_indices].astype(np.float64)
        if train.size == 0:
            raise ValueError("Cannot compute normalization statistics for empty nodes")
        median = np.median(train, axis=0)
        q25, q75 = np.percentile(train, [25.0, 75.0], axis=0)
        scale = np.maximum(q75 - q25, 1e-6)
        result = median.astype(np.float32), scale.astype(np.float32)
        self._stats_cache[key] = result
        return result[0].copy(), result[1].copy()

    def window_features(
        self,
        node_indices: np.ndarray,
        train_end: int,
        daily_period: int,
        weekly_period: int,
    ) -> np.ndarray:
        """构造并缓存 ``[T, N, 5]`` 的归一化输入特征。

        原实现把这部分工作放在 ``__getitem__`` 中，导致每个 epoch 的每个
        样本都重复执行 NumPy 运算。缓存后 Dataset 只做连续时间切片，
        CPU 开销显著降低，且不改变样本内容或数据划分。
        """

        indices = np.asarray(node_indices, dtype=np.int64)
        key = (
            int(train_end),
            tuple(int(index) for index in indices.tolist()),
            int(daily_period),
            int(weekly_period),
        )
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached

        median, scale = self.robust_stats(indices, train_end)
        normalized = (self.load_ts[:, indices] - median[None, :]) / scale[None, :]
        steps = np.arange(self.time_steps, dtype=np.float32)
        phase_day = 2.0 * np.pi * (steps % daily_period) / daily_period
        phase_cycle = 2.0 * np.pi * steps / weekly_period
        periodic = np.stack(
            [
                np.sin(phase_day),
                np.cos(phase_day),
                np.sin(phase_cycle),
                np.cos(phase_cycle),
            ],
            axis=-1,
        )
        periodic = np.broadcast_to(
            periodic[:, None, :], (self.time_steps, len(indices), 4)
        )
        features = np.concatenate(
            [normalized[..., None], periodic], axis=-1
        ).astype(np.float32, copy=False)
        self._feature_cache[key] = features
        return features

    def client_partitions(self, clients: int) -> list[np.ndarray]:
        """按空间顺序划分客户端，并保证完全相同的负荷曲线不被拆开。

        SmartDS 中存在多个节点共享同一条负荷序列。若直接对节点编号
        ``array_split``，相同曲线可能落在不同客户端，使客户端之间的
        Non-IID 统计失真。这里把每个重复曲线组视为不可拆分单元，再用
        连续空间分段的动态规划寻找接近等量的 K 个分区。分区仍是合成
        网络上的空间聚合，不表示真实台区边界。
        """

        if clients < 1:
            raise ValueError("clients must be >= 1")
        active = self.active_indices
        if len(active) == 0:
            raise ValueError("SmartDS contains no non-zero load nodes")
        if clients > len(active):
            raise ValueError("clients cannot exceed the number of active load nodes")

        groups = [np.sort(group.astype(np.int64)) for group in self.duplicate_groups]
        groups.sort(
            key=lambda group: (
                float(self.node_coords[group, 0].mean()),
                float(self.node_coords[group, 1].mean()),
                int(group[0]),
            )
        )
        if clients > len(groups):
            raise ValueError(
                "clients cannot exceed the number of indivisible duplicate-curve groups"
            )

        sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
        prefix = np.concatenate(([0], np.cumsum(sizes)))
        target = float(len(active)) / clients
        # dp[k, j]：前 j 个空间连续组切成 k 个客户端的最小负载偏差。
        dp = np.full((clients + 1, len(groups) + 1), np.inf, dtype=np.float64)
        prev = np.full((clients + 1, len(groups) + 1), -1, dtype=np.int64)
        dp[0, 0] = 0.0
        for k in range(1, clients + 1):
            for j in range(k, len(groups) + 1):
                # 至少给每个剩余客户端保留一个不可拆分组。
                lower = k - 1
                upper = j - 1
                for split in range(lower, upper + 1):
                    segment_size = int(prefix[j] - prefix[split])
                    cost = dp[k - 1, split] + (segment_size - target) ** 2
                    if cost < dp[k, j]:
                        dp[k, j] = cost
                        prev[k, j] = split

        boundaries: list[int] = [len(groups)]
        j = len(groups)
        for k in range(clients, 0, -1):
            split = int(prev[k, j])
            if split < 0:
                raise RuntimeError("failed to construct duplicate-group-aware partitions")
            boundaries.append(split)
            j = split
        boundaries.reverse()
        return [
            np.concatenate(groups[boundaries[k] : boundaries[k + 1]]).astype(np.int64)
            for k in range(clients)
        ]


class LoadWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """某一客户端和时间切分上的惰性滑动窗口数据集。"""

    def __init__(
        self,
        data: SmartDS,
        node_indices: np.ndarray,
        split: Literal["train", "val", "test"],
        history: int,
        horizon: int,
        train_end: int,
        val_end: int,
        daily_period: int,
        weekly_period: int,
        max_windows: int | None = None,
    ) -> None:
        if history < 1 or horizon < 1:
            raise ValueError("history and horizon must be positive")
        if daily_period < 1 or weekly_period < daily_period:
            raise ValueError("period lengths must be positive and weekly_period >= daily_period")
        self.data = data
        self.node_indices = np.asarray(node_indices, dtype=np.int64)
        self.history = history
        self.horizon = horizon
        self.daily_period = daily_period
        self.weekly_period = weekly_period
        self.median, self.scale = data.robust_stats(self.node_indices, train_end)
        # MAPE 只对不接近零的负荷点统计；阈值按训练集每个节点的平均绝对负荷
        # 的 1% 确定，并随结果一并记录覆盖率，避免低负荷点制造虚高误差。
        train_abs = np.abs(data.load_ts[:train_end, self.node_indices].astype(np.float64))
        self.mape_floor = np.maximum(np.mean(train_abs, axis=0) * 0.01, 1e-6).astype(np.float32)
        self.feature_ts = data.window_features(
            self.node_indices,
            train_end,
            daily_period,
            weekly_period,
        )
        self.origins = self._origins(split, train_end, val_end)
        if max_windows is not None:
            if max_windows < 1:
                raise ValueError("max_windows 必须是正整数")
            if max_windows < len(self.origins):
                # 在整个时间划分上等间隔抽样，不能只截取最前面的窗口；否则
                # 调参集会偏向单一季节或单一负荷阶段，验证排序不可靠。
                positions = np.linspace(
                    0, len(self.origins) - 1, num=max_windows, dtype=np.int64
                )
                self.origins = self.origins[positions]

    def _origins(self, split: str, train_end: int, val_end: int) -> np.ndarray:
        first = self.history - 1
        last = self.data.time_steps - self.horizon - 1
        candidates = np.arange(first, last + 1, dtype=np.int64)
        target_start = candidates + 1
        target_end = candidates + self.horizon + 1
        if split == "train":
            mask = target_end <= train_end
        elif split == "val":
            mask = (target_start >= train_end) & (target_end <= val_end)
        elif split == "test":
            mask = target_start >= val_end
        else:
            raise ValueError(f"Unknown split: {split}")
        return candidates[mask]

    def __len__(self) -> int:
        return int(len(self.origins))

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """把最后一维为节点的张量还原到原始负荷尺度。"""

        median = torch.as_tensor(self.median, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        return values * scale + median

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tau = int(self.origins[index])
        start = tau - self.history + 1
        stop = tau + self.horizon + 1
        features = self.feature_ts[start : tau + 1]
        target = self.feature_ts[tau + 1 : stop, :, 0]
        # 时间维切片保持共享缓存；target 的最后一维切片可能非连续，
        # 仅对这一小块做连续化，保证 DataLoader 能高效堆叠 batch。
        return torch.from_numpy(features), torch.from_numpy(np.ascontiguousarray(target))


def make_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    training_config: dict,
) -> DataLoader:
    """按统一训练配置构造 DataLoader。

    pinned memory 与 non_blocking 搬运配合使用；persistent workers 只在
    ``num_workers > 0`` 时启用，避免 CPU 模式或单 worker 模式报错。
    """

    num_workers = int(training_config.get("num_workers", 0))
    if not shuffle:
        # 验证集和测试集每轮只遍历一次。如果每个客户端、每个划分都保留
        # worker，会产生大量空闲进程并争抢 CPU；这里使用专门的评估配置。
        num_workers = int(training_config.get("eval_num_workers", 0))
    if num_workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        # CPU 运行时不请求 pinned memory，避免无加速器环境产生无意义警告。
        "pin_memory": bool(training_config.get("pin_memory", False))
        and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            training_config.get("persistent_workers", True) and shuffle
        )
        kwargs["prefetch_factor"] = max(
            1, int(training_config.get("prefetch_factor", 2))
        )
    return DataLoader(dataset, **kwargs)


def archive_sha256(source: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(source).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
