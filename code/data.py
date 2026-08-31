"""SmartDS 图数据、已验证日历 sidecar 与时间窗口构造。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ``topology_knn`` 是 Stage 1 起的正式图模式。其余模式仅为历史兼容：
# ``raw``/``inf`` 在 273 节点图上寻路；``forest``/``mst_no_tag``/``mst_tag``
# 属于旧投影与启发式补边方案，不再注册到正式实验矩阵。
GraphMode = Literal[
    "raw",
    "topology_knn",
    "inf",
    "legacy_inf",
    "forest",
    "mst_no_tag",
    "mst_tag",
    "projected",
    "projected_inf",
]


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
    target_mask: np.ndarray
    source: Path
    calendar_source: Path | None = None
    timestamp: np.ndarray | None = None
    calendar_values: np.ndarray | None = None
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
    # NumPy 切片、归一化和日历特征拼接。不同节点子集仍保持独立归一化。
    _feature_cache: dict[tuple[int, tuple[int, ...], int, int], np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )
    _topology_knn_cache: dict[
        int, tuple[np.ndarray, np.ndarray, float, float]
    ] = field(default_factory=dict, init=False, repr=False)
    _topology_knn_contract_checked: set[int] = field(
        default_factory=set, init=False, repr=False
    )

    @classmethod
    def load(cls, source: str | Path, calendar_source: str | Path | None = None) -> "SmartDS":
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SmartDS archive not found: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "node_coords",
                "adj",
                "edge_index",
                "load_ts",
                "node_ids",
                "target_mask",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Missing NPZ fields: {sorted(missing)}")

            data = cls(
                node_coords=np.asarray(archive["node_coords"], dtype=np.float64),
                adj=np.asarray(archive["adj"], dtype=np.float32),
                edge_index=np.asarray(archive["edge_index"], dtype=np.int64),
                load_ts=np.asarray(archive["load_ts"], dtype=np.float32),
                node_ids=np.asarray(archive["node_ids"]),
                target_mask=np.asarray(archive["target_mask"], dtype=bool),
                source=path,
            )
        if calendar_source is not None:
            calendar_path = Path(calendar_source).expanduser().resolve()
            if not calendar_path.is_file():
                raise FileNotFoundError(f"SmartDS calendar sidecar not found: {calendar_path}")
            with np.load(calendar_path, allow_pickle=False) as calendar:
                required_calendar = {
                    "timestamp", "hour_of_day", "day_of_week", "weekend", "month"
                }
                missing_calendar = required_calendar.difference(calendar.files)
                if missing_calendar:
                    raise ValueError(f"Missing calendar fields: {sorted(missing_calendar)}")
                timestamp = np.asarray(calendar["timestamp"]).astype("datetime64[s]")
                calendar_values = np.stack(
                    [
                        np.asarray(calendar["hour_of_day"], dtype=np.float32),
                        np.asarray(calendar["day_of_week"], dtype=np.float32),
                        np.asarray(calendar["weekend"], dtype=np.float32),
                        np.asarray(calendar["month"], dtype=np.float32),
                    ],
                    axis=-1,
                )
                if "canonical_source_sha256" in calendar.files:
                    expected_sha = str(np.asarray(calendar["canonical_source_sha256"]).item())
                    if expected_sha != archive_sha256(path):
                        raise ValueError("calendar sidecar 与 canonical NPZ 的 SHA256 不匹配")
            data.calendar_source = calendar_path
            data.timestamp = timestamp
            data.calendar_values = calendar_values
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
        if self.target_mask.shape != (n,):
            raise ValueError("target_mask must have shape (N,)")
        if not np.isfinite(self.node_coords).all():
            raise ValueError("node_coords must contain only finite values")
        if not np.isfinite(self.adj).all() or np.any(self.adj < 0):
            raise ValueError("adjacency must be finite and non-negative")
        if not np.isfinite(self.load_ts).all():
            raise ValueError("load_ts must contain only finite values")
        if self.calendar_values is not None:
            if self.timestamp is None or self.timestamp.shape != (self.time_steps,):
                raise ValueError("timestamp must align one-to-one with load_ts")
            if self.calendar_values.shape != (self.time_steps, 4):
                raise ValueError("calendar fields must have shape (T, 4)")
            if np.isnat(self.timestamp).any() or not np.isfinite(self.calendar_values).all():
                raise ValueError("timestamp/calendar must not contain NaT, NaN or Inf")
            hour, day, weekend, month = self.calendar_values.T
            if np.any((hour < 0) | (hour > 23)):
                raise ValueError("hour_of_day must be in [0, 23]")
            if np.any((day < 0) | (day > 6)):
                raise ValueError("day_of_week must be in [0, 6]")
            if np.any((weekend != 0) & (weekend != 1)):
                raise ValueError("weekend must be binary")
            if np.any((month < 1) | (month > 12)):
                raise ValueError("month must be in [1, 12]")
            if self.time_steps > 1:
                gaps = np.diff(self.timestamp).astype("timedelta64[m]").astype(np.int64)
                if not np.all(gaps == 15):
                    raise ValueError("timestamp must be continuous at 15-minute intervals")
            dates = self.timestamp.astype("datetime64[D]")
            expected_hour = (self.timestamp - dates).astype("timedelta64[h]").astype(np.int64)
            expected_day = (dates.astype(np.int64) + 3) % 7  # 1970-01-01 是星期四。
            expected_month = self.timestamp.astype("datetime64[M]").astype(np.int64) % 12 + 1
            if not np.array_equal(hour.astype(np.int64), expected_hour):
                raise ValueError("hour_of_day 与 timestamp 不一致")
            if not np.array_equal(day.astype(np.int64), expected_day):
                raise ValueError("day_of_week 与 timestamp 不一致")
            if not np.array_equal(weekend.astype(bool), expected_day >= 5):
                raise ValueError("weekend 与 timestamp 不一致")
            if not np.array_equal(month.astype(np.int64), expected_month):
                raise ValueError("month 与 timestamp 不一致")
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
        """返回 canonical target 标记，不从负荷数值重新推断节点角色。"""

        return self.target_mask.copy()

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
        mode: GraphMode = "topology_knn",
        hop_radius: int = 2,
        target_knn_k: int = 6,
    ) -> GraphView:
        indices = np.asarray(list(node_indices), dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("node_indices must be a non-empty one-dimensional sequence")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("node_indices must not contain duplicates")
        if mode == "topology_knn":
            self.assert_topology_knn_contract(target_knn_k)
            targets = self.active_indices
            target_lookup = {
                int(node): position for position, node in enumerate(targets.tolist())
            }
            if any(int(node) not in target_lookup for node in indices.tolist()):
                raise ValueError("topology_knn node_indices must contain target nodes only")
            global_adjacency, global_hops, hop_scale, geo_scale = self.topology_knn_graph(
                target_knn_k
            )
            local_lookup = {int(node): position for position, node in enumerate(indices.tolist())}
            n_local = len(indices)
            adjacency = np.eye(n_local, dtype=np.float32)
            features = np.zeros((n_local, n_local, 3), dtype=np.float32)
            for local_i, global_i in enumerate(indices.tolist()):
                source_position = target_lookup[int(global_i)]
                for target_position in np.flatnonzero(global_adjacency[source_position] > 0):
                    global_j = int(targets[int(target_position)])
                    local_j = local_lookup.get(global_j)
                    if local_j is None:
                        continue
                    adjacency[local_i, local_j] = 1.0
                    features[local_i, local_j, 0] = (
                        global_hops[source_position, int(target_position)] / hop_scale
                    )
                    features[local_i, local_j, 1] = (
                        np.linalg.norm(self.node_coords[global_i] - self.node_coords[global_j])
                        / geo_scale
                    )
                    # 第三维仅为兼容模型输入的占位，不表示 MST 或其他推断边。
                    features[local_i, local_j, 2] = 0.0
            return GraphView(indices, adjacency, features, ())

        if hop_radius < 1:
            raise ValueError("hop_radius must be >= 1 for legacy graph modes")

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
        """构造并缓存负荷与 7 维周期日历编码，共 8 个输入通道。

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
        if self.calendar_values is not None and self.timestamp is not None:
            # 日内、星期和月份是周期变量，使用 sin/cos 可避免 23:45 与
            # 00:00、周日与周一、12 月与 1 月在线性数轴上被错误拉远。
            dates = self.timestamp.astype("datetime64[D]")
            minute_of_day = (
                self.timestamp - dates
            ).astype("timedelta64[m]").astype(np.float64)
            day_of_week = self.calendar_values[:, 1].astype(np.float64)
            month_zero_based = self.calendar_values[:, 3].astype(np.float64) - 1.0
            time_phase = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
            week_phase = 2.0 * np.pi * day_of_week / 7.0
            month_phase = 2.0 * np.pi * month_zero_based / 12.0
            periodic = np.stack(
                [
                    np.sin(time_phase),
                    np.cos(time_phase),
                    np.sin(week_phase),
                    np.cos(week_phase),
                    np.sin(month_phase),
                    np.cos(month_phase),
                    self.calendar_values[:, 2],
                ],
                axis=-1,
            ).astype(np.float32)
        else:
            # 仅供未配置 sidecar 的 legacy 工具兼容；正式配置不走此分支。
            # 保持同样的 7 维通道契约，month/weekend 无可靠日期时置零。
            steps = np.arange(self.time_steps, dtype=np.float32)
            phase_day = 2.0 * np.pi * (steps % daily_period) / daily_period
            phase_cycle = 2.0 * np.pi * steps / weekly_period
            periodic = np.stack(
                [
                    np.sin(phase_day),
                    np.cos(phase_day),
                    np.sin(phase_cycle),
                    np.cos(phase_cycle),
                    np.zeros_like(phase_day),
                    np.zeros_like(phase_day),
                    np.zeros_like(phase_day),
                ],
                axis=-1,
            )
        periodic = np.broadcast_to(
            periodic[:, None, :], (self.time_steps, len(indices), 7)
        )
        features = np.concatenate(
            [normalized[..., None], periodic], axis=-1
        ).astype(np.float32, copy=False)
        self._feature_cache[key] = features
        return features

    def topology_client_partition(
        self, clients: int
    ) -> tuple[list[np.ndarray], tuple[tuple[int, int], ...]]:
        """通过切除官方树边构造确定性的拓扑连续客户端区域。

        划分只使用官方邻接、target 位置和 ``node_id``，不读取负荷序列
        数值。树动态规划保留一个可与父节点连接的开放分量，并决定每条
        父子边保留或切断。对于当前 92 targets / 8 clients，目标按以下
        字典序优化：相对 10--13 的总超界幅度、相对 11.5 的总平方
        偏差、超界分量数。该顺序避免用一个严重失衡区域换取更少的超界
        区域。完全并列时按被切边端点的 ``node_id`` 排序。

        返回每个客户端的 target 全图索引，以及被切除的官方无向树边。
        客户端编号按各区域最小 ``node_id`` 确定，因而可跨运行复现。
        """

        if clients < 1:
            raise ValueError("clients must be >= 1")
        targets = self.active_indices
        target_count = len(targets)
        if target_count == 0:
            raise ValueError("SmartDS contains no non-zero load nodes")
        if clients > target_count:
            raise ValueError("clients cannot exceed the number of target nodes")

        graph = nx.from_numpy_array((self.adj > 0).astype(np.uint8))
        if not nx.is_tree(graph):
            raise ValueError(
                "topology client partition requires one connected official tree"
            )

        node_ids = [str(node_id) for node_id in self.node_ids.tolist()]
        root = min(graph.nodes, key=lambda node: (node_ids[int(node)], int(node)))
        target_weights = self.active_mask.astype(np.int64)
        average = target_count / clients
        preferred_low = max(1, int(np.floor(average)) - 1)
        preferred_high = int(np.ceil(average)) + 1

        # 候选值为 (已闭合分量的三项代价, 已切官方边)。开放分量的代价
        # 只有在切断父边或到达根节点时才计入，保证动态规划的最优子结构。
        Candidate = tuple[tuple[int, int, int], tuple[tuple[int, int], ...]]

        def component_score(size: int) -> tuple[int, int, int]:
            below = max(preferred_low - size, 0)
            above = max(size - preferred_high, 0)
            range_distance = below + above
            # 用整数形式 clients^2 * (size - average)^2 避免浮点并列误差。
            squared_deviation = (size * clients - target_count) ** 2
            return range_distance, squared_deviation, int(range_distance > 0)

        def edge_key(edge: tuple[int, int]) -> tuple[str, str, int, int]:
            left, right = edge
            if (node_ids[left], left) > (node_ids[right], right):
                left, right = right, left
            return node_ids[left], node_ids[right], left, right

        def canonical_edge(left: int, right: int) -> tuple[int, int]:
            if (node_ids[left], left) <= (node_ids[right], right):
                return left, right
            return right, left

        def merge_cuts(
            *groups: tuple[tuple[int, int], ...],
        ) -> tuple[tuple[int, int], ...]:
            return tuple(sorted((edge for group in groups for edge in group), key=edge_key))

        def candidate_key(candidate: Candidate) -> tuple[object, ...]:
            score, cuts = candidate
            return (*score, tuple(edge_key(edge) for edge in cuts))

        def add_scores(
            left: tuple[int, int, int], right: tuple[int, int, int]
        ) -> tuple[int, int, int]:
            return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]

        def keep_best(
            states: dict[tuple[int, int], Candidate],
            state: tuple[int, int],
            candidate: Candidate,
        ) -> None:
            previous = states.get(state)
            if previous is None or candidate_key(candidate) < candidate_key(previous):
                states[state] = candidate

        def solve_subtree(node: int, parent: int | None) -> dict[tuple[int, int], Candidate]:
            # state=(已闭合分量数, 与 node 相连的开放分量 target 数)
            states: dict[tuple[int, int], Candidate] = {
                (0, int(target_weights[node])): ((0, 0, 0), ())
            }
            children = sorted(
                (int(neighbor) for neighbor in graph.neighbors(node) if neighbor != parent),
                key=lambda child: (node_ids[child], child),
            )
            for child in children:
                child_states = solve_subtree(child, node)
                combined: dict[tuple[int, int], Candidate] = {}
                for (closed_left, open_left), left_candidate in states.items():
                    for (closed_right, open_right), right_candidate in child_states.items():
                        base_closed = closed_left + closed_right
                        if base_closed > clients - 1:
                            continue
                        base_score = add_scores(left_candidate[0], right_candidate[0])
                        base_cuts = merge_cuts(left_candidate[1], right_candidate[1])

                        # 保留父子边：两个开放分量合并。
                        keep_best(
                            combined,
                            (base_closed, open_left + open_right),
                            (base_score, base_cuts),
                        )

                        # 切断父子边：child 的开放分量在此闭合。
                        if base_closed + 1 <= clients - 1:
                            cut_score = add_scores(base_score, component_score(open_right))
                            cut_edge = canonical_edge(node, child)
                            keep_best(
                                combined,
                                (base_closed + 1, open_left),
                                (cut_score, merge_cuts(base_cuts, (cut_edge,))),
                            )
                states = combined
            return states

        root_states = solve_subtree(int(root), None)
        finalists: list[tuple[tuple[object, ...], Candidate]] = []
        for (closed_count, open_size), candidate in root_states.items():
            if closed_count != clients - 1:
                continue
            final_score = add_scores(candidate[0], component_score(open_size))
            final_candidate = (final_score, candidate[1])
            finalists.append((candidate_key(final_candidate), final_candidate))
        if not finalists:
            raise RuntimeError(f"unable to partition official tree into {clients} regions")

        _, best = min(finalists, key=lambda item: item[0])
        cut_edges = best[1]
        if len(cut_edges) != clients - 1:
            raise RuntimeError("tree partition did not select clients - 1 cut edges")

        partitioned_graph = graph.copy()
        partitioned_graph.remove_edges_from(cut_edges)
        components = [set(component) for component in nx.connected_components(partitioned_graph)]
        components.sort(
            key=lambda component: min((node_ids[node], node) for node in component)
        )
        if len(components) != clients:
            raise RuntimeError("cut edges did not produce the requested number of regions")

        target_set = set(int(index) for index in targets.tolist())
        partitions = [
            np.asarray(
                sorted(
                    target_set.intersection(component),
                    key=lambda node: (node_ids[node], node),
                ),
                dtype=np.int64,
            )
            for component in components
        ]
        flattened = [node for partition in partitions for node in partition.tolist()]
        if len(flattened) != target_count or set(flattened) != target_set:
            raise RuntimeError("tree regions do not cover every target exactly once")
        return partitions, cut_edges

    def client_partitions(self, clients: int) -> list[np.ndarray]:
        """返回基于官方树切边得到的拓扑连续 target 客户端划分。"""

        partitions, _ = self.topology_client_partition(clients)
        return partitions

    def legacy_duplicate_aware_client_partitions(self, clients: int) -> list[np.ndarray]:
        """按空间顺序划分客户端，并保证完全相同的负荷曲线不被拆开。

        SmartDS 中存在多个节点共享同一条负荷序列。若直接对节点编号
        ``array_split``，相同曲线可能落在不同客户端，使客户端之间的
        Non-IID 统计失真。这里把每个重复曲线组视为不可拆分单元，再用
        连续空间分段的动态规划寻找接近等量的 K 个分区。分区仍是合成
        网络上的空间聚合，不表示真实台区边界。该方法仅保留用于与新的
        官方树切边划分做历史审计，不再是 ``client_partitions`` 默认实现。
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

    def topology_knn_graph(
        self, k: int
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """在完整官方图上构造 target symmetric topology-kNN。

        返回 target 节点顺序下的邻接矩阵、官方最短 hop 矩阵，以及由全局
        kNN 边集合计算的 hop/坐标距离固定归一化尺度。客户端只允许对该
        全局图取诱导子图，不能重新计算本地 kNN 或本地尺度。
        """

        k = int(k)
        cached = self._topology_knn_cache.get(k)
        if cached is not None:
            adjacency, hops, hop_scale, geo_scale = cached
            return adjacency.copy(), hops.copy(), hop_scale, geo_scale
        targets = self.active_indices
        if not 1 <= k < len(targets):
            raise ValueError(f"target_knn_k must be in [1, {len(targets) - 1}]")
        full_graph = nx.from_numpy_array((self.adj > 0).astype(np.uint8))
        if not nx.is_connected(full_graph):
            raise ValueError("topology_knn requires a connected official full graph")
        target_ids = [str(node_id) for node_id in self.node_ids[targets].tolist()]
        target_lookup = {int(node): position for position, node in enumerate(targets.tolist())}
        distances = np.full((len(targets), len(targets)), np.inf, dtype=np.float64)
        for source_position, source_node in enumerate(targets.tolist()):
            lengths = nx.single_source_shortest_path_length(full_graph, int(source_node))
            for target_node, target_position in target_lookup.items():
                if target_node not in lengths:
                    raise ValueError("official graph has unreachable target nodes")
                distances[source_position, target_position] = lengths[target_node]
        adjacency = np.zeros((len(targets), len(targets)), dtype=np.float32)
        for source in range(len(targets)):
            candidates = [position for position in range(len(targets)) if position != source]
            candidates.sort(
                key=lambda position: (int(distances[source, position]), target_ids[position])
            )
            for target in candidates[:k]:
                adjacency[source, target] = 1.0
                adjacency[target, source] = 1.0
        edge_positions = np.argwhere(np.triu(adjacency > 0, k=1))
        if edge_positions.size == 0:
            raise RuntimeError("topology_knn produced no target edges")
        geo_values = np.asarray(
            [
                np.linalg.norm(self.node_coords[targets[left]] - self.node_coords[targets[right]])
                for left, right in edge_positions.tolist()
            ],
            dtype=np.float64,
        )
        hop_values = np.asarray(
            [distances[left, right] for left, right in edge_positions.tolist()],
            dtype=np.float64,
        )
        hop_scale = max(float(np.median(hop_values)), 1e-6)
        geo_scale = max(float(np.median(geo_values)), 1e-6)
        result = (adjacency, distances.astype(np.float32), hop_scale, geo_scale)
        self._topology_knn_cache[k] = result
        return result[0].copy(), result[1].copy(), result[2], result[3]

    def assert_topology_knn_contract(self, k: int) -> None:
        """锁定 Stage 0 的全局图和 8-client 拓扑划分不变量。"""

        k = int(k)
        if k in self._topology_knn_contract_checked:
            return
        adjacency, _, _, _ = self.topology_knn_graph(k)
        graph = nx.from_numpy_array((adjacency > 0).astype(np.uint8))
        if k == 6:
            assert graph.number_of_nodes() == 92, "global target graph must have 92 nodes"
            assert graph.number_of_edges() == 352, "global target graph must have 352 edges"
            assert nx.number_connected_components(graph) == 1, "global target graph must be connected"
            partitions = self.client_partitions(8)
            counts = [len(partition) for partition in partitions]
            assert counts == [15, 12, 9, 10, 11, 12, 11, 12], (
                f"unexpected frozen client target counts: {counts}"
            )
            for partition in partitions:
                induced = graph.subgraph(
                    [int(np.flatnonzero(self.active_indices == node)[0]) for node in partition.tolist()]
                )
                assert nx.number_connected_components(induced) == 1, (
                    "each frozen client induced target graph must be connected"
                )
                assert nx.number_of_isolates(induced) == 0, (
                    "each frozen client induced target graph must have no isolates"
                )
        self._topology_knn_contract_checked.add(k)


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
