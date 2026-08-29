"""审计官方物理图上的 target topology-kNN，不执行任何模型训练。

距离在完整 273 节点 Line+Transformer 图上计算，零负荷结构节点保留为
最短路径中继。每个 target 选择 hop 距离最近的 k 个其他 target；距离相同
时按 node_id 字符串排序，任一方向选中即保留无向边。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data import SmartDS  # noqa: E402


K_VALUES = (2, 4, 6, 8)


@dataclass(frozen=True)
class GraphSummary:
    edge_count: int
    density: float
    average_degree: float
    max_degree: int
    components: int
    isolated_nodes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit deterministic topology-distance target kNN graphs."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "processed" / "smartds_full_graph_v2.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "target_topology_audit.md",
    )
    parser.add_argument(
        "--knn-stats",
        type=Path,
        default=ROOT / "reports" / "supporting" / "target_knn_stats.csv",
    )
    parser.add_argument(
        "--client-stats",
        type=Path,
        default=ROOT / "reports" / "supporting" / "client_topology_stats.csv",
    )
    return parser.parse_args()


def full_target_hop_distances(
    adjacency: np.ndarray, target_indices: np.ndarray
) -> np.ndarray:
    """返回 target 两两最短 hop；路径可经过任意结构节点。"""

    graph = nx.from_numpy_array((adjacency > 0).astype(np.uint8))
    target_indices = np.asarray(target_indices, dtype=np.int64)
    distances = np.full((len(target_indices), len(target_indices)), np.inf)
    full_to_target = {
        int(full_index): target_index
        for target_index, full_index in enumerate(target_indices.tolist())
    }

    for source_target, source_full in enumerate(target_indices.tolist()):
        path_lengths = nx.single_source_shortest_path_length(graph, int(source_full))
        for target_full, target_position in full_to_target.items():
            if target_full in path_lengths:
                distances[source_target, target_position] = path_lengths[target_full]

    if not np.isfinite(distances).all():
        unreachable = int(np.count_nonzero(~np.isfinite(np.triu(distances, k=1))))
        raise RuntimeError(
            "Official full graph contains unreachable target pairs: "
            f"{unreachable} upper-triangle entries"
        )
    if not np.array_equal(distances, distances.T):
        raise RuntimeError("Shortest-hop distance matrix is not symmetric")
    if not np.all(np.diag(distances) == 0):
        raise RuntimeError("Shortest-hop distance diagonal must be zero")
    return distances.astype(np.int64)


def symmetric_topology_knn(
    distances: np.ndarray, target_node_ids: np.ndarray, k: int
) -> nx.Graph:
    """按 (hop distance, node_id) 确定性排序并取 symmetric kNN。"""

    node_count = len(target_node_ids)
    if not 1 <= k < node_count:
        raise ValueError(f"k must be in [1, {node_count - 1}], got {k}")

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    node_ids = [str(node_id) for node_id in target_node_ids.tolist()]
    for source in range(node_count):
        candidates = [target for target in range(node_count) if target != source]
        candidates.sort(key=lambda target: (int(distances[source, target]), node_ids[target]))
        for target in candidates[:k]:
            graph.add_edge(min(source, target), max(source, target))
    return graph


def summarize_graph(graph: nx.Graph) -> GraphSummary:
    node_count = graph.number_of_nodes()
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.int64)
    return GraphSummary(
        edge_count=int(graph.number_of_edges()),
        density=float(nx.density(graph)),
        average_degree=float(degrees.mean()) if node_count else 0.0,
        max_degree=int(degrees.max()) if node_count else 0,
        components=int(nx.number_connected_components(graph)) if node_count else 0,
        isolated_nodes=int(nx.number_of_isolates(graph)),
    )


def hop_summary(graph: nx.Graph, distances: np.ndarray) -> dict[str, float]:
    values = np.asarray(
        [distances[left, right] for left, right in sorted(graph.edges())],
        dtype=np.float64,
    )
    if values.size == 0:
        raise RuntimeError("topology-kNN graph unexpectedly contains no edges")
    return {
        "hop_min": float(values.min()),
        "hop_median": float(np.median(values)),
        "hop_mean": float(values.mean()),
        "hop_p75": float(np.percentile(values, 75)),
        "hop_p90": float(np.percentile(values, 90)),
        "hop_max": float(values.max()),
    }


def partition_sha256(partitions: list[np.ndarray]) -> str:
    payload = [
        np.asarray(partition, dtype=np.int64).reshape(-1).tolist()
        for partition in partitions
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def duplicate_groups_crossing_clients(
    duplicate_groups: list[np.ndarray], partitions: list[np.ndarray]
) -> int:
    """划分完成后记录重复曲线组跨客户端数量，不参与划分优化。"""

    node_to_client = {
        int(node): client_id
        for client_id, partition in enumerate(partitions)
        for node in partition.tolist()
    }
    return sum(
        len({node_to_client[int(node)] for node in group.tolist()}) > 1
        for group in duplicate_groups
    )


def official_cut_edge_records(
    source: Path,
    cut_edges: tuple[tuple[int, int], ...],
) -> list[dict[str, object]]:
    """从 canonical NPZ 读取七条切边的官方设备类型和来源证据。"""

    with np.load(source, allow_pickle=False) as archive:
        edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
        edge_type = np.asarray(archive["edge_type"]).astype(str)
        edge_source = np.asarray(archive["edge_source"]).astype(str)
        node_ids = np.asarray(archive["node_ids"]).astype(str)

    lookup: dict[tuple[int, int], tuple[str, str]] = {}
    for position in range(edge_index.shape[1]):
        left, right = (int(value) for value in edge_index[:, position])
        key = (min(left, right), max(left, right))
        evidence = (str(edge_type[position]), str(edge_source[position]))
        previous = lookup.get(key)
        if previous is not None and previous != evidence:
            raise RuntimeError(f"inconsistent official evidence for edge {key}")
        lookup[key] = evidence

    records: list[dict[str, object]] = []
    for left, right in cut_edges:
        key = (min(left, right), max(left, right))
        if key not in lookup:
            raise RuntimeError(f"cut edge {key} lacks official source evidence")
        device_type, source_reference = lookup[key]
        records.append(
            {
                "left_index": left,
                "left_node_id": str(node_ids[left]),
                "right_index": right,
                "right_node_id": str(node_ids[right]),
                "edge_type": device_type,
                "edge_source": source_reference,
            }
        )
    return records


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(
    path: Path,
    source: Path,
    partition_hash: str,
    target_counts: list[int],
    legacy_target_counts: list[int],
    cut_edge_records: list[dict[str, object]],
    duplicate_group_count: int,
    duplicate_cross_client_count: int,
    legacy_duplicate_cross_client_count: int,
    knn_rows: list[dict[str, object]],
    client_rows: list[dict[str, object]],
) -> None:
    total_range_violation = sum(
        max(10 - count, 0) + max(count - 13, 0) for count in target_counts
    )
    lines = [
        "# Target Topology Audit",
        "",
        "## Protocol",
        "",
        f"- Canonical source: `{source}`",
        "- Distance graph: official 273-node Line+Transformer graph.",
        "- Targets: 92 non-zero load nodes; structural nodes remain shortest-path relays.",
        "- Candidate k values: 2, 4, 6, 8 (diagnostic only; no k is selected here).",
        "- Symmetrization: retain an undirected edge when either endpoint selects the other.",
        "- Tie break: ascending `node_id` after ascending official shortest-path hop distance.",
        "- Client partition: deterministic official-tree partition from seven cut edges.",
        "- Partition objective: minimize total 10--13 range violation, then total squared deviation from 11.5, then the number of out-of-range regions.",
        "- Partition inputs: official adjacency, target positions, and node IDs only.",
        f"- Client partition SHA-256: `{partition_hash}`",
        "",
        "## Topology-based Client Partition",
        "",
        f"- Target counts by client: `{target_counts}`",
        f"- Legacy duplicate-aware target counts: `{legacy_target_counts}`",
        f"- Duplicate load-curve groups: {duplicate_group_count}",
        f"- Duplicate groups crossing new clients: {duplicate_cross_client_count}",
        f"- Duplicate groups crossing legacy clients: {legacy_duplicate_cross_client_count}",
        "",
        "The exact tree dynamic program found no seven-edge cut whose eight region counts all lie in 10--13. "
        f"The minimum total range violation is {total_range_violation} targets, yielding {target_counts}.",
        "",
        "The duplicate-curve statistics are computed only after the topology partition is frozen; they do not affect any cut decision.",
        "",
        "### Seven Cut Official Tree Edges",
        "",
        "| Left index | Left node ID | Right index | Right node ID | Type | Official source |",
        "|---:|---|---:|---|---|---|",
    ]
    for record in cut_edge_records:
        escaped_source = str(record["edge_source"]).replace("|", "\\|")
        lines.append(
            f"| {record['left_index']} | `{record['left_node_id']}` | "
            f"{record['right_index']} | `{record['right_node_id']}` | "
            f"{record['edge_type']} | `{escaped_source}` |"
        )

    lines.extend(
        [
        "",
        "## Global Target Graph Statistics",
        "",
        "| k | Edges | Density | Avg degree | Max degree | Components | Isolated | Hop min | Hop median | Hop mean | Hop P75 | Hop P90 | Hop max | Cross-client ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in knn_rows:
        lines.append(
            "| {k} | {edge_count} | {density} | {average_degree} | {max_degree} | "
            "{components} | {isolated_nodes} | {hop_min} | {hop_median} | "
            "{hop_mean} | {hop_p75} | {hop_p90} | {hop_max} | "
            "{cross_client_edge_ratio} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## Client-induced Subgraph Statistics",
            "",
            "| k | Client | Targets | Edges | Density | Avg degree | Max degree | Components | Isolated |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in client_rows:
        lines.append(
            "| {k} | {client_id} | {target_count} | {induced_edge_count} | "
            "{density} | {average_degree} | {max_degree} | {components} | "
            "{isolated_nodes} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "These tables are topology diagnostics only. They contain no model accuracy, "
            "do not select k, and do not change the formal graph configuration.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    data = SmartDS.load(source)
    target_indices = data.active_indices
    if len(target_indices) != 92:
        raise RuntimeError(f"Expected 92 target nodes, found {len(target_indices)}")

    distances = full_target_hop_distances(data.adj, target_indices)
    target_node_ids = data.node_ids[target_indices]
    partitions, cut_edges = data.topology_client_partition(8)
    legacy_partitions = data.legacy_duplicate_aware_client_partitions(8)
    target_lookup = {
        int(full_index): target_index
        for target_index, full_index in enumerate(target_indices.tolist())
    }
    client_target_positions: list[np.ndarray] = []
    target_to_client: dict[int, int] = {}
    for client_id, full_indices in enumerate(partitions):
        positions = np.asarray(
            [target_lookup[int(full_index)] for full_index in full_indices.tolist()],
            dtype=np.int64,
        )
        client_target_positions.append(positions)
        for position in positions.tolist():
            if position in target_to_client:
                raise RuntimeError(f"Target {position} appears in multiple clients")
            target_to_client[position] = client_id
    if len(target_to_client) != len(target_indices):
        raise RuntimeError("Client partitions do not cover all target nodes exactly once")

    knn_rows: list[dict[str, object]] = []
    client_rows: list[dict[str, object]] = []
    for k in K_VALUES:
        graph = symmetric_topology_knn(distances, target_node_ids, k)
        graph_stats = summarize_graph(graph)
        hops = hop_summary(graph, distances)
        cross_client_edges = sum(
            target_to_client[left] != target_to_client[right]
            for left, right in graph.edges()
        )
        knn_rows.append(
            {
                "k": k,
                "target_nodes": len(target_indices),
                "edge_count": graph_stats.edge_count,
                "density": graph_stats.density,
                "average_degree": graph_stats.average_degree,
                "max_degree": graph_stats.max_degree,
                "components": graph_stats.components,
                "isolated_nodes": graph_stats.isolated_nodes,
                **hops,
                "cross_client_edges": int(cross_client_edges),
                "cross_client_edge_ratio": float(
                    cross_client_edges / graph_stats.edge_count
                ),
            }
        )

        for client_id, positions in enumerate(client_target_positions):
            induced = graph.subgraph(positions.tolist()).copy()
            summary = summarize_graph(induced)
            client_rows.append(
                {
                    "k": k,
                    "client_id": client_id,
                    "target_count": len(positions),
                    "induced_edge_count": summary.edge_count,
                    "density": summary.density,
                    "average_degree": summary.average_degree,
                    "max_degree": summary.max_degree,
                    "components": summary.components,
                    "isolated_nodes": summary.isolated_nodes,
                }
            )

    write_csv(
        args.knn_stats,
        knn_rows,
        [
            "k",
            "target_nodes",
            "edge_count",
            "density",
            "average_degree",
            "max_degree",
            "components",
            "isolated_nodes",
            "hop_min",
            "hop_median",
            "hop_mean",
            "hop_p75",
            "hop_p90",
            "hop_max",
            "cross_client_edges",
            "cross_client_edge_ratio",
        ],
    )
    write_csv(
        args.client_stats,
        client_rows,
        [
            "k",
            "client_id",
            "target_count",
            "induced_edge_count",
            "density",
            "average_degree",
            "max_degree",
            "components",
            "isolated_nodes",
        ],
    )
    write_report(
        args.report,
        source,
        partition_sha256(partitions),
        [len(partition) for partition in partitions],
        [len(partition) for partition in legacy_partitions],
        official_cut_edge_records(source, cut_edges),
        sum(len(group) > 1 for group in data.duplicate_groups),
        duplicate_groups_crossing_clients(
            [group for group in data.duplicate_groups if len(group) > 1], partitions
        ),
        duplicate_groups_crossing_clients(
            [group for group in data.duplicate_groups if len(group) > 1], legacy_partitions
        ),
        knn_rows,
        client_rows,
    )
    print(f"target topology audit complete: {args.report}")


if __name__ == "__main__":
    main()
