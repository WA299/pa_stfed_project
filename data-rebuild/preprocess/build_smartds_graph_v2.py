#!/usr/bin/env python3
"""从已审计的官方 OpenDSS 文件生成 SmartDS v2 图数据。

若官方文件不能覆盖当前 legacy 节点并提供设备来源证据，脚本只写出
``BLOCKED`` 元数据，不会把旧 adj、欧氏 MST 或节点名称关系写入物理图。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

from audit_smartds_data import DEFAULT_FEEDER_ROOT, audit_dataset, parse_official_raw, sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY = ROOT / "data" / "legacy" / "smartds_graph_legacy.npz"
DEFAULT_RAW = ROOT / "data" / "raw" / "SMARTDS"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
DEFAULT_REPORTS = ROOT / "reports"


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _quarantine_provisional_artifacts(output_dir: Path) -> list[str]:
    """将 blocker 之前遗留的 v2 NPZ 移入显式隔离目录。

    这些文件是旧审计状态下生成的 provisional 产物，保留用于追溯但不能
    与当前 blocker 元数据并列放在 canonical 文件名下，避免误送入训练。
    """

    quarantine = output_dir / "blocked_provisional"
    moved: list[str] = []
    for name in ("smartds_full_graph_v2.npz", "smartds_target_graph_v2.npz"):
        source = output_dir / name
        if not source.exists():
            continue
        quarantine.mkdir(parents=True, exist_ok=True)
        destination = quarantine / f"{name}.blocked"
        if destination.exists():
            destination = quarantine / f"{name}.{sha256(source)[:12]}.blocked"
        source.replace(destination)
        moved.append(destination.relative_to(output_dir).as_posix())
    return moved


def _normalise_bus(value: str) -> str:
    return value.strip().lower()


def _build_full_graph(legacy: dict[str, np.ndarray], official: dict) -> tuple[dict, dict]:
    legacy_ids = legacy["node_ids"].astype(str)
    legacy_lookup = {node.lower(): index for index, node in enumerate(legacy_ids)}
    official_nodes = set(official["coords"]).union(official["load_buses"])
    for record in official["records"]:
        official_nodes.add(record["source"])
        official_nodes.add(record["target"])
    matched = set(legacy_lookup).intersection(official_nodes)
    if len(matched) != len(legacy_ids):
        raise RuntimeError(
            "官方 OpenDSS 节点未完整覆盖 legacy NPZ，拒绝生成 canonical 图；"
            f" matched={len(matched)}/{len(legacy_ids)}"
        )

    # 当前负荷序列与坐标只覆盖 legacy 节点；匹配成功后按 legacy 顺序保留，
    # 这样 target_load_ts 和 full_to_target_mapping 不会发生隐式重排。
    node_ids = legacy_ids.copy()
    n_nodes = len(node_ids)
    coords = legacy["node_coords"].astype(np.float32).copy()
    for name, index in legacy_lookup.items():
        if name in official["coords"]:
            coords[index] = np.asarray(official["coords"][name], dtype=np.float32)

    lookup = {node.lower(): index for index, node in enumerate(node_ids)}
    edge_records: list[dict] = []
    seen: set[tuple[int, int, str, str]] = set()
    for record in official["records"]:
        source = lookup.get(_normalise_bus(record["source"]))
        target = lookup.get(_normalise_bus(record["target"]))
        if source is None or target is None or source == target:
            continue
        left, right = sorted((source, target))
        key = (left, right, record["device_type"], record["device_name"])
        if key in seen:
            continue
        seen.add(key)
        edge_records.append(record | {"source_index": left, "target_index": right})

    if not edge_records:
        raise RuntimeError("官方文件未提供可映射到 legacy 节点的 Line/Transformer 边")

    # 同一端点若有多个设备，保留各条来源证据；图边集合去重但 edge_type
    # 优先保留 transformer（更具体），否则保留 line。
    pair_records: dict[tuple[int, int], list[dict]] = {}
    for record in edge_records:
        pair_records.setdefault((record["source_index"], record["target_index"]), []).append(record)
    pairs = sorted(pair_records)
    edge_index = np.asarray(
        [[u, v] for u, v in pairs for u, v in ((u, v), (v, u))],
        dtype=np.int64,
    ).T
    edge_type: list[str] = []
    edge_source: list[str] = []
    edge_attr_json: list[str] = []
    edge_length: list[float] = []
    edge_length_units: list[str] = []
    edge_phases: list[int] = []
    edge_enabled: list[str] = []
    edge_switch: list[str] = []
    for pair in pairs:
        records = pair_records[pair]
        selected = sorted(records, key=lambda item: (item["device_type"] != "transformer", item["device_name"]))[0]
        edge_type.extend([selected["device_type"], selected["device_type"]])
        edge_source.extend([f"{selected['source_file']}:{selected['source_line']}"] * 2)
        attrs = selected.get("attributes", {})
        encoded_attrs = json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        edge_attr_json.extend([encoded_attrs, encoded_attrs])
        try:
            length = float(attrs.get("length", "nan"))
        except ValueError:
            length = float("nan")
        try:
            phases = int(float(attrs.get("phases", "-1")))
        except ValueError:
            phases = -1
        edge_length.extend([length, length])
        edge_length_units.extend([attrs.get("units", ""), attrs.get("units", "")])
        edge_phases.extend([phases, phases])
        edge_enabled.extend([attrs.get("enabled", ""), attrs.get("enabled", "")])
        edge_switch.extend([attrs.get("switch", ""), attrs.get("switch", "")])

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for source, target in pairs:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    load_mask = np.asarray([node.lower() in official["load_buses"] for node in node_ids], dtype=bool)
    target_mask = np.any(np.abs(legacy["load_ts"]) > 0, axis=0)
    full = {
        "node_ids": node_ids,
        "node_coords": coords,
        "edge_index": edge_index,
        "edge_type": np.asarray(edge_type, dtype="U16"),
        "edge_source": np.asarray(edge_source, dtype="U512"),
        "edge_attr_json": np.asarray(edge_attr_json, dtype="U2048"),
        "edge_length": np.asarray(edge_length, dtype=np.float32),
        "edge_length_units": np.asarray(edge_length_units, dtype="U16"),
        "edge_phases": np.asarray(edge_phases, dtype=np.int16),
        "edge_enabled": np.asarray(edge_enabled, dtype="U8"),
        "edge_switch": np.asarray(edge_switch, dtype="U8"),
        "adj": adjacency,
        "target_mask": target_mask.astype(bool),
        "load_mask": load_mask.astype(bool),
    }
    evidence = {"records": edge_records, "pair_records": pair_records}
    return full, evidence


def _project_target_graph(full: dict[str, np.ndarray], load_ts: np.ndarray) -> dict[str, np.ndarray]:
    target_indices = np.flatnonzero(full["target_mask"]).astype(np.int64)
    if np.any(~full["load_mask"][target_indices]):
        raise RuntimeError("至少一个 target node 没有官方 Load 设备证据")
    target_ids = full["node_ids"][target_indices]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(full["node_ids"])))
    graph.add_edges_from(full["edge_index"].T.tolist())

    projected_pairs: list[tuple[int, int]] = []
    projected_attr: list[list[float]] = []
    projected_paths: list[str] = []
    hops = np.full((len(target_indices), len(target_indices)), np.inf, dtype=np.float32)
    np.fill_diagonal(hops, 0.0)
    for left_position, source in enumerate(target_indices.tolist()):
        for right_position in range(left_position + 1, len(target_indices)):
            target = int(target_indices[right_position])
            try:
                shortest_paths = list(nx.all_shortest_paths(graph, source, target))
            except nx.NetworkXNoPath:
                continue
            selected_path = sorted(shortest_paths, key=lambda path: tuple(path))[0]
            hops[left_position, right_position] = hops[right_position, left_position] = float(len(selected_path) - 1)
            interior_targets = set(selected_path[1:-1]).intersection(set(target_indices.tolist()))
            if interior_targets:
                continue
            projected_pairs.append((left_position, right_position))
            projected_attr.append([float(len(selected_path) - 1), float("nan")])
            projected_paths.append(json.dumps(selected_path, separators=(",", ":")))

    projected_edge_index = (
        np.asarray(
            [[u, v] for u, v in projected_pairs for u, v in ((u, v), (v, u))],
            dtype=np.int64,
        ).T
        if projected_pairs
        else np.empty((2, 0), dtype=np.int64)
    )
    projected_edge_attr = np.asarray([attr for attr in projected_attr for attr in (attr, attr)], dtype=np.float32) if projected_attr else np.empty((0, 2), dtype=np.float32)
    return {
        "target_node_ids": target_ids,
        "target_node_coords": full["node_coords"][target_indices],
        "target_load_ts": load_ts[:, target_indices],
        "target_shortest_hops": hops,
        "topology_projected_edge_index": projected_edge_index,
        "topology_projected_edge_attr": projected_edge_attr,
        "topology_projected_path_nodes": np.asarray(projected_paths, dtype="U4096"),
        "full_to_target_mapping": np.full(len(full["node_ids"]), -1, dtype=np.int64),
        "target_to_full_mapping": target_indices,
    }


def build(legacy_path: Path, raw_root: Path, output_dir: Path, reports_dir: Path, feeder_root: str = DEFAULT_FEEDER_ROOT) -> dict:
    audit = audit_dataset(legacy_path, raw_root, reports_dir, feeder_root=feeder_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict = {
        "processing_version": "smartds_graph_v2_evidence_first_20260829",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": audit["status"],
        "legacy_sha256": sha256(legacy_path),
        "raw_root": str(raw_root),
        "canonical_topology": False,
        "feeder_root": feeder_root,
        "profile_mapping_status": audit["time_series"]["official_profile_mapping"]["status"],
        "profile_mapping_verified": bool(audit["constraints"]["load_series_mapping_verified"]),
    }
    if audit["status"] != "RAW_SOURCE_VERIFIED_CANONICAL":
        metadata["quarantined_provisional_artifacts"] = _quarantine_provisional_artifacts(output_dir)
        if audit["status"] == "BLOCKED_LOAD_SERIES_MAPPING_UNVERIFIED":
            metadata["blocker"] = (
                "官方拓扑证据已匹配，但 legacy load_ts 无法按官方 OpenDSS-native "
                "(Loads.dss kW × LoadShapes.dss mult) 或 parquet-native 口径逐点复现；"
                "未生成可用于正式实验的 canonical graph"
            )
        else:
            metadata["blocker"] = "官方 OpenDSS 原始文件未能完整覆盖 legacy 节点或设备来源证据不足，未生成 canonical graph"
        metadata["topology_evidence_available"] = bool(not audit["official_source"].get("official_graph", {}).get("components", 0) == 0)
        (output_dir / "smartds_metadata_v2.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return metadata

    with np.load(legacy_path, allow_pickle=False) as data:
        legacy = {name: np.asarray(data[name]) for name in data.files}
    official = parse_official_raw(raw_root, feeder_root=feeder_root)
    full, evidence = _build_full_graph(legacy, official)
    target = _project_target_graph(full, legacy["load_ts"])
    for index in np.flatnonzero(full["target_mask"]):
        target["full_to_target_mapping"][index] = int(np.flatnonzero(target["target_to_full_mapping"] == index)[0])

    # 一致性断言：任何 canonical 边都必须能回到官方记录，且不允许 MST edge。
    official_pairs = {(r["source_index"], r["target_index"]) for r in evidence["records"]}
    for source, target_index in zip(full["edge_index"][0], full["edge_index"][1]):
        if tuple(sorted((int(source), int(target_index)))) not in official_pairs:
            raise AssertionError("canonical edge lacks official source evidence")
    if len(np.unique(full["node_ids"])) != len(full["node_ids"]):
        raise AssertionError("node_id not unique")
    if np.any(full["edge_index"] < 0) or np.any(full["edge_index"] >= len(full["node_ids"])):
        raise AssertionError("edge_index out of range")
    if not np.array_equal(full["adj"] > 0, (full["adj"] > 0).T):
        raise AssertionError("adjacency not symmetric")
    if np.isnan(target["target_load_ts"]).any() or np.isinf(target["target_load_ts"]).any():
        raise AssertionError("target load contains NaN/Inf")
    if target["target_load_ts"].shape[1] != target["target_node_ids"].shape[0]:
        raise AssertionError("target load shape mismatch")
    if not np.array_equal(full["node_ids"][target["target_to_full_mapping"]], target["target_node_ids"]):
        raise AssertionError("target mapping is not reversible")

    np.savez_compressed(output_dir / "smartds_full_graph_v2.npz", **full)
    np.savez_compressed(output_dir / "smartds_target_graph_v2.npz", **target)
    graph = nx.from_numpy_array(full["adj"])
    projected_graph = nx.Graph()
    projected_graph.add_nodes_from(range(int(target["target_node_ids"].shape[0])))
    projected_graph.add_edges_from(target["topology_projected_edge_index"].T.tolist())
    finite_hops = target["target_shortest_hops"][np.triu(np.isfinite(target["target_shortest_hops"]) & (target["target_shortest_hops"] > 0), 1)]
    projected_edges = int(target["topology_projected_edge_index"].shape[1] // 2)
    projected_density = float(projected_edges / max(target["target_node_ids"].shape[0] * (target["target_node_ids"].shape[0] - 1) / 2, 1))
    finite_edge_lengths = int(np.isfinite(full["edge_length"]).sum() // 2)
    metadata.update(
        {
            "canonical_topology": True,
            "source": "official SMART-DS/OpenDSS files under data/raw/SMARTDS",
            "official_scope": audit["official_source"]["scope"],
            "official_scope_files": audit["official_source"]["scope_files"],
            "official_load_shape_count": audit["time_series"]["official_load_shape_count"],
            "official_profile_count": audit["time_series"]["official_profile_count"],
            "official_profiles_present": audit["time_series"]["official_profiles_present"],
            "profile_mapping_status": audit["time_series"]["official_profile_mapping"]["status"],
            "profile_mapping_verified": bool(audit["constraints"]["load_series_mapping_verified"]),
            "full_nodes": int(len(full["node_ids"])),
            "target_nodes": int(full["target_mask"].sum()),
            "zero_load_structural_nodes": int((~full["target_mask"]).sum()),
            "line_edges_undirected": int(sum(t == "line" for t in full["edge_type"]) // 2),
            "transformer_edges_undirected": int(sum(t == "transformer" for t in full["edge_type"]) // 2),
            "full_edges_undirected": int(graph.number_of_edges()),
            "full_components": int(nx.number_connected_components(graph)),
            "target_projected_edges_undirected": int(target["topology_projected_edge_index"].shape[1] // 2),
            "target_projected_components": int(nx.number_connected_components(projected_graph)),
            "target_projected_density": projected_density,
            "target_projected_mean_degree": float(2 * projected_edges / max(target["target_node_ids"].shape[0], 1)),
            "target_projected_max_degree": int(max(dict(projected_graph.degree()).values(), default=0)),
            "target_shortest_hops_finite_pairs": int(finite_hops.size),
            "target_shortest_hops_min": float(np.min(finite_hops)) if finite_hops.size else None,
            "target_shortest_hops_median": float(np.median(finite_hops)) if finite_hops.size else None,
            "target_shortest_hops_p90": float(np.percentile(finite_hops, 90)) if finite_hops.size else None,
            "target_shortest_hops_max": float(np.max(finite_hops)) if finite_hops.size else None,
            "full_edge_length_available_undirected": finite_edge_lengths,
            "target_path_distance_available": False,
            "target_projection_overdense": bool(projected_density > 0.5),
            "sampling_interval_minutes": 15,
            "timesteps": int(target["target_load_ts"].shape[0]),
            "target_load_ts_shape": list(target["target_load_ts"].shape),
            "target_projection_rule": "lexicographically first shortest path whose interior contains no third target node",
        }
    )
    (output_dir / "smartds_metadata_v2.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--feeder-root", default=DEFAULT_FEEDER_ROOT)
    args = parser.parse_args()
    metadata = build(args.legacy.resolve(), args.raw_root.resolve(), args.output.resolve(), args.reports.resolve(), feeder_root=args.feeder_root)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
