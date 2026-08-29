#!/usr/bin/env python3
"""从官方 SMART-DS/OpenDSS 文件独立生成 canonical 负荷序列和图数据。

本脚本不读取 legacy ``smartds_graph.npz`` 的负荷数值，也不把 legacy
邻接矩阵、节点命名规则或欧氏 MST 当作物理证据。节点集合来自当前 feeder
自己的 ``Buscoords.dss``，设备边来自该 feeder 的 ``Lines.dss`` 和
``Transformers.dss``，负荷序列严格按官方文档定义的
``Loads.dss kW * LoadShapes.dss mult`` 逐母线汇总。

输出目录应与旧的 ``candidate_*`` blocker 目录分开。这样即使 legacy
序列无法追溯，仍可得到一份来源明确、可重跑的官方序列候选；两者不得
混为同一 ground truth。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

from audit_smartds_data import DEFAULT_FEEDER_ROOT, parse_official_raw, sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "SMARTDS"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "official_v1_2018_P10R_p10rdt7719"
DEFAULT_REPORTS = ROOT / "reports" / "official_v1_2018_P10R_p10rdt7719"


def _normalise_bus(value: str) -> str:
    value = value.strip().strip("[](){}'\"").rstrip(",;")
    return value.split(".", 1)[0].strip().lower()


def _read_feeder_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """按官方 Buscoords.dss 文件顺序返回 node_ids 和坐标。"""

    node_ids: list[str] = []
    coords: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            tokens = raw.strip().split()
            if len(tokens) < 3:
                continue
            try:
                x, y = float(tokens[1]), float(tokens[2])
            except ValueError:
                continue
            node_ids.append(_normalise_bus(tokens[0]))
            coords.append([x, y])
    ids = np.asarray(node_ids, dtype="U256")
    coordinates = np.asarray(coords, dtype=np.float32)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("feeder Buscoords.dss 的 node_id 为空或不唯一")
    if coordinates.shape != (len(ids), 2) or not np.isfinite(coordinates).all():
        raise ValueError("feeder Buscoords.dss 坐标形状或有限性校验失败")
    return ids, coordinates


def _profile_name(shape: dict) -> str | None:
    match = re.search(r"file\s*=\s*([^\)\s]+)", shape.get("mult_file", ""), flags=re.I)
    return Path(match.group(1)).name if match else None


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _line_intermediates(official: dict) -> dict[str, dict]:
    """把 Intermediates.txt 作为线段几何证据附着到同名 Line。"""

    return {
        str(row.get("line_name", "")).lower(): row
        for row in official.get("intermediates", [])
        if row.get("line_name")
    }


def _edge_payload(records: list[dict], node_lookup: dict[str, int], intermediates: dict[str, dict]) -> tuple[dict[str, np.ndarray], dict]:
    """构造只含官方设备证据的双向边数组。"""

    pair_records: dict[tuple[int, int], list[dict]] = defaultdict(list)
    excluded: list[dict] = []
    for record in records:
        source = node_lookup.get(_normalise_bus(record["source"]))
        target = node_lookup.get(_normalise_bus(record["target"]))
        if source is None or target is None:
            # feeder Lines.dss 中可能有一条指向相邻 feeder 的边界开关，
            # 它不是当前 feeder 的节点，保留审计证据但不写入当前图。
            excluded.append(record)
            continue
        if source == target:
            continue
        pair_records[tuple(sorted((source, target)))].append(record)

    pairs = sorted(pair_records)
    edge_index = np.asarray(
        [[u, v] for u, v in pairs for u, v in ((u, v), (v, u))],
        dtype=np.int64,
    ).T if pairs else np.empty((2, 0), dtype=np.int64)
    edge_type: list[str] = []
    edge_source: list[str] = []
    edge_attr_json: list[str] = []
    edge_intermediates_json: list[str] = []
    edge_length: list[float] = []
    edge_length_units: list[str] = []
    edge_phases: list[int] = []
    edge_enabled: list[str] = []
    edge_switch: list[str] = []
    for pair in pairs:
        # 同一端点多设备时，保留 transformer 优先的稳定代表；完整设备
        # 证据仍写入 metadata/report，不因去重而宣称只有一台设备。
        candidates = pair_records[pair]
        selected = sorted(candidates, key=lambda row: (row["device_type"] != "transformer", row["device_name"]))[0]
        attrs = dict(selected.get("attributes", {}))
        intermediate = intermediates.get(str(selected["device_name"]).lower(), {})
        attrs["intermediate_points"] = intermediate.get("intermediate_points", [])
        attrs["intermediate_source"] = (
            f"{intermediate.get('source_file', '')}:{intermediate.get('source_line', '')}"
            if intermediate
            else ""
        )
        encoded = json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        intermediate_encoded = json.dumps(intermediate.get("intermediate_points", []), separators=(",", ":"))
        try:
            length = float(selected.get("attributes", {}).get("length", "nan"))
        except (TypeError, ValueError):
            length = float("nan")
        try:
            phases = int(float(selected.get("attributes", {}).get("phases", "-1")))
        except (TypeError, ValueError):
            phases = -1
        for _ in range(2):
            edge_type.append(str(selected["device_type"]))
            edge_source.append(f"{selected['source_file']}:{selected['source_line']}")
            edge_attr_json.append(encoded)
            edge_intermediates_json.append(intermediate_encoded)
            edge_length.append(length)
            edge_length_units.append(str(selected.get("attributes", {}).get("units", "")))
            edge_phases.append(phases)
            edge_enabled.append(str(selected.get("attributes", {}).get("enabled", "")))
            edge_switch.append(str(selected.get("attributes", {}).get("switch", "")))

    payload = {
        "edge_index": edge_index,
        "edge_type": np.asarray(edge_type, dtype="U16"),
        "edge_source": np.asarray(edge_source, dtype="U512"),
        "edge_attr_json": np.asarray(edge_attr_json, dtype="U4096"),
        "edge_intermediates_json": np.asarray(edge_intermediates_json, dtype="U4096"),
        "edge_length": np.asarray(edge_length, dtype=np.float32),
        "edge_length_units": np.asarray(edge_length_units, dtype="U16"),
        "edge_phases": np.asarray(edge_phases, dtype=np.int16),
        "edge_enabled": np.asarray(edge_enabled, dtype="U8"),
        "edge_switch": np.asarray(edge_switch, dtype="U8"),
    }
    return payload, {"pair_records": pair_records, "excluded_boundary_records": excluded}


def _build_series(
    raw_root: Path,
    feeder_root: str,
    node_ids: np.ndarray,
    official: dict,
) -> tuple[np.ndarray, dict, dict[str, list[dict]]]:
    """按官方 OpenDSS-native 规则生成 [T,N] kW 序列。"""

    prefix = feeder_root.rstrip("/").lower() + "/"
    load_records = [
        row for row in official["load_records"]
        if str(row.get("source_file", "")).lower().startswith(prefix)
    ]
    shapes = {
        name: shape
        for name, shape in official.get("load_shapes", {}).items()
        if str(shape.get("source_file", "")).lower().startswith(prefix)
    }
    if not load_records:
        raise RuntimeError("当前 feeder 的 Loads.dss 未解析到任何 Load 记录")
    node_lookup = {str(node).lower(): index for index, node in enumerate(node_ids.astype(str))}
    by_bus: dict[str, list[dict]] = defaultdict(list)
    for row in load_records:
        bus = _normalise_bus(row["bus"])
        if bus not in node_lookup:
            raise RuntimeError(f"Load bus 不在 feeder Buscoords.dss: {bus}")
        by_bus[bus].append(row)

    profile_cache: dict[str, np.ndarray] = {}
    profile_evidence: dict[str, dict] = {}
    required_shapes = sorted({str(row.get("yearly_shape", "")) for row in load_records})
    for shape_name in required_shapes:
        if not shape_name or shape_name not in shapes:
            raise RuntimeError(f"缺少 feeder LoadShapes.dss 定义: {shape_name}")
        profile = _profile_name(shapes[shape_name])
        if not profile:
            raise RuntimeError(f"LoadShape 未提供 mult file: {shape_name}")
        path = raw_root / "profiles" / profile
        if not path.exists():
            raise FileNotFoundError(f"缺少官方 profile: {path}")
        values = np.loadtxt(path, dtype=np.float64)
        if values.ndim != 1 or values.size != 35040 or not np.isfinite(values).all():
            raise ValueError(f"profile 长度/有限性失败: {path} shape={values.shape}")
        profile_cache[shape_name] = values
        profile_evidence[shape_name] = {
            "profile": profile,
            "source_file": shapes[shape_name].get("source_file", ""),
            "source_line": shapes[shape_name].get("source_line", 0),
            "npts": shapes[shape_name].get("npts", ""),
            "interval": shapes[shape_name].get("interval", ""),
            "sha256": sha256(path),
        }

    series = np.zeros((35040, len(node_ids)), dtype=np.float64)
    mapping_rows: dict[str, list[dict]] = {}
    for bus, rows in sorted(by_bus.items()):
        index = node_lookup[bus]
        row_evidence: list[dict] = []
        for row in rows:
            try:
                kw = float(row["kW"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Loads.dss kW 非数值: {row}") from exc
            if not np.isfinite(kw):
                raise ValueError(f"Loads.dss kW 非有限: {row}")
            shape_name = str(row.get("yearly_shape", ""))
            # 官方文档：_1/_2 在 Loads.dss 中已经是拆分后的半客户峰值；
            # 所以 OpenDSS-native 口径直接使用 kW * mult，不再乘 0.5。
            series[:, index] += kw * profile_cache[shape_name]
            row_evidence.append(
                {
                    "device_name": row.get("device_name", ""),
                    "profile": profile_evidence[shape_name]["profile"],
                    "kW": kw,
                    "center_tap_split": str(row.get("device_name", "")).lower().endswith(("_1", "_2")),
                    "formula": "Loads.dss kW * LoadShapes.dss mult",
                    "source_file": row.get("source_file", ""),
                    "source_line": row.get("source_line", 0),
                }
            )
        mapping_rows[str(node_ids[index])] = row_evidence
    if not np.isfinite(series).all():
        raise AssertionError("官方负荷序列含 NaN/Inf")
    target_mask = np.any(np.abs(series) > 0, axis=0)
    if int(target_mask.sum()) != len(by_bus):
        raise AssertionError("非零目标节点数与官方 Load bus 数不一致")
    return series, {"target_mask": target_mask, "profile_evidence": profile_evidence, "load_records": load_records}, mapping_rows


def _project_targets(full: dict[str, np.ndarray], target_mask: np.ndarray, series: np.ndarray) -> dict[str, np.ndarray]:
    target_to_full = np.flatnonzero(target_mask).astype(np.int64)
    target_ids = full["node_ids"][target_to_full]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(full["node_ids"])))
    graph.add_edges_from(full["edge_index"].T.tolist())
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes)
    edge_lengths = full["edge_length"]
    for position, (source, target) in enumerate(full["edge_index"].T.tolist()):
        if position % 2:
            continue
        length = float(edge_lengths[position])
        if np.isfinite(length):
            weighted.add_edge(int(source), int(target), weight=length)
    projected_pairs: list[tuple[int, int]] = []
    projected_attr: list[list[float]] = []
    projected_paths: list[str] = []
    hops = np.full((len(target_to_full), len(target_to_full)), np.inf, dtype=np.float32)
    path_distance = np.full_like(hops, np.nan)
    np.fill_diagonal(hops, 0.0)
    np.fill_diagonal(path_distance, 0.0)
    target_set = set(target_to_full.tolist())
    for left_position, source in enumerate(target_to_full.tolist()):
        for right_position in range(left_position + 1, len(target_to_full)):
            target = int(target_to_full[right_position])
            try:
                shortest = sorted(nx.all_shortest_paths(graph, source, target), key=tuple)[0]
            except nx.NetworkXNoPath:
                continue
            hop = float(len(shortest) - 1)
            hops[left_position, right_position] = hops[right_position, left_position] = hop
            if all(node in weighted for node in shortest) and all(weighted.has_edge(int(a), int(b)) for a, b in zip(shortest, shortest[1:])):
                distance = float(sum(weighted[int(a)][int(b)]["weight"] for a, b in zip(shortest, shortest[1:])))
                path_distance[left_position, right_position] = path_distance[right_position, left_position] = distance
            if target_set.intersection(shortest[1:-1]):
                continue
            projected_pairs.append((left_position, right_position))
            projected_attr.append([hop, float(path_distance[left_position, right_position])])
            projected_paths.append(json.dumps(shortest, separators=(",", ":")))
    projected_edge_index = (
        np.asarray([[u, v] for u, v in projected_pairs for u, v in ((u, v), (v, u))], dtype=np.int64).T
        if projected_pairs
        else np.empty((2, 0), dtype=np.int64)
    )
    projected_edge_attr = (
        np.asarray([attr for attr in projected_attr for attr in (attr, attr)], dtype=np.float32)
        if projected_attr
        else np.empty((0, 2), dtype=np.float32)
    )
    full_to_target = np.full(len(full["node_ids"]), -1, dtype=np.int64)
    for position, index in enumerate(target_to_full.tolist()):
        full_to_target[index] = position
    return {
        "target_node_ids": target_ids,
        "target_node_coords": full["node_coords"][target_to_full],
        "target_load_ts": series[:, target_to_full].astype(np.float32),
        "target_shortest_hops": hops,
        "target_path_distance": path_distance,
        "topology_projected_edge_index": projected_edge_index,
        "topology_projected_edge_attr": projected_edge_attr,
        "topology_projected_path_nodes": np.asarray(projected_paths, dtype="U4096"),
        "full_to_target_mapping": full_to_target,
        "target_to_full_mapping": target_to_full,
    }


def build(raw_root: Path, output_dir: Path, reports_dir: Path, feeder_root: str = DEFAULT_FEEDER_ROOT) -> dict:
    feeder_dir = raw_root / feeder_root
    coords_path = feeder_dir / "Buscoords.dss"
    if not coords_path.exists():
        raise FileNotFoundError(f"缺少 feeder Buscoords.dss: {coords_path}")
    node_ids, node_coords = _read_feeder_coords(coords_path)
    official = parse_official_raw(raw_root, feeder_root=feeder_root)
    prefix = feeder_root.rstrip("/").lower() + "/"
    records = [
        row for row in official["records"]
        if str(row.get("source_file", "")).lower().startswith(prefix)
        and row.get("device_type") in {"line", "transformer"}
    ]
    node_lookup = {str(node).lower(): index for index, node in enumerate(node_ids.astype(str))}
    edge_payload, edge_evidence = _edge_payload(records, node_lookup, _line_intermediates(official))
    if len(edge_evidence["excluded_boundary_records"]) != 1:
        raise AssertionError(
            "当前 feeder 预期只有一条指向相邻 feeder 的边界记录；"
            f"实际排除 {len(edge_evidence['excluded_boundary_records'])} 条"
        )
    adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
    for source, target in edge_payload["edge_index"].T.tolist():
        adjacency[int(source), int(target)] = 1.0
    series, series_evidence, mapping_rows = _build_series(raw_root, feeder_root, node_ids, official)
    full = {
        "node_ids": node_ids,
        "node_coords": node_coords,
        "adj": adjacency,
        "target_mask": series_evidence["target_mask"].astype(bool),
        "load_mask": series_evidence["target_mask"].astype(bool),
        "load_ts": series.astype(np.float32),
        **edge_payload,
    }
    target = _project_targets(full, full["target_mask"], series)

    # 构建前的最小一致性断言，失败时不写入半成品。
    if len(np.unique(full["node_ids"])) != len(full["node_ids"]):
        raise AssertionError("node_id 不唯一")
    if np.any(full["edge_index"] < 0) or np.any(full["edge_index"] >= len(node_ids)):
        raise AssertionError("edge_index 越界")
    if not np.array_equal(full["adj"] > 0, (full["adj"] > 0).T):
        raise AssertionError("adjacency 不对称")
    if not np.array_equal(
        {tuple(sorted((int(u), int(v)))) for u, v in full["edge_index"].T.tolist()},
        {tuple(sorted((int(u), int(v)))) for u, v in zip(*np.where(np.triu(full["adj"] > 0, 1)))},
    ):
        raise AssertionError("adjacency 与 edge_index 不一致")
    if any("mst" in str(source).lower() for source in full["edge_source"]):
        raise AssertionError("canonical 图禁止包含 MST 来源")
    if not np.isfinite(full["load_ts"]).all() or not np.isfinite(target["target_load_ts"]).all():
        raise AssertionError("输出负荷序列含 NaN/Inf")
    if not np.array_equal(full["node_ids"][target["target_to_full_mapping"]], target["target_node_ids"]):
        raise AssertionError("target 映射不可逆")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "smartds_full_graph_v2.npz", **full)
    np.savez_compressed(output_dir / "smartds_target_graph_v2.npz", **target)
    graph = nx.from_numpy_array(full["adj"])
    projected_graph = nx.Graph()
    projected_graph.add_nodes_from(range(len(target["target_node_ids"])))
    projected_graph.add_edges_from(target["topology_projected_edge_index"].T.tolist())
    line_edges = int(sum(value == "line" for value in full["edge_type"]) // 2)
    transformer_edges = int(sum(value == "transformer" for value in full["edge_type"]) // 2)
    projected_edges = int(target["topology_projected_edge_index"].shape[1] // 2)
    metadata = {
        "processing_version": "smartds_official_native_v1_20260829",
        "status": "RAW_SOURCE_VERIFIED_CANONICAL",
        "canonical_topology": True,
        "canonical_dataset_source": "official_opendss_native",
        "official_series_verified": True,
        "legacy_load_ts_mapping_status": "BLOCKED_VALUES_MISMATCH",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open Energy Data Initiative (OEDI)",
        "documentation_url": "https://github.com/openEDI/documentation/blob/main/SMART-DS/Readme.md",
        "documentation_commit": "9cdf598733f94d72de09ce0015f4dda671982f9f",
        "version": "v0.9",
        "year": 2018,
        "dataset": "Full_Texas",
        "region": "P10R",
        "scenario": "base_timeseries",
        "substation": feeder_root.split("/", 1)[0],
        "feeder": feeder_root.rsplit("/", 1)[-1],
        "feeder_root": feeder_root,
        "raw_root": str(raw_root.resolve()),
        "full_nodes": int(len(node_ids)),
        "target_nodes": int(full["target_mask"].sum()),
        "zero_load_structural_nodes": int((~full["target_mask"]).sum()),
        "timesteps": int(series.shape[0]),
        "sampling_interval_minutes": 15,
        "line_edges_undirected": line_edges,
        "transformer_edges_undirected": transformer_edges,
        "full_edges_undirected": int(graph.number_of_edges()),
        "full_components": int(nx.number_connected_components(graph)),
        "excluded_boundary_edge_count": len(edge_evidence["excluded_boundary_records"]),
        "excluded_boundary_edges": edge_evidence["excluded_boundary_records"],
        "intermediates_count": int(len(official.get("intermediates", []))),
        "intermediates_with_points_count": int(sum(bool(row.get("intermediate_points")) for row in official.get("intermediates", []))),
        "target_projected_edges_undirected": projected_edges,
        "target_projected_components": int(nx.number_connected_components(projected_graph)),
        "target_projected_density": float(projected_edges / max(len(target["target_node_ids"]) * (len(target["target_node_ids"]) - 1) / 2, 1)),
        "target_projected_mean_degree": float(2 * projected_edges / max(len(target["target_node_ids"]), 1)),
        "target_projected_max_degree": int(max(dict(projected_graph.degree()).values(), default=0)),
        "target_projection_rule": "lexicographically first shortest path whose interior contains no third target node",
        "load_series_formula": "sum over official Loads.dss records on each bus of kW * referenced LoadShapes.dss mult",
        "center_tap_rule": "Loads.dss _1/_2 kW already split; parquet whole-customer values would use 0.5 per split element",
        "load_series_sha256": _array_hash(series.astype(np.float32)),
        "full_npz_sha256": sha256(output_dir / "smartds_full_graph_v2.npz"),
        "target_npz_sha256": sha256(output_dir / "smartds_target_graph_v2.npz"),
        "edge_source_policy": "every edge is from feeder-scoped Lines.dss or Transformers.dss; Intermediates.txt is geometry-only evidence",
    }
    (output_dir / "smartds_metadata_v2.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (reports_dir / "official_load_mapping.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["node_id", "load_element_count", "load_elements"])
        writer.writeheader()
        for node_id in node_ids[full["target_mask"]].astype(str):
            rows = mapping_rows[node_id]
            writer.writerow({"node_id": node_id, "load_element_count": len(rows), "load_elements": json.dumps(rows, ensure_ascii=False, separators=(",", ":"))})
    report = [
        "# 官方 SMART-DS 负荷序列重建报告",
        "",
        "状态：`RAW_SOURCE_VERIFIED_CANONICAL`（独立官方序列候选）",
        "",
        "本报告不把 legacy `smartds_graph.npz` 的 `load_ts` 当作输入；旧序列与官方序列的映射审计仍为 `BLOCKED_VALUES_MISMATCH`，两者不得混用。",
        "",
        "## 来源与公式",
        "",
        f"- 官方来源：OEDI SMART-DS v0.9 / 2018 / Full_Texas / P10R / base_timeseries。",
        f"- feeder：`{feeder_root}`；节点顺序来自 feeder `Buscoords.dss`，共 `{len(node_ids)}` 个。",
        "- 物理边仅来自该 feeder 的 `Lines.dss` 与 `Transformers.dss`；`Intermediates.txt` 只保存线段中间坐标，不产生额外节点或边。",
        "- 每个目标母线的序列为 `sum(Loads.dss kW * LoadShapes.dss mult)`；中心抽头 `_1/_2` 的 `kW` 已按官方说明拆分，不重复乘 `0.5`。",
        "- profile 长度统一为 `35040`，采样间隔为 `15` 分钟；未引入天气、节假日或真实日期特征。",
        "",
        "## 重建统计",
        "",
        f"- full graph：`{len(node_ids)}` 节点，`{graph.number_of_edges()}` 条无向边，`{nx.number_connected_components(graph)}` 个连通分量。",
        f"- 边类型：Line `{line_edges}`，Transformer `{transformer_edges}`。",
        f"- 预测目标：`{int(full['target_mask'].sum())}` 个；零负荷结构节点：`{int((~full['target_mask']).sum())}` 个，保留在 full graph 但不进入 loss。",
        f"- feeder 边界外记录：`{len(edge_evidence['excluded_boundary_records'])}` 条，未写入当前 feeder 图，完整证据见 metadata。",
        f"- target projection：`{projected_edges}` 条无向边，密度 `{metadata['target_projected_density']:.4f}`；若过密，后续模型应优先使用 full graph。",
        "",
        "## 审计边界",
        "",
        "- SmartDS 是合成配电网；该产物可支持合成网络实验，不能外推为真实台区实测结果。",
        "- 旧 NPZ 负荷列与官方序列无法逐点复现，不能通过自由缩放、平移或列重排解除 blocker。",
        "- 本脚本只完成数据重建和一致性检查，不启动 PA-STFed、FedAvg、FedProx 或任何论文结论实验。",
    ]
    (reports_dir / "official_series_rebuild.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--feeder-root", default=DEFAULT_FEEDER_ROOT)
    args = parser.parse_args()
    metadata = build(args.raw_root.resolve(), args.output.resolve(), args.reports.resolve(), args.feeder_root)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
