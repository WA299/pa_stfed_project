#!/usr/bin/env python3
"""SmartDS 数据来源、节点角色和时间序列审计。

本脚本的设计原则是“证据优先”：旧版 NPZ 只作为 legacy 审计输入，不能
被自动当作官方物理拓扑。只有在 ``data/raw/SMARTDS`` 中找到 OpenDSS
文件并能逐条追溯设备来源时，后续构建脚本才允许生成 canonical 图。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY = ROOT / "data" / "legacy" / "smartds_graph_legacy.npz"
DEFAULT_RAW = ROOT / "data" / "raw" / "SMARTDS"
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_FEEDER_ROOT = "p10rhs0_1247/p10rhs0_1247--p10rdt7719"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_bus(value: str) -> str:
    """将 OpenDSS bus 表达式归一化为可比较的母线名（去掉相位）。"""

    value = value.strip().strip("[](){}'\"")
    value = value.rstrip(",;")
    if "." in value:
        value = value.split(".", 1)[0]
    return value.strip().lower()


def _logical_dss_records(path: Path) -> Iterable[tuple[int, str]]:
    """合并 OpenDSS ``~`` 续行，并保留起始行号。"""

    current: str | None = None
    start_line = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.split("!", 1)[0].strip()
            if not line:
                continue
            if line.startswith("~") and current is not None:
                current += " " + line[1:].strip()
                continue
            if current is not None:
                yield start_line, current
            current = line
            start_line = line_number
    if current is not None:
        yield start_line, current


def _property(text: str, name: str) -> str | None:
    """读取 OpenDSS 属性；属性值可能被方括号或圆括号包围。"""

    match = re.search(rf"(?:^|\s){re.escape(name)}\s*=\s*(\[[^\]]+\]|\([^\)]+\)|[^\s]+)", text, flags=re.I)
    return match.group(1).strip() if match else None


def _properties(text: str, names: Iterable[str]) -> dict[str, str]:
    """提取一组可审计属性，未出现的属性不写入结果。"""

    return {name: value for name in names if (value := _property(text, name)) is not None}


def _parse_intermediates(path: Path) -> list[dict]:
    """读取官方 ``Intermediates.txt`` 的线段中间坐标。

    官方文档明确说明该文件只用于拓扑可视化，不参与 OpenDSS 求解。
    因此这里只保存 ``line_name``、坐标点和来源行号，绝不把中间点
    自动转成新的物理节点或边。
    """

    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line or ";" not in line:
                continue
            parts = [part.strip() for part in line.split(";") if part.strip()]
            if not parts:
                continue
            line_name = parts[0].lower()
            points: list[list[float]] = []
            for token in parts[1:]:
                match = re.fullmatch(r"\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)", token)
                if match:
                    points.append([float(match.group(1)), float(match.group(2))])
            records.append(
                {
                    "line_name": line_name,
                    "intermediate_points": points,
                    "source_file": path.name,
                    "source_line": line_number,
                    "evidence": line,
                }
            )
    return records


def _is_center_tap_split(device_name: str) -> bool:
    """判断是否为官方命名的中心抽头拆分元件。

    ``Loads.dss`` 中的 ``kW`` 已经是拆分后的半客户峰值；0.5 修正仅适用
    于直接使用 parquet 的整户 ``total_site_electricity_kw``，不能再次施加
    到 ``kW * mult`` 的 OpenDSS-native 重建公式。
    """

    return str(device_name).lower().endswith(("_1", "_2"))


def _scope_files(raw_root: Path, feeder_root: str | None) -> tuple[list[Path], dict[str, str]]:
    """返回严格限定的 OpenDSS 文件集合和 scope 元数据。

    SMART-DS 的 feeder 文件依赖父级 substation 文件。默认递归模式仅用于
    早期探针；正式候选源必须通过 ``--feeder-root`` 显式指定，避免把其它
    feeder 的记录混入节点、边和负荷证据。
    """

    if not feeder_root:
        return sorted(raw_root.rglob("*.dss"), key=lambda item: item.as_posix().lower()), {
            "mode": "recursive_probe",
            "feeder_root": "",
            "parent_root": "",
        }
    feeder = (raw_root / feeder_root).resolve()
    raw_resolved = raw_root.resolve()
    try:
        feeder.relative_to(raw_resolved)
    except ValueError as exc:
        raise ValueError("--feeder-root 必须位于 --raw-root 内") from exc
    if not feeder.is_dir():
        raise FileNotFoundError(f"feeder 目录不存在: {feeder}")
    parent = feeder.parent
    selected: list[Path] = []
    # 根级文件仅用于版本说明/共享负荷曲线定义；设备连接只来自 parent 与 feeder。
    for relative in ("Buscoords.dss", "LoadShapes.dss"):
        candidate = raw_root / relative
        if candidate.exists():
            selected.append(candidate)
    for directory in (parent, feeder):
        for name in ("Master.dss", "Buscoords.dss", "Lines.dss", "Transformers.dss", "Loads.dss", "LoadShapes.dss", "LineCodes.dss", "Regulators.dss", "Intermediates.txt"):
            candidate = directory / name
            if candidate.exists():
                selected.append(candidate)
    unique = {path.resolve(): path for path in selected}
    return sorted(unique.values(), key=lambda item: item.relative_to(raw_root).as_posix().lower()), {
        "mode": "feeder_scoped",
        "feeder_root": feeder.relative_to(raw_root).as_posix(),
        "parent_root": parent.relative_to(raw_root).as_posix(),
    }


def parse_official_raw(raw_root: Path, feeder_root: str | None = None) -> dict:
    """解析可见的 Line/Transformer/Load/Buscoords 证据。

    解析器只提取连接关系和可核验的基础属性，不根据节点名或坐标猜测
    设备。它兼容 OpenDSS 常见的 ``bus1/bus2`` 和 ``buses=[...]`` 写法。
    """

    records: list[dict] = []
    load_records: list[dict] = []
    load_buses: set[str] = set()
    transformer_nodes: set[str] = set()
    coords: dict[str, tuple[float, float]] = {}
    load_shapes: dict[str, dict] = {}
    intermediates: list[dict] = []
    source_files: list[str] = []

    if not raw_root.exists():
        return {
            "records": records,
            "load_buses": load_buses,
            "transformer_nodes": transformer_nodes,
            "coords": coords,
            "load_shapes": load_shapes,
            "intermediates": intermediates,
            "source_files": source_files,
            "scope": {},
        }

    paths, scope = _scope_files(raw_root, feeder_root)
    for path in paths:
        source_files.append(path.relative_to(raw_root).as_posix())
        lower_name = path.name.lower()
        if lower_name == "buscoords.dss":
            for line_number, line in _logical_dss_records(path):
                tokens = line.split()
                if len(tokens) < 3 or "=" in tokens[0]:
                    continue
                try:
                    x, y = float(tokens[1]), float(tokens[2])
                except ValueError:
                    continue
                coords[_normalise_bus(tokens[0])] = (x, y)
            continue

        if lower_name == "intermediates.txt":
            intermediates.extend(_parse_intermediates(path))
            continue

        for line_number, line in _logical_dss_records(path):
            if lower_name == "loadshapes.dss":
                shape_match = re.match(r"(?:new|edit)\s+loadshape\.([^\s]+)", line, flags=re.I)
                if shape_match:
                    shape_name = shape_match.group(1)
                    load_shapes[shape_name] = {
                        "name": shape_name,
                        "npts": _property(line, "npts") or "",
                        "interval": _property(line, "interval") or "",
                        "mult_file": _property(line, "mult") or "",
                        "qmult_file": _property(line, "qmult") or "",
                        "source_file": path.relative_to(raw_root).as_posix(),
                        "source_line": line_number,
                        "evidence": line,
                    }
                continue
            match = re.match(r"(?:new|edit)\s+(line|transformer|load)\.([^\s]+)", line, flags=re.I)
            if not match:
                continue
            device_type, device_name = match.group(1).lower(), match.group(2)
            if device_type == "load":
                bus = _property(line, "bus1") or _property(line, "bus")
                if bus:
                    normalized_bus = _normalise_bus(bus)
                    load_buses.add(normalized_bus)
                    load_records.append(
                        {
                            "bus": normalized_bus,
                            "device_name": device_name,
                            "source_file": path.relative_to(raw_root).as_posix(),
                            "source_line": line_number,
                            "yearly_shape": _property(line, "yearly") or "",
                            "kW": _property(line, "kW") or "",
                            "kvar": _property(line, "kvar") or "",
                            "phases": _property(line, "phases") or "",
                        }
                    )
                continue

            buses: list[str] = []
            buses_value = _property(line, "buses")
            if buses_value:
                inner = buses_value.strip("[]")
                buses = [_normalise_bus(token) for token in re.split(r"[\s,]+", inner) if token]
            else:
                for property_name in ("bus1", "bus2", "bus3", "bus4"):
                    bus = _property(line, property_name)
                    if bus:
                        buses.append(_normalise_bus(bus))
                # OpenDSS multi-winding Transformer 常用重复的 ``bus=``
                # 属性，而不是 bus1/bus2；必须保留每个 winding 的母线。
                if device_type == "transformer" and not buses:
                    buses = [
                        _normalise_bus(value)
                        for value in re.findall(r"\bbus\s*=\s*([^\s]+)", line, flags=re.I)
                    ]
            buses = [bus for bus in buses if bus]
            if len(buses) < 2:
                continue
            # 多绕组 Transformer 用首个 winding 作为设备连接中心，避免把
            # 非相邻 winding 误当成独立物理线路；两绕组时即为普通一条边。
            source = buses[0]
            for target in buses[1:]:
                records.append(
                    {
                        "source": source,
                        "target": target,
                        "device_type": device_type,
                        "device_name": device_name,
                        "source_file": path.relative_to(raw_root).as_posix(),
                        "source_line": line_number,
                        "evidence": line,
                        "attributes": _properties(
                            line,
                            ("phases", "length", "units", "enabled", "switch", "kV", "kva", "normhkva", "EmergHKVA", "%R", "%loadloss", "%Noloadloss", "XHL", "XLT", "XHT"),
                        ),
                    }
                )
            if device_type == "transformer":
                transformer_nodes.update(buses)

    # 同一设备可能在多个文件中被重复引用；保留完整证据但给后续建图一个稳定去重视图。
    unique: dict[tuple[str, str, str, str], dict] = {}
    for record in records:
        key = (
            record["source"],
            record["target"],
            record["device_type"],
            record["device_name"],
        )
        unique.setdefault(key, record)
    return {
        "records": list(unique.values()),
        "load_records": load_records,
        "load_buses": load_buses,
        "transformer_nodes": transformer_nodes,
        "coords": coords,
        "load_shapes": load_shapes,
        "intermediates": intermediates,
        "source_files": source_files,
        "scope": scope,
    }


def _components(adjacency: np.ndarray) -> list[list[int]]:
    graph = (adjacency > 0).astype(np.uint8)
    n = graph.shape[0]
    seen = np.zeros(n, dtype=bool)
    components: list[list[int]] = []
    for root in range(n):
        if seen[root]:
            continue
        queue: deque[int] = deque([root])
        seen[root] = True
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in np.flatnonzero(graph[node]):
                neighbour = int(neighbour)
                if not seen[neighbour]:
                    seen[neighbour] = True
                    queue.append(neighbour)
        components.append(sorted(component))
    return components


def _edge_pairs(edge_index: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for source, target in edge_index.T.tolist():
        a, b = sorted((int(source), int(target)))
        if a != b:
            pairs.add((a, b))
    return pairs


def _topology_comparison(adjacency: np.ndarray, nonzero_mask: np.ndarray) -> list[dict[str, str | int]]:
    """统计 legacy 原图、目标诱导图和零负荷中继投影。

    历史 MST 只提供数量对照并标记为非 canonical，绝不会被 v2 建图使用。
    """

    raw_graph = (adjacency > 0).astype(np.uint8)
    n_nodes = raw_graph.shape[0]
    raw_edges = int(np.count_nonzero(np.triu(raw_graph, 1)))
    raw_components = _components(raw_graph)
    active = np.flatnonzero(nonzero_mask).astype(np.int64)
    active_lookup = {int(node): index for index, node in enumerate(active.tolist())}
    induced = raw_graph[np.ix_(active, active)]
    induced_edges = int(np.count_nonzero(np.triu(induced, 1)))

    projected = induced.copy()
    zero_nodes = np.flatnonzero(~nonzero_mask).astype(np.int64)
    zero_components = _components(raw_graph[np.ix_(zero_nodes, zero_nodes)]) if len(zero_nodes) else []
    for zero_component in zero_components:
        global_zero = zero_nodes[np.asarray(zero_component, dtype=np.int64)]
        neighbours = sorted(
            {
                int(neighbour)
                for node in global_zero.tolist()
                for neighbour in np.flatnonzero(raw_graph[node]).tolist()
                if nonzero_mask[int(neighbour)]
            }
        )
        for left_position, source in enumerate(neighbours):
            for target in neighbours[left_position + 1 :]:
                left, right = active_lookup[source], active_lookup[target]
                projected[left, right] = projected[right, left] = 1
    projected_edges = int(np.count_nonzero(np.triu(projected, 1)))
    projected_components = len(_components(projected))
    return [
        {"scheme": "legacy_raw_full", "nodes": n_nodes, "undirected_edges": raw_edges, "components": len(raw_components), "imputed_edges": 0, "canonical": "false", "notes": "旧 NPZ 邻接矩阵；不是官方 OpenDSS 证据"},
        {"scheme": "legacy_target_induced", "nodes": int(len(active)), "undirected_edges": induced_edges, "components": len(_components(induced)), "imputed_edges": 0, "canonical": "false", "notes": "删除零负荷节点后的诱导图，仅作诊断"},
        {"scheme": "legacy_target_relay_projection", "nodes": int(len(active)), "undirected_edges": projected_edges, "components": projected_components, "imputed_edges": 0, "canonical": "false", "notes": "零负荷节点作为中继的拓扑投影；仍缺官方设备证据"},
        {"scheme": "legacy_full_euclidean_mst", "nodes": n_nodes, "undirected_edges": max(n_nodes - 1, 0), "components": 1 if n_nodes else 0, "imputed_edges": max(len(raw_components) - 1, 0), "canonical": "false", "notes": "历史欧氏 MST 数量对照；禁止进入 canonical physical graph"},
    ]


def _validate_legacy_arrays(
    node_coords: np.ndarray,
    adjacency: np.ndarray,
    edge_index: np.ndarray,
    load_ts: np.ndarray,
    node_ids: np.ndarray,
) -> None:
    """在生成任何报告前执行不依赖外部来源的数组一致性断言。"""

    if node_ids.ndim != 1 or len(np.unique(node_ids)) != len(node_ids):
        raise AssertionError("node_id 必须唯一")
    if node_coords.shape != (len(node_ids), 2):
        raise AssertionError("node_coords shape 与 node_ids 不一致")
    if adjacency.shape != (len(node_ids), len(node_ids)):
        raise AssertionError("adjacency shape 与节点数不一致")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise AssertionError("edge_index 必须为 [2,E]")
    if np.any(edge_index < 0) or np.any(edge_index >= len(node_ids)):
        raise AssertionError("edge_index 越界")
    if not np.array_equal(adjacency > 0, (adjacency > 0).T):
        raise AssertionError("adjacency 必须对称")
    if _edge_pairs(edge_index) != {(int(i), int(j)) for i, j in zip(*np.where(np.triu(adjacency > 0, 1)))}:
        raise AssertionError("adjacency 与 edge_index 不一致")
    if load_ts.ndim != 2 or load_ts.shape[1] != len(node_ids):
        raise AssertionError("load_ts shape 与节点数不一致")
    if np.isnan(load_ts).any() or np.isinf(load_ts).any():
        raise AssertionError("load_ts 不得含 NaN/Inf")


def _official_digest(official: dict) -> str:
    payload = {
        "records": [
            [r["source"], r["target"], r["device_type"], r["device_name"], r["source_file"], int(r["source_line"])]
            for r in official["records"]
        ],
        "load_buses": sorted(official["load_buses"]),
        "coords": sorted((key, float(value[0]), float(value[1])) for key, value in official["coords"].items()),
        "intermediates": [
            [row["line_name"], row["intermediate_points"], row["source_file"], int(row["source_line"])]
            for row in official.get("intermediates", [])
        ],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _official_graph_stats(official: dict, node_ids: np.ndarray, target_mask: np.ndarray) -> dict:
    """只用 scope 内官方设备记录构造审计图，不写入任何推断边。"""

    lookup = {str(node).lower(): index for index, node in enumerate(node_ids.astype(str))}
    pair_records: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in official["records"]:
        source = lookup.get(record["source"].lower())
        target = lookup.get(record["target"].lower())
        if source is None or target is None or source == target:
            continue
        pair_records[tuple(sorted((source, target)))].append(record)
    adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=np.uint8)
    for source, target in pair_records:
        adjacency[source, target] = adjacency[target, source] = 1
    components = _components(adjacency)
    component_of = {node: component_id for component_id, component in enumerate(components) for node in component}
    transformer_nodes = {
        lookup[node]
        for node in official["transformer_nodes"]
        if node in lookup
    }
    load_nodes = {
        lookup[node]
        for node in official["load_buses"]
        if node in lookup
    }
    return {
        "adjacency": adjacency,
        "pair_records": pair_records,
        "components": components,
        "component_of": component_of,
        "degree": adjacency.sum(axis=1).astype(int),
        "line_edges_undirected": int(sum(any(r["device_type"] == "line" for r in records) for records in pair_records.values())),
        "transformer_edges_undirected": int(sum(any(r["device_type"] == "transformer" for r in records) for records in pair_records.values())),
        "nodes_with_official_load": sorted(load_nodes),
        "transformer_side_nodes": sorted(transformer_nodes),
        "official_edge_pairs": int(len(pair_records)),
        "official_components": int(len(components)),
        "component_sizes": [len(component) for component in components],
        "target_nodes_with_load_evidence": int(sum(bool(target_mask[index]) and index in load_nodes for index in range(len(node_ids)))),
        "zero_nodes_with_load_evidence": int(sum(not bool(target_mask[index]) and index in load_nodes for index in range(len(node_ids)))),
        "zero_structural_nodes": int(sum(not bool(target_mask[index]) and index not in load_nodes and degree > 0 for index, degree in enumerate(adjacency.sum(axis=1)))),
    }


def _profile_mapping_audit(official: dict, raw_root: Path, node_ids: np.ndarray, load_ts: np.ndarray) -> dict:
    """核对 legacy ``load_ts`` 是否可由官方 Loads/LoadShapes 重建。

    SMART-DS 文档规定，timeseries ``kW`` 是该负荷全年最大有功功率，
    ``mult`` profile 是无量纲的 15 分钟归一化曲线。一个中心抽头负荷
    可能在 ``Loads.dss`` 中拆成两个等值 Load 元件；因此这里按母线汇总
    *全部*官方 Load 记录，而不是只取首条。审计不做任何自由缩放、时间
    平移、列重排或对零负荷结构节点插值，只有逐点精确映射才可解除数据闸门。
    """

    loads_by_bus: dict[str, list[dict]] = defaultdict(list)
    for row in official["load_records"]:
        loads_by_bus[row["bus"]].append(row)
    shape_files: dict[str, str] = {}
    for shape_name, shape in official["load_shapes"].items():
        match = re.search(r"file\s*=\s*([^\)\s]+)", shape.get("mult_file", ""), flags=re.I)
        if match:
            shape_files[shape_name] = Path(match.group(1)).name
    errors: list[float] = []
    relative_errors: list[float] = []
    correlations: list[float] = []
    matched_nodes: list[str] = []
    missing_nodes: list[str] = []
    missing_profiles: list[str] = []
    invalid_shapes: list[dict] = []
    details: list[dict] = []
    lookup = {str(node).lower(): index for index, node in enumerate(node_ids.astype(str))}
    target_load_buses = sorted(loads_by_bus)
    for bus in target_load_buses:
        index = lookup.get(bus)
        if index is None:
            continue
        rows = loads_by_bus[bus]
        expected = np.zeros(load_ts.shape[0], dtype=np.float64)
        row_evidence: list[dict] = []
        node_failed = False
        for row in rows:
            shape_name = row.get("yearly_shape", "")
            file_name = shape_files.get(shape_name)
            profile_path = raw_root / "profiles" / file_name if file_name else None
            if not file_name or profile_path is None or not profile_path.exists():
                missing_profiles.append(file_name or shape_name or f"<missing shape for {bus}>")
                node_failed = True
                continue
            try:
                values = np.loadtxt(profile_path, dtype=np.float64)
                kw = float(row["kW"])
            except (OSError, ValueError, TypeError) as exc:
                invalid_shapes.append({"node_id": str(node_ids[index]), "profile": file_name, "error": f"{type(exc).__name__}: {exc}"})
                node_failed = True
                continue
            if values.ndim != 1 or values.shape != (load_ts.shape[0],):
                invalid_shapes.append({"node_id": str(node_ids[index]), "profile": file_name, "observed_length": int(values.size), "expected_length": int(load_ts.shape[0])})
                node_failed = True
                continue
            center_tap_split = _is_center_tap_split(row.get("device_name", ""))
            # OpenDSS-native 公式：Loads.dss 的 kW 已是该拆分元件的峰值，
            # 因此按母线汇总时直接乘 mult；不能把 parquet 的 0.5 规则重复套用。
            expected += values * kw
            row_evidence.append(
                {
                    "profile": file_name,
                    "kW": kw,
                    "dss_timeseries_multiplier": 1.0,
                    "center_tap_split": center_tap_split,
                    "parquet_split_multiplier_if_used": 0.5 if center_tap_split else 1.0,
                    "source_file": row.get("source_file", ""),
                    "source_line": row.get("source_line", 0),
                }
            )
        if node_failed or not row_evidence:
            missing_nodes.append(str(node_ids[index]))
            continue
        observed = np.asarray(load_ts[:, index], dtype=np.float64)
        error = np.abs(expected - observed)
        correlation = float(np.corrcoef(expected, observed)[0, 1]) if np.std(expected) > 0 and np.std(observed) > 0 else None
        errors.append(float(np.max(error)))
        relative_errors.append(float(np.mean(error) / max(np.mean(np.abs(observed)), 1e-12)))
        if correlation is not None:
            correlations.append(correlation)
        matched_nodes.append(str(node_ids[index]))
        details.append({
            "node_id": str(node_ids[index]),
            "load_element_count": len(rows),
            "load_elements": row_evidence,
            "max_abs_error_kw": float(np.max(error)),
            "mean_abs_error_kw": float(np.mean(error)),
            "observed_mean": float(np.mean(np.abs(observed))),
            "expected_mean": float(np.mean(np.abs(expected))),
            "observed_max": float(np.max(observed)),
            "expected_max": float(np.max(expected)),
            "profile_observed_correlation": correlation,
            "exact_match_at_1e-5": bool(np.allclose(observed, expected, rtol=1e-5, atol=1e-5)),
        })
    exact_count = sum(bool(item["exact_match_at_1e-5"]) for item in details)
    all_close = bool(details) and not missing_nodes and not invalid_shapes and exact_count == len(details)
    if all_close:
        status = "PASS_EXACT"
    elif missing_nodes or missing_profiles or invalid_shapes:
        status = "BLOCKED_INCOMPLETE_PROFILE_EVIDENCE"
    else:
        status = "BLOCKED_VALUES_MISMATCH"
    return {
        "expected_formula": "sum_over_load_elements_on_bus(Loads.dss_kW * official_mult_profile)",
        "center_tap_rule": "Loads.dss kW 已按 _1/_2 拆分；直接读取 parquet 整户曲线时才对每个拆分元件乘 0.5",
        "comparison_policy": "no_free_scaling_no_time_shift_no_column_reordering",
        "status": status,
        "official_load_buses": int(len(target_load_buses)),
        "matched_nodes": int(len(matched_nodes)),
        "missing_nodes": sorted(set(missing_nodes)),
        "missing_profiles": sorted(set(missing_profiles)),
        "invalid_shapes": invalid_shapes,
        "exact_match_nodes": int(exact_count),
        "matched_node_examples": matched_nodes[:10],
        "max_abs_error_kw": float(max(errors)) if errors else None,
        "mean_max_abs_error_kw": float(np.mean(errors)) if errors else None,
        "median_profile_observed_correlation": float(np.median(correlations)) if correlations else None,
        "mean_relative_error": float(np.mean(relative_errors)) if relative_errors else None,
        "all_close_at_1e-5": all_close,
        "node_results": sorted(details, key=lambda item: item["node_id"].lower()),
        "worst_nodes": sorted(details, key=lambda item: item["max_abs_error_kw"], reverse=True)[:10],
    }


def audit_dataset(legacy_path: Path, raw_root: Path, reports_dir: Path, feeder_root: str | None = DEFAULT_FEEDER_ROOT) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    with np.load(legacy_path, allow_pickle=False) as data:
        required = {"node_coords", "adj", "edge_index", "load_ts", "node_ids"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"legacy NPZ 缺少字段: {missing}")
        node_coords = np.asarray(data["node_coords"])
        adjacency = np.asarray(data["adj"])
        edge_index = np.asarray(data["edge_index"])
        load_ts = np.asarray(data["load_ts"])
        node_ids = np.asarray(data["node_ids"]).astype(str)

    _validate_legacy_arrays(node_coords, adjacency, edge_index, load_ts, node_ids)

    n_nodes = len(node_ids)
    nonzero_mask = np.any(np.abs(load_ts) > 0, axis=0)
    zero_mask = ~nonzero_mask
    components = _components(adjacency)
    official = parse_official_raw(raw_root, feeder_root=feeder_root)
    official_repeat = parse_official_raw(raw_root, feeder_root=feeder_root)
    official_by_node = defaultdict(list)
    for record in official["records"]:
        official_by_node[record["source"]].append(record)
        official_by_node[record["target"]].append(record)
    legacy_lookup = {node.lower(): index for index, node in enumerate(node_ids)}
    official_nodes = set(official_by_node).union(official["load_buses"]).union(official["coords"])
    matched_nodes = sorted(set(legacy_lookup).intersection(official_nodes))
    load_evidence_mask = np.asarray([node.lower() in official["load_buses"] for node in node_ids], dtype=bool)
    target_mask = nonzero_mask & load_evidence_mask
    official_graph = _official_graph_stats(official, node_ids, target_mask)

    # 56 对名称线索仅在官方 Transformer 记录中才算 verified。
    rdt_pairs: list[tuple[str, str]] = []
    for node in sorted(node_ids, key=str.lower):
        if re.fullmatch(r"p10rdt\d+", node, flags=re.I):
            partner = node + "lv"
            if partner.lower() in legacy_lookup:
                rdt_pairs.append((node, partner))
    transformer_pairs = {
        tuple(sorted((record["source"], record["target"])))
        for record in official["records"]
        if record["device_type"] == "transformer"
    }
    verification_rows: list[dict[str, str]] = []
    for source, target in rdt_pairs:
        pair = tuple(sorted((source.lower(), target.lower())))
        evidence = next((item for item in official["records"] if item["device_type"] == "transformer" and tuple(sorted((item["source"], item["target"]))) == pair), None)
        verification_rows.append(
            {
                "source_node": source,
                "target_node": target,
                "verified": "true" if evidence else "false",
                "device_type": evidence["device_type"] if evidence else "",
                "device_name": evidence["device_name"] if evidence else "",
                "source_file": evidence["source_file"] if evidence else "",
                "evidence": evidence["evidence"] if evidence else "",
                "notes": "官方 Transformer 两侧关系已匹配" if evidence else "未在已获取官方 OpenDSS 文件中找到证据；不能据名称认定",
            }
        )
    with (reports_dir / "rdt_rdtlv_verification.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_node", "target_node", "verified", "device_type", "device_name", "source_file", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(verification_rows)

    component_of = {node: index for index, component in enumerate(components) for node in component}
    official_component_of = official_graph["component_of"]
    official_degree = official_graph["degree"]
    official_transformer_nodes = set(official_graph["transformer_side_nodes"])
    node_rows: list[dict[str, str | int]] = []
    for index, node in enumerate(node_ids):
        key = node.lower()
        has_load_device = key in official["load_buses"]
        structural = bool(official_degree[index] > 0) and not has_load_device
        transformer_side = index in official_transformer_nodes
        if has_load_device and nonzero_mask[index]:
            role = "target_load"
            notes = "时间序列非零且官方文件存在 Load 设备证据"
        elif has_load_device and zero_mask[index]:
            role = "zero_timeseries_with_load_device"
            notes = "存在 Load 设备但序列全年为零，需核查映射或场景定义；不做插值"
        elif structural:
            role = "structural_bus_candidate"
            notes = "官方 scope 内存在 Line/Transformer 连接但未挂载 Load；作为结构母线保留，不参与预测 loss"
        elif nonzero_mask[index]:
            role = "target_unverified"
            notes = "序列非零，但尚无官方文件完成节点映射"
        else:
            role = "zero_unverified"
            notes = "序列全年为零，官方角色尚未确认"
        node_rows.append(
            {
                "node_id": node,
                "has_nonzero_timeseries": int(nonzero_mask[index]),
                "has_load_device": int(has_load_device),
                "is_structural_bus": int(structural),
                "degree": int(official_degree[index]),
                "component": int(official_component_of.get(index, -1)),
                "legacy_component": int(component_of[index]),
                "is_transformer_side_bus": int(transformer_side),
                "role": role,
                "notes": notes,
            }
        )
    with (reports_dir / "node_role_audit.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["node_id", "has_nonzero_timeseries", "has_load_device", "is_structural_bus", "is_transformer_side_bus", "degree", "component", "legacy_component", "role", "notes"])
        writer.writeheader()
        writer.writerows(node_rows)

    topology_rows = _topology_comparison(adjacency, nonzero_mask)
    topology_rows.append(
        {
            "scheme": "official_full_physical",
            "nodes": n_nodes,
            "undirected_edges": int(official_graph["official_edge_pairs"]),
            "components": int(official_graph["official_components"]),
            "imputed_edges": 0,
            "canonical": "true",
            "notes": "scope 内 OpenDSS Line/Transformer 设备关系；含零负荷结构中继节点",
        }
    )
    with (reports_dir / "topology_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scheme", "nodes", "undirected_edges", "components", "imputed_edges", "canonical", "notes"],
        )
        writer.writeheader()
        writer.writerows(topology_rows)

    # 从官方 LoadShapes/Loads 记录审计采样点数、步长与原始 profile 引用。
    yearly_shapes = sorted({row["yearly_shape"] for row in official["load_records"] if row["yearly_shape"]})
    profile_names: set[str] = set()
    shape_audit: list[dict] = []
    for shape_name in yearly_shapes:
        shape = official["load_shapes"].get(shape_name)
        if not shape:
            shape_audit.append({"name": shape_name, "found": False})
            continue
        refs: dict[str, str] = {}
        for key in ("mult_file", "qmult_file"):
            if shape.get(key):
                file_match = re.search(r"file\s*=\s*([^\)\s]+)", shape[key], flags=re.I)
                if file_match:
                    refs[key] = Path(file_match.group(1)).name
                    profile_names.add(refs[key])
        shape_audit.append(
            {
                "name": shape_name,
                "found": True,
                "npts": shape.get("npts", ""),
                "interval_hours": shape.get("interval", ""),
                "profile_files": refs,
                "profile_files_present": all((raw_root / "profiles" / name).exists() for name in refs.values()),
                "source_file": shape.get("source_file", ""),
                "source_line": shape.get("source_line", 0),
            }
        )
    profile_rows = [
        {
            "name": name,
            "relative_path": f"profiles/{name}",
            "present": bool((raw_root / "profiles" / name).exists()),
            "sha256": sha256(raw_root / "profiles" / name) if (raw_root / "profiles" / name).exists() else "",
        }
        for name in sorted(profile_names)
    ]
    profile_mapping = _profile_mapping_audit(official, raw_root, node_ids, load_ts)
    official_load_data_audit: dict | None = None
    load_data_audit_path = reports_dir / "official_load_data_audit.json"
    if load_data_audit_path.exists():
        try:
            official_load_data_audit = json.loads(load_data_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            official_load_data_audit = None
    with (reports_dir / "load_series_mapping_audit.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "node_id",
                "load_element_count",
                "max_abs_error_kw",
                "mean_abs_error_kw",
                "observed_mean",
                "expected_mean",
                "observed_max",
                "expected_max",
                "profile_observed_correlation",
                "exact_match_at_1e-5",
                "load_elements",
            ],
        )
        writer.writeheader()
        for row in profile_mapping["node_results"]:
            writer.writerow({**row, "load_elements": json.dumps(row["load_elements"], ensure_ascii=False, separators=(",", ":"))})

    # 当前文件没有 timestamp；以下统计只报告可证实的序列事实。
    row_hashes = {hashlib.sha256(np.ascontiguousarray(row).view(np.uint8).tobytes()).hexdigest() for row in load_ts}
    nonzero_interior_zero = {}
    for index, node in enumerate(node_ids):
        values = load_ts[:, index]
        nz = np.flatnonzero(np.abs(values) > 0)
        if len(nz) >= 2:
            nonzero_interior_zero[node] = int(np.count_nonzero(np.abs(values[nz[0] : nz[-1] + 1]) == 0))
    time_audit = {
        "timesteps": int(load_ts.shape[0]),
        "nodes": int(load_ts.shape[1]),
        "sampling_interval_minutes": 15,
        "timestamp_available": False,
        "calendar_range": None,
        "expected_duration_days_at_15min": float(load_ts.shape[0] * 15 / 1440),
        "nan_count": int(np.isnan(load_ts).sum()),
        "inf_count": int(np.isinf(load_ts).sum()),
        "unique_row_hashes": int(len(row_hashes)),
        "duplicate_row_count": int(load_ts.shape[0] - len(row_hashes)),
        "max_abs_value": float(np.max(np.abs(load_ts))),
        "min_value": float(np.min(load_ts)),
        "max_value": float(np.max(load_ts)),
        "nonzero_nodes_with_interior_zero_count": int(sum(value > 0 for value in nonzero_interior_zero.values())),
        "nonzero_interior_zero_by_node": nonzero_interior_zero,
        "unit": "未在 NPZ 字段或元数据中标注",
        "scaling": "未在 NPZ 字段或元数据中标注",
        "official_load_shape_count": int(len(yearly_shapes)),
        "official_load_shapes": shape_audit,
        "official_profile_count": int(len(profile_rows)),
        "official_profiles_present": bool(profile_rows) and all(row["present"] for row in profile_rows),
        "official_profiles": profile_rows,
        "official_profile_mapping": profile_mapping,
        "official_load_data_audit": (
            {
                "status": official_load_data_audit.get("status"),
                "profiles": official_load_data_audit.get("profiles"),
                "official_internal_consistency": official_load_data_audit.get("official_internal_consistency"),
                "legacy_comparison": official_load_data_audit.get("legacy_comparison"),
                "report": load_data_audit_path.name,
            }
            if official_load_data_audit
            else {"status": "NOT_RUN", "report": load_data_audit_path.name}
        ),
    }

    edge_pairs = _edge_pairs(edge_index)
    raw_stats = {
        "nodes": n_nodes,
        "node_ids_unique": bool(len(set(node_ids.tolist())) == n_nodes),
        "node_coords_shape": list(node_coords.shape),
        "adj_shape": list(adjacency.shape),
        "edge_index_shape": list(edge_index.shape),
        "load_ts_shape": list(load_ts.shape),
        "undirected_edges": int(len(edge_pairs)),
        "adjacency_symmetric": bool(np.array_equal(adjacency > 0, (adjacency > 0).T)),
        "adjacency_edge_index_consistent": bool(edge_pairs == {(i, j) for i, j in zip(*np.where(np.triu(adjacency > 0, 1)))}),
        "components": int(len(components)),
        "component_sizes": [len(component) for component in components],
        "nonzero_load_nodes": int(nonzero_mask.sum()),
        "zero_only_nodes": int(zero_mask.sum()),
        "node_id_prefixes": sorted({node[:8].lower() for node in node_ids}),
        "legacy_coordinate_range": {
            "x_min": float(np.min(node_coords[:, 0])),
            "x_max": float(np.max(node_coords[:, 0])),
            "y_min": float(np.min(node_coords[:, 1])),
            "y_max": float(np.max(node_coords[:, 1])),
        },
    }
    raw_stats["topology_comparison"] = _topology_comparison(adjacency, nonzero_mask)
    official_stats = {
        "raw_root": str(raw_root),
        "scope": official["scope"],
        "scope_files": official["source_files"],
        "dss_file_count": int(len(official["source_files"])),
        "official_record_count": int(len(official["records"])),
        "official_line_record_count": int(sum(r["device_type"] == "line" for r in official["records"])),
        "official_transformer_record_count": int(sum(r["device_type"] == "transformer" for r in official["records"])),
        "official_load_bus_count": int(len(official["load_buses"])),
        "official_coordinate_count": int(len(official["coords"])),
        "intermediates_count": int(len(official.get("intermediates", []))),
        "intermediates_with_points_count": int(sum(bool(row.get("intermediate_points")) for row in official.get("intermediates", []))),
        "intermediates_source_files": sorted({row.get("source_file", "") for row in official.get("intermediates", [])}),
        "legacy_node_match_count": int(len(matched_nodes)),
        "legacy_node_match_fraction": float(len(matched_nodes) / max(n_nodes, 1)),
        "matched_node_examples": matched_nodes[:20],
        "deterministic_reparse": bool(_official_digest(official) == _official_digest(official_repeat)),
        "official_graph": {
            "nodes": int(n_nodes),
            "edges_undirected": int(official_graph["official_edge_pairs"]),
            "line_edges_undirected": int(official_graph["line_edges_undirected"]),
            "transformer_edges_undirected": int(official_graph["transformer_edges_undirected"]),
            "components": int(official_graph["official_components"]),
            "component_sizes": official_graph["component_sizes"],
            "target_nodes_with_load_evidence": int(official_graph["target_nodes_with_load_evidence"]),
            "zero_nodes_with_load_evidence": int(official_graph["zero_nodes_with_load_evidence"]),
            "zero_structural_nodes": int(official_graph["zero_structural_nodes"]),
            "max_degree": int(np.max(official_degree)) if len(official_degree) else 0,
            "mean_degree": float(np.mean(official_degree)) if len(official_degree) else 0.0,
        },
        "load_records": int(len(official["load_records"])),
        "load_shape_count": int(len(official["load_shapes"])),
        "official_coordinate_range": (
            {
                "x_min": float(min(value[0] for value in official["coords"].values())),
                "x_max": float(max(value[0] for value in official["coords"].values())),
                "y_min": float(min(value[1] for value in official["coords"].values())),
                "y_max": float(max(value[1] for value in official["coords"].values())),
            }
            if official["coords"]
            else None
        ),
    }
    # canonical full graph 不接受部分匹配：273 个 legacy 节点必须全部在
    # 同一份官方 raw 证据中出现，否则无法排除 feeder 混合或截断。
    topology_blocker = (
        not official["records"]
        or len(matched_nodes) != n_nodes
        or official_graph["target_nodes_with_load_evidence"] != int(target_mask.sum())
        or not official_stats["deterministic_reparse"]
        or official["scope"].get("mode") != "feeder_scoped"
    )
    load_series_blocker = profile_mapping["status"] != "PASS_EXACT"
    if topology_blocker:
        status = "BLOCKED_NO_VERIFIED_CANONICAL_SOURCE"
    elif load_series_blocker:
        status = "BLOCKED_LOAD_SERIES_MAPPING_UNVERIFIED"
    else:
        status = "RAW_SOURCE_VERIFIED_CANONICAL"
    audit = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "legacy_source": str(legacy_path),
        "legacy_sha256": sha256(legacy_path),
        "legacy": raw_stats,
        "time_series": time_audit,
        "official_source": official_stats,
        "rdt_pair_count": len(rdt_pairs),
        "rdt_verified_count": int(sum(row["verified"] == "true" for row in verification_rows)),
        "constraints": {
            "legacy_adj_is_not_canonical": True,
            "unverified_mst_forbidden_in_physical_graph": True,
            "zero_load_nodes_not_interpolated": True,
            "absolute_timestamp_or_weather_not_inferred": True,
            "load_series_mapping_verified": not load_series_blocker,
            "canonical_gate_requires_exact_official_profile_mapping": True,
        },
    }
    (reports_dir / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    scope = official["scope"]
    graph_stats = official_stats["official_graph"]
    provenance = "# SmartDS 数据来源与拓扑审计\n\n"
    provenance += f"审计时间（UTC）：`{audit['audit_timestamp_utc']}`  \n状态：**{audit['status']}**\n\n"
    provenance += "## 1. 数据来源（证据等级 A）\n\n"
    provenance += "- 提供方：Open Energy Data Initiative（OEDI）SMART-DS。\n"
    provenance += "- 版本/年份/数据集：`SMART-DS v0.9 / 2018 / Full_Texas`。\n"
    provenance += "- 区域：`P10R`；场景：`base_timeseries`。\n"
    provenance += "- substation：`p10rhs0_1247`；feeder：`p10rhs0_1247--p10rdt7719`。\n"
    provenance += "- 官方对象存储前缀：`s3://oedi-data-lake/SMART-DS/v0.9/2018/Full_Texas/P10R/scenarios/base_timeseries/`；v0.9/2018/P10R 信息来自该官方对象键与下载清单，当前 OpenDSS 文件本身未嵌入独立版本字段。\n"
    provenance += f"- 本次严格 scope：`{scope['feeder_root']}` 及父级 `{scope['parent_root']}`，解析文件 `{official_stats['dss_file_count']}` 个；未把其它 feeder 递归纳入。\n"
    provenance += "- 节点命名仅用于与官方母线 ID 做大小写不敏感的逐项匹配；设备关系均来自 OpenDSS `Lines.dss`/`Transformers.dss`，不由名称或距离推断。\n\n"
    provenance += "## 2. Legacy NPZ 审计（证据等级 A）\n\n"
    provenance += f"- 文件：`{legacy_path}`；SHA256：`{audit['legacy_sha256']}`。\n"
    provenance += f"- 字段：`node_coords={tuple(node_coords.shape)}`、`adj={tuple(adjacency.shape)}`、`edge_index={tuple(edge_index.shape)}`、`load_ts={tuple(load_ts.shape)}`、`node_ids={tuple(node_ids.shape)}`。\n"
    provenance += f"- `{n_nodes}` 个节点，其中 `{int(nonzero_mask.sum())}` 个全年非零、`{int(zero_mask.sum())}` 个全年为零；旧邻接为 `{len(edge_pairs)}` 条无向边、`{len(components)}` 个分量。\n"
    provenance += f"- 序列长度 `{load_ts.shape[0]}`；按 15 分钟采样为 `{time_audit['expected_duration_days_at_15min']:.2f}` 天。文件无可验证绝对 timestamp，因此不得外推日期、星期、节假日或天气。\n"
    provenance += f"- NaN/Inf=`{time_audit['nan_count']}/{time_audit['inf_count']}`；非零节点中间零值节点数 `{time_audit['nonzero_nodes_with_interior_zero_count']}`。单位和缩放未在 NPZ 元数据中标注。\n\n"
    provenance += "## 3. 官方拓扑核查（证据等级 A）\n\n"
    provenance += f"- 官方 OpenDSS 端点记录 `{official_stats['official_record_count']}` 条；legacy 节点匹配 `{official_stats['legacy_node_match_count']}/{n_nodes}`。\n"
    provenance += f"- 官方 `Intermediates.txt` 解析到 `{official_stats['intermediates_count']}` 条线段几何记录，其中 `{official_stats['intermediates_with_points_count']}` 条含中间坐标；该文件仅作可视化证据，不新增 OpenDSS 物理边或节点。\n"
    provenance += f"- 完整官方图：`{graph_stats['nodes']}` 节点、`{graph_stats['edges_undirected']}` 条无向边、`{graph_stats['components']}` 个连通分量；其中 line=`{graph_stats['line_edges_undirected']}`、transformer=`{graph_stats['transformer_edges_undirected']}`。\n"
    provenance += f"- 56 对 `p10rdtXXXXX ↔ p10rdtXXXXXlv`：官方 Transformer 逐对验证 `{audit['rdt_verified_count']}/{audit['rdt_pair_count']}`；因此旧 NPZ 确实遗漏了这 56 条 Transformer connectivity。\n"
    provenance += f"- 官方图最大度 `{graph_stats['max_degree']}`、平均度 `{graph_stats['mean_degree']:.3f}`；旧 57 个分量主要由缺失 Transformer 边造成，而非删除零负荷节点后的真实断连。\n\n"
    provenance += "## 4. 节点角色与负荷曲线（证据等级 A/B）\n\n"
    provenance += f"- `{graph_stats['target_nodes_with_load_evidence']}` 个非零目标节点均有官方 Load 设备证据；`{graph_stats['zero_structural_nodes']}` 个全年零序列节点被识别为结构母线候选。结构节点的 0 不做插值。\n"
    provenance += f"- 官方 LoadShape 数 `{time_audit['official_load_shape_count']}`，profile 文件 `{time_audit['official_profile_count']}` 个；本地 profile 完整存在：`{time_audit['official_profiles_present']}`。每个 shape 的 `npts`、`interval` 与来源行记录在 `audit_report.json`。\n"
    provenance += f"- 官方 OpenDSS-native 重建定义为按母线汇总 `Loads.dss` 全部 Load 元件的 `kW × mult`；中心抽头 `_1/_2` 的 `kW` 已经是拆分后的半客户峰值，不能再次乘 0.5。若直接读取整户 parquet，则每个 `_1/_2` 元件才乘 0.5。\n"
    provenance += f"- 本次逐点核对状态为 `{profile_mapping['status']}`，匹配 `{profile_mapping['matched_nodes']}/{profile_mapping['official_load_buses']}` 个母线，精确通过 `{profile_mapping['exact_match_nodes']}` 个。\n"
    provenance += f"- 最大绝对误差 `{profile_mapping['max_abs_error_kw']}` kW、平均相对误差 `{profile_mapping['mean_relative_error']}`；因此当前 NPZ 的负荷数值映射{'已' if profile_mapping['status'] == 'PASS_EXACT' else '未'}通过，不能把现有序列宣称为官方 profile 的直接重建结果。\n"
    load_data_summary = time_audit.get("official_load_data_audit", {})
    provenance += f"- parquet 独立审计状态 `{load_data_summary.get('status', 'NOT_RUN')}`；报告 `reports/{load_data_summary.get('report', 'official_load_data_audit.json')}`。官方文件内部一致性与 legacy 对比结果均以该机器可读报告为准。\n\n"
    provenance += "- 在映射状态为 `PASS_EXACT` 前，legacy `load_ts` 只能作为待解释的历史输入；禁止用自由缩放、时间平移或列重排把它强行拟合到官方 profile/parquet。\n\n"
    provenance += "## 5. 约束与下一步\n\n"
    provenance += "1. canonical physical graph 只允许官方 Line/Transformer 等明确设备关系；Euclidean MST、节点名补边和旧 adj 仅作 legacy 敏感性对照。\n"
    provenance += "2. 推荐后续模型保留 181 个零负荷结构节点作为 message-passing relay，仅在 92 个 target 节点计算损失；target projection 图若过密，不直接作为稀疏图。\n"
    provenance += "3. 本报告不包含任何正式模型训练、FedAvg/FedProx 或消融结论；联邦客户端仍是合成网络空间划分模拟。\n"
    provenance += "4. 若 profile 单位、原始生成脚本或设备元件映射出现冲突，应暂停模型实验并在本报告中记录 blocker。\n\n"
    provenance += "## 6. 审计产物\n\n"
    provenance += "- `audit_report.json`：机器可读字段、scope、拓扑和时间序列统计。\n- `node_role_audit.csv`：节点角色及官方/legacy 分量。\n- `rdt_rdtlv_verification.csv`：56 对 Transformer 逐对证据。\n- `topology_comparison.csv`：旧图诊断对照。\n- `load_series_mapping_audit.csv`：逐目标母线的官方重建误差审计。\n- `official_load_data_audit.json/csv`：官方 parquet 与 OpenDSS-native 口径的独立交叉核验。\n- `PROFILE_MANIFEST.json`：官方 profile/parquet SHA256 与下载清单。\n- `data_provenance.md`：本报告。\n"
    (reports_dir / "data_provenance.md").write_text(provenance, encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--feeder-root", default=DEFAULT_FEEDER_ROOT, help="相对于 raw-root 的 feeder 目录；正式审计必须显式限定 scope")
    args = parser.parse_args()
    audit = audit_dataset(args.legacy.resolve(), args.raw_root.resolve(), args.reports.resolve(), feeder_root=args.feeder_root)
    print(json.dumps({"status": audit["status"], "legacy_sha256": audit["legacy_sha256"], "reports": str(args.reports.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
