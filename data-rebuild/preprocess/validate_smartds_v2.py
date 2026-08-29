#!/usr/bin/env python3
"""SmartDS v2 处理结果的独立一致性与来源审计。

该脚本只读取 processed/raw 文件，不训练模型。任何一项失败都会以非零
状态退出，避免把未通过数据闸门的图送入 PA-STFed。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from audit_smartds_data import DEFAULT_FEEDER_ROOT, parse_official_raw


def _pairs(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(source), int(target)))) for source, target in edge_index.T.tolist() if source != target}


def validate(processed: Path, raw_root: Path, feeder_root: str, report_path: Path | None = None) -> dict:
    metadata_path = processed / "smartds_metadata_v2.json"
    full_path = processed / "smartds_full_graph_v2.npz"
    target_path = processed / "smartds_target_graph_v2.npz"
    if not metadata_path.exists():
        result = {
            "status": "FAIL",
            "reason": "processed 目录缺少 smartds_metadata_v2.json；未通过数据闸门",
            "checks": {"metadata_exists": False, "canonical_gate": False},
            "processed": str(processed.resolve()),
            "raw_root": str(raw_root.resolve()),
            "feeder_root": feeder_root,
        }
        if report_path:
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # 两种 canonical 来源都允许通过闸门：
    # 1) 旧流程已证明 legacy load_ts 与官方 profile 逐点一致；
    # 2) 新流程直接由官方 OpenDSS-native 公式生成 load_ts。
    # 第二种不要求 legacy 序列匹配，但必须显式记录数据源和独立校验状态。
    legacy_series_gate = bool(metadata.get("profile_mapping_verified"))
    official_series_gate = metadata.get("canonical_dataset_source") == "official_opendss_native" and bool(metadata.get("official_series_verified"))
    gate = metadata.get("status") == "RAW_SOURCE_VERIFIED_CANONICAL" and bool(metadata.get("canonical_topology")) and (legacy_series_gate or official_series_gate)
    if not gate:
        unquarantined = [
            name
            for name in ("smartds_full_graph_v2.npz", "smartds_target_graph_v2.npz")
            if (processed / name).exists()
        ]
        result = {
            "status": "FAIL",
            "reason": f"canonical 闸门未通过: {metadata.get('status')}; profile_mapping_status={metadata.get('profile_mapping_status')}",
            "checks": {
                "metadata_exists": True,
                "canonical_gate": False,
                "profile_mapping_verified": bool(metadata.get("profile_mapping_verified")),
                "official_series_verified": bool(metadata.get("official_series_verified")),
                "no_unquarantined_provisional_artifacts": not bool(unquarantined),
            },
            "unquarantined_provisional_artifacts": unquarantined,
            "processed": str(processed.resolve()),
            "raw_root": str(raw_root.resolve()),
            "feeder_root": feeder_root,
        }
        if report_path:
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    if not full_path.exists() or not target_path.exists():
        result = {
            "status": "FAIL",
            "reason": "canonical metadata 已通过但 processed v2 NPZ 缺失",
            "checks": {
                "metadata_exists": True,
                "canonical_gate": True,
                "full_npz_exists": full_path.exists(),
                "target_npz_exists": target_path.exists(),
            },
            "processed": str(processed.resolve()),
            "raw_root": str(raw_root.resolve()),
            "feeder_root": feeder_root,
        }
        if report_path:
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    with np.load(full_path, allow_pickle=False) as full_data, np.load(target_path, allow_pickle=False) as target_data:
        full = {name: np.asarray(full_data[name]) for name in full_data.files}
        target = {name: np.asarray(target_data[name]) for name in target_data.files}

    checks: dict[str, bool] = {}
    node_ids = full["node_ids"].astype(str)
    checks["node_ids_unique"] = len(np.unique(node_ids)) == len(node_ids)
    checks["edge_index_in_bounds"] = bool(np.all(full["edge_index"] >= 0) and np.all(full["edge_index"] < len(node_ids)))
    checks["adjacency_symmetric"] = bool(np.array_equal(full["adj"] > 0, (full["adj"] > 0).T))
    checks["adjacency_edge_index_consistent"] = _pairs(full["edge_index"]) == {
        (int(source), int(target)) for source, target in zip(*np.where(np.triu(full["adj"] > 0, 1)))
    }
    edge_fields = ("edge_type", "edge_source", "edge_attr_json", "edge_length", "edge_length_units", "edge_phases", "edge_enabled", "edge_switch")
    if "edge_intermediates_json" in full:
        edge_fields = edge_fields + ("edge_intermediates_json",)
    checks["edge_attributes_aligned"] = all(full[name].shape[0] == full["edge_index"].shape[1] for name in edge_fields)
    checks["edge_types_are_official"] = set(full["edge_type"].astype(str)).issubset({"line", "transformer"})
    checks["mst_not_present"] = not any("mst" in str(value).lower() for value in full["edge_source"].astype(str))
    checks["target_load_shape"] = target["target_load_ts"].ndim == 2 and target["target_load_ts"].shape[1] == target["target_node_ids"].shape[0]
    full_load = full.get("load_ts")
    checks["full_load_shape"] = full_load is None or (full_load.ndim == 2 and full_load.shape[0] == target["target_load_ts"].shape[0] and full_load.shape[1] == len(node_ids))
    checks["target_load_finite"] = bool(np.isfinite(target["target_load_ts"]).all())
    # 仅有官方 Load 元件证据还不够：每个 target 必须确实拥有至少一个
    # 非零时间点，避免把全年零序列结构母线误纳入预测目标。
    checks["targets_have_nonzero_series"] = bool(
        target["target_load_ts"].ndim == 2
        and target["target_load_ts"].shape[1] > 0
        and np.all(np.any(np.abs(target["target_load_ts"]) > 0, axis=0))
    )
    checks["full_load_finite"] = full_load is None or bool(np.isfinite(full_load).all())
    checks["metadata_timesteps_match"] = int(metadata["timesteps"]) == int(target["target_load_ts"].shape[0])
    checks["mapping_reversible"] = bool(np.array_equal(node_ids[target["target_to_full_mapping"]], target["target_node_ids"]))
    checks["full_to_target_inverse"] = bool(all(target["full_to_target_mapping"][index] == position for position, index in enumerate(target["target_to_full_mapping"])))

    official = parse_official_raw(raw_root, feeder_root=feeder_root)
    lookup = {node.lower(): index for index, node in enumerate(node_ids)}
    official_evidence = {
        tuple(sorted((lookup[record["source"]], lookup[record["target"]]))): record
        for record in official["records"]
        if record["source"] in lookup and record["target"] in lookup and record["source"] != record["target"]
    }
    source_evidence_ok = True
    for source, target_index, source_ref in zip(full["edge_index"][0], full["edge_index"][1], full["edge_source"]):
        pair = tuple(sorted((int(source), int(target_index))))
        ref = str(source_ref)
        record = official_evidence.get(pair)
        if record is None or ref != f"{record['source_file']}:{record['source_line']}":
            # 同一端点存在多个设备时允许匹配任一官方证据，但仍要求类型一致。
            candidates = [r for r in official["records"] if tuple(sorted((lookup.get(r["source"], -1), lookup.get(r["target"], -1)))) == pair and ref == f"{r['source_file']}:{r['source_line']}"]
            if not candidates:
                source_evidence_ok = False
                break
    checks["every_edge_has_official_source"] = source_evidence_ok
    target_ids = {node.lower() for node in target["target_node_ids"].astype(str)}
    checks["targets_have_official_load"] = target_ids.issubset(official["load_buses"])
    numeric_values = [full["node_coords"], full["adj"], target["target_load_ts"], target["target_shortest_hops"]]
    if full_load is not None:
        numeric_values.append(full_load)
    checks["no_nan_inf_any_numeric"] = all(np.isfinite(values).all() for values in numeric_values)
    graph = nx.from_numpy_array(full["adj"])
    checks["full_graph_connected_components_metadata"] = nx.number_connected_components(graph) == int(metadata["full_components"])
    checks["projection_density_metadata"] = int(target["topology_projected_edge_index"].shape[1] // 2) == int(metadata["target_projected_edges_undirected"])
    checks["all_checks_passed"] = all(checks.values())
    result = {"status": "PASS" if checks["all_checks_passed"] else "FAIL", "checks": checks, "processed": str(processed.resolve()), "raw_root": str(raw_root.resolve()), "feeder_root": feeder_root}
    if report_path:
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--feeder-root", default=DEFAULT_FEEDER_ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.processed.resolve(), args.raw_root.resolve(), args.feeder_root, args.report.resolve() if args.report else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
