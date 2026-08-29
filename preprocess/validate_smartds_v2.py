#!/usr/bin/env python3
"""校验 canonical SmartDS full-graph 文件，不执行模型训练。

默认检查可提交的 canonical artifact、元数据和 SHA-256。官方 OpenDSS
原始文件通常不进入 Git；若本地仍提供 ``--raw-root``，脚本会额外逐边
复核源文件证据，但不会把“未保留 raw 文件”误判为 artifact 损坏。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_RAW = ROOT / "data" / "raw" / "SMARTDS"
DEFAULT_FEEDER_ROOT = "p10rhs0_1247/p10rhs0_1247--p10rdt7719"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(source), int(target))))
        for source, target in edge_index.T.tolist()
        if source != target
    }


def _raw_scope_available(raw_root: Path, feeder_root: str) -> bool:
    scope = raw_root / feeder_root
    return all((scope / name).is_file() for name in ("Lines.dss", "Transformers.dss"))


def _validate_raw_sources(
    full: dict[str, np.ndarray], raw_root: Path, feeder_root: str
) -> bool:
    # 延迟导入，保证只保留 manifest 的常规工程仍可完成 artifact 校验。
    from audit_smartds_data import parse_official_raw

    official = parse_official_raw(raw_root, feeder_root=feeder_root)
    node_ids = full["node_ids"].astype(str)
    lookup = {node.lower(): index for index, node in enumerate(node_ids)}
    evidence: set[tuple[tuple[int, int], str]] = set()
    for record in official["records"]:
        source = lookup.get(record["source"])
        target = lookup.get(record["target"])
        if source is None or target is None or source == target:
            continue
        pair = tuple(sorted((int(source), int(target))))
        reference = f"{record['source_file']}:{record['source_line']}"
        evidence.add((pair, reference))
    return all(
        (tuple(sorted((int(source), int(target)))), str(reference)) in evidence
        for source, target, reference in zip(
            full["edge_index"][0], full["edge_index"][1], full["edge_source"]
        )
    )


def validate(
    processed: Path,
    raw_root: Path | None = None,
    feeder_root: str = DEFAULT_FEEDER_ROOT,
    report_path: Path | None = None,
) -> dict:
    metadata_path = processed / "smartds_metadata_v2.json"
    full_path = processed / "smartds_full_graph_v2.npz"
    if not metadata_path.is_file() or not full_path.is_file():
        result = {
            "status": "FAIL",
            "checks": {
                "metadata_exists": metadata_path.is_file(),
                "full_npz_exists": full_path.is_file(),
            },
            "processed": str(processed.resolve()),
        }
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return result

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(full_path, allow_pickle=False) as archive:
        full = {name: np.asarray(archive[name]) for name in archive.files}

    required = {
        "node_ids",
        "node_coords",
        "adj",
        "edge_index",
        "edge_type",
        "edge_source",
        "target_mask",
        "load_mask",
        "load_ts",
    }
    checks: dict[str, bool] = {
        "canonical_gate": (
            metadata.get("status") == "RAW_SOURCE_VERIFIED_CANONICAL"
            and bool(metadata.get("canonical_topology"))
            and metadata.get("canonical_dataset_source") == "official_opendss_native"
            and bool(metadata.get("official_series_verified"))
        ),
        "required_fields_present": required.issubset(full),
    }
    if not checks["required_fields_present"]:
        result = {
            "status": "FAIL",
            "checks": checks,
            "missing_fields": sorted(required.difference(full)),
            "processed": str(processed.resolve()),
        }
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return result

    node_ids = full["node_ids"].astype(str)
    node_count = len(node_ids)
    edge_index = full["edge_index"]
    adjacency = full["adj"]
    load_ts = full["load_ts"]
    target_mask = full["target_mask"].astype(bool)
    load_mask = full["load_mask"].astype(bool)
    graph = nx.from_numpy_array(adjacency > 0)
    edge_fields = (
        "edge_type",
        "edge_source",
        "edge_attr_json",
        "edge_length",
        "edge_length_units",
        "edge_phases",
        "edge_enabled",
        "edge_switch",
    )
    if "edge_intermediates_json" in full:
        edge_fields += ("edge_intermediates_json",)

    observed_target_mask = np.any(np.abs(load_ts) > 0, axis=0)
    source_refs = full["edge_source"].astype(str)
    edge_types = full["edge_type"].astype(str)
    checks.update(
        {
            "node_ids_unique": len(np.unique(node_ids)) == node_count,
            "node_count_matches_metadata": node_count == int(metadata["full_nodes"]),
            "node_coords_shape": full["node_coords"].shape == (node_count, 2),
            "edge_index_shape": edge_index.ndim == 2 and edge_index.shape[0] == 2,
            "edge_index_in_bounds": bool(
                np.all(edge_index >= 0) and np.all(edge_index < node_count)
            ),
            "adjacency_shape": adjacency.shape == (node_count, node_count),
            "adjacency_symmetric": bool(
                np.array_equal(adjacency > 0, (adjacency > 0).T)
            ),
            "adjacency_edge_index_consistent": _pairs(edge_index)
            == {
                (int(source), int(target))
                for source, target in zip(
                    *np.where(np.triu(adjacency > 0, k=1))
                )
            },
            "edge_attributes_aligned": all(
                name in full and full[name].shape[0] == edge_index.shape[1]
                for name in edge_fields
            ),
            "edge_types_are_official": set(edge_types).issubset(
                {"line", "transformer"}
            ),
            "edge_type_counts_match_metadata": (
                int(np.count_nonzero(edge_types == "line") // 2)
                == int(metadata["line_edges_undirected"])
                and int(np.count_nonzero(edge_types == "transformer") // 2)
                == int(metadata["transformer_edges_undirected"])
            ),
            "every_edge_has_source_reference": bool(
                len(source_refs) == edge_index.shape[1]
                and np.all(np.char.str_len(source_refs) > 0)
            ),
            "mst_not_present": not any(
                "mst" in value.lower() for value in source_refs
            ),
            "load_shape": load_ts.ndim == 2
            and load_ts.shape == (int(metadata["timesteps"]), node_count),
            "load_finite": bool(np.isfinite(load_ts).all()),
            "target_mask_shape": target_mask.shape == (node_count,),
            "target_mask_matches_nonzero_series": bool(
                np.array_equal(target_mask, observed_target_mask)
            ),
            "target_count_matches_metadata": int(target_mask.sum())
            == int(metadata["target_nodes"]),
            "targets_have_official_load_device": bool(
                np.all(load_mask[target_mask])
            ),
            "structural_nodes_are_zero_load": bool(
                np.all(load_ts[:, ~target_mask] == 0)
            ),
            "full_edge_count_matches_metadata": graph.number_of_edges()
            == int(metadata["full_edges_undirected"]),
            "component_count_matches_metadata": nx.number_connected_components(graph)
            == int(metadata["full_components"]),
            "full_npz_sha256_matches_metadata": _sha256(full_path).lower()
            == str(metadata["full_npz_sha256"]).lower(),
        }
    )

    raw_available = raw_root is not None and _raw_scope_available(raw_root, feeder_root)
    raw_revalidated = (
        _validate_raw_sources(full, raw_root, feeder_root) if raw_available else None
    )
    if raw_revalidated is not None:
        checks["raw_edge_sources_revalidated"] = raw_revalidated

    checks["all_checks_passed"] = all(checks.values())
    result = {
        "status": "PASS" if checks["all_checks_passed"] else "FAIL",
        "checks": checks,
        "raw_source_revalidation": (
            "PASS" if raw_revalidated else "FAIL" if raw_revalidated is False else "SKIPPED_RAW_NOT_PRESENT"
        ),
        "processed": str(processed.resolve()),
        "raw_root": str(raw_root.resolve()) if raw_root is not None else None,
        "feeder_root": feeder_root,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--feeder-root", default=DEFAULT_FEEDER_ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(
        args.processed.resolve(),
        args.raw_root.resolve() if args.raw_root else None,
        args.feeder_root,
        args.report.resolve() if args.report else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
