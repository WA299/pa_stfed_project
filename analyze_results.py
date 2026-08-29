"""PA-STFed 结果汇总与配对时间块 Bootstrap。

该脚本只读取已完成的 ``*_result.json``，不重新训练模型。它使用训练过程中
保存的固定时间块 WAPE 充分统计量，计算两个实验的配对差异均值和置信区间。
负差值表示右侧实验误差更低；若区间跨过 0，只能写“数值较低”，不能写“显著优于”。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return payload


def _validate_block_record(record: dict[str, Any], split: str) -> None:
    """校验 WAPE 充分统计量，防止 NaN、错位或自行归一化结果进入检验。"""

    required = {
        "aggregation_level",
        "origin_sha256",
        "n_windows",
        "block_windows",
        "block_n_windows",
        "error_sum",
        "target_sum",
        "wape",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"{split} WAPE block metadata missing: {missing}")
    arrays = {
        name: np.asarray(record[name], dtype=np.float64)
        for name in ("error_sum", "target_sum", "wape")
    }
    if any(array.ndim != 1 for array in arrays.values()):
        raise ValueError(f"{split} WAPE block arrays must be one-dimensional")
    lengths = {int(array.size) for array in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) < 2:
        raise ValueError(f"{split} WAPE block arrays must have equal length >= 2")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError(f"{split} WAPE block arrays contain NaN or Inf")
    if np.any(arrays["error_sum"] < 0.0) or np.any(arrays["target_sum"] <= 0.0):
        raise ValueError(f"{split} WAPE sums have invalid signs")
    expected = 100.0 * arrays["error_sum"] / arrays["target_sum"]
    if not np.allclose(expected, arrays["wape"], rtol=1e-6, atol=1e-6):
        raise ValueError(f"{split} WAPE values do not match error_sum/target_sum")
    if int(record["n_windows"]) < int(sum(record["block_n_windows"])):
        raise ValueError(f"{split} n_windows is smaller than block coverage")


def _block_record(payload: dict, split: str, aggregation: str) -> dict[str, Any]:
    """读取严格标注的全局 micro 或客户端 macro 区块统计量。"""

    if aggregation == "micro":
        record = payload.get(f"{split}_wape_blocks_micro")
        if record is None:
            # 集中式结果使用无后缀键；新版本仍要求显式 aggregation_level。
            record = payload.get(f"{split}_wape_blocks")
        if not isinstance(record, dict):
            raise KeyError(
                f"{split} micro WAPE blocks not found; rerun the audited v2 pipeline"
            )
        if record.get("aggregation_level") != "micro_global":
            raise ValueError(
                f"{split} record is not micro_global; got {record.get('aggregation_level')!r}"
            )
        _validate_block_record(record, split)
        return record

    if aggregation != "macro":
        raise ValueError("aggregation must be 'micro' or 'macro'")
    clients = payload.get(f"{split}_wape_blocks_clients")
    if not isinstance(clients, list) or not clients:
        raise KeyError(f"{split} client WAPE blocks not found")
    if payload.get("evaluation_metadata", {}).get("client_count") not in {
        None,
        len(clients),
    }:
        raise ValueError(f"{split} client count does not match evaluation metadata")
    for index, item in enumerate(clients):
        if not isinstance(item, dict):
            raise ValueError(f"{split} client {index} block is not an object")
        if item.get("aggregation_level") != "micro_client":
            raise ValueError(f"{split} client {index} lacks micro_client annotation")
        _validate_block_record(item, split)
    alignment_keys = (
        "source_sha256",
        "history",
        "horizon",
        "origin_sha256",
        "first_origin",
        "last_origin",
        "n_windows",
        "block_windows",
        "block_n_windows",
    )
    reference = clients[0]
    for index, item in enumerate(clients[1:], start=1):
        for key in alignment_keys:
            if item.get(key) != reference.get(key):
                raise ValueError(
                    f"{split} client block alignment differs at client {index}, field {key!r}"
                )
    errors = np.mean(
        np.stack([np.asarray(item["error_sum"], dtype=np.float64) for item in clients]),
        axis=0,
    )
    targets = np.mean(
        np.stack([np.asarray(item["target_sum"], dtype=np.float64) for item in clients]),
        axis=0,
    )
    return {
        **{key: reference[key] for key in alignment_keys},
        "aggregation_level": "macro_client",
        "client_count": len(clients),
        "error_sum": errors.tolist(),
        "target_sum": targets.tolist(),
        "wape": (100.0 * errors / targets).tolist(),
    }


def _blocks(payload: dict, split: str, aggregation: str = "micro") -> np.ndarray:
    record = _block_record(payload, split, aggregation)
    return np.asarray(record["wape"], dtype=np.float64)


def _validate_pair(
    left: dict,
    right: dict,
    split: str,
    aggregation: str,
) -> None:
    """检查 Bootstrap 两侧是否属于同一可配对评估口径。"""

    if left.get("code_revision") != right.get("code_revision"):
        raise ValueError("code_revision differs; do not bootstrap across code versions")
    if left.get("code_revision") != "20260829_topology_projection_v2":
        raise ValueError("result is not from the audited v2 pipeline")
    if left.get("seed") != right.get("seed"):
        raise ValueError("paired bootstrap requires the same random seed on both sides")
    for key in ("data_source_sha256", "node_indices_sha256", "split_bounds"):
        if left.get(key) != right.get(key):
            raise ValueError(f"paired bootstrap requires identical {key}")
    left_eval = left.get("evaluation_metadata", {})
    right_eval = right.get("evaluation_metadata", {})
    for key in ("source_sha256", "history", "horizon", "node_indices_sha256"):
        if left_eval.get(key) != right_eval.get(key):
            raise ValueError(f"evaluation metadata differs in {key}")
    if left.get("mode") == right.get("mode") == "federated":
        if left.get("client_partition_sha256") != right.get("client_partition_sha256"):
            raise ValueError("federated client partitions differ")
        if left_eval.get("client_ids") != right_eval.get("client_ids"):
            raise ValueError("federated client IDs differ")
    if left.get("bootstrap_block_windows") != right.get("bootstrap_block_windows"):
        raise ValueError("bootstrap block length differs")
    if left.get("split_bounds") != right.get("split_bounds"):
        raise ValueError("chronological split bounds differ")
    left_record = _block_record(left, split, aggregation)
    right_record = _block_record(right, split, aggregation)
    for key in (
        "source_sha256",
        "history",
        "horizon",
        "origin_sha256",
        "first_origin",
        "last_origin",
        "n_windows",
        "block_windows",
        "block_n_windows",
    ):
        if left_record.get(key) != right_record.get(key):
            raise ValueError(f"{split} block alignment differs in {key}")


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if left.size != right.size or left.size < 2:
        raise ValueError("paired block arrays must have equal length >= 2")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    differences = right - left
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "blocks": int(differences.size),
        "bootstrap_samples": int(samples),
        "difference_definition": "right_minus_left_wape_percentage_points",
        "observed_mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "seed": int(seed),
        "confidence": float(confidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PA-STFed 配对区块 Bootstrap")
    parser.add_argument("--left", type=Path, required=True, help="左侧实验结果 JSON")
    parser.add_argument("--right", type=Path, required=True, help="右侧实验结果 JSON")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--aggregation",
        choices=("micro", "macro"),
        default="micro",
        help="micro=全局误差（默认，可与集中式配对）；macro=客户端等权描述统计",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_payload = _load(args.left)
    right_payload = _load(args.right)
    _validate_pair(left_payload, right_payload, args.split, args.aggregation)
    left = _blocks(left_payload, args.split, args.aggregation)
    right = _blocks(right_payload, args.split, args.aggregation)
    result = {
        "left": str(args.left),
        "right": str(args.right),
        "split": args.split,
        "aggregation": args.aggregation,
        "left_experiment": left_payload.get("experiment_name"),
        "right_experiment": right_payload.get("experiment_name"),
        "bootstrap": paired_bootstrap(left, right, args.samples, args.seed),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
