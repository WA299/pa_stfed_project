#!/usr/bin/env python3
"""审计 SMART-DS 官方 profile、load_data parquet 与 legacy load_ts 的一致性。

此脚本只做来源核验，不修改任何输入数据，也不启动模型训练。它同时保留
两种官方文档定义的计算口径：

* OpenDSS-native：``Loads.dss`` 的拆分后 ``kW`` 乘 ``LoadShapes.dss`` 的
  无量纲 ``mult``，再按母线汇总；
* parquet-native：``load_data`` 的整户 ``total_site_electricity_kw``，对
  中心抽头 ``_1/_2`` 元件各乘 0.5 后按母线汇总。

两种口径都必须先与官方文件自身相互核对，不能用 legacy NPZ 的数值反向
修改或校准官方数据。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from audit_smartds_data import DEFAULT_FEEDER_ROOT, parse_official_raw, sha256


def _is_center_tap(name: str) -> bool:
    return str(name).lower().endswith(("_1", "_2"))


def _profile_file(shape: dict) -> str | None:
    match = re.search(r"file\s*=\s*([^\)\s]+)", shape.get("mult_file", ""), flags=re.I)
    return Path(match.group(1)).name if match else None


def _parquet_name(profile: str) -> str | None:
    match = re.fullmatch(r"(res|com)_kw_([^_]+)-South-Central_pu\.csv", profile)
    return f"{match.group(1)}_{match.group(2)}-South-Central.parquet" if match else None


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def audit(legacy_path: Path, raw_root: Path, reports_dir: Path, feeder_root: str) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    with np.load(legacy_path, allow_pickle=False) as archive:
        node_ids = np.asarray(archive["node_ids"]).astype(str)
        legacy_ts = np.asarray(archive["load_ts"], dtype=np.float64)

    official = parse_official_raw(raw_root, feeder_root=feeder_root)
    shape_names = sorted({row["yearly_shape"] for row in official["load_records"] if row.get("yearly_shape")})
    profile_cache: dict[str, np.ndarray] = {}
    parquet_cache: dict[str, np.ndarray] = {}
    profile_rows: list[dict] = []
    for shape_name in shape_names:
        shape = official["load_shapes"].get(shape_name, {})
        profile = _profile_file(shape)
        parquet_name = _parquet_name(profile or "")
        row = {
            "shape": shape_name,
            "profile": profile or "",
            "parquet": parquet_name or "",
            "shape_source": f"{shape.get('source_file', '')}:{shape.get('source_line', '')}",
            "profile_sha256": "",
            "parquet_sha256": "",
            "profile_n": 0,
            "parquet_n": 0,
            "time_start": "",
            "time_end": "",
            "time_15min_contiguous": False,
            "profile_parquet_corr": None,
            "profile_parquet_max_abs_error_after_max_scaling": None,
            "status": "FAIL",
            "notes": "",
        }
        if not profile or not parquet_name:
            row["notes"] = "无法从 LoadShapes/文件名得到对应 profile 或 parquet"
            profile_rows.append(row)
            continue
        profile_path = raw_root / "profiles" / profile
        parquet_path = raw_root / "load_data" / parquet_name
        if not profile_path.exists() or not parquet_path.exists():
            row["notes"] = f"缺少文件 profile={profile_path.exists()} parquet={parquet_path.exists()}"
            profile_rows.append(row)
            continue
        try:
            profile_values = np.loadtxt(profile_path, dtype=np.float64)
            table = pq.read_table(parquet_path, columns=["Time", "total_site_electricity_kw"])
            parquet_values = table["total_site_electricity_kw"].to_numpy(zero_copy_only=False).astype(np.float64)
            times = [str(value.as_py()) for value in table["Time"]]
        except Exception as exc:  # 文件损坏时保留结构化证据
            row["notes"] = f"读取失败: {type(exc).__name__}: {exc}"
            profile_rows.append(row)
            continue
        row.update(
            {
                "profile_sha256": sha256(profile_path),
                "parquet_sha256": sha256(parquet_path),
                "profile_n": int(profile_values.size),
                "parquet_n": int(parquet_values.size),
                "time_start": times[0] if times else "",
                "time_end": times[-1] if times else "",
            }
        )
        contiguous = False
        if len(times) == 35040:
            # SMART-DS parquet 时间戳含有时区字符串；转为 numpy datetime64
            # 时可能丢失时区，因此只检查相邻时间差的字符串解析结果。
            try:
                parsed = np.array([np.datetime64(value[:19].replace(" ", "T")) for value in times])
                delta = np.diff(parsed).astype("timedelta64[m]").astype(np.int64)
                contiguous = bool(np.all(delta == 15))
            except (TypeError, ValueError, OverflowError):
                contiguous = False
        row["time_15min_contiguous"] = contiguous
        if profile_values.ndim == 1 and profile_values.size == parquet_values.size and parquet_values.size:
            row["profile_parquet_corr"] = _corr(profile_values, parquet_values)
            scale = float(np.max(parquet_values))
            row["profile_parquet_max_abs_error_after_max_scaling"] = float(np.max(np.abs(profile_values * scale - parquet_values)))
        if profile_values.ndim == 1 and profile_values.size == 35040 and parquet_values.size == 35040 and np.isfinite(profile_values).all() and np.isfinite(parquet_values).all() and contiguous:
            row["status"] = "PASS"
            profile_cache[shape_name] = profile_values
            parquet_cache[shape_name] = parquet_values
        else:
            row["notes"] = "长度、有限性或 15 分钟连续性未通过"
        profile_rows.append(row)

    lookup = {node.lower(): index for index, node in enumerate(node_ids)}
    loads_by_bus: dict[str, list[dict]] = {}
    for record in official["load_records"]:
        loads_by_bus.setdefault(record["bus"], []).append(record)
    node_rows: list[dict] = []
    for bus in sorted(loads_by_bus):
        if bus not in lookup:
            continue
        index = lookup[bus]
        dss_native = np.zeros(legacy_ts.shape[0], dtype=np.float64)
        parquet_native = np.zeros(legacy_ts.shape[0], dtype=np.float64)
        complete = True
        for record in loads_by_bus[bus]:
            shape_name = record.get("yearly_shape", "")
            profile = profile_cache.get(shape_name)
            parquet = parquet_cache.get(shape_name)
            try:
                kw = float(record["kW"])
            except (TypeError, ValueError):
                complete = False
                continue
            if profile is None or parquet is None:
                complete = False
                continue
            dss_native += kw * profile
            # parquet 记录的是整户曲线；中心抽头在 Loads.dss 中拆为两个
            # 元件，因此每个拆分元件只贡献整户曲线的 0.5。
            parquet_native += parquet * (0.5 if _is_center_tap(record["device_name"]) else 1.0)
        observed = legacy_ts[:, index]
        node_rows.append(
            {
                "node_id": node_ids[index],
                "bus": bus,
                "load_element_count": len(loads_by_bus[bus]),
                "complete_official_files": complete,
                "dss_vs_parquet_max_abs_error_kw": float(np.max(np.abs(dss_native - parquet_native))) if complete else None,
                "dss_vs_parquet_mean_abs_error_kw": float(np.mean(np.abs(dss_native - parquet_native))) if complete else None,
                "dss_vs_parquet_corr": _corr(dss_native, parquet_native) if complete else None,
                "legacy_vs_dss_max_abs_error_kw": float(np.max(np.abs(observed - dss_native))) if complete else None,
                "legacy_vs_parquet_max_abs_error_kw": float(np.max(np.abs(observed - parquet_native))) if complete else None,
                "legacy_vs_dss_corr": _corr(observed, dss_native) if complete else None,
                "legacy_vs_parquet_corr": _corr(observed, parquet_native) if complete else None,
            }
        )

    profile_pass = sum(row["status"] == "PASS" for row in profile_rows)
    official_internal = [row for row in node_rows if row["complete_official_files"]]
    legacy_dss_corr = [row["legacy_vs_dss_corr"] for row in official_internal if row["legacy_vs_dss_corr"] is not None]
    legacy_parquet_corr = [row["legacy_vs_parquet_corr"] for row in official_internal if row["legacy_vs_parquet_corr"] is not None]
    result = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_LEGACY_LOAD_TS_UNVERIFIED",
        "legacy_source": str(legacy_path.resolve()),
        "legacy_sha256": sha256(legacy_path),
        "official_raw_root": str(raw_root.resolve()),
        "feeder_root": feeder_root,
        "official_document_rule": {
            "dss_native": "sum(Loads.dss kW * LoadShapes.dss mult) per bus; _1/_2 kW already split",
            "parquet_native": "sum(total_site_electricity_kw * 0.5 for center-tap _1/_2) per bus",
        },
        "profiles": {
            "referenced_shapes": len(shape_names),
            "passed_files": profile_pass,
            "all_files_passed": profile_pass == len(profile_rows),
        },
        "official_internal_consistency": {
            "complete_nodes": len(official_internal),
            "dss_parquet_max_abs_error_kw_max": max((row["dss_vs_parquet_max_abs_error_kw"] for row in official_internal), default=None),
            "dss_parquet_mean_abs_error_kw_median": float(np.median([row["dss_vs_parquet_mean_abs_error_kw"] for row in official_internal])) if official_internal else None,
            "dss_parquet_corr_median": float(np.median([row["dss_vs_parquet_corr"] for row in official_internal if row["dss_vs_parquet_corr"] is not None])) if official_internal else None,
        },
        "legacy_comparison": {
            "nodes": len(node_rows),
            "legacy_vs_dss_corr_median": float(np.median(legacy_dss_corr)) if legacy_dss_corr else None,
            "legacy_vs_parquet_corr_median": float(np.median(legacy_parquet_corr)) if legacy_parquet_corr else None,
            "interpretation": "legacy load_ts 不是官方 OpenDSS/parquet 口径可逐点复现的序列；禁止用缩放、平移或列重排解除 blocker",
        },
        "profile_rows": profile_rows,
        "node_rows": node_rows,
    }
    (reports_dir / "official_load_data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (reports_dir / "official_load_data_audit.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = list(node_rows[0]) if node_rows else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(node_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--feeder-root", default=DEFAULT_FEEDER_ROOT)
    args = parser.parse_args()
    result = audit(args.legacy.resolve(), args.raw_root.resolve(), args.reports.resolve(), args.feeder_root)
    print(json.dumps({"status": result["status"], "reports": str(args.reports.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
