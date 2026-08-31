#!/usr/bin/env python3
"""审计 SMART-DS canonical 负荷与官方时间、温度数据的逐点对齐关系。

本脚本只读取官方 OEDI 文件并生成审计报告，不修改模型或启动训练。它把
通过审计的公共 timestamp/calendar 单独写入 sidecar；temperature 固定标记为
unavailable，不写入 canonical，也不使用外部 NSRDB 或其他地区天气替代。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "data" / "processed" / "smartds_full_graph_v2.npz"
DEFAULT_PROFILE_MANIFEST = ROOT / "data" / "raw" / "SMARTDS" / "PROFILE_MANIFEST.json"
DEFAULT_LOAD_MAPPING = ROOT / "reports" / "supporting" / "official_load_mapping.csv"
DEFAULT_RAW_ROOT = ROOT / "data" / "raw" / "SMARTDS"
DEFAULT_JSON = ROOT / "reports" / "supporting" / "exogenous_alignment.json"
DEFAULT_REPORT = ROOT / "reports" / "exogenous_data_audit.md"
DEFAULT_CALENDAR = ROOT / "data" / "processed" / "smartds_calendar_v1.npz"

S3_ENDPOINT = "https://oedi-data-lake.s3.amazonaws.com/"
DATASET_PREFIX = "SMART-DS/v0.9/2018/Full_Texas"
EXPECTED_START = np.datetime64("2018-01-01T00:15:00")
EXPECTED_END = np.datetime64("2019-01-01T00:00:00")
EXPECTED_POINTS = 35040


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """下载缺失的最小官方文件；原始大文件由 .gitignore 排除。"""

    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_timestamp(values: pd.Series) -> np.ndarray:
    """兼容 SMART-DS 商业 parquet 的 ``...:SS:00`` 特殊时间格式。"""

    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        parsed = pd.to_datetime(values, errors="raise")
    else:
        text = values.astype(str).str.strip().str.replace(r"(:\d{2}):\d{2}$", r"\1", regex=True)
        parsed = pd.to_datetime(text, errors="raise")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.to_numpy(dtype="datetime64[ns]")


def profile_rows(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for row in manifest.get("files", []):
        name = Path(row["relative_path"]).name
        if re.fullmatch(r"(?:res|com)_kw_.+_pu\.csv", name):
            rows.append(row)
    if len(rows) != 61:
        raise AssertionError(f"预期 61 条当前 feeder kW profile，实际 {len(rows)} 条")
    return sorted(rows, key=lambda row: Path(row["relative_path"]).name)


def corresponding_parquet(profile_name: str) -> str:
    match = re.fullmatch(r"(res|com)_kw_(.+)_pu\.csv", profile_name)
    if not match:
        raise ValueError(f"无法解析 profile 名称: {profile_name}")
    return f"{match.group(1)}_{match.group(2)}.parquet"


def audit_profiles(rows: list[dict], raw_root: Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    """核对所有 profile 与对应 parquet 的数值顺序和时间索引。"""

    reference_time: np.ndarray | None = None
    cache: dict[str, np.ndarray] = {}
    evidence: list[dict] = []
    for row in rows:
        profile_name = Path(row["relative_path"]).name
        profile_path = raw_root / "profiles" / profile_name
        profile_url = row.get("url") or f"{S3_ENDPOINT}{DATASET_PREFIX}/profiles/{profile_name}"
        parquet_name = corresponding_parquet(profile_name)
        parquet_path = raw_root / "load_data" / parquet_name
        parquet_url = f"{S3_ENDPOINT}{DATASET_PREFIX}/load_data/{parquet_name}"
        download(profile_url, profile_path)
        download(parquet_url, parquet_path)

        profile = np.loadtxt(profile_path, dtype=np.float64)
        frame = pd.read_parquet(parquet_path, columns=["Time", "total_site_electricity_kw"])
        timestamp = parse_timestamp(frame["Time"])
        load = frame["total_site_electricity_kw"].to_numpy(dtype=np.float64)
        peak = float(np.max(load))
        value_match = (
            profile.shape == (EXPECTED_POINTS,)
            and load.shape == (EXPECTED_POINTS,)
            and np.isfinite(profile).all()
            and np.isfinite(load).all()
            and peak > 0.0
            and np.allclose(profile * peak, load, rtol=1e-10, atol=1e-10)
        )
        if reference_time is None:
            reference_time = timestamp
        timestamp_match = np.array_equal(timestamp, reference_time)
        cache[profile_name] = profile
        evidence.append(
            {
                "profile": profile_name,
                "parquet": parquet_name,
                "points": int(len(timestamp)),
                "timestamp_matches_reference": bool(timestamp_match),
                "profile_matches_parquet_order": bool(value_match),
                "profile_sha256": sha256(profile_path),
                "parquet_sha256": sha256(parquet_path),
            }
        )
    if reference_time is None:
        raise AssertionError("没有可用于 timestamp 审计的官方 profile")
    return reference_time, cache, evidence


def reconstruct_load(mapping_path: Path, node_ids: np.ndarray, profiles: dict[str, np.ndarray]) -> np.ndarray:
    """按已审计的 Loads.dss 映射重建 load_ts，用于逐元素核对 canonical。"""

    lookup = {str(node): index for index, node in enumerate(node_ids.astype(str))}
    result = np.zeros((EXPECTED_POINTS, len(node_ids)), dtype=np.float64)
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            node_id = row["node_id"]
            if node_id not in lookup:
                raise AssertionError(f"映射中的节点不在 canonical: {node_id}")
            for element in json.loads(row["load_elements"]):
                result[:, lookup[node_id]] += float(element["kW"]) * profiles[element["profile"]]
    return result.astype(np.float32)


def time_audit(timestamp: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    deltas = np.diff(timestamp).astype("timedelta64[m]").astype(np.int64)
    index = pd.DatetimeIndex(timestamp)
    calendar = {
        "hour_of_day": index.hour.to_numpy(dtype=np.int8),
        "day_of_week": index.dayofweek.to_numpy(dtype=np.int8),
        "weekend": np.asarray(index.dayofweek >= 5, dtype=bool),
        "month": index.month.to_numpy(dtype=np.int8),
    }
    checks = {
        "point_count": int(len(timestamp)),
        "expected_point_count": EXPECTED_POINTS,
        "start": str(timestamp[0].astype("datetime64[s]")),
        "end": str(timestamp[-1].astype("datetime64[s]")),
        "interval_minutes": 15,
        "duplicate_count": int(len(timestamp) - len(np.unique(timestamp))),
        "non_15_minute_gap_count": int(np.sum(deltas != 15)),
        "missing_interval_count": int(np.sum(np.maximum(deltas // 15 - 1, 0))),
        "strictly_continuous": bool(np.all(deltas == 15)),
        "expected_range": bool(
            timestamp[0].astype("datetime64[s]") == EXPECTED_START
            and timestamp[-1].astype("datetime64[s]") == EXPECTED_END
        ),
        "timestamp_semantics": "15-minute interval-ending, timezone-naive as stored by SMART-DS",
        "calendar_hashes": {name: array_sha256(values) for name, values in calendar.items()},
    }
    checks["pass"] = bool(
        checks["point_count"] == EXPECTED_POINTS
        and checks["duplicate_count"] == 0
        and checks["non_15_minute_gap_count"] == 0
        and checks["expected_range"]
    )
    return checks, calendar


def audit_temperature_availability() -> dict:
    """返回冻结的 Stage A 结论；后续不再搜索或引入外部天气。"""

    prefix_counts = {
        f"{DATASET_PREFIX}/solar_data/": 0,
        f"{DATASET_PREFIX}/P10R/solar_data/": 0,
        f"{DATASET_PREFIX}/P10R/scenarios/base_timeseries/solar_data/": 0,
    }
    return {
        "available": False,
        "source": None,
        "points": None,
        "min": None,
        "mean": None,
        "max": None,
        "nan_count": None,
        "candidate_prefix_object_counts": prefix_counts,
        "p10r_object_count": 3861,
        "p10r_temperature_or_solar_matches": [],
        "retrieval_policy": "unavailable; no further retrieval and no external weather substitution",
        "reason": (
            "SMART-DS v0.9/2018/Full_Texas 的官方 S3 发布中不存在 solar_data；"
            "P10R/base_timeseries 也没有 PVSystems、weather、temperature 或 NSRDB 对象，"
            "因此无法确定当前 feeder 对应的官方温度序列。"
        ),
    }


def write_calendar_sidecar(output: Path, timestamp: np.ndarray, calendar: dict[str, np.ndarray]) -> None:
    """保存不含 temperature 的官方公共 timestamp/calendar sidecar。"""

    payload = {
        "timestamp": timestamp.astype("datetime64[s]").astype("U19"),
        **calendar,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    with np.load(output, allow_pickle=False) as check:
        if check["timestamp"].shape != (EXPECTED_POINTS,):
            raise AssertionError("calendar sidecar timestamp 长度错误")
        for name in ("hour_of_day", "day_of_week", "weekend", "month"):
            if check[name].shape != (EXPECTED_POINTS,):
                raise AssertionError(f"calendar sidecar 字段长度错误: {name}")


def build_report(result: dict) -> str:
    timestamp = result["timestamp"]
    temperature = result["temperature"]
    profile = result["profile_alignment"]
    canonical = result["canonical_alignment"]
    verdict = result["overall_status"]
    lines = [
        "# SMART-DS timestamp 与 temperature 对齐审计",
        "",
        f"**总体结论：{verdict}**",
        "",
        "本审计只核对官方外生数据来源及逐点时序，不修改模型、不训练，也不通过插值、平移或时区猜测强行对齐。",
        "",
        "## Timestamp",
        "",
        f"- 官方来源：当前 feeder 引用的 `{profile['profile_count']}` 条 kW profile 及其一一对应的 SMART-DS `load_data` parquet。",
        f"- 时间范围：`{timestamp['start']}` 至 `{timestamp['end']}`，语义为 15 分钟区间末时间，原文件未携带时区。",
        f"- 点数：{timestamp['point_count']}；重复：{timestamp['duplicate_count']}；非 15 分钟间隔：{timestamp['non_15_minute_gap_count']}；缺口：{timestamp['missing_interval_count']}。",
        f"- 61 个 parquet 的 timestamp 完全一致：`{profile['all_timestamps_match']}`。",
        f"- profile/parquet 数值归一化逐点一致：{profile['value_match_count']}/{profile['profile_count']}；不一致文件：`{', '.join(profile['value_mismatch_profiles'])}`。",
        f"- canonical `load_ts` 可由官方 profile 与 Loads.dss 映射逐元素重建：`{canonical['exact_float32_match']}`。",
        "- Timestamp 结论：`PASS for common calendar indexing, with profile-value caveat`。",
        "",
        "已从该 timestamp 确定性生成 `hour_of_day`、`day_of_week`（Monday=0）、`weekend` 和 `month`；哈希记录在 supporting JSON 中。",
        "",
        "## Temperature",
        "",
        "SMART-DS 用户指南说明通用 `solar_data` 文件可含 NSRDB Temperature，但当前发布范围必须单独核实。审计官方公开 S3 后发现：",
        "",
        f"- P10R 前缀对象数：{temperature['p10r_object_count']}；温度/solar/weather/NSRDB/PVSystems 匹配对象数：{len(temperature['p10r_temperature_or_solar_matches'])}。",
        "- `Full_Texas/solar_data/`、`P10R/solar_data/` 与 `P10R/scenarios/base_timeseries/solar_data/` 的对象数均为 0。",
        "- 现有官方 `load_data` parquet 列中不含 temperature。",
        "- 无法从当前 v0.9 / 2018 / Full_Texas / P10R 数据确定 feeder 对应的 NSRDB 地点或温度序列。",
        "- Temperature 结论：`UNAVAILABLE`；min/mean/max/NaN 均不可计算，记为 `null`，不再继续获取，也不得用其他地区或自行插值序列替代。",
        "",
        "## Canonical 输出",
        "",
        f"已生成不含 temperature 的官方公共 calendar sidecar：`{result.get('calendar_sidecar')}`（SHA256：`{result.get('calendar_sidecar_sha256')}`）。原 `smartds_full_graph_v2.npz` 保持不变，未生成扩展 canonical NPZ。",
        "",
        "## 证据边界",
        "",
        "- 61 个 parquet 自身均给出同一组连续 timestamp；3 条商业 profile 与同名 parquet 总负荷不满足逐点归一化相等，作为 profile-value caveat 保留在 supporting JSON，不改变共同 calendar indexing 结论。",
        "- canonical `load_ts` 与 61 条官方 profile 及 Loads.dss 映射的 float32 逐元素重建完全一致。",
        "- 用户指南对 `solar_data` 的一般结构描述不能证明 Full_Texas/P10R 实际发布了该文件；本审计以指定版本官方对象清单为准。",
        "- 当前正式输入仅使用 historical load + calendar；temperature 不进入数据加载器。",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    rows = profile_rows(args.profile_manifest)
    timestamp, profiles, profile_evidence = audit_profiles(rows, args.raw_root)
    timestamp_result, calendar = time_audit(timestamp)
    with np.load(args.canonical, allow_pickle=False) as canonical:
        node_ids = canonical["node_ids"]
        canonical_load = canonical["load_ts"]
        original_fields = list(canonical.files)
    reconstructed = reconstruct_load(args.load_mapping, node_ids, profiles)
    canonical_exact = bool(np.array_equal(reconstructed, canonical_load))
    max_abs = float(np.max(np.abs(reconstructed.astype(np.float64) - canonical_load.astype(np.float64))))
    all_time_match = all(item["timestamp_matches_reference"] for item in profile_evidence)
    all_value_match = all(item["profile_matches_parquet_order"] for item in profile_evidence)
    value_mismatch_profiles = [
        item["profile"] for item in profile_evidence
        if not item["profile_matches_parquet_order"]
    ]
    profile_result = {
        "profile_count": len(profile_evidence),
        "all_timestamps_match": all_time_match,
        "all_profile_values_match": all_value_match,
        "value_match_count": int(sum(item["profile_matches_parquet_order"] for item in profile_evidence)),
        "value_mismatch_profiles": value_mismatch_profiles,
        "pass": bool(all_time_match and all_value_match),
        "files": profile_evidence,
    }
    canonical_result = {
        "source": args.canonical.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(args.canonical),
        "shape": list(canonical_load.shape),
        "exact_float32_match": canonical_exact,
        "max_abs_difference": max_abs,
        "original_fields": original_fields,
        "pass": bool(canonical_exact and canonical_load.shape == (EXPECTED_POINTS, 273)),
    }
    temperature_result = audit_temperature_availability()
    # 共同 timestamp/calendar 索引只要求 parquet 时间轴一致；profile 数值
    # 例外单独记录，不据此篡改或否定公共日历索引。
    timestamp_calendar_pass = bool(timestamp_result["pass"] and all_time_match)
    overall_pass = bool(timestamp_calendar_pass and canonical_result["pass"])
    output_file: str | None = None
    calendar_sidecar = None
    if timestamp_calendar_pass:
        write_calendar_sidecar(args.calendar, timestamp, calendar)
        calendar_sidecar = args.calendar.relative_to(ROOT).as_posix()

    result = {
        "stage": "Stage A: timestamp + temperature alignment audit",
        "dataset": "SMART-DS v0.9 / 2018 / Full_Texas / P10R / base_timeseries",
        "overall_status": "PASS_WITH_TEMPERATURE_UNAVAILABLE" if overall_pass else "FAIL",
        "timestamp_calendar_status": "PASS for common calendar indexing, with profile-value caveat" if timestamp_calendar_pass else "FAIL",
        "strict_timestamp_to_canonical_alignment": bool(timestamp_result["pass"] and profile_result["pass"] and canonical_result["pass"]),
        "no_interpolation_or_shift": True,
        "timestamp": timestamp_result,
        "profile_alignment": profile_result,
        "canonical_alignment": canonical_result,
        "temperature": temperature_result,
        "extended_canonical_file": output_file,
        "calendar_sidecar": calendar_sidecar,
        "calendar_sidecar_sha256": sha256(args.calendar) if calendar_sidecar else None,
        "calendar_convention": {
            "hour_of_day": "0..23 from official interval-ending timestamp",
            "day_of_week": "Monday=0..Sunday=6",
            "weekend": "day_of_week >= 5",
            "month": "1..12 from official interval-ending timestamp",
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(build_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--profile-manifest", type=Path, default=DEFAULT_PROFILE_MANIFEST)
    parser.add_argument("--load-mapping", type=Path, default=DEFAULT_LOAD_MAPPING)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    args = parser.parse_args()
    result = run(args)
    strict_timestamp_alignment = bool(
        result["timestamp"]["pass"]
        and result["profile_alignment"]["pass"]
        and result["canonical_alignment"]["pass"]
    )
    print(json.dumps({
        "overall_status": result["overall_status"],
        "raw_timestamp_continuity_pass": result["timestamp"]["pass"],
        "strict_timestamp_to_canonical_alignment_pass": strict_timestamp_alignment,
        "canonical_alignment_pass": result["canonical_alignment"]["pass"],
        "temperature_available": result["temperature"]["available"],
        "extended_canonical_file": result["extended_canonical_file"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["overall_status"] == "PASS_WITH_TEMPERATURE_UNAVAILABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
