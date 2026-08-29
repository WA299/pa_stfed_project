#!/usr/bin/env python3
"""逐文件核对本地 OpenDSS 原始文件与 OEDI 官方对象存储。

该脚本只下载当前 feeder scope 的 OpenDSS 文件（不下载 profile/parquet），
计算远程内容 SHA256 并与本地文件比较。结果写入报告，便于审计者确认
``Master.dss`` 等文件确实来自官方对象，而不是由节点名称反推得到。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from audit_smartds_data import DEFAULT_FEEDER_ROOT, _scope_files, sha256


DEFAULT_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "SMARTDS"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "official_raw_remote_verification.json"
DEFAULT_PREFIX = "https://oedi-data-lake.s3.amazonaws.com/SMART-DS/v0.9/2018/Full_Texas/P10R/scenarios/base_timeseries/opendss"


def _remote_hash(url: str) -> tuple[str, int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size, "PASS"
    except Exception as exc:  # 网络故障也要以结构化状态落盘
        return "", size, f"ERROR:{type(exc).__name__}:{exc}"


def verify(raw_root: Path, feeder_root: str, report_path: Path, prefix: str = DEFAULT_PREFIX) -> dict:
    paths, scope = _scope_files(raw_root, feeder_root)
    rows: list[dict] = []

    def one(path: Path) -> dict:
        relative = path.relative_to(raw_root).as_posix()
        url = f"{prefix.rstrip('/')}/{relative}"
        remote_sha, remote_bytes, status = _remote_hash(url)
        local_sha = sha256(path)
        local_bytes = path.stat().st_size
        return {
            "relative_path": relative,
            "official_url": url,
            "local_sha256": local_sha,
            "remote_sha256": remote_sha,
            "local_bytes": local_bytes,
            "remote_bytes": remote_bytes,
            "status": "PASS" if status == "PASS" and local_sha == remote_sha and local_bytes == remote_bytes else status if status != "PASS" else "FAIL_HASH_OR_SIZE",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(len(paths), 1))) as executor:
        rows = list(executor.map(one, paths))
    result = {
        "verification_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open Energy Data Initiative (OEDI)",
        "official_object_prefix": prefix,
        "version": "v0.9",
        "year": 2018,
        "dataset": "Full_Texas",
        "region": "P10R",
        "scenario": "base_timeseries",
        "feeder_root": feeder_root,
        "scope": scope,
        "files": rows,
        "all_files_verified": bool(rows) and all(row["status"] == "PASS" for row in rows),
        "notes": [
            "仅核对当前 feeder scope 的 OpenDSS 文件；profile/parquet 由 PROFILE_MANIFEST.json 单独记录。",
            "Intermediates.txt 只作为官方几何证据，不被解释为额外物理节点或边。",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with report_path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        import csv

        fields = ["relative_path", "official_url", "local_sha256", "remote_sha256", "local_bytes", "remote_bytes", "status"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--feeder-root", default=DEFAULT_FEEDER_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    result = verify(args.raw_root.resolve(), args.feeder_root, args.report.resolve(), args.prefix)
    print(json.dumps({"all_files_verified": result["all_files_verified"], "report": str(args.report.resolve())}, ensure_ascii=False, indent=2))
    return 0 if result["all_files_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
