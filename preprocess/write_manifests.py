#!/usr/bin/env python3
"""生成 SmartDS raw 与 data-rebuild 根目录的可追溯 MANIFEST。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(root: Path, legacy: Path, source: Path, doc_url: str, doc_commit: str, audit_report: Path | None = None) -> dict:
    audit_status = "UNVERIFIED_PENDING_AUDIT"
    profile_mapping_status = "UNVERIFIED_PENDING_AUDIT"
    if audit_report and audit_report.exists():
        audit = json.loads(audit_report.read_text(encoding="utf-8"))
        audit_status = str(audit.get("status", audit_status))
        profile_mapping_status = str(audit.get("time_series", {}).get("official_profile_mapping", {}).get("status", profile_mapping_status))
    files = []
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.name in {"MANIFEST.json", "PROFILE_MANIFEST.json"}:
            continue
        files.append({
            "relative_path": path.relative_to(source).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    remote_report = root / "reports" / "official_raw_remote_verification.json"
    remote_summary = None
    if remote_report.exists():
        try:
            remote = json.loads(remote_report.read_text(encoding="utf-8"))
            remote_summary = {
                "report": remote_report.relative_to(root).as_posix(),
                "official_object_prefix": remote.get("official_object_prefix", ""),
                "all_files_verified": bool(remote.get("all_files_verified")),
                "file_count": len(remote.get("files", [])),
            }
        except (OSError, json.JSONDecodeError):
            remote_summary = {"report": remote_report.relative_to(root).as_posix(), "all_files_verified": False}
    return {
        "manifest_version": "2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open Energy Data Initiative (OEDI)",
        "official_source": {
            "version": "v0.9",
            "year": 2018,
            "dataset": "Full_Texas",
            "region": "P10R",
            "scenario": "base_timeseries",
            "substation": "p10rhs0_1247",
            "feeder": "p10rhs0_1247--p10rdt7719",
            "object_prefix": "s3://oedi-data-lake/SMART-DS/v0.9/2018/Full_Texas/P10R/scenarios/base_timeseries/",
            "source_status": audit_status,
            "profile_mapping_status": profile_mapping_status,
            "audit_report": audit_report.relative_to(root).as_posix() if audit_report else None,
            "documentation_url": doc_url,
            "documentation_commit": doc_commit,
            "documentation_sha256": sha256(source / "reference" / "SMART-DS_Readme.md") if (source / "reference" / "SMART-DS_Readme.md").exists() else "",
            "download_date": "2026-08-29",
            "openDSS_remote_verification": remote_summary,
        },
        "legacy": {
            "relative_path": "data/legacy/smartds_graph_legacy.npz",
            "sha256": sha256(legacy),
        },
        "source_root": source.relative_to(root).as_posix(),
        "files": files,
        "profile_manifest": (source / "PROFILE_MANIFEST.json").relative_to(root).as_posix() if (source / "PROFILE_MANIFEST.json").exists() else None,
        "notes": [
            "设备边只来自 feeder-scoped OpenDSS Line/Transformer 文件及父级 substation 文件。",
            "profile 文件只下载 Loads.dss/LoadShapes.dss 实际引用的最小集合。",
            "未验证的 MST、欧氏距离和节点命名关系不得进入 canonical physical graph。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, help="机器可读审计报告；用于避免 manifest 误标为 canonical")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    source = args.source.resolve()
    legacy = args.legacy.resolve()
    audit_report = args.audit_report.resolve() if args.audit_report else None
    value = manifest(workspace, legacy, source, "https://github.com/openEDI/documentation/blob/main/SMART-DS/Readme.md", "9cdf598733f94d72de09ce0015f4dda671982f9f", audit_report)
    (workspace / "MANIFEST.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_manifest = {key: value[key] for key in ("manifest_version", "created_at_utc", "provider", "official_source", "source_root", "files", "profile_manifest", "notes")}
    (workspace / "data" / "raw" / "SMARTDS" / "MANIFEST.json").write_text(json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"root_manifest": str((workspace / "MANIFEST.json").resolve()), "raw_manifest": str((workspace / "data" / "raw" / "SMARTDS" / "MANIFEST.json").resolve()), "file_count": len(value["files"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
