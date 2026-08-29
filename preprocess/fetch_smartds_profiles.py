#!/usr/bin/env python3
"""下载当前 SMART-DS feeder 所引用的最小官方负荷曲线子集。

脚本只根据 ``Loads.dss``/``LoadShapes.dss`` 中的 ``file=`` 引用下载曲线，
不会下载整个 SMART-DS 数据湖。下载清单和 SHA256 写入 raw 根目录，便于
审计时确认序列来源及后续 clean rerun。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_lines(path: Path) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("~"):
            current += " " + line[1:].strip()
        else:
            if current:
                lines.append(current)
            current = line
    if current:
        lines.append(current)
    return lines


def _references(raw_root: Path, feeder_root: str) -> dict[str, str]:
    feeder = raw_root / feeder_root
    loads = feeder / "Loads.dss"
    shapes = feeder / "LoadShapes.dss"
    if not loads.exists() or not shapes.exists():
        raise FileNotFoundError("feeder 必须同时包含 Loads.dss 和 LoadShapes.dss")
    yearly = sorted({
        match.group(1)
        for line in _logical_lines(loads)
        if (match := re.search(r"\byearly\s*=\s*([^\s]+)", line, flags=re.I))
    })
    references: dict[str, str] = {}
    for line in _logical_lines(shapes):
        shape_match = re.match(r"(?:new|edit)\s+loadshape\.([^\s]+)", line, flags=re.I)
        if not shape_match:
            continue
        shape_name = shape_match.group(1)
        if shape_name not in yearly:
            continue
        for kind in ("mult", "qmult"):
            file_match = re.search(rf"\b{kind}\s*=\s*\(\s*file\s*=\s*([^\)\s]+)", line, flags=re.I)
            if file_match:
                file_name = Path(file_match.group(1)).name
                references[file_name] = f"{feeder_root}/LoadShapes.dss"
    return references


def fetch(
    raw_root: Path,
    feeder_root: str,
    version: str,
    year: int,
    dataset: str,
    scenario: str,
    include_load_data: bool = False,
) -> dict:
    references = _references(raw_root, feeder_root)
    profile_dir = raw_root / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    base_key = f"SMART-DS/{version}/{year}/{dataset}/profiles"
    def download(item: tuple[str, str]) -> dict:
        file_name, evidence_file = item
        if not re.fullmatch(r"(?:res|com)_(?:kw|kvar)_[^/]+\.csv", file_name):
            raise ValueError(f"拒绝下载非预期 profile 名称: {file_name}")
        local = profile_dir / file_name
        url = f"https://oedi-data-lake.s3.amazonaws.com/{base_key}/{file_name}"
        status = "cached"
        error = ""
        if not local.exists() or local.stat().st_size == 0:
            temporary = local.with_suffix(local.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as stream:
                    stream.write(response.read())
                temporary.replace(local)
                status = "downloaded"
            except Exception as exc:  # 网络抖动时记录失败，不阻塞其它曲线
                error = f"{type(exc).__name__}: {exc}"
                temporary.unlink(missing_ok=True)
                local.unlink(missing_ok=True)
                status = "failed"
        row = {
            "relative_path": local.relative_to(raw_root).as_posix(),
            "official_key": f"{base_key}/{file_name}",
            "url": url,
            "sha256": sha256(local) if local.exists() and local.stat().st_size else "",
            "bytes": local.stat().st_size if local.exists() else 0,
            "evidence_file": evidence_file,
            "status": status,
            "error": error,
        }
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(download, sorted(references.items())))
    load_data_rows: list[dict] = []
    if include_load_data:
        # SMART-DS 文档规定 load_data 文件名为 <class>_<profile id>-<location>.parquet。
        # 只下载当前 feeder 的 yearly profile 所需文件，避免拉取完整数据湖。
        load_data_dir = raw_root / "load_data"
        load_data_dir.mkdir(parents=True, exist_ok=True)
        load_data_names: set[str] = set()
        for profile_name in references:
            match = re.fullmatch(r"(res|com)_kw_([^_]+)-South-Central_pu\.csv", profile_name)
            if match:
                load_data_names.add(f"{match.group(1)}_{match.group(2)}-South-Central.parquet")

        def download_load_data(file_name: str) -> dict:
            local = load_data_dir / file_name
            url = (
                f"https://oedi-data-lake.s3.amazonaws.com/SMART-DS/{version}/{year}/"
                f"{dataset}/load_data/{file_name}"
            )
            status = "cached"
            error = ""
            if not local.exists() or local.stat().st_size == 0:
                temporary = local.with_suffix(local.suffix + ".part")
                try:
                    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                        stream.write(response.read())
                    temporary.replace(local)
                    status = "downloaded"
                except Exception as exc:  # 网络问题逐条记录，不吞掉审计证据
                    error = f"{type(exc).__name__}: {exc}"
                    temporary.unlink(missing_ok=True)
                    local.unlink(missing_ok=True)
                    status = "failed"
            return {
                "relative_path": local.relative_to(raw_root).as_posix(),
                "official_key": f"SMART-DS/{version}/{year}/{dataset}/load_data/{file_name}",
                "url": url,
                "sha256": sha256(local) if local.exists() and local.stat().st_size else "",
                "bytes": local.stat().st_size if local.exists() else 0,
                "status": status,
                "error": error,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            load_data_rows = list(executor.map(download_load_data, sorted(load_data_names)))

    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open Energy Data Initiative (OEDI)",
        "version": version,
        "year": year,
        "dataset": dataset,
        "scenario": scenario,
        "feeder_root": feeder_root,
        "profile_count": len(rows),
        "files": rows,
        "load_data_count": len(load_data_rows),
        "load_data_files": load_data_rows,
    }
    (raw_root / "PROFILE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--feeder-root", required=True)
    parser.add_argument("--version", default="v0.9")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--dataset", default="Full_Texas")
    parser.add_argument("--scenario", default="base_timeseries")
    parser.add_argument(
        "--include-load-data",
        action="store_true",
        help="额外下载当前 feeder 引用的最小 load_data parquet 子集（约 110 MB）",
    )
    args = parser.parse_args()
    manifest = fetch(
        args.raw_root.resolve(),
        args.feeder_root,
        args.version,
        args.year,
        args.dataset,
        args.scenario,
        include_load_data=args.include_load_data,
    )
    print(
        json.dumps(
            {
                "profile_count": manifest["profile_count"],
                "load_data_count": manifest.get("load_data_count", 0),
                "manifest": str((args.raw_root / "PROFILE_MANIFEST.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
