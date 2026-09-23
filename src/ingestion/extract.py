"""
One-time Velodyne zip extraction to local disk.

Never deletes data_odometry_velodyne.zip or extracted .bin files.
After a successful extract, loaders should ignore the zip and memmap on-disk scans.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import List, Optional, Tuple

SENTINEL_NAME = ".extract_complete"
DEFAULT_EXTRACT_DIR = "data/data_odometry_velodyne"
DEFAULT_ZIP_CANDIDATES = [
    "data_odometry_velodyne.zip",
    os.path.join(os.path.dirname(__file__), "../../data_odometry_velodyne.zip"),
    "data/data_odometry_velodyne.zip",
]


def _abs(path: str) -> str:
    return os.path.abspath(path)


def find_velodyne_zip(explicit: Optional[str] = None) -> Optional[str]:
    if explicit and os.path.isfile(explicit) and zipfile.is_zipfile(explicit):
        return _abs(explicit)
    for candidate in DEFAULT_ZIP_CANDIDATES:
        abs_c = _abs(candidate)
        if os.path.isfile(abs_c) and zipfile.is_zipfile(abs_c):
            return abs_c
    return None


def extract_dir_has_scans(extract_dir: str) -> bool:
    if not os.path.isdir(extract_dir):
        return False
    velo_hints = [
        os.path.join(extract_dir, "dataset", "sequences", "00", "velodyne"),
        os.path.join(extract_dir, "sequences", "00", "velodyne"),
        os.path.join(extract_dir, "00", "velodyne"),
        os.path.join(extract_dir, "velodyne"),
    ]
    for d in velo_hints:
        if os.path.isdir(d) and any(name.endswith(".bin") for name in os.listdir(d)):
            return True
    # Any sequence folder with bins
    seq_roots = [
        os.path.join(extract_dir, "dataset", "sequences"),
        os.path.join(extract_dir, "sequences"),
        extract_dir,
    ]
    for root in seq_roots:
        if not os.path.isdir(root):
            continue
        for seq in os.listdir(root):
            velo = os.path.join(root, seq, "velodyne")
            if os.path.isdir(velo) and any(n.endswith(".bin") for n in os.listdir(velo)):
                return True
    return False


def _sentinel_path(extract_dir: str) -> str:
    return os.path.join(extract_dir, SENTINEL_NAME)


def _read_sentinel(extract_dir: str) -> Optional[dict]:
    path = _sentinel_path(extract_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_sentinel(extract_dir: str, zip_path: str, member_count: int) -> None:
    os.makedirs(extract_dir, exist_ok=True)
    payload = {
        "zip_path": _abs(zip_path),
        "zip_mtime": os.path.getmtime(zip_path),
        "zip_size": os.path.getsize(zip_path),
        "member_count": member_count,
    }
    with open(_sentinel_path(extract_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _clear_sentinel_only(extract_dir: str) -> None:
    """Clears the sentinel file only. Never deletes the zip or extracted bins."""
    path = _sentinel_path(extract_dir)
    if os.path.isfile(path):
        os.unlink(path)


def _sentinel_matches(extract_dir: str, zip_path: str, member_count: int) -> bool:
    data = _read_sentinel(extract_dir)
    if not data:
        return False
    try:
        return (
            abs(float(data.get("zip_mtime", -1)) - os.path.getmtime(zip_path)) < 1.0
            and int(data.get("zip_size", -1)) == os.path.getsize(zip_path)
            and int(data.get("member_count", -1)) == member_count
        )
    except Exception:
        return False


def _missing_members(zf: zipfile.ZipFile, extract_dir: str) -> List[str]:
    missing = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        dest = os.path.join(extract_dir, info.filename)
        if not os.path.isfile(dest) or os.path.getsize(dest) != info.file_size:
            missing.append(info.filename)
    return missing


def ensure_velodyne_extracted(
    zip_path: Optional[str] = None,
    extract_dir: Optional[str] = None,
    progress: bool = True,
) -> Tuple[str, bool]:
    """
    Extracts the Velodyne zip to extract_dir if needed.

    Returns:
        (extract_dir, did_extract)
        extract_dir is empty string if neither zip nor a valid extract tree exists.

    Never deletes the zip archive or extracted scan files.
    """
    target = _abs(extract_dir or DEFAULT_EXTRACT_DIR)
    archive = find_velodyne_zip(zip_path)

    if extract_dir_has_scans(target):
        if archive:
            try:
                with zipfile.ZipFile(archive, "r") as zf:
                    n = len(zf.infolist())
                    if _sentinel_matches(target, archive, n):
                        return target, False
                    missing = _missing_members(zf, target)
                    if not missing and not _read_sentinel(target):
                        _write_sentinel(target, archive, n)
                        return target, False
                    if not missing:
                        _write_sentinel(target, archive, n)
                        return target, False
                    if progress:
                        print(
                            f"[INFO] Resuming Velodyne extract: {len(missing)} missing members -> {target}",
                            flush=True,
                        )
                    _clear_sentinel_only(target)
                    for i, name in enumerate(missing, 1):
                        zf.extract(name, target)
                        if progress and (i % 200 == 0 or i == len(missing)):
                            print(f"[INFO] Extract progress {i}/{len(missing)} missing", flush=True)
                    _write_sentinel(target, archive, n)
                    return target, True
            except zipfile.BadZipFile:
                return target, False
        return target, False

    if not archive:
        return "", False

    os.makedirs(target, exist_ok=True)
    if progress:
        size_gb = os.path.getsize(archive) / (1024 ** 3)
        print(
            f"[INFO] Extracting Velodyne archive '{os.path.basename(archive)}' "
            f"({size_gb:.1f} GB compressed) to '{target}'. "
            f"Uncompressed KITTI odometry Velodyne can require tens of GB. "
            f"The zip file is kept on disk and ignored after extract.",
            flush=True,
        )

    try:
        with zipfile.ZipFile(archive, "r") as zf:
            members = zf.infolist()
            n = len(members)
            _clear_sentinel_only(target)
            for i, info in enumerate(members, 1):
                dest = os.path.join(target, info.filename)
                if info.is_dir():
                    os.makedirs(dest, exist_ok=True)
                    continue
                if os.path.isfile(dest) and os.path.getsize(dest) == info.file_size:
                    continue
                zf.extract(info, target)
                if progress and (i % 500 == 0 or i == n):
                    print(f"[INFO] Extract progress {i}/{n}", flush=True)
            _write_sentinel(target, archive, n)
    except Exception:
        _clear_sentinel_only(target)
        raise

    if not extract_dir_has_scans(target):
        _clear_sentinel_only(target)
        print(
            f"[WARN] Extract finished but no Velodyne .bin scans were found under '{target}'.",
            flush=True,
        )
        return target, True

    return target, True
