"""
KITTI / SemanticKITTI Ingestion Module for FoveaMap.

Loads binary point clouds (.bin) and ground-truth semantic labels (.label).
Supports direct zero-extraction streaming from dataset zip archives
(e.g., data_odometry_velodyne.zip) as well as extracted sequence directories.
Formats scans into unified (N, 5) float32 arrays: [x, y, z, intensity, class_id].
"""

import os
import re
import glob
import zipfile
from typing import Optional, List, Tuple, Union, Iterator
import numpy as np

from src.ingestion.extract import (
    DEFAULT_EXTRACT_DIR,
    ensure_velodyne_extracted,
    extract_dir_has_scans,
    find_velodyne_zip,
)


def load_velodyne_bin(bin_path: str) -> np.ndarray:
    """
    Load a raw Velodyne binary point cloud file.
    
    Args:
        bin_path: Path to .bin file.
        
    Returns:
        (N, 4) float32 array: [x, y, z, intensity]
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Velodyne binary file not found: {bin_path}")
    
    points = np.memmap(bin_path, dtype=np.float32, mode='r')
    if points.size % 4 != 0:
        raise ValueError(f"Corrupted binary file (size {points.size} not divisible by 4): {bin_path}")
    
    return points.reshape(-1, 4)


def load_labels(label_path: str) -> np.ndarray:
    """
    Load a SemanticKITTI .label file.
    
    Format: 32-bit unsigned integers where:
      - Lower 16 bits: semantic class ID (label & 0xFFFF)
      - Upper 16 bits: instance ID (label >> 16)
      
    Args:
        label_path: Path to .label file.
        
    Returns:
        (N,) int32 array: raw semantic class IDs.
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    raw_labels = np.memmap(label_path, dtype=np.uint32, mode='r')
    semantic_labels = (raw_labels & 0xFFFF).astype(np.int32)
    return semantic_labels


def load_kitti_scan(bin_path: str, label_path: Optional[str] = None) -> np.ndarray:
    """
    Load a single KITTI scan and optional ground-truth label, returning an (N, 5) array.
    
    Args:
        bin_path: Path to .bin file.
        label_path: Optional path to .label file.
        
    Returns:
        (N, 5) float32 array: [x, y, z, intensity, class_id].
        class_id is -1 if label_path is None or missing.
    """
    points = load_velodyne_bin(bin_path)
    n_points = points.shape[0]
    
    scan = np.empty((n_points, 5), dtype=np.float32)
    scan[:, :4] = points
    
    if label_path is not None and os.path.exists(label_path):
        labels = load_labels(label_path)
        if labels.shape[0] != n_points:
            raise ValueError(
                f"Mismatch between points count ({n_points}) and labels count ({labels.shape[0]})"
            )
        scan[:, 4] = labels.astype(np.float32)
    else:
        scan[:, 4] = -1.0  # Unclassified
        
    return scan


class KITTILoader:
    """
    Dataset loader for KITTI / SemanticKITTI LiDAR sequences.
    
    Supports:
    - Direct streaming from .zip archives (e.g. data_odometry_velodyne.zip) without disk extraction.
    - Extracted sequence directories (e.g. data/data_odometry_velodyne/dataset/sequences/00/).
    - Sequence selection ('00', '01', ..., '21').
    - Synthetic signature detection guard to distinguish procedural stand-ins from genuine sensor data.
    """
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        sequence: str = "00",
        warn_on_synthetic: bool = True,
    ):
        self.sequence = str(sequence).zfill(2)
        self.warn_on_synthetic = warn_on_synthetic
        self.dataset_path = self._resolve_dataset_path(dataset_path)

        self.is_zip = zipfile.is_zipfile(self.dataset_path) if os.path.exists(self.dataset_path) else False
        self._zip_ref: Optional[zipfile.ZipFile] = None
        
        self.bin_files: List[str] = []
        self.label_files: List[Optional[str]] = []
        self.is_synthetic_signature: bool = False

        self._discover_files()

    @staticmethod
    def _resolve_dataset_path(path: Optional[str]) -> str:
        """Resolves provided path, preferring an extracted directory over the zip."""
        if path:
            abs_p = os.path.abspath(path)
            if os.path.isdir(abs_p):
                return abs_p
            if os.path.isfile(abs_p) and zipfile.is_zipfile(abs_p):
                extract_dir, _ = ensure_velodyne_extracted(zip_path=abs_p)
                if extract_dir and extract_dir_has_scans(extract_dir):
                    return extract_dir
                return abs_p

        extract_dir = os.path.abspath(DEFAULT_EXTRACT_DIR)
        zip_path = find_velodyne_zip()
        if zip_path or extract_dir_has_scans(extract_dir):
            resolved, _ = ensure_velodyne_extracted(zip_path=zip_path, extract_dir=extract_dir)
            if resolved and extract_dir_has_scans(resolved):
                return resolved

        dir_candidates = [
            "data_odometry_velodyne",
            "data/data_odometry_velodyne",
            "dataset",
            "data/dataset",
            "data/kitti",
        ]
        for candidate in dir_candidates:
            abs_c = os.path.abspath(candidate)
            if os.path.isdir(abs_c) and extract_dir_has_scans(abs_c):
                return abs_c
            if os.path.isdir(abs_c):
                return abs_c

        synthetic_candidates = [
            "data/synthetic_kitti_like",
            "data/kitti_sample",
        ]
        for candidate in synthetic_candidates:
            abs_c = os.path.abspath(candidate)
            if os.path.exists(abs_c):
                return abs_c

        if zip_path:
            return zip_path
        return path or "data_odometry_velodyne.zip"

    def _discover_files(self) -> None:
        """Discovers scans for the configured sequence."""
        if not os.path.exists(self.dataset_path):
            return

        if self.is_zip:
            if self._zip_ref is None:
                self._zip_ref = zipfile.ZipFile(self.dataset_path, "r")
            all_names = self._zip_ref.namelist()
            
            # Match dataset/sequences/{seq}/velodyne/*.bin or sequences/{seq}/velodyne/*.bin
            patterns = [
                f"dataset/sequences/{self.sequence}/velodyne/",
                f"sequences/{self.sequence}/velodyne/",
                f"{self.sequence}/velodyne/",
                "velodyne/",
            ]
            matched_prefix = None
            for p in patterns:
                if any(name.startswith(p) and name.endswith(".bin") for name in all_names):
                    matched_prefix = p
                    break

            if matched_prefix:
                self.bin_files = sorted([
                    name for name in all_names
                    if name.startswith(matched_prefix) and name.endswith(".bin")
                ])
            else:
                # Fallback: any bin files
                self.bin_files = sorted([name for name in all_names if name.endswith(".bin")])

            # Labels in zip if available
            label_prefix = matched_prefix.replace("velodyne/", "labels/") if matched_prefix else None
            name_set = set(all_names)
            self.label_files = []
            for bin_f in self.bin_files:
                if label_prefix:
                    base = os.path.splitext(os.path.basename(bin_f))[0]
                    lbl = f"{label_prefix}{base}.label"
                    self.label_files.append(lbl if lbl in name_set else None)
                else:
                    self.label_files.append(None)

        else:
            # Extracted directory
            # Check multiple directory layouts
            possible_velo_dirs = [
                os.path.join(self.dataset_path, "dataset", "sequences", self.sequence, "velodyne"),
                os.path.join(self.dataset_path, "sequences", self.sequence, "velodyne"),
                os.path.join(self.dataset_path, self.sequence, "velodyne"),
                os.path.join(self.dataset_path, "velodyne"),
                self.dataset_path,
            ]
            
            velo_dir = None
            for d in possible_velo_dirs:
                if os.path.isdir(d) and glob.glob(os.path.join(d, "*.bin")):
                    velo_dir = d
                    break

            if not velo_dir:
                return

            self.bin_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
            labels_dir = velo_dir.replace("velodyne", "labels")
            self.label_files = []
            
            for bin_f in self.bin_files:
                basename = os.path.splitext(os.path.basename(bin_f))[0]
                label_f = os.path.join(labels_dir, f"{basename}.label")
                if os.path.exists(label_f):
                    self.label_files.append(label_f)
                else:
                    self.label_files.append(None)

        # Inspect file sizes for synthetic vs physical LiDAR signatures
        self._check_synthetic_signature()

    def _check_synthetic_signature(self) -> None:
        """
        Inspects file sizes. Genuine LiDAR captures have variable beam returns per rotation.
        Uniform byte sizes across all frames indicate procedurally generated synthetic stand-ins.
        """
        if len(self.bin_files) < 2:
            return

        sample_files = self.bin_files[:10]
        if self.is_zip and self._zip_ref:
            sizes = [self._zip_ref.getinfo(f).file_size for f in sample_files]
        else:
            sizes = [os.path.getsize(f) for f in sample_files]

        if len(set(sizes)) == 1:
            self.is_synthetic_signature = True
            if self.warn_on_synthetic:
                print(
                    f"\n[WARNING] [KITTILoader] SYNTHETIC DATA SIGNATURE DETECTED in '{self.dataset_path}':\n"
                    f"  All {len(self.bin_files)} scans are byte-identical in size ({sizes[0]:,} bytes / {sizes[0]//16:,} points).\n"
                    f"  This directory contains procedurally generated stand-ins, NOT genuine physical Velodyne sensor captures.\n",
                    flush=True,
                )
        else:
            self.is_synthetic_signature = False
            first_pts = sizes[0] // 16
            print(
                f"[INFO] [KITTILoader] Genuine physical LiDAR capture loaded from '{os.path.basename(self.dataset_path)}' "
                f"(Sequence {self.sequence}: {len(self.bin_files)} scans, ~{first_pts:,} pts/frame, variable beam returns).",
                flush=True,
            )

    def set_sequence(self, sequence: Union[int, str]) -> "KITTILoader":
        """
        Switches loader to another sequence in the dataset.
        
        Args:
            sequence: Sequence ID string or int (e.g. '00', 4, '21').
        """
        self.sequence = str(sequence).zfill(2)
        self._discover_files()
        return self

    def iter_playlist(
        self,
        sequences: Optional[List[str]] = None,
    ) -> Iterator[Tuple[str, int]]:
        """Yields (sequence_id, frame_idx) across sequences without loading scans."""
        seqs = sequences if sequences is not None else self.list_sequences()
        original = self.sequence
        try:
            for seq in seqs:
                self.set_sequence(seq)
                for idx in range(len(self.bin_files)):
                    yield seq, idx
        finally:
            if self.sequence != original:
                self.set_sequence(original)

    def list_sequences(self) -> List[str]:
        """Returns list of sequence IDs available in this dataset."""
        if self.is_zip:
            if self._zip_ref is None and os.path.exists(self.dataset_path):
                self._zip_ref = zipfile.ZipFile(self.dataset_path, "r")
            if self._zip_ref:
                seqs = set(re.findall(r"(?:dataset/)?sequences/(\d+)/", "\n".join(self._zip_ref.namelist())))
                return sorted(list(seqs))
        elif os.path.isdir(self.dataset_path):
            seq_dir = os.path.join(self.dataset_path, "dataset", "sequences")
            if not os.path.exists(seq_dir):
                seq_dir = os.path.join(self.dataset_path, "sequences")
            if os.path.exists(seq_dir):
                return sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))])
        return [self.sequence]

    def vehicle_state_at(self, idx: int, dt_s: float = 0.1) -> "VehicleState":
        """Best-effort ego kinematics from poses.txt if present; else a stable default."""
        from src.grid.grid_types import VehicleState

        poses = self._load_poses()
        if poses is None or idx >= len(poses):
            return VehicleState(speed_mps=10.0, steering_angle_rad=0.0)

        t = poses[idx]
        if idx > 0:
            prev = poses[idx - 1]
            dx = float(t[0, 3] - prev[0, 3])
            dy = float(t[1, 3] - prev[1, 3])
            speed = float(np.hypot(dx, dy) / max(dt_s, 1e-3))
            yaw = float(np.arctan2(t[1, 0], t[0, 0]))
            prev_yaw = float(np.arctan2(prev[1, 0], prev[0, 0]))
            d_yaw = (yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi
            steer = float(np.clip(d_yaw, -0.8, 0.8))
            return VehicleState(speed_mps=speed, steering_angle_rad=steer)
        return VehicleState(speed_mps=10.0, steering_angle_rad=0.0)

    def _load_poses(self) -> Optional[np.ndarray]:
        if getattr(self, "_poses_cache_seq", None) == self.sequence and getattr(self, "_poses", None) is not None:
            return self._poses
        self._poses_cache_seq = self.sequence
        self._poses = None
        if self.is_zip:
            return None
        candidates = [
            os.path.join(self.dataset_path, "dataset", "poses", f"{self.sequence}.txt"),
            os.path.join(self.dataset_path, "poses", f"{self.sequence}.txt"),
            os.path.join(self.dataset_path, "dataset", "sequences", self.sequence, "poses.txt"),
            os.path.join(self.dataset_path, "sequences", self.sequence, "poses.txt"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                raw = np.loadtxt(p)
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                if raw.shape[1] >= 12:
                    mats = raw[:, :12].reshape(-1, 3, 4)
                    self._poses = mats
                    return self._poses
        return None

    def __len__(self) -> int:
        return len(self.bin_files)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.get_scan(idx)

    def get_scan(self, idx: int) -> np.ndarray:
        """
        Loads a single scan as an (N, 5) float32 array: [x, y, z, intensity, class_id].
        """
        if idx < 0 or idx >= len(self.bin_files):
            raise IndexError(f"Scan index {idx} out of range (0 to {len(self.bin_files)-1})")

        bin_f = self.bin_files[idx]
        label_f = self.label_files[idx]

        if self.is_zip and self._zip_ref:
            raw_bytes = self._zip_ref.read(bin_f)
            points = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 4)
            n_points = points.shape[0]

            scan = np.empty((n_points, 5), dtype=np.float32)
            scan[:, :4] = points

            if label_f:
                label_bytes = self._zip_ref.read(label_f)
                raw_labels = np.frombuffer(label_bytes, dtype=np.uint32)
                scan[:, 4] = (raw_labels & 0xFFFF).astype(np.float32)
            else:
                scan[:, 4] = -1.0  # Unclassified

            return scan
        else:
            return load_kitti_scan(bin_f, label_f)

    def close(self) -> None:
        """Closes any open zip archive handles."""
        if self._zip_ref is not None:
            self._zip_ref.close()
            self._zip_ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
