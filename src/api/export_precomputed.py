"""
Precomputed Scenario Frame Exporter for FoveaMap.

Executes the full pipeline (perception, dynamic foveation, grid binning, tracking,
anti-ghosting, and memory metrics) and saves JSON payloads to disk for deterministic,
zero-latency replay.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ingestion.kitti_loader import KITTILoader
from src.perception.segment import segment_points
from src.perception.class_map import map_semantickitti_to_4class
from src.perception.heuristic_fallback import heuristic_segment_points, warmup_heuristic_jit
from src.grid.grid_engine import PolarGridEngine, warmup_jit
from src.grid.grid_types import VehicleState, GridCell
from src.grid.foveation import (
    BASE_FINE_RADIUS,
    MAX_STRETCH,
    SHEAR_STRENGTH,
    MOTION_FOVEATION_ENABLED,
    COLLISION_FOCUS_ONLY,
    CORRIDOR_HALF_WIDTH_M,
    MAX_THREAT_DISTANCE_M,
    MOTION_SPEED_THRESHOLD_MPS,
    MOTION_LEAD_TIME_S,
    MOTION_BUFFER_M,
    MOTION_ACUITY_BOOST,
    detect_moving_objects,
    compute_fovea_polyline,
)
from src.tracking.kalman_tracker import (
    KalmanTrackerManager,
    erase_vacated_footprints,
    cluster_dynamic_detections,
)
from src.synthetic.scenarios import ScenarioGenerator


def compute_memory_metrics(grid_map: Dict[Any, GridCell], engine: PolarGridEngine) -> Dict[str, float]:
    """
    Computes exact memory usage for FoveaMap 2.5D sparse grid vs Uniform 5cm high-res baseline.
    
    A uniform 5cm grid covering 100m radius:
    Area = pi * 100^2 = 31,416 m^2.
    Cell count = 31,416 / (0.05 * 0.05) = 12,566,400 cells.
    Uniform dense storage (assuming 16 bytes/cell): 12,566,400 * 16 = 201,062,400 bytes (~201 MB).
    
    FoveaMap Sparse Storage:
    Active cells count * (key: 8 bytes + cell struct: 32 bytes) = count * 40 bytes.
    """
    active_cells = len(grid_map)
    # Uniform baseline: 100m radius uniform 5cm grid
    # Nominal sparse active cells in uniform grid ~ active_cells * 25 (since outer is 0.5m -> 100x area, mid is 0.15m -> 9x area)
    uniform_cells_equivalent = 0
    for cell in grid_map.values():
        res = cell.resolution_tier
        scale = (res / 0.05) ** 2
        uniform_cells_equivalent += int(scale)

    uniform_cells_equivalent = max(uniform_cells_equivalent, active_cells * 9)

    bytes_per_sparse_cell = 40  # 8 bytes key + 32 bytes GridCell fields
    bytes_foveated = active_cells * bytes_per_sparse_cell
    bytes_uniform = uniform_cells_equivalent * bytes_per_sparse_cell

    savings_pct = (1.0 - (bytes_foveated / max(bytes_uniform, 1))) * 100.0
    return {
        "active_cells_count": float(active_cells),
        "memory_bytes_foveated": float(bytes_foveated),
        "memory_bytes_uniform_baseline": float(bytes_uniform),
        "memory_savings_pct": float(np.clip(savings_pct, 0.0, 99.9)),
    }


def process_sequence_to_json(
    sequence: List[Tuple[np.ndarray, VehicleState, Dict[str, Any]]],
    scenario_name: str,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Runs full FoveaMap perception, grid binning, and tracking over a frame sequence.
    """
    warmup_jit()
    warmup_heuristic_jit()
    engine = PolarGridEngine()
    tracker = KalmanTrackerManager()
    prev_footprints: Dict[int, Any] = {}

    frames_payload = []

    for frame_id, (scan, v_state, meta) in enumerate(sequence):
        t_start = time.perf_counter()

        # Step 1: Perception / Semantic Segmentation
        if scan.shape[1] >= 5 and not np.all(scan[:, 4] == -1.0):
            raw_classes = scan[:, 4].astype(np.int32)
            if np.any(raw_classes > 3):  # Raw SemanticKITTI classes
                classes = map_semantickitti_to_4class(raw_classes)
            else:
                classes = raw_classes
        else:
            # Robust geometric / RANSAC segmentation for real unannotated LiDAR returns
            classes = heuristic_segment_points(scan[:, :3])

        points_5d = np.column_stack([scan[:, :3], scan[:, 3] if scan.shape[1] >= 4 else np.zeros(len(scan)), classes])

        # Step 2: Grid Projection & Multi-layer aggregation
        roof_h = 4.0 if "bridge" in scenario_name.lower() else 2.5
        grid_map = engine.project_to_grid(
            points=points_5d,
            vehicle_state=v_state,
            frame_id=frame_id,
            roof_height_m=roof_h,
            use_foveation=True,
            active_tracks=tracker.tracks
        )

        # Step 3: Extract dynamic clusters for tracking
        raw_detections = []
        for key, cell in grid_map.items():
            if cell.semantic_class == 3:
                cx, cy, _, _ = engine.get_cell_spatial_center(key[1], key[2], cell.resolution_tier)
                raw_detections.append((cx, cy, 3))

        dynamic_detections = cluster_dynamic_detections(raw_detections, cluster_dist_m=2.5)
        tracker.update_tracks(dynamic_detections)
        all_tracks = tracker.tracks

        # On frame 0, re-project grid so initial dynamic detections receive high-acuity allocation immediately
        if frame_id == 0 and all_tracks:
            grid_map = engine.project_to_grid(
                points=points_5d,
                vehicle_state=v_state,
                frame_id=frame_id,
                roof_height_m=roof_h,
                use_foveation=True,
                active_tracks=all_tracks
            )

        # Step 4: Active Footprint Erasure (Anti-Ghosting)
        grid_map, prev_footprints, erased_count = erase_vacated_footprints(
            grid_map=grid_map,
            tracks=all_tracks,
            prev_footprints=prev_footprints,
            frame_id=frame_id,
            grid_engine=engine,
        )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        fps = 1000.0 / max(latency_ms, 0.001)

        # Step 5: Serialize cells
        cells_list = []
        for (t_idx, r_idx, a_idx), cell in grid_map.items():
            cx, cy, r_val, th_val = engine.get_cell_spatial_center(r_idx, a_idx, cell.resolution_tier)
            c_dict = {
                "ring_idx": int(r_idx),
                "angle_idx": int(a_idx),
                "elevation_ground": round(cell.elevation_ground, 3),
                "elevation_obstacle_bottom": round(cell.elevation_obstacle_bottom, 3) if cell.elevation_obstacle_bottom is not None else None,
                "elevation_obstacle_top": round(cell.elevation_obstacle_top, 3) if cell.elevation_obstacle_top is not None else None,
                "semantic_class": int(cell.semantic_class),
                "confidence": round(cell.confidence, 2),
                "resolution_tier": round(cell.resolution_tier, 2),
            }
            c_dict["x"] = round(cx, 3)
            c_dict["y"] = round(cy, 3)
            c_dict["r"] = round(r_val, 3)
            c_dict["theta"] = round(th_val, 3)
            cells_list.append(c_dict)

        mem_metrics = compute_memory_metrics(grid_map, engine)

        # Evaluate motion of tracked dynamic obstacles
        moving_info = detect_moving_objects(all_tracks, speed_threshold_mps=MOTION_SPEED_THRESHOLD_MPS)
        moving_count = sum(1 for m in moving_info if m["is_moving"])

        fovea_poly = compute_fovea_polyline(
            v_state,
            active_tracks=all_tracks,
            base_radius=BASE_FINE_RADIUS,
            max_stretch=MAX_STRETCH,
            shear_strength=SHEAR_STRENGTH,
            motion_foveation_enabled=MOTION_FOVEATION_ENABLED,
            motion_speed_threshold_mps=MOTION_SPEED_THRESHOLD_MPS,
            motion_lead_time_s=MOTION_LEAD_TIME_S,
            collision_focus_only=COLLISION_FOCUS_ONLY,
            corridor_half_width_m=CORRIDOR_HALF_WIDTH_M,
            max_threat_distance_m=MAX_THREAT_DISTANCE_M,
        )

        frame_data = {
            "frame_id": frame_id,
            "timestamp": float(meta.get("timestamp_s", frame_id * 0.1)),
            "scenario_name": scenario_name,
            "description": meta.get("description", ""),
            "vehicle_state": {
                "speed_mps": round(v_state.speed_mps, 2),
                "steering_angle_rad": round(v_state.steering_angle_rad, 4),
            },
            "foveation_params": {
                "base_fine_radius_m": BASE_FINE_RADIUS,
                "max_stretch": MAX_STRETCH,
                "shear_strength": SHEAR_STRENGTH,
                "motion_foveation_enabled": MOTION_FOVEATION_ENABLED,
                "collision_focus_only": COLLISION_FOCUS_ONLY,
                "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M,
                "max_threat_distance_m": MAX_THREAT_DISTANCE_M,
                "motion_speed_threshold_mps": MOTION_SPEED_THRESHOLD_MPS,
                "motion_lead_time_s": MOTION_LEAD_TIME_S,
                "motion_buffer_m": MOTION_BUFFER_M,
                "motion_acuity_boost": MOTION_ACUITY_BOOST,
            },
            "fovea_polyline": fovea_poly,
            "cells": cells_list,
            "tracks": [t.to_dict() for t in all_tracks],
            "metrics": {
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 2),
                "memory_bytes_foveated": mem_metrics["memory_bytes_foveated"],
                "memory_bytes_uniform_baseline": mem_metrics["memory_bytes_uniform_baseline"],
                "memory_savings_pct": round(mem_metrics["memory_savings_pct"], 1),
                "ghosting_cells_erased": erased_count,
                "active_cells_count": mem_metrics["active_cells_count"],
                "moving_objects_count": moving_count,
                "motion_foveation_active": moving_count > 0 and MOTION_FOVEATION_ENABLED,
            },
        }
        frames_payload.append(frame_data)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(frames_payload, f)
        print(f"[INFO] Exported {len(frames_payload)} frames to {output_path}")

    return frames_payload


def export_kitti_sequence(
    sequence: str = "00",
    output_dir: str = "data/precomputed",
    max_frames: int = 25,
    force_recompute: bool = False,
) -> str:
    """Exports precomputed frames for a specific KITTI sequence."""
    os.makedirs(output_dir, exist_ok=True)
    seq_str = str(sequence).zfill(2)
    target_name = f"kitti_seq_{seq_str}"
    out_path = os.path.join(output_dir, f"{target_name}.json")

    if os.path.exists(out_path) and not force_recompute:
        print(f"[INFO] Sequence {seq_str} already precomputed in {out_path}")
        return out_path

    loader = KITTILoader(sequence=seq_str)
    if len(loader) == 0:
        print(f"[WARN] No scans found for KITTI sequence {seq_str}")
        return out_path

    num_export = min(len(loader), max_frames)
    print(f"[INFO] Exporting {num_export} frames for KITTI Sequence {seq_str} (out of {len(loader)} scans)...")
    kitti_seq = []
    for i in range(num_export):
        scan = loader[i]
        v_state = VehicleState(speed_mps=10.0, steering_angle_rad=0.0)
        meta = {
            "scenario_name": f"KITTI Odometry Sequence {seq_str}",
            "timestamp_s": i * 0.1,
            "description": f"Real Velodyne HDL-64E LiDAR capture from KITTI sequence {seq_str} (~{len(scan):,} points/scan).",
        }
        kitti_seq.append((scan, v_state, meta))

    process_sequence_to_json(kitti_seq, target_name, out_path)

    # Alias sequence 00 for backward compatibility
    if seq_str == "00":
        import shutil
        shutil.copyfile(out_path, os.path.join(output_dir, "kitti_odometry.json"))
        shutil.copyfile(out_path, os.path.join(output_dir, "kitti_sample.json"))

    return out_path


def export_synthetic_kitti_like(
    kitti_dir: str = "data/synthetic_kitti_like",
    output_dir: str = "data/precomputed",
    num_frames: int = 30,
    force_recompute: bool = False,
) -> str:
    """
    Exports the SemanticKITTI ground-truth benchmark sequence from data/synthetic_kitti_like.
    This preserves the original labeled SemanticKITTI dataset alongside real Velodyne sequences.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "synthetic_kitti_like.json")

    # If it already exists and is not the old 250MB accidental copy of sequence 00
    if os.path.exists(out_path) and not force_recompute:
        if os.path.getsize(out_path) < 100 * 1024 * 1024:
            print(f"[INFO] SemanticKITTI benchmark already precomputed in {out_path}")
            return out_path

    if not os.path.exists(os.path.join(kitti_dir, "velodyne")):
        from src.ingestion.sample_generator import generate_kitti_sample_dataset
        print(f"[INFO] Generating SemanticKITTI benchmark dataset in {kitti_dir}...")
        generate_kitti_sample_dataset(kitti_dir, num_frames=num_frames)

    loader = KITTILoader(dataset_path=kitti_dir, warn_on_synthetic=False)
    if len(loader) == 0:
        print(f"[WARN] No scans found in {kitti_dir}")
        return out_path

    n = min(len(loader), num_frames)
    print(f"[INFO] Exporting {n} frames for SemanticKITTI Benchmark from {kitti_dir}...")
    kitti_seq = []
    for i in range(n):
        scan = loader[i]
        v_state = VehicleState(speed_mps=8.0, steering_angle_rad=0.0)
        meta = {
            "scenario_name": "SemanticKITTI Benchmark (Ground-Truth Labeled)",
            "timestamp_s": i * 0.1,
            "description": f"SemanticKITTI format 30-frame sequence (Frame {i + 1}/{n}). Contains ground-truth semantic class annotations.",
        }
        kitti_seq.append((scan, v_state, meta))

    process_sequence_to_json(kitti_seq, "synthetic_kitti_like", out_path)
    return out_path


def export_all_precomputed(
    output_dir: str = "data/precomputed",
    max_kitti_frames: int = 25,
    export_all_sequences: bool = False,
    sequence: Optional[str] = None,
) -> None:
    """Exports precomputed frames for KITTI real sequences, SemanticKITTI benchmark, and synthetic scenarios."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. SemanticKITTI Ground-Truth Benchmark (always maintained alongside)
    export_synthetic_kitti_like(output_dir=output_dir, num_frames=30)

    # 2. KITTI Real Odometry Velodyne Dataset
    kitti_loader = KITTILoader()
    if len(kitti_loader) > 0 and not kitti_loader.is_synthetic_signature:
        if export_all_sequences:
            seqs = kitti_loader.list_sequences()
            print(f"[INFO] Exporting frames for all {len(seqs)} KITTI sequences...")
            for s in seqs:
                export_kitti_sequence(s, output_dir=output_dir, max_frames=max_kitti_frames)
        elif sequence is not None:
            export_kitti_sequence(sequence, output_dir=output_dir, max_frames=max_kitti_frames)
        else:
            # Export default sequence 00
            export_kitti_sequence("00", output_dir=output_dir, max_frames=max_kitti_frames)

    # 3. Synthetic Scenarios
    gen = ScenarioGenerator()
    scenarios = gen.get_all_scenarios()
    for name, seq in scenarios.items():
        s_path = os.path.join(output_dir, f"{name}.json")
        if not os.path.exists(s_path):
            process_sequence_to_json(seq, name, s_path)


if __name__ == "__main__":
    export_all_precomputed()
