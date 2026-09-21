"""
Vectorized Rule-Based / Heuristic Segmentation Fallback for FoveaMap.

Executes real-time CPU point cloud classification when deep learning weights
are unavailable or during high-throughput baseline runs:
  - RANSAC Ground Plane Fitting
  - Terrain splitting: Drivable Surface (Class 0) vs Non-Drivable / Sidewalk / Curb (Class 1)
  - Height & Density Bounding: Static Obstacles (Class 2) vs Dynamic Objects (Class 3)
"""

import os
from typing import Optional
import numpy as np
import yaml


def fit_ground_plane_ransac(
    points_xyz: np.ndarray,
    distance_threshold: float = 0.15,
    max_iterations: int = 50,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    RANSAC plane fitting on bottom candidate points to estimate road ground plane [a, b, c, d].
    Plane equation: a*x + b*y + c*z + d = 0.

    Args:
        seed: Random seed for reproducibility. None = non-deterministic (production use).
    """
    if len(points_xyz) < 3:
        return np.array([0.0, 0.0, 1.0, 1.5], dtype=np.float32)

    # Candidate points for ground (lower quartile of z)
    z_thresh = np.percentile(points_xyz[:, 2], 35)
    candidates = points_xyz[points_xyz[:, 2] <= z_thresh]
    if len(candidates) < 10:
        candidates = points_xyz

    best_inliers_count = 0
    best_plane = np.array([0.0, 0.0, 1.0, 1.5], dtype=np.float32)

    n_cand = len(candidates)
    rng = np.random.default_rng(seed)

    for _ in range(max_iterations):
        sample_indices = rng.choice(n_cand, size=3, replace=False)
        p1, p2, p3 = candidates[sample_indices, :3]

        # Normal vector
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue

        normal = normal / norm_len
        # Ensure normal points upwards (c > 0)
        if normal[2] < 0:
            normal = -normal

        # Plane must be approximately horizontal (z-component dominant, e.g. c > 0.85)
        if normal[2] < 0.80:
            continue

        d = -np.dot(normal, p1)
        # Vectorized distance of all candidates to plane
        distances = np.abs(np.dot(candidates[:, :3], normal) + d)
        inliers_count = np.sum(distances < distance_threshold)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

    return best_plane


def _load_heuristic_config(config_path: str = "configs/default.yaml") -> dict:
    """Loads perception config for heuristic fallback."""
    defaults = {"road_half_width_m": 4.2}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                p_cfg = cfg.get("perception", {})
                if "road_half_width_m" in p_cfg:
                    defaults["road_half_width_m"] = float(p_cfg["road_half_width_m"])
        except Exception:
            pass
    return defaults

_HEURISTIC_CFG = _load_heuristic_config()


def heuristic_segment_points(
    points_xyz: np.ndarray,
    road_half_width_m: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Vectorized rule-based segmentation returning 4-class labels:
      0: Drivable Terrain
      1: Non-Drivable Terrain
      2: Static Obstacle
      3: Dynamic Object
      
    Args:
        points_xyz: (N, 3) or (N, >=3) float array of LiDAR coordinates [x, y, z, ...].
        road_half_width_m: Estimated lateral corridor boundary for drivable road surface.
        
    Returns:
        (N,) int32 array with values in {0, 1, 2, 3}.
    """
    if road_half_width_m is None:
        road_half_width_m = _HEURISTIC_CFG["road_half_width_m"]

    n_points = len(points_xyz)
    if n_points == 0:
        return np.empty(0, dtype=np.int32)

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    labels = np.full(n_points, -1, dtype=np.int32)

    # 1. Fit ground plane
    plane = fit_ground_plane_ransac(points_xyz[:, :3], seed=seed)
    a, b, c, d = plane

    # Compute signed height above estimated ground plane
    # h = (a*x + b*y + c*z + d) / c  (since normal is normalized, h ~ distance above ground)
    plane_z = -(a * x + b * y + d) / max(abs(c), 1e-4)
    height_above_ground = z - plane_z

    # 2. Identify Ground Surface Band (-0.25m <= height <= 0.20m)
    ground_mask = (height_above_ground >= -0.25) & (height_above_ground <= 0.20)

    # Drivable Terrain (Class 0): Ground points inside roadway corridor (|y| <= road_half_width)
    # with flat surface profile (height_above_ground between -0.15m and +0.08m)
    drivable_mask = ground_mask & (np.abs(y) <= road_half_width_m) & (height_above_ground <= 0.08)
    labels[drivable_mask] = 0

    # Non-Drivable Terrain (Class 1): Sidewalks, curbs, verges, roadside terrain
    # Either ground points outside roadway (|y| > road_half_width) or elevated curb step (+0.08m to +0.25m)
    non_drivable_ground = ground_mask & (~drivable_mask)
    labels[non_drivable_ground] = 1

    # 3. Non-Ground Objects (height_above_ground > 0.20m)
    non_ground_mask = ~ground_mask

    # Separate dynamic objects (cars, pedestrians) from static obstacles (poles, buildings, trees)
    # Dynamic objects criteria:
    # - Height span: 0.20m <= height_above_ground <= 2.20m
    # - Situated within or immediately adjacent to roadway corridor (|y| <= 7.0m)
    dynamic_candidate = non_ground_mask & (height_above_ground <= 2.20) & (np.abs(y) <= 7.0)

    # Static Obstacles (Class 2):
    # - Tall structures (height > 2.20m: poles, buildings, walls, trees)
    # - OR structures located far from roadway (|y| > 7.0m: buildings, fences)
    static_mask = non_ground_mask & ((height_above_ground > 2.20) | (np.abs(y) > 7.0))
    labels[static_mask] = 2

    # Dynamic Objects (Class 3):
    labels[dynamic_candidate] = 3

    # Any remaining unassigned points defaulted to static obstacle
    labels[labels == -1] = 2

    return labels
