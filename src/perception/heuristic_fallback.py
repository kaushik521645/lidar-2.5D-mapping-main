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
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    labels = np.full(n_points, 2, dtype=np.int32)

    # 1. Primary RANSAC plane fit near vehicle (base orientation)
    plane = fit_ground_plane_ransac(points_xyz[:, :3], seed=seed)
    a, b, c, d = plane
    c_safe = max(abs(c), 1e-4) * (1.0 if c >= 0 else -1.0)
    plane_z = -(a * x + b * y + d) / c_safe
    h_plane = z - plane_z

    # 2. Multi-Sector Radial Ground Profile
    # Roadways undulate and slope across range r, so we combine RANSAC base plane
    # with a radial-sector elevation profile to track ground elevation changes accurately.
    num_sectors = 24
    sector_idx = np.clip(
        np.floor((theta + np.pi) / (2.0 * np.pi) * num_sectors).astype(np.int32),
        0, num_sectors - 1
    )
    r_bin_size = 2.5
    r_bins = np.clip(np.floor(r / r_bin_size).astype(np.int32), 0, 40)

    # Candidate points for ground per bin: points with h_plane near zero
    candidate_mask = (h_plane >= -0.6) & (h_plane <= 0.5)

    ground_grid = np.zeros((num_sectors, 41), dtype=np.float32)

    for s in range(num_sectors):
        last_z_offset = 0.0
        for b in range(41):
            mask = candidate_mask & (sector_idx == s) & (r_bins == b)
            n_in_bin = np.sum(mask)
            if n_in_bin >= 4:
                # 10th percentile offset from plane
                bin_offset = float(np.percentile(h_plane[mask], 10))
                # Enforce physical continuity (< 18 degree ground slope relative to plane)
                if abs(bin_offset - last_z_offset) <= (0.32 * r_bin_size):
                    last_z_offset = bin_offset
            ground_grid[s, b] = last_z_offset

    z_ground_offset = ground_grid[sector_idx, r_bins]
    height_above_ground = h_plane - z_ground_offset

    # 3. Identify Ground Surface Band (-0.28m <= height <= 0.22m)
    ground_mask = (height_above_ground >= -0.28) & (height_above_ground <= 0.22)

    # Drivable Terrain (Class 0): Ground points inside roadway corridor (|y| <= road_half_width)
    # with flat surface profile (|height_above_ground| <= 0.12m)
    drivable_mask = ground_mask & (np.abs(y) <= road_half_width_m) & (height_above_ground <= 0.12)
    labels[drivable_mask] = 0

    # Non-Drivable Terrain (Class 1): Sidewalks, curbs, verges, roadside terrain
    # Either ground points outside roadway (|y| > road_half_width) or elevated curb step (+0.12m to +0.28m)
    non_drivable_ground = ground_mask & (~drivable_mask)
    labels[non_drivable_ground] = 1

    # 4. Non-Ground Objects (height_above_ground > 0.22m)
    non_ground_mask = ~ground_mask

    # Default for all non-ground objects is Static Obstacle (Class 2)
    # (poles, traffic signs, walls, buildings, trees, fences, guardrails, roadside clutter)
    labels[non_ground_mask] = 2

    # Dynamic Objects (Class 3):
    # Strictly isolated vehicle/pedestrian scale objects within or immediately adjacent to roadway
    # - Within active roadway corridor (|y| <= road_half_width_m + 0.3m)
    # - True obstacle height: 0.35m <= height_above_ground <= 2.20m
    # - Range horizon: 1.8m <= r <= 45.0m (excludes ego vehicle self-returns r < 1.8m)
    dynamic_candidate = (
        non_ground_mask &
        (r >= 1.8) &
        (r <= 45.0) &
        (np.abs(y) <= (road_half_width_m + 0.3)) &
        (height_above_ground >= 0.35) &
        (height_above_ground <= 2.20)
    )
    labels[dynamic_candidate] = 3

    return labels
