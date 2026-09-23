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

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True)
def _jit_sector_ground_profile(h_plane, sector_idx, r_bins, candidate_mask, num_sectors=24, num_bins=41, r_bin_size=2.5):
    counts = np.zeros((num_sectors, num_bins), dtype=np.int32)
    n_pts = len(h_plane)
    for i in range(n_pts):
        if candidate_mask[i]:
            s = sector_idx[i]
            b = r_bins[i]
            counts[s, b] += 1

    offsets = np.zeros((num_sectors, num_bins), dtype=np.int32)
    total_cand = 0
    for s in range(num_sectors):
        for b in range(num_bins):
            offsets[s, b] = total_cand
            total_cand += counts[s, b]

    cand_vals = np.empty(total_cand, dtype=np.float32)
    cur_pos = np.copy(offsets)
    for i in range(n_pts):
        if candidate_mask[i]:
            s = sector_idx[i]
            b = r_bins[i]
            pos = cur_pos[s, b]
            cand_vals[pos] = h_plane[i]
            cur_pos[s, b] = pos + 1

    ground_grid = np.zeros((num_sectors, num_bins), dtype=np.float32)
    slope_limit = 0.32 * r_bin_size

    for s in range(num_sectors):
        last_z_offset = 0.0
        for b in range(num_bins):
            cnt = counts[s, b]
            if cnt >= 4:
                start = offsets[s, b]
                end = start + cnt
                vals = cand_vals[start:end]
                vals.sort()
                idx_float = 0.10 * (cnt - 1)
                lo = int(idx_float)
                hi = min(lo + 1, cnt - 1)
                weight = idx_float - lo
                bin_offset = vals[lo] * (1.0 - weight) + vals[hi] * weight
                if abs(bin_offset - last_z_offset) <= slope_limit:
                    last_z_offset = bin_offset
            ground_grid[s, b] = last_z_offset

    return ground_grid


_HEURISTIC_JIT_WARMED = False

def warmup_heuristic_jit():
    global _HEURISTIC_JIT_WARMED
    if NUMBA_AVAILABLE and not _HEURISTIC_JIT_WARMED:
        dummy_h = np.array([0.0, 0.1, -0.1], dtype=np.float32)
        dummy_s = np.array([0, 0, 1], dtype=np.int32)
        dummy_b = np.array([0, 1, 0], dtype=np.int32)
        dummy_m = np.array([True, True, True], dtype=np.bool_)
        _jit_sector_ground_profile(dummy_h, dummy_s, dummy_b, dummy_m, 24, 41, 2.5)
        _HEURISTIC_JIT_WARMED = True


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

    if NUMBA_AVAILABLE:
        h_plane_f32 = np.ascontiguousarray(h_plane, dtype=np.float32)
        ground_grid = _jit_sector_ground_profile(
            h_plane_f32, sector_idx, r_bins, candidate_mask, num_sectors, 41, float(r_bin_size)
        )
    else:
        ground_grid = np.zeros((num_sectors, 41), dtype=np.float32)
        for s in range(num_sectors):
            last_z_offset = 0.0
            for b in range(41):
                mask = candidate_mask & (sector_idx == s) & (r_bins == b)
                n_in_bin = np.sum(mask)
                if n_in_bin >= 4:
                    bin_offset = float(np.percentile(h_plane[mask], 10))
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
    # Strictly isolated vehicle/pedestrian scale objects within or immediately adjacent to roadway.
    # Lacking temporal baseline/motion history, single-frame geometry must NOT default
    # obstacles to dynamic. Tall structures (buildings, poles, trees, walls) whose column
    # extent exceeds passenger vehicle height (> 2.30m) stay Static Obstacle (Class 2).
    #
    # Vectorized 2D column max-height profile (0.5m grid) to distinguish vehicle-scale obstacles:
    bin_size = 0.5
    gx = np.floor(x / bin_size).astype(np.int32)
    gy = np.floor(y / bin_size).astype(np.int32)
    col_id = (gx + 1000).astype(np.int64) * 2000 + (gy + 1000).astype(np.int64)

    sort_order = np.argsort(col_id)
    sorted_cols = col_id[sort_order]
    sorted_h = height_above_ground[sort_order]

    boundaries = np.where(sorted_cols[:-1] != sorted_cols[1:])[0] + 1
    boundaries = np.concatenate(([0], boundaries, [len(sorted_cols)]))

    max_h_per_col = np.maximum.reduceat(sorted_h, boundaries[:-1])
    col_lengths = np.diff(boundaries)
    point_max_h = np.repeat(max_h_per_col, col_lengths)

    inv_sort = np.empty_like(sort_order)
    inv_sort[sort_order] = np.arange(len(sort_order))
    column_max_height = point_max_h[inv_sort]

    # Vehicle-height ceiling constraint: true vehicles do not exceed 2.30m above ground
    vehicle_height_mask = column_max_height <= 2.30

    # Ego vehicle self-return exclusion: returns on ego vehicle hood/mirrors (r < 2.5m)
    ego_exclusion_mask = r >= 2.5

    dynamic_candidate = (
        non_ground_mask &
        ego_exclusion_mask &
        (r <= 45.0) &
        (np.abs(y) <= (road_half_width_m + 0.3)) &
        (height_above_ground >= 0.35) &
        (height_above_ground <= 2.20) &
        vehicle_height_mask
    )
    labels[dynamic_candidate] = 3

    return labels
