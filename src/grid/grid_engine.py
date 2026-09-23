"""
Core spatial data structure for a variable-resolution 2.5D LiDAR map.
"""

import numpy as np
import math
from typing import Dict, Any, Tuple, Optional, List

from src.grid.grid_types import GridCell, GridMap, VehicleState
from src.grid.resolution import get_resolution, angular_step, DEFAULT_RESOLUTION_TIERS
from src.grid.foveation import fine_radius_at_angle, BASE_FINE_RADIUS_M
from src.grid.hysteresis import BoundaryHysteresisManager


def robust_z_split(z_array: np.ndarray, min_gap: float = 0.4) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Sorts Z-values and uses 1D 2-Means clustering to robustly find ground and overhang layers.
    Returns (ground_z, overhang_bottom_z, overhang_top_z).
    """
    if len(z_array) == 0:
        return 0.0, None, None

    min_z = np.min(z_array)
    max_z = np.max(z_array)

    if max_z - min_z < min_gap:
        return min_z, None, None

    # 1D 2-Means (Lloyd's algorithm, up to 5 iterations)
    c1, c2 = min_z, max_z
    mask = np.zeros(len(z_array), dtype=bool)

    for _ in range(5):
        dist1 = np.abs(z_array - c1)
        dist2 = np.abs(z_array - c2)
        new_mask = dist1 < dist2
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

        if np.sum(mask) == 0 or np.sum(~mask) == 0:
            break

        c1 = np.mean(z_array[mask])
        c2 = np.mean(z_array[~mask])

    pts1 = z_array[mask]
    pts2 = z_array[~mask]

    if len(pts1) == 0 or len(pts2) == 0:
        return min_z, None, None

    if np.mean(pts1) < np.mean(pts2):
        gnd_pts, obs_pts = pts1, pts2
    else:
        gnd_pts, obs_pts = pts2, pts1

    gap = np.min(obs_pts) - np.max(gnd_pts)
    if gap > min_gap:
        return float(np.min(gnd_pts)), float(np.min(obs_pts)), float(np.max(obs_pts))

    return float(min_z), None, None


def project_to_grid(
    points: np.ndarray,
    vehicle_state: VehicleState,
    frame_id: int = 0,
    roof_height_m: float = 2.5,
    tiers: Optional[List[Tuple[float, float]]] = None,
    base_fine_radius: float = BASE_FINE_RADIUS_M,
    active_tracks: Optional[list] = None,
    hysteresis_manager: Optional[BoundaryHysteresisManager] = None,
) -> GridMap:
    """
    Projects (N, 5) point cloud to a variable-resolution 2.5D GridMap.
    """
    if tiers is None:
        tiers = DEFAULT_RESOLUTION_TIERS
        
    if len(points) == 0:
        return {}
        
    # Vectorized computation
    # 1. Z-CLIPPING
    valid_mask = points[:, 2] <= roof_height_m
    pts = points[valid_mask]
    
    if len(pts) == 0:
        return {}
        
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    # intensity = pts[:, 3]
    classes = pts[:, 4].astype(int)
    
    # 2. Convert to polar
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    
    # 3. Determine resolution tier
    # fine_radius_at_angle supports vectorized numpy arrays
    fine_radii = fine_radius_at_angle(
        theta, vehicle_state, base_radius=base_fine_radius, active_tracks=active_tracks
    )

    # Damp frame-to-frame jitter of the dynamic foveation boundary so points near
    # it don't flip resolution tier (and therefore flicker) every frame.
    if hysteresis_manager is not None:
        fine_radii = hysteresis_manager.smooth_fine_radius(theta, fine_radii)

    res = np.zeros_like(r)
    fine_mask = r < fine_radii
    res[fine_mask] = tiers[0][1]
    
    # For non-fine points, check the remaining tiers
    non_fine_mask = ~fine_mask
    if np.any(non_fine_mask):
        r_nf = r[non_fine_mask]
        res_nf = np.full_like(r_nf, tiers[-1][1])
        # Skip the first tier
        for max_range, tier_res in tiers[1:]:
            mask = r_nf < max_range
            # only update those not already assigned a smaller tier
            update_mask = mask & (res_nf == tiers[-1][1])
            res_nf[update_mask] = tier_res
        res[non_fine_mask] = res_nf

    # Determine tier index for uniqueness
    # Tier 0: 0.05, Tier 1: 0.15, Tier 2: 0.50
    tier_idx = np.zeros_like(r, dtype=np.int64)
    tier_idx[res == tiers[-1][1]] = len(tiers) - 1
    for i, (max_range, tier_res) in enumerate(tiers[:-1]):
        tier_idx[res == tier_res] = i
        
    # 4. Compute angular step and bin
    clamped_r = np.maximum(r, 0.5)
    ang_step = res / clamped_r
    
    ring_idx = (r / res).astype(np.int64)
    angle_idx = (theta / ang_step).astype(np.int64)
    
    # 5. Accumulate per-cell via Vectorization
    MAX_ANGLES = 1000000
    MAX_RINGS = 1000000
    # Offset angle_idx to handle negative indices safely
    offset = 500000
    # Create a globally unique cell_id by combining tier_idx, ring_idx, angle_idx
    cell_ids = tier_idx * (MAX_RINGS * MAX_ANGLES) + ring_idx * MAX_ANGLES + (angle_idx + offset)
    
    sort_idx = np.argsort(cell_ids)
    sorted_cell_ids = cell_ids[sort_idx]
    sorted_z = z[sort_idx]
    sorted_classes = classes[sort_idx]
    sorted_res = res[sort_idx]
    sorted_r = r[sort_idx]
    
    boundaries = np.where(sorted_cell_ids[:-1] != sorted_cell_ids[1:])[0] + 1
    boundaries = np.concatenate(([0], boundaries, [len(pts)]))
    
    grid_map: GridMap = {}
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        
        cell_id = sorted_cell_ids[start]
        t_idx = int(cell_id // (MAX_RINGS * MAX_ANGLES))
        rem = cell_id % (MAX_RINGS * MAX_ANGLES)
        r_idx = int(rem // MAX_ANGLES)
        a_idx = int(rem % MAX_ANGLES) - offset
        
        cell_z = sorted_z[start:end]
        cell_cls = sorted_classes[start:end]
        cell_res = float(sorted_res[start])
        cell_r = sorted_r[start:end]
        
        gnd, obs_bot, obs_top = robust_z_split(cell_z, min_gap=0.4)
        
        shifted_cls = cell_cls + 1
        best_cls = int(np.bincount(shifted_cls).argmax()) - 1
        
        mean_r = max(1.0, float(np.mean(cell_r)))
        expected_points = max(1.0, 150.0 / mean_r)
        conf = min(1.0, len(cell_z) / expected_points)
        
        # Use a 3-tuple key to prevent collision
        grid_map[(t_idx, r_idx, a_idx)] = GridCell(
            elevation_ground=gnd,
            elevation_obstacle_bottom=obs_bot,
            elevation_obstacle_top=obs_top,
            semantic_class=best_cls,
            point_count=len(cell_z),
            confidence=conf,
            last_updated_frame=frame_id,
            resolution_tier=cell_res
        )
        
    return grid_map


def grid_memory_bytes(grid: GridMap) -> int:
    """Approximate byte size as len(grid) * 40"""
    return len(grid) * 40


def uniform_grid_baseline_bytes(radius_m: float = 100.0, cell_size_m: float = 0.05) -> int:
    """
    Computes memory for a uniform grid covering a circle.
    Area = pi * r^2. Cells = Area / (cell_size^2).
    """
    area = math.pi * (radius_m ** 2)
    cells = int(area / (cell_size_m ** 2))
    return cells * 40


# --- BACKWARD COMPATIBILITY WRAPPER ---
# To prevent breaking existing pipeline code that expects PolarGridEngine.
class PolarGridEngine:
    def __init__(self, config_path: str = "configs/default.yaml"):
        # One hysteresis manager per engine instance so its per-sector boundary
        # history persists across consecutive project_to_grid() calls (i.e. across
        # frames of a sequence), which is required for it to actually suppress
        # flicker. Previously this state was never created/threaded through, so
        # the anti-flicker logic in hysteresis.py had no effect on the pipeline.
        self.hysteresis_manager = BoundaryHysteresisManager()

    def project_to_grid(
        self,
        points: np.ndarray,
        vehicle_state: Optional[VehicleState] = None,
        frame_id: int = 0,
        roof_height_m: float = 2.5,
        use_foveation: bool = True,
        active_tracks: Optional[list] = None
    ) -> GridMap:
        if vehicle_state is None:
            vehicle_state = VehicleState(0.0, 0.0)
        return project_to_grid(
            points, vehicle_state, frame_id=frame_id, roof_height_m=roof_height_m,
            active_tracks=active_tracks, hysteresis_manager=self.hysteresis_manager,
        )

    def get_cell_spatial_center(self, ring_idx: int, angle_idx: int, res: float) -> Tuple[float, float, float, float]:
        r = (ring_idx + 0.5) * res
        theta = (angle_idx + 0.5) * angular_step(res, r)
        return r * math.cos(theta), r * math.sin(theta), r, theta
