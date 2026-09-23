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
def _jit_robust_z_split_single(z_slice, min_gap=0.4):
    n = len(z_slice)
    if n == 0:
        return 0.0, 0.0, 0.0, False

    min_z = z_slice[0]
    max_z = z_slice[0]
    for idx in range(1, n):
        val = z_slice[idx]
        if val < min_z:
            min_z = val
        if val > max_z:
            max_z = val

    if (max_z - min_z) < min_gap:
        return min_z, 0.0, 0.0, False

    # 1D 2-Means (Lloyd's algorithm, up to 5 iterations)
    c1 = min_z
    c2 = max_z

    for _ in range(5):
        sum1 = 0.0
        cnt1 = 0
        sum2 = 0.0
        cnt2 = 0
        for idx in range(n):
            val = z_slice[idx]
            d1 = abs(val - c1)
            d2 = abs(val - c2)
            if d1 < d2:
                sum1 += val
                cnt1 += 1
            else:
                sum2 += val
                cnt2 += 1

        if cnt1 == 0 or cnt2 == 0:
            break

        new_c1 = sum1 / cnt1
        new_c2 = sum2 / cnt2
        if abs(new_c1 - c1) < 1e-4 and abs(new_c2 - c2) < 1e-4:
            c1 = new_c1
            c2 = new_c2
            break
        c1 = new_c1
        c2 = new_c2

    # Partition clusters
    sum1 = 0.0
    cnt1 = 0
    min1 = 1e9
    max1 = -1e9
    sum2 = 0.0
    cnt2 = 0
    min2 = 1e9
    max2 = -1e9

    for idx in range(n):
        val = z_slice[idx]
        d1 = abs(val - c1)
        d2 = abs(val - c2)
        if d1 < d2:
            sum1 += val
            cnt1 += 1
            if val < min1: min1 = val
            if val > max1: max1 = val
        else:
            sum2 += val
            cnt2 += 1
            if val < min2: min2 = val
            if val > max2: max2 = val

    if cnt1 == 0 or cnt2 == 0:
        return min_z, 0.0, 0.0, False

    mean1 = sum1 / cnt1
    mean2 = sum2 / cnt2

    if mean1 < mean2:
        gap = min2 - max1
        if gap > min_gap:
            return min1, min2, max2, True
    else:
        gap = min1 - max2
        if gap > min_gap:
            return min2, min1, max1, True

    return min_z, 0.0, 0.0, False


@njit(fastmath=True)
def _jit_aggregate_cells(sorted_z, sorted_r, boundaries, min_gap=0.4):
    n_cells = len(boundaries) - 1
    gnd_arr = np.empty(n_cells, dtype=np.float64)
    obs_bot_arr = np.empty(n_cells, dtype=np.float64)
    obs_top_arr = np.empty(n_cells, dtype=np.float64)
    has_obs_arr = np.empty(n_cells, dtype=np.bool_)
    conf_arr = np.empty(n_cells, dtype=np.float64)
    point_count_arr = np.empty(n_cells, dtype=np.int32)

    for i in range(n_cells):
        start = boundaries[i]
        end = boundaries[i + 1]
        cnt = end - start
        point_count_arr[i] = cnt

        z_slice = sorted_z[start:end]
        r_slice = sorted_r[start:end]

        gnd, obs_b, obs_t, has_obs = _jit_robust_z_split_single(z_slice, min_gap)
        gnd_arr[i] = gnd
        obs_bot_arr[i] = obs_b
        obs_top_arr[i] = obs_t
        has_obs_arr[i] = has_obs

        sum_r = 0.0
        for j in range(cnt):
            sum_r += r_slice[j]
        mean_r = max(1.0, sum_r / cnt)
        expected_points = max(1.0, 150.0 / mean_r)
        conf = min(1.0, cnt / expected_points)
        conf_arr[i] = conf

    return gnd_arr, obs_bot_arr, obs_top_arr, has_obs_arr, conf_arr, point_count_arr


_JIT_WARMED = False

def warmup_jit():
    """Warms up the Numba JIT compiler on a tiny synthetic array to eliminate first-frame latency."""
    global _JIT_WARMED
    if NUMBA_AVAILABLE and not _JIT_WARMED:
        dummy_z = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        dummy_r = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        dummy_b = np.array([0, 3], dtype=np.int64)
        _jit_aggregate_cells(dummy_z, dummy_r, dummy_b, 0.4)
        _JIT_WARMED = True


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
    n_cells = len(boundaries) - 1

    # Step (a): Global majority-class computation via vectorized bincount
    NUM_CLASSES = 5  # shifted classes: -1..3 -> 0..4
    point_cell_idx = np.repeat(np.arange(n_cells), np.diff(boundaries))
    shifted_classes = sorted_classes + 1
    encoded_cls = point_cell_idx * NUM_CLASSES + shifted_classes
    cls_counts = np.bincount(encoded_cls, minlength=n_cells * NUM_CLASSES).reshape((n_cells, NUM_CLASSES))
    best_classes = cls_counts.argmax(axis=1) - 1
    
    # Step (b): Numba-JIT robust_z_split and per-cell aggregation
    sorted_z_c = np.ascontiguousarray(sorted_z, dtype=np.float64)
    sorted_r_c = np.ascontiguousarray(sorted_r, dtype=np.float64)
    boundaries_c = boundaries.astype(np.int64)

    gnd_arr, obs_bot_arr, obs_top_arr, has_obs_arr, conf_arr, pt_cnt_arr = _jit_aggregate_cells(
        sorted_z_c, sorted_r_c, boundaries_c, 0.4
    )

    grid_map: GridMap = {}

    for i in range(n_cells):
        start = boundaries[i]
        cell_id = sorted_cell_ids[start]
        t_idx = int(cell_id // (MAX_RINGS * MAX_ANGLES))
        rem = cell_id % (MAX_RINGS * MAX_ANGLES)
        r_idx = int(rem // MAX_ANGLES)
        a_idx = int(rem % MAX_ANGLES) - offset

        obs_b = float(obs_bot_arr[i]) if has_obs_arr[i] else None
        obs_t = float(obs_top_arr[i]) if has_obs_arr[i] else None

        grid_map[(t_idx, r_idx, a_idx)] = GridCell(
            elevation_ground=float(gnd_arr[i]),
            elevation_obstacle_bottom=obs_b,
            elevation_obstacle_top=obs_t,
            semantic_class=int(best_classes[i]),
            point_count=int(pt_cnt_arr[i]),
            confidence=float(conf_arr[i]),
            last_updated_frame=frame_id,
            resolution_tier=float(sorted_res[start])
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
