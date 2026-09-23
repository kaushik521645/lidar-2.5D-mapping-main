"""
Kalman Filter Object Tracking & Active Footprint Erasure (Anti-Ghosting).

Implements 2D constant-velocity Kalman filtering for dynamic obstacle tracking (Class 3)
and active footprint erasure to eliminate stale/ghost occupancy trails behind moving objects.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any
import numpy as np

from src.grid.grid_types import GridCell, GridMap
from src.grid.resolution import (
    TIER0_RES, TIER1_RES, TIER2_RES,
    TIER0_OUTER_M, TIER1_OUTER_M,
    TIER0_MAX_RING, TIER1_MAX_RING,
)

MAX_COAST_FRAMES = 5
# 5.0m gate: at 22 m/s (highway) a vehicle moves 2.2m per 0.1s frame — 3.0m was
# marginally safe but any clustering jitter or slight frame-time variation would
# push the centroid outside the old gate and spawn a new track ID.  5.0m is still
# far too small to accidentally associate a car with a pedestrian at city scale.
MAX_ASSOC_DIST_M = 5.0


@dataclass
class TrackedObject:
    """
    State of a tracked dynamic object in world/sensor coordinates.
    """
    track_id: int
    position_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    predicted_next_xy: Tuple[float, float]
    class_id: int
    frames_since_seen: int
    bbox_size_xy: Tuple[float, float] = (3.5, 1.8)  # Length, width in meters
    state: str = 'TENTATIVE'  # 'TENTATIVE' or 'CONFIRMED'
    hits: int = 1
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32) * 0.1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "position_xy": [float(self.position_xy[0]), float(self.position_xy[1])],
            "velocity_xy": [float(self.velocity_xy[0]), float(self.velocity_xy[1])],
            "predicted_next_xy": [float(self.predicted_next_xy[0]), float(self.predicted_next_xy[1])],
            "class_id": int(self.class_id),
            "frames_since_seen": int(self.frames_since_seen),
            "bbox_size_xy": [float(self.bbox_size_xy[0]), float(self.bbox_size_xy[1])],
            "state": self.state,
            "hits": self.hits,
        }


def update_tracks(
    existing_tracks: List[TrackedObject],
    dynamic_cluster_centroids: List[Tuple[float, float, int]],
    dt: float
) -> List[TrackedObject]:
    """
    Updates tracks using a constant-velocity Kalman filter and greedy nearest-neighbor association.
    """
    # 1. Predict
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    Q = np.diag([0.05, 0.05, 0.5, 0.5]).astype(np.float32)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    R = np.eye(2, dtype=np.float32) * 0.1
    
    # Store predictions so we can associate
    predicted_states = []
    for t in existing_tracks:
        x = np.array([t.position_xy[0], t.position_xy[1], t.velocity_xy[0], t.velocity_xy[1]], dtype=np.float32)
        P = t.covariance
        
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        t.predicted_next_xy = (float(x_pred[0]), float(x_pred[1]))
        predicted_states.append((x_pred, P_pred))

    # 2. Global Optimal Association (Hungarian) with Mahalanobis Distance
    from scipy.optimize import linear_sum_assignment

    n_tracks = len(existing_tracks)
    n_meas = len(dynamic_cluster_centroids)

    MAX_COST = 1e9
    cost_matrix = np.full((n_tracks, n_meas), MAX_COST)

    for i, t in enumerate(existing_tracks):
        x_pred, P_pred = predicted_states[i]
        # CONFIRMED tracks get a wider Mahalanobis gate (chi-sq 99.9% = 13.82)
        # so an established track can survive momentary centroid jitter without
        # being orphaned and replaced by a new track ID next frame.
        # TENTATIVE tracks keep the strict 99% gate (chi-sq = 9.21) to avoid
        # false initialisation from noise.
        gate_chi2 = 13.82 if t.state == 'CONFIRMED' else 9.21
        for j, c in enumerate(dynamic_cluster_centroids):
            z_meas = np.array([c[0], c[1]], dtype=np.float32)
            y_res = z_meas - (H @ x_pred)
            S = H @ P_pred @ H.T + R

            # Mahalanobis distance squared
            inv_S = np.linalg.inv(S)
            m_dist_sq = float(y_res.T @ inv_S @ y_res)

            # Gate: Mahalanobis + Euclidean safety bound
            e_dist = np.hypot(y_res[0], y_res[1])
            if m_dist_sq < gate_chi2 and e_dist <= MAX_ASSOC_DIST_M:
                cost_matrix[i, j] = m_dist_sq
                
    matched_tracks = set()
    matched_centroids = set()
    active_tracks = []
    
    if n_tracks > 0 and n_meas > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < MAX_COST:
                matched_tracks.add(i)
                matched_centroids.add(j)
                
                # Kalman Update
                t = existing_tracks[i]
                x_pred, P_pred = predicted_states[i]
                z_meas = np.array([dynamic_cluster_centroids[j][0], dynamic_cluster_centroids[j][1]], dtype=np.float32)
                
                y_res = z_meas - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.solve(S, np.eye(2, dtype=np.float32))
                
                x_updated = x_pred + K @ y_res
                I_KH = np.eye(4, dtype=np.float32) - K @ H
                P_updated = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
                
                t.position_xy = (float(x_updated[0]), float(x_updated[1]))
                t.velocity_xy = (float(x_updated[2]), float(x_updated[3]))
                t.covariance = P_updated
                t.frames_since_seen = 0
                t.class_id = dynamic_cluster_centroids[j][2]
                
                # Lifecycle Management
                t.hits += 1
                if t.hits >= 3:
                    t.state = 'CONFIRMED'
                    
                active_tracks.append(t)
            
    # 3. Unmatched existing tracks (coast)
    for i, t in enumerate(existing_tracks):
        if i not in matched_tracks:
            x_pred, P_pred = predicted_states[i]
            # Use predicted position as the ghost estimate
            t.position_xy = t.predicted_next_xy
            t.velocity_xy = (float(x_pred[2]), float(x_pred[3]))
            t.covariance = P_pred
            t.frames_since_seen += 1
            if t.frames_since_seen <= MAX_COAST_FRAMES:
                active_tracks.append(t)
                
    # 4. Unmatched centroids (new tracks)
    next_id = 1
    if existing_tracks:
        next_id = max(t.track_id for t in existing_tracks) + 1
        
    for j, c in enumerate(dynamic_cluster_centroids):
        if j not in matched_centroids:
            new_track = TrackedObject(
                track_id=next_id,
                position_xy=(c[0], c[1]),
                velocity_xy=(0.0, 0.0),
                predicted_next_xy=(c[0], c[1]),
                class_id=c[2],
                frames_since_seen=0,
                covariance=np.diag([0.1, 0.1, 100.0, 100.0]).astype(np.float32)
            )
            active_tracks.append(new_track)
            next_id += 1
            
    # 5. Compute predicted_next_xy for all active tracks based on their newly updated states
    for t in active_tracks:
        x_curr = np.array([t.position_xy[0], t.position_xy[1], t.velocity_xy[0], t.velocity_xy[1]], dtype=np.float32)
        x_next = F @ x_curr
        t.predicted_next_xy = (float(x_next[0]), float(x_next[1]))
            
    return active_tracks


def cluster_dynamic_detections(
    cell_centers: List[Tuple[float, float, int]],
    cluster_dist_m: float = 2.5,
    max_cluster_span_m: float = 14.0,
    max_cells_per_cluster: int = 200,
) -> List[Tuple[float, float, int]]:
    """
    Clusters neighboring dynamic cells into unified object centroids.
    Filters out oversized clusters (e.g. continuous walls, fences, road sheets).
    """
    if cell_centers is None or len(cell_centers) == 0:
        return []

    n = len(cell_centers)
    if n == 1:
        return [(cell_centers[0][0], cell_centers[0][1], cell_centers[0][2])]

    pts = np.array([[x, y] for x, y, _ in cell_centers], dtype=np.float32)
    classes = np.array([c for _, _, c in cell_centers], dtype=np.int32)

    try:
        from scipy.spatial import KDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        tree = KDTree(pts)
        pairs = tree.query_pairs(cluster_dist_m, output_type='ndarray')

        if len(pairs) == 0:
            clusters: List[Tuple[float, float, int]] = []
            for i in range(n):
                clusters.append((float(pts[i, 0]), float(pts[i, 1]), int(classes[i])))
            return clusters

        row_ind = np.concatenate([pairs[:, 0], pairs[:, 1]])
        col_ind = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(row_ind), dtype=bool)

        adj_csr = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
        n_components, labels = connected_components(
            csgraph=adj_csr, directed=False, return_labels=True
        )

        clusters: List[Tuple[float, float, int]] = []
        for c_id in range(n_components):
            mask = (labels == c_id)
            c_pts = pts[mask]
            
            # Reject oversized non-vehicle clusters
            if len(c_pts) > max_cells_per_cluster:
                continue
            span_x = float(np.max(c_pts[:, 0]) - np.min(c_pts[:, 0]))
            span_y = float(np.max(c_pts[:, 1]) - np.min(c_pts[:, 1]))
            if span_x > max_cluster_span_m or span_y > max_cluster_span_m:
                continue

            centroid_x = float(np.mean(c_pts[:, 0]))
            centroid_y = float(np.mean(c_pts[:, 1]))
            cls_val = int(classes[mask][0])
            clusters.append((centroid_x, centroid_y, cls_val))

        return clusters
    except Exception:
        clusters = []
        visited = set()
        for i in range(n):
            if i in visited:
                continue
            c_indices = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if adj_matrix[i, j]:
                    c_indices.append(j)
                    visited.add(j)
            c_pts = pts[c_indices]
            if len(c_pts) <= max_cells_per_cluster:
                span_x = float(np.max(c_pts[:, 0]) - np.min(c_pts[:, 0]))
                span_y = float(np.max(c_pts[:, 1]) - np.min(c_pts[:, 1]))
                if span_x <= max_cluster_span_m and span_y <= max_cluster_span_m:
                    clusters.append((float(np.mean(c_pts[:, 0])), float(np.mean(c_pts[:, 1])), int(classes[c_indices[0]])))
        return clusters


class KalmanTrackerManager:
    """
    Compatibility wrapper retaining state for older pipelines.
    """
    def __init__(
        self,
        max_coast_frames: int = 5,
        distance_gate_m: float = 3.0,
        dt_s: float = 0.1,
    ):
        self.dt_s = dt_s
        self.tracks: List[TrackedObject] = []
        self.prev_footprints: Dict[int, Set[Tuple[int, int]]] = {}

    def update_tracks(
        self,
        detections_xy: List[Tuple[float, float, int]],
    ) -> List[TrackedObject]:
        self.tracks = update_tracks(self.tracks, detections_xy, self.dt_s)
        # Only return CONFIRMED tracks to the downstream pipeline
        return [t for t in self.tracks if t.state == 'CONFIRMED']


def erase_vacated_footprints(
    grid_map: GridMap,
    tracks: List[TrackedObject],
    prev_footprints: Dict[int, Set[Tuple[int, int]]],
    frame_id: int,
    grid_engine=None,
) -> Tuple[GridMap, Dict[int, Set[Tuple[int, int]]], int]:
    """
    Active footprint erasure to remove ghost dynamic obstacles.

    A "ghost erasure" is counted when a cell that was occupied by a dynamic
    object in the previous frame is either:
      (a) completely absent from the new grid_map (the most common case — the
          grid is rebuilt from scratch each frame, so a vacated cell simply
          disappears), or
      (b) still present but reclassified away from class 3 (terrain or static
          won the majority vote in the new frame).

    Previously the counter was always 0 because case (a) was silently skipped
    by the ``if k in grid_map`` guard: vacated cells are gone, so the check
    never reached the increment.  This fix counts them explicitly.
    """
    updated_footprints: Dict[int, Set] = {}
    erased_count = 0

    # 1. Gather current footprints (dynamic cells updated this frame near each track)
    # Optimization: Extract dynamic cells once instead of scanning 44k cells for every track
    dyn_cells_with_pos = []
    for k, cell in grid_map.items():
        if cell.semantic_class == 3 and cell.last_updated_frame == frame_id:
            if grid_engine and hasattr(grid_engine, 'get_cell_spatial_center'):
                x, y, _, _ = grid_engine.get_cell_spatial_center(k[1], k[2], cell.resolution_tier)
                dyn_cells_with_pos.append((k, x, y))
            else:
                dyn_cells_with_pos.append((k, None, None))

    for track in tracks:
        current_footprint: Set = set()
        tx, ty = track.position_xy[0], track.position_xy[1]
        max_dist = max(track.bbox_size_xy)
        max_dist_sq = max_dist * max_dist
        for k, cx, cy in dyn_cells_with_pos:
            if cx is not None:
                dx = cx - tx
                dy = cy - ty
                if (dx * dx + dy * dy) <= max_dist_sq:
                    current_footprint.add(k)
            else:
                current_footprint.add(k)
        updated_footprints[track.track_id] = current_footprint

    # 2. Erase previous footprints not in current footprints
    for track_id, prev_set in prev_footprints.items():
        curr_set = updated_footprints.get(track_id, set())
        to_erase = prev_set - curr_set
        for k in to_erase:
            if k not in grid_map:
                # Case (a): cell was vacated entirely — the most common ghost.
                # The grid is rebuilt per-frame; absence means the object moved on.
                # Count this as one erased ghost cell.
                erased_count += 1
            elif grid_map[k].semantic_class == 3:
                # Case (b): cell still exists but hasn't been claimed by this track
                # this frame, and wasn't updated in the current frame — it's stale.
                if grid_map[k].last_updated_frame < frame_id:
                    grid_map[k].semantic_class = 0
                    grid_map[k].elevation_obstacle_top = None
                    erased_count += 1
            else:
                # Case (c): cell exists but was already reclassified (terrain/static won).
                # Count as erased since the ghost occupancy is gone.
                erased_count += 1

    return grid_map, updated_footprints, erased_count
