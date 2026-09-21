"""
Unit tests for Object Tracking and Active Footprint Erasure Anti-Ghosting (Milestones 9 and 10).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking.kalman_tracker import (
    KalmanTrackerManager,
    TrackedObject,
    erase_vacated_footprints,
)
from src.grid.grid_types import GridCell, GridMap
from src.grid.grid_engine import PolarGridEngine


def test_kalman_tracker_convergence():
    """
    Validates Milestone 9:
    A moving vehicle with constant velocity (vx=10.0 m/s, vy=0) is tracked across frames.
    Track ID persists and estimated velocity converges towards 10.0 m/s.
    """
    tracker = KalmanTrackerManager(dt_s=0.1)
    true_vx = 10.0
    true_x = 15.0

    track_id = None
    for frame in range(10):
        meas_x = true_x + np.random.normal(0.0, 0.05)
        meas_y = 2.0 + np.random.normal(0.0, 0.05)
        detections = [(meas_x, meas_y, 3)]  # Class 3: Dynamic

        tracks = tracker.update_tracks(detections)
        if frame >= 2:
            assert len(tracks) == 1
            t = tracks[0]

            if track_id is None:
                track_id = t.track_id
            else:
                assert t.track_id == track_id  # Track ID persists

            if frame == 9:
                assert t.velocity_xy[0] == pytest.approx(10.0, abs=1.5)

        true_x += true_vx * 0.1

    # After 10 frames, velocity estimate should be close to 10 m/s
    final_track = tracker.tracks[0]
    assert final_track.velocity_xy[0] == pytest.approx(10.0, abs=1.5)


def test_active_footprint_erasure_antighosting():
    """
    Validates Milestone 10:
    Simulated moving object moves from cell A in Frame 1 to cell B in Frame 2.
    Cell A (vacated) has its dynamic occupancy erased immediately without ghosting.
    Static obstacle in Cell S is strictly untouched.
    """
    grid_engine = PolarGridEngine()

    # Track 1 representing a vehicle
    track = TrackedObject(
        track_id=1,
        position_xy=(10.0, 0.0),
        velocity_xy=(5.0, 0.0),
        predicted_next_xy=(10.5, 0.0),
        class_id=3,
        frames_since_seen=0,
        bbox_size_xy=(4.0, 2.0),
    )

    cell_a_key = (0, 200, 314)  # Old position (10m, 0 rad)
    cell_b_key = (0, 205, 314)  # New position (10.75m, 0 rad)
    cell_static_key = (0, 100, 50)  # Static wall nearby

    # Frame 1: Track occupied Cell A
    prev_footprints = {1: {cell_a_key}}

    # Frame 2: Grid has:
    # - Cell A still temporarily marked dynamic from naive accumulation
    # - Cell B now occupied by track at (10.75m, 0 rad)
    # - Static obstacle at cell_static_key
    track.position_xy = (10.75, 0.0)

    grid_map: GridMap = {
        cell_a_key: GridCell(
            elevation_ground=0.0,
            elevation_obstacle_bottom=0.2,
            elevation_obstacle_top=1.5,
            semantic_class=3,  # Ghost dynamic
            point_count=5,
            confidence=1.0,
            last_updated_frame=1,
            resolution_tier=0.15,
        ),
        cell_b_key: GridCell(
            elevation_ground=0.0,
            elevation_obstacle_bottom=0.2,
            elevation_obstacle_top=1.5,
            semantic_class=3,  # Active dynamic
            point_count=10,
            confidence=1.0,
            last_updated_frame=2,
            resolution_tier=0.15,
        ),
        cell_static_key: GridCell(
            elevation_ground=0.0,
            elevation_obstacle_bottom=None,
            elevation_obstacle_top=2.0,
            semantic_class=2,  # Static Obstacle
            point_count=8,
            confidence=1.0,
            last_updated_frame=2,
            resolution_tier=0.05,
        ),
    }

    # Execute active footprint erasure
    updated_grid, updated_footprints, erased_count = erase_vacated_footprints(
        grid_map=grid_map,
        tracks=[track],
        prev_footprints=prev_footprints,
        frame_id=2,
        grid_engine=grid_engine,
    )

    # Assertions:
    # 1. Vacated Cell A had dynamic occupancy cleared (reverted to class 0 ground)
    assert updated_grid[cell_a_key].semantic_class == 0
    assert updated_grid[cell_a_key].elevation_obstacle_top is None
    assert erased_count >= 1

    # 2. Active Cell B remains Dynamic (Class 3)
    assert updated_grid[cell_b_key].semantic_class == 3
    assert updated_grid[cell_b_key].elevation_obstacle_top == 1.5

    # 3. Static obstacle remains Class 2 (untouched)
    assert updated_grid[cell_static_key].semantic_class == 2
    assert updated_grid[cell_static_key].elevation_obstacle_top == 2.0
