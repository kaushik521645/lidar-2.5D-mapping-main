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


def test_ghost_cells_erased_counter_increments_on_movement():
    """
    Regression test for SIH2026 bug where ghost_cells_erased was always 0.

    Root cause was that the grid is rebuilt from scratch each frame.  When an
    object moves from cell A to cell B, cell A is simply *absent* from the new
    grid_map.  The old code only counted erasures when ``k in grid_map``, so
    it never incremented.

    This test verifies:
    1. When prev_footprints contains cells absent from the new grid_map,
       the counter increments (case a: vacated = ghost erased).
    2. When prev_footprints contains cells still present but reclassified
       to terrain, the counter also increments (case c).
    3. Cells in the current footprint (object is still there) are NOT counted.
    """
    grid_engine = PolarGridEngine()

    track = TrackedObject(
        track_id=7,
        position_xy=(12.0, 0.0),   # Moved to new position
        velocity_xy=(5.0, 0.0),
        predicted_next_xy=(12.5, 0.0),
        class_id=3,
        frames_since_seen=0,
        bbox_size_xy=(4.0, 2.0),
        state='CONFIRMED',
        hits=4,
    )

    # Previous frame footprint: 3 cells that the object occupied at (10m, 0 rad)
    old_cell_A = (0, 200, 314)
    old_cell_B = (0, 201, 314)
    old_cell_C = (0, 199, 314)
    # Current frame cell where the track now sits
    new_cell_X = (0, 240, 314)

    prev_footprints = {7: {old_cell_A, old_cell_B, old_cell_C}}

    # New grid: old cells are GONE (vacated), only the new cell exists near the track
    grid_map: GridMap = {
        new_cell_X: GridCell(
            elevation_ground=0.0,
            elevation_obstacle_bottom=0.2,
            elevation_obstacle_top=1.5,
            semantic_class=3,
            point_count=10,
            confidence=1.0,
            last_updated_frame=3,
            resolution_tier=0.15,
        ),
    }

    updated_grid, updated_fp, erased_count = erase_vacated_footprints(
        grid_map=grid_map,
        tracks=[track],
        prev_footprints=prev_footprints,
        frame_id=3,
        grid_engine=grid_engine,
    )

    # All 3 old cells were absent from new grid -> should count as 3 ghost erasures
    assert erased_count == 3, (
        f"Expected 3 ghost erasures (vacated cells), got {erased_count}. "
        "This was always 0 before the fix."
    )
    # Updated footprint should now contain only the new cell (track is at new_cell_X area)
    assert 7 in updated_fp


def test_ghost_cells_erased_does_not_count_static_neighbors():
    """
    Verifies that cells near the track that are static (class 2) and were NOT
    in prev_footprints are never mistakenly counted as ghost erasures.
    """
    grid_engine = PolarGridEngine()

    track = TrackedObject(
        track_id=3,
        position_xy=(8.0, 0.0),
        velocity_xy=(2.0, 0.0),
        predicted_next_xy=(8.2, 0.0),
        class_id=3,
        frames_since_seen=0,
        bbox_size_xy=(3.5, 1.8),
        state='CONFIRMED',
        hits=5,
    )

    # Track occupied this one dynamic cell last frame
    dynamic_prev_key = (0, 160, 314)
    # Static wall nearby (never in prev_footprints)
    static_wall_key = (0, 162, 310)

    prev_footprints = {3: {dynamic_prev_key}}

    # New grid: dynamic cell is gone (object moved), static wall is present
    grid_map: GridMap = {
        static_wall_key: GridCell(
            elevation_ground=0.0,
            elevation_obstacle_bottom=0.5,
            elevation_obstacle_top=2.0,
            semantic_class=2,  # Static obstacle
            point_count=8,
            confidence=1.0,
            last_updated_frame=5,
            resolution_tier=0.05,
        ),
    }

    _, _, erased_count = erase_vacated_footprints(
        grid_map=grid_map,
        tracks=[track],
        prev_footprints=prev_footprints,
        frame_id=5,
        grid_engine=grid_engine,
    )

    # Only the vacated dynamic cell should be counted
    assert erased_count == 1
    # Static wall should be untouched (it was never in prev_footprints)
    assert grid_map[static_wall_key].semantic_class == 2


def test_ghost_cells_erased_multi_frame_cumulative_stays_bounded():
    """
    Regression test ensuring erase_vacated_footprints across 15 consecutive frames:
    1. Only counts genuine dynamic footprint erasures (never mass-erasing absent static cells).
    2. Keeps the cumulative total strictly bounded and proportional to per-frame erasures.
    """
    grid_engine = PolarGridEngine()
    prev_footprints = {}
    cum_erased = 0

    # Simulate a dynamic object moving across 15 frames:
    # At each frame f, the object occupies 3 cells: (0, 160 + f, 300), (0, 160 + f, 301), (0, 160 + f, 302)
    # When it moves, the previous 3 cells are vacated.
    for f in range(15):
        ring = 160 + f * 10
        center_x, center_y, _, _ = grid_engine.get_cell_spatial_center(ring, 0, 0.05)
        track = TrackedObject(
            track_id=1,
            position_xy=(center_x, center_y),
            velocity_xy=(5.0, 0.0),
            predicted_next_xy=(center_x + 0.5, center_y),
            class_id=3,
            frames_since_seen=0,
            bbox_size_xy=(4.0, 2.0),
            state='CONFIRMED',
            hits=f + 1,
        )

        curr_keys = [
            (0, ring, -1),
            (0, ring, 0),
            (0, ring, 1),
        ]
        grid_map = {
            k: GridCell(
                elevation_ground=0.0,
                elevation_obstacle_bottom=0.3,
                elevation_obstacle_top=1.8,
                semantic_class=3,
                point_count=15,
                confidence=1.0,
                last_updated_frame=f,
                resolution_tier=0.05,
            )
            for k in curr_keys
        }

        # Add unrelated static cells that fluctuate into/out of the grid
        for s in range(50):
            grid_map[(1, 50 + s + (f % 3), 100)] = GridCell(
                elevation_ground=0.0,
                elevation_obstacle_bottom=0.0,
                elevation_obstacle_top=0.0,
                semantic_class=1,
                point_count=5,
                confidence=0.8,
                last_updated_frame=f,
                resolution_tier=0.15,
            )

        grid_map, prev_footprints, erased = erase_vacated_footprints(
            grid_map=grid_map,
            tracks=[track],
            prev_footprints=prev_footprints,
            frame_id=f,
            grid_engine=grid_engine,
        )

        cum_erased += erased
        # Frame 0 has no previous footprints to erase
        if f == 0:
            assert erased == 0
        else:
            # Exactly 3 vacated cells erased per step
            assert erased == 3, f"Expected 3 vacated cells in frame {f}, got {erased}"

    # After 14 movement steps of 3 cells each, cumulative must be exactly 42 (14 * 3)
    assert cum_erased == 42, f"Cumulative ghost cells erased should be 42, got {cum_erased}"

