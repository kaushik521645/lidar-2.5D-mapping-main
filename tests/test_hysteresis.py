"""
Unit tests for Boundary Hysteresis & Sensor Degradation (Milestone 7).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.hysteresis import BoundaryHysteresisManager, SensorDegradationTracker
from src.grid.grid_types import GridCell, GridMap


def test_boundary_hysteresis_anti_flicker():
    """
    Validates that a point oscillating between r=9.9m and r=10.1m across frames
    does NOT flip resolution tier every frame due to 0.5m hysteresis margin.
    """
    mgr = BoundaryHysteresisManager(hysteresis_margin_m=0.5)
    cell_key = (199, 10)
    boundary = 10.0  # Nominal boundary between 0.05m and 0.15m

    # Frame 1: starts at r=9.9m (< 10.0m) -> Tier 0 (0.05m)
    tier_f1 = mgr.get_tier_with_hysteresis(9.9, boundary, cell_key)
    assert tier_f1 == pytest.approx(0.05)

    # Frame 2: moves to r=10.1m (crosses nominal 10.0m, but 10.1 < 10.0 + 0.5 = 10.5m)
    # MUST stay at 0.05m to prevent flicker
    tier_f2 = mgr.get_tier_with_hysteresis(10.1, boundary, cell_key)
    assert tier_f2 == pytest.approx(0.05)

    # Frame 3: moves to r=9.9m -> stays 0.05m
    tier_f3 = mgr.get_tier_with_hysteresis(9.9, boundary, cell_key)
    assert tier_f3 == pytest.approx(0.05)

    # Frame 4: moves far out to r=10.8m (> 10.5m) -> now switches to Tier 1 (0.15m)
    tier_f4 = mgr.get_tier_with_hysteresis(10.8, boundary, cell_key)
    assert tier_f4 == pytest.approx(0.15)

    # Frame 5: moves slightly inward to r=9.8m (below 10.0m, but 9.8 > 10.0 - 0.5 = 9.5m)
    # MUST stay at 0.15m until it crosses below 9.5m
    tier_f5 = mgr.get_tier_with_hysteresis(9.8, boundary, cell_key)
    assert tier_f5 == pytest.approx(0.15)

    # Frame 6: moves deep inward to r=9.2m (< 9.5m) -> now switches back to Tier 0 (0.05m)
    tier_f6 = mgr.get_tier_with_hysteresis(9.2, boundary, cell_key)
    assert tier_f6 == pytest.approx(0.05)


def test_sensor_degradation_fallback():
    """
    Validates that a cell with < 3 points over 3 consecutive frames is marked
    confidence=0.3 and forced to coarsest resolution tier 0.50m.
    """
    tracker = SensorDegradationTracker(min_points=3, consecutive_frames_threshold=3)

    key = (0, 50, 100)
    # Frame 1: 1 point (sparse)
    grid_f1 = {key: GridCell(0.0, None, None, 0, point_count=1, confidence=0.1, last_updated_frame=1, resolution_tier=0.05)}
    grid_f1 = tracker.process_grid_degradation(grid_f1, frame_id=1)
    assert grid_f1[key].confidence == 0.1  # Not yet 3 consecutive frames

    # Frame 2: 2 points (sparse)
    grid_f2 = {key: GridCell(0.0, None, None, 0, point_count=2, confidence=0.2, last_updated_frame=2, resolution_tier=0.05)}
    grid_f2 = tracker.process_grid_degradation(grid_f2, frame_id=2)
    assert grid_f2[key].confidence == 0.2

    # Frame 3: 1 point (sparse, 3rd consecutive frame)
    grid_f3 = {key: GridCell(0.0, None, None, 0, point_count=1, confidence=0.1, last_updated_frame=3, resolution_tier=0.05)}
    grid_f3 = tracker.process_grid_degradation(grid_f3, frame_id=3)
    assert grid_f3[key].confidence == pytest.approx(0.3)
    assert grid_f3[key].resolution_tier == pytest.approx(0.50)
