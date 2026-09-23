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


def test_smooth_fine_radius_suppresses_small_oscillation():
    """
    Validates that smooth_fine_radius() suppresses frame-to-frame boundary
    jitter smaller than the hysteresis_margin_m.

    A boundary oscillating between 10.0m and 10.4m (delta = 0.4m) should be
    held constant at the first seen value, since 0.4m < 0.5m (the margin).
    This directly tests the fix for cell classification flickering at the
    fine/coarse tier boundary.
    """
    mgr = BoundaryHysteresisManager(hysteresis_margin_m=0.5)
    sector_width_rad = 0.02

    # Angular sector at ~0 radians
    theta = np.array([0.0])

    # Frame 1: boundary at 10.0m
    r1 = mgr.smooth_fine_radius(theta, np.array([10.0]), sector_width_rad)
    assert float(r1[0]) == pytest.approx(10.0, abs=1e-6), "Frame 1 should accept 10.0m"

    # Frame 2: boundary moves to 10.4m (delta = 0.4m < 0.5m margin)
    # Should be suppressed — returned as prev value (10.0m)
    r2 = mgr.smooth_fine_radius(theta, np.array([10.4]), sector_width_rad)
    assert float(r2[0]) == pytest.approx(10.0, abs=1e-6), (
        f"Frame 2 boundary should stay at 10.0m (suppressed), got {float(r2[0]):.3f}m"
    )

    # Frame 3: boundary jumps to 10.8m (delta from 10.0 = 0.8m > 0.5m margin)
    # This change is large enough to pass through
    r3 = mgr.smooth_fine_radius(theta, np.array([10.8]), sector_width_rad)
    assert float(r3[0]) == pytest.approx(10.8, abs=1e-6), (
        f"Frame 3 large jump should be allowed through, got {float(r3[0]):.3f}m"
    )

    # Frame 4: boundary drops to 10.4m (delta from 10.8 = 0.4m < 0.5m margin)
    # Suppressed again
    r4 = mgr.smooth_fine_radius(theta, np.array([10.4]), sector_width_rad)
    assert float(r4[0]) == pytest.approx(10.8, abs=1e-6), (
        f"Frame 4 should stay at 10.8m (suppressed), got {float(r4[0]):.3f}m"
    )


def test_z_clip_hysteresis_suppresses_single_frame_spike():
    """
    Validates ZClipHysteresisFilter: a cell whose z_max is slightly above
    roof_height_m for only ONE frame should NOT be clipped (min_frames=2).
    """
    from src.grid.hysteresis import ZClipHysteresisFilter

    filt = ZClipHysteresisFilter(margin_m=0.10, min_frames=2)
    key = (150, 200)
    roof = 2.5

    # Frame 1: z slightly above roof (within margin)
    assert filt.should_clip(key, 2.55, roof) is False, (
        "First frame in margin band should NOT clip (only 1 frame, need 2)"
    )

    # Frame 2: still slightly above — now meets min_frames threshold
    assert filt.should_clip(key, 2.55, roof) is True, (
        "Second consecutive frame in margin band SHOULD clip"
    )

    # Frame 3: z drops back below roof — counter resets
    assert filt.should_clip(key, 2.4, roof) is False, (
        "After dropping below roof, should NOT clip"
    )

    # Frame 4: z spikes slightly above again (only 1 frame) — not clipped
    assert filt.should_clip(key, 2.52, roof) is False, (
        "Single-frame re-entry should NOT clip"
    )


def test_z_clip_hysteresis_clips_hard_above_margin():
    """
    Validates ZClipHysteresisFilter: a cell whose z_max clearly exceeds
    roof_height_m + margin_m is clipped immediately (no waiting).
    """
    from src.grid.hysteresis import ZClipHysteresisFilter

    filt = ZClipHysteresisFilter(margin_m=0.10, min_frames=2)
    key = (200, 100)
    roof = 2.5

    # z clearly above hard limit (2.5 + 0.1 = 2.6) -> clip immediately
    assert filt.should_clip(key, 3.2, roof) is True, (
        "z >> roof_height_m + margin should clip immediately on frame 1"
    )
