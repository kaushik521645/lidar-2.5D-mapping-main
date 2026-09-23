"""
Tests for Dynamic Foveation module (Elliptical + Attention).
"""

import sys
import os
import pytest
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.foveation import (
    fine_radius_at_angle,
    attention_modifier,
    BASE_FINE_RADIUS_M,
    MAX_STRETCH,
    STRETCH_SPEED_REF_MPS,
)
from src.grid.grid_types import VehicleState

@dataclass
class MockTrack:
    class_id: int
    position_xy: tuple
    bbox_size_xy: tuple = (4.0, 2.0)
    velocity_xy: tuple = (0.0, 0.0)
    predicted_next_xy: tuple = None


def test_stationary_vehicle():
    """
    stationary vehicle (speed=0, steering=0): fine_radius_at_angle returns
    base_radius at theta=0, theta=pi/2, and theta=pi (all equal, within 1e-6)
    """
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)
    
    r_front = fine_radius_at_angle(0.0, state)
    r_side = fine_radius_at_angle(np.pi / 2, state)
    r_back = fine_radius_at_angle(np.pi, state)
    
    assert r_front == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6)
    assert r_side == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6)
    assert r_back == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6)


def test_forward_stretch():
    """
    vehicle at STRETCH_SPEED_REF_MPS, steering=0: radius at theta=0 equals
    base_radius * MAX_STRETCH (within 1e-6)
    """
    state = VehicleState(speed_mps=STRETCH_SPEED_REF_MPS, steering_angle_rad=0.0)
    
    r_front = fine_radius_at_angle(0.0, state)
    assert r_front == pytest.approx(BASE_FINE_RADIUS_M * MAX_STRETCH, abs=1e-6)


def test_backward_stretch_unchanged():
    """
    same vehicle: radius at theta=pi (directly behind) still equals base_radius
    unchanged (forward stretch shouldn't apply behind the vehicle)
    """
    state = VehicleState(speed_mps=STRETCH_SPEED_REF_MPS, steering_angle_rad=0.0)
    
    r_back = fine_radius_at_angle(np.pi, state)
    assert r_back == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6)


def test_turn_ellipse_rotation():
    """
    vehicle turning right (steering_angle_rad=0.5): The ellipse's major axis 
    aligns with 0.5 rad. So radius at 0.5 should be max stretched, and radius 
    at -0.5 should be smaller.
    """
    state = VehicleState(speed_mps=STRETCH_SPEED_REF_MPS, steering_angle_rad=0.5)
    
    r_turn_side = fine_radius_at_angle(0.5, state)
    r_opp_side = fine_radius_at_angle(-0.5, state)
    
    # At the steering angle, it should be the max stretch
    assert r_turn_side == pytest.approx(BASE_FINE_RADIUS_M * MAX_STRETCH, abs=1e-6)
    assert r_turn_side > r_opp_side


def test_attention_modifier():
    """
    Test that a dynamic track at a specific angle causes a gaussian spike in radius.
    """
    # Track exactly at 0 rad, 20m away without velocity (backward compatibility)
    track = MockTrack(class_id=3, position_xy=(20.0, 0.0), velocity_xy=(0.0, 0.0))
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)
    
    # Check without tracks
    r_no_track = fine_radius_at_angle(0.0, state)
    assert r_no_track == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6)
    
    # Check with track
    r_with_track = fine_radius_at_angle(0.0, state, active_tracks=[track])
    
    # The spike height for an object at 20m is 20 + 5 = 25m
    assert r_with_track == pytest.approx(25.0, abs=1e-6)
    
    # Check falloff at 90 degrees (should be back to base radius)
    r_falloff = fine_radius_at_angle(np.pi/2, state, active_tracks=[track])
    assert r_falloff == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6)


def test_detect_moving_objects():
    """
    Verify detect_moving_objects correctly separates stationary from moving obstacles.
    """
    from src.grid.foveation import detect_moving_objects
    
    stationary_car = MockTrack(class_id=3, position_xy=(10.0, 2.0), velocity_xy=(0.0, 0.0))
    slow_drift = MockTrack(class_id=3, position_xy=(12.0, -1.0), velocity_xy=(0.2, 0.1)) # 0.22 m/s < 0.8 m/s
    moving_car = MockTrack(class_id=3, position_xy=(18.0, -5.0), velocity_xy=(0.0, 10.0)) # 10 m/s >= 0.8 m/s
    static_tree = MockTrack(class_id=2, position_xy=(5.0, 5.0), velocity_xy=(0.0, 0.0)) # Non-dynamic class
    
    tracks = [stationary_car, slow_drift, moving_car, static_tree]
    info = detect_moving_objects(tracks, speed_threshold_mps=0.8)
    
    # Only dynamic class 3 should be analyzed
    assert len(info) == 3
    assert not info[0]["is_moving"]  # Stationary
    assert not info[1]["is_moving"]  # Below threshold
    assert info[2]["is_moving"]      # Moving at 10 m/s
    assert info[2]["speed_mps"] == pytest.approx(10.0, abs=1e-3)


def test_motion_foveation_trajectory_lookahead():
    """
    Test that when collision_focus_only is False, a moving vehicle triggers a 
    dynamic lobe stretching ahead along its velocity vector.
    """
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0) # Ego stationary (base radius = 10m)
    
    # Cross traffic moving in +Y direction: pos=(15.0, 0.0), vel=(0.0, 12.0) m/s
    # In 1.0s lead time, predicted pos=(15.0, 12.0), distance=19.2m, bearing=atan2(12, 15) ~ 0.67 rad (~38.6 deg)
    moving_track = MockTrack(
        class_id=3,
        position_xy=(15.0, 0.0),
        velocity_xy=(0.0, 12.0),
        bbox_size_xy=(4.0, 2.0)
    )
    
    # Bearing ahead along trajectory (~0.67 rad)
    traj_angle = np.arctan2(12.0, 15.0)
    r_traj = fine_radius_at_angle(traj_angle, state, active_tracks=[moving_track], collision_focus_only=False)
    
    # Verify that foveation extends well beyond the 10m base radius along the trajectory
    assert r_traj > 20.0, f"Expected fine radius > 20m along moving trajectory, got {r_traj}"
    
    # Verify directional asymmetry: opposite direction should not have high reach
    r_opp = fine_radius_at_angle(-traj_angle, state, active_tracks=[moving_track], collision_focus_only=False)
    assert r_traj > r_opp + 8.0, "Trajectory lobe should be much larger than opposite direction"


def test_oncoming_vs_receding_foveation():
    """
    Verify that collision-only focus:
    - Focuses on oncoming vehicles (closing speed > 0, vx < 0)
    - Rejects receding vehicles moving away (vx > 0)
    - Rejects distant obstacles outside threat horizon (> 35m)
    - Rejects off-corridor sidewalk pedestrians
    """
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)
    
    # 1. Oncoming vehicle closing in at 20m: should expand fovea
    oncoming_track = MockTrack(class_id=3, position_xy=(20.0, 0.0), velocity_xy=(-8.0, 0.0))
    r_oncoming = fine_radius_at_angle(0.0, state, active_tracks=[oncoming_track])
    assert r_oncoming >= 25.0, f"Expected oncoming vehicle to expand fovea >= 25m, got {r_oncoming}"
    
    # 2. Receding vehicle pulling away at 20m: MUST NOT expand fovea
    receding_track = MockTrack(class_id=3, position_xy=(20.0, 0.0), velocity_xy=(8.0, 0.0))
    r_receding = fine_radius_at_angle(0.0, state, active_tracks=[receding_track])
    assert r_receding == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6), "Receding vehicle should not expand fovea"
    
    # 3. Distant vehicle at 45m (> 35m threat horizon): MUST NOT expand fovea
    distant_track = MockTrack(class_id=3, position_xy=(45.0, 0.0), velocity_xy=(-8.0, 0.0))
    r_distant = fine_radius_at_angle(0.0, state, active_tracks=[distant_track])
    assert r_distant == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6), "Distant vehicle should not expand fovea"
    
    # 4. Sidewalk pedestrian at y=6.0m (outside corridor): MUST NOT expand fovea
    sidewalk_ped = MockTrack(class_id=3, position_xy=(15.0, 6.0), velocity_xy=(0.0, 0.0))
    ped_angle = np.arctan2(6.0, 15.0)
    r_ped = fine_radius_at_angle(ped_angle, state, active_tracks=[sidewalk_ped])
    assert r_ped == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6), "Sidewalk pedestrian should not expand fovea"


def test_motion_foveation_toggle_disable():
    """
    Test that setting motion_foveation_enabled=False disables moving obstacle foveation lobes.
    """
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)
    moving_track = MockTrack(
        class_id=3,
        position_xy=(15.0, 0.0),
        velocity_xy=(0.0, 12.0)
    )
    
    # When disabled, should remain at base radius (10.0m)
    r_disabled = fine_radius_at_angle(0.0, state, active_tracks=[moving_track], motion_foveation_enabled=False)
    assert r_disabled == pytest.approx(BASE_FINE_RADIUS_M, abs=1e-6)


def test_motion_adaptive_grid_resolution_upgrade():
    """
    Test that PolarGridEngine dynamically assigns Tier 0 (0.05m) resolution
    to points near an ONCOMING obstacle at 25m, which would otherwise be Tier 1 (0.15m),
    while keeping receding vehicles at Tier 1.
    """
    from src.grid.grid_engine import PolarGridEngine
    
    engine = PolarGridEngine()
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)
    
    # Oncoming obstacle at x=25.0, y=0.0 closing in at -5 m/s
    oncoming_track = MockTrack(
        class_id=3,
        position_xy=(25.0, 0.0),
        velocity_xy=(-5.0, 0.0),
        bbox_size_xy=(4.0, 2.0)
    )
    # Receding obstacle at x=25.0, y=0.0 pulling away at +5 m/s
    receding_track = MockTrack(
        class_id=3,
        position_xy=(25.0, 0.0),
        velocity_xy=(5.0, 0.0),
        bbox_size_xy=(4.0, 2.0)
    )
    
    # Points on the obstacle at 25m
    obs_points = np.array([
        [25.0, 0.0, -0.5, 0.8, 3],
        [25.5, 0.2, -0.5, 0.8, 3],
        [24.8, -0.2, -0.5, 0.8, 3],
    ], dtype=np.float32)
    
    # Without tracks: points at 25m are in Tier 1 (res = 0.15)
    grid_without = engine.project_to_grid(obs_points, vehicle_state=state, active_tracks=[])
    for cell in grid_without.values():
        assert cell.resolution_tier == pytest.approx(0.15, abs=1e-4)
        
    # With receding track: points at 25m stay in Tier 1 (0.15) because it is moving away!
    grid_receding = engine.project_to_grid(obs_points, vehicle_state=state, active_tracks=[receding_track])
    for cell in grid_receding.values():
        assert cell.resolution_tier == pytest.approx(0.15, abs=1e-4)

    # With oncoming track: points at 25m are upgraded to Tier 0 (0.05m)
    grid_oncoming = engine.project_to_grid(obs_points, vehicle_state=state, active_tracks=[oncoming_track])
    for cell in grid_oncoming.values():
        assert cell.resolution_tier == pytest.approx(0.05, abs=1e-4)


def test_compute_fovea_polyline():
    """Verify compute_fovea_polyline produces 120 valid [x, y] coordinates."""
    from src.grid.foveation import compute_fovea_polyline

    state = VehicleState(speed_mps=15.0, steering_angle_rad=0.2)
    poly = compute_fovea_polyline(state, num_samples=120)

    assert len(poly) == 120
    for pt in poly:
        assert len(pt) == 2
        assert isinstance(pt[0], float) and isinstance(pt[1], float)
        # Radius should be between base_radius (10m) and stretched radius (~25m)
        r = float(np.hypot(pt[0], pt[1]))
        assert 9.5 <= r <= 26.0


def test_foveation_shear_strength():
    """Verify shear_strength modulates the steering deflection angle."""
    from src.grid.foveation import fine_radius_at_angle

    state = VehicleState(speed_mps=20.0, steering_angle_rad=0.4)
    # With shear 1.0, maximum stretch is at 0.4 rad
    r_full_shear = fine_radius_at_angle(0.4, state, shear_strength=1.0)
    assert r_full_shear == pytest.approx(BASE_FINE_RADIUS_M * MAX_STRETCH, abs=1e-5)

    # With shear 0.5, effective steering angle is 0.2 rad
    r_half_shear = fine_radius_at_angle(0.2, state, shear_strength=0.5)
    assert r_half_shear == pytest.approx(BASE_FINE_RADIUS_M * MAX_STRETCH, abs=1e-5)


