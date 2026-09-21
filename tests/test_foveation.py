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
    # Track exactly at 0 rad, 20m away
    track = MockTrack(class_id=3, position_xy=(20.0, 0.0))
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
