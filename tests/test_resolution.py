"""
Tests for the FoveaMap grid resolution module.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.resolution import get_resolution, angular_step


def test_get_resolution():
    """Validates get_resolution() across all tiers and fallback behavior."""
    # Point in the fine tier (<10m)
    assert get_resolution(5.0) == pytest.approx(0.05)
    
    # Point in the mid tier (10m <= range < 30m)
    assert get_resolution(15.0) == pytest.approx(0.15)
    
    # Point in the coarse tier (30m <= range < 100m)
    assert get_resolution(60.0) == pytest.approx(0.50)
    
    # Point beyond all tiers (>= 100m) -> fallback to coarsest
    assert get_resolution(500.0) == pytest.approx(0.50)


def test_angular_step():
    """Validates angular step scales with range to keep bins roughly square."""
    step_near = angular_step(0.15, 5.0)
    step_far = angular_step(0.15, 50.0)
    
    assert step_far < step_near
    
    # At 5m, step is 0.15 / 5 = 0.03
    assert step_near == pytest.approx(0.03)
    
    # At 50m, step is 0.15 / 50 = 0.003
    assert step_far == pytest.approx(0.003)
    
    # Check min_range_m clamping (e.g. range=0.1 clamped to 0.5)
    step_origin = angular_step(0.05, 0.1, min_range_m=0.5)
    assert step_origin == pytest.approx(0.1)  # 0.05 / 0.5 = 0.1
