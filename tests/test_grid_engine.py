"""
Tests for Grid Engine.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.grid_engine import project_to_grid
from src.grid.grid_types import VehicleState


def test_single_point():
    """A single point lands in exactly one grid cell with the correct height."""
    pts = np.array([[5.0, 0.0, 1.2, 0.0, 1]])  # x=5, y=0, z=1.2, intensity=0, class=1
    state = VehicleState(0.0, 0.0)
    grid = project_to_grid(pts, state, roof_height_m=2.5)
    
    assert len(grid) == 1
    cell = list(grid.values())[0]
    assert cell.elevation_ground == pytest.approx(1.2)
    assert cell.elevation_obstacle_bottom is None
    assert cell.elevation_obstacle_top is None
    assert cell.semantic_class == 1


def test_z_clipping_excluded():
    """A point with z above roof_height_m is excluded entirely (grid stays empty)."""
    pts = np.array([[5.0, 0.0, 3.0, 0.0, 1]])  # z=3.0 > 2.5
    state = VehicleState(0.0, 0.0)
    grid = project_to_grid(pts, state, roof_height_m=2.5)
    
    assert len(grid) == 0


def test_multi_layer_overhang_detected():
    """Two points in the same (x,y) with gap > 0.5m produce obstacle layers."""
    # z=0.0 and z=1.0 (gap 1.0 > 0.5)
    pts = np.array([
        [5.0, 0.0, 0.0, 0.0, 0],
        [5.0, 0.0, 1.0, 0.0, 1]
    ])
    state = VehicleState(0.0, 0.0)
    grid = project_to_grid(pts, state, roof_height_m=2.5)
    
    assert len(grid) == 1
    cell = list(grid.values())[0]
    assert cell.elevation_ground == pytest.approx(0.0)
    assert cell.elevation_obstacle_bottom == pytest.approx(1.0)
    assert cell.elevation_obstacle_top == pytest.approx(1.0)


def test_multi_layer_close_points_no_overhang():
    """Two points close in height (gap < 0.5m) produce NO obstacle layers."""
    # z=0.0 and z=0.4 (gap 0.4 < 0.5)
    pts = np.array([
        [5.0, 0.0, 0.0, 0.0, 0],
        [5.0, 0.0, 0.4, 0.0, 1]
    ])
    state = VehicleState(0.0, 0.0)
    grid = project_to_grid(pts, state, roof_height_m=2.5)
    
    assert len(grid) == 1
    cell = list(grid.values())[0]
    assert cell.elevation_ground == pytest.approx(0.0)
    assert cell.elevation_obstacle_bottom is None
    assert cell.elevation_obstacle_top is None


def test_majority_class_voting():
    """Majority class voting picks the class with the most points in a cell."""
    pts = np.array([
        [5.0, 0.0, 0.0, 0.0, 1],
        [5.0, 0.0, 0.1, 0.0, 2],
        [5.0, 0.0, 0.2, 0.0, 2]
    ])
    state = VehicleState(0.0, 0.0)
    grid = project_to_grid(pts, state, roof_height_m=2.5)
    
    assert len(grid) == 1
    cell = list(grid.values())[0]
    assert cell.semantic_class == 2  # class 2 has 2 points, class 1 has 1


def test_z_clip_overhang_tradeoff_missing():
    """
    IMPORTANT EDGE CASE: An overhang point taller than roof_height_m 
    gets silently Z-clipped and is therefore invisible to the multi-layer detection.
    """
    pts = np.array([
        [5.0, 0.0, 0.0, 0.0, 0],  # Ground
        [5.0, 0.0, 3.0, 0.0, 1]   # Overhang (z=3.0)
    ])
    state = VehicleState(0.0, 0.0)
    # With default roof_height_m = 2.5, the overhang is clipped
    grid = project_to_grid(pts, state, roof_height_m=2.5)
    
    assert len(grid) == 1
    cell = list(grid.values())[0]
    assert cell.elevation_ground == pytest.approx(0.0)
    # Because z=3.0 was clipped, the cell only saw z=0.0, so no overhang detected
    assert cell.elevation_obstacle_bottom is None


def test_z_clip_overhang_tradeoff_restored():
    """
    Passing a higher roof_height_m override restores correct detection of the overhang.
    """
    pts = np.array([
        [5.0, 0.0, 0.0, 0.0, 0],  # Ground
        [5.0, 0.0, 3.0, 0.0, 1]   # Overhang (z=3.0)
    ])
    state = VehicleState(0.0, 0.0)
    # Override roof_height_m to 3.5, now the overhang is included
    grid = project_to_grid(pts, state, roof_height_m=3.5)
    
    assert len(grid) == 1
    cell = list(grid.values())[0]
    assert cell.elevation_ground == pytest.approx(0.0)
    # The overhang is now properly detected
    assert cell.elevation_obstacle_bottom == pytest.approx(3.0)
    assert cell.elevation_obstacle_top == pytest.approx(3.0)
