"""
Unit tests for Multi-Layer Elevation / Overhang Detection (Milestone 4).
Validates that vertical gaps > 0.5m separate ground surface from overhead structures (bridges/tunnels).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.grid_engine import PolarGridEngine
from src.grid.grid_types import VehicleState


def test_multilayer_overhang_detection():
    """
    Validates exact committed fixture:
    - 4 ground points with z in [-0.05, 0.02] -> elevation_ground = -0.05
    - 4 overhang points with z in [2.80, 3.20] -> elevation_obstacle_bottom = 2.80, top = 3.20
    - Gap = 2.78m > 0.50m threshold
    """
    engine = PolarGridEngine()
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)

    # Combined 8 points in cell [5.00, 5.05)
    points = np.array([
        # Ground cluster (Class 0: Drivable)
        [5.02, 0.0, -0.05, 0.5, 0],
        [5.03, 0.0,  0.00, 0.6, 0],
        [5.01, 0.0,  0.02, 0.5, 0],
        [5.02, 0.0, -0.02, 0.4, 0],
        # Overhang cluster (Class 2: Static Obstacle)
        [5.02, 0.0,  2.80, 0.8, 2],
        [5.03, 0.0,  3.00, 0.7, 2],
        [5.01, 0.0,  3.10, 0.9, 2],
        [5.02, 0.0,  3.20, 0.8, 2],
    ], dtype=np.float32)

    # Use roof_height_m = 4.0 so the 3.20m bridge points are retained
    grid_map = engine.project_to_grid(
        points, vehicle_state=state, frame_id=1, roof_height_m=4.0, use_foveation=False
    )

    assert len(grid_map) == 1
    cell = list(grid_map.values())[0]

    assert cell.point_count == 8
    assert cell.elevation_ground == pytest.approx(-0.05, abs=1e-3)
    assert cell.elevation_obstacle_bottom == pytest.approx(2.80, abs=1e-3)
    assert cell.elevation_obstacle_top == pytest.approx(3.20, abs=1e-3)


def test_continuous_height_no_overhang():
    """
    Validates that a solid vertical structure (e.g. wall/pillar) with continuous
    points (max gap <= 0.35m) does NOT trigger an overhang gap.
    """
    engine = PolarGridEngine()
    state = VehicleState(speed_mps=0.0, steering_angle_rad=0.0)

    # 7 points spaced by ~0.35m vertically: no single gap > 0.50m
    points = np.array([
        [5.02, 0.0, -0.05, 0.5, 2],
        [5.02, 0.0,  0.30, 0.5, 2],
        [5.02, 0.0,  0.65, 0.5, 2],
        [5.02, 0.0,  1.00, 0.5, 2],
        [5.02, 0.0,  1.35, 0.5, 2],
        [5.02, 0.0,  1.70, 0.5, 2],
        [5.02, 0.0,  2.05, 0.5, 2],
    ], dtype=np.float32)

    grid_map = engine.project_to_grid(
        points, vehicle_state=state, frame_id=1, roof_height_m=4.0, use_foveation=False
    )

    assert len(grid_map) == 1
    cell = list(grid_map.values())[0]

    assert cell.point_count == 7
    assert cell.elevation_ground == pytest.approx(-0.05, abs=1e-3)
    assert cell.elevation_obstacle_bottom is None
    assert cell.elevation_obstacle_top is None
