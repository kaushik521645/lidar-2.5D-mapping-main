"""
Shared test fixtures for FoveaMap test suite.

Provides reusable fixtures for PolarGridEngine, KalmanTrackerManager,
and synthetic scan generation to eliminate boilerplate across test files.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid.grid_engine import PolarGridEngine
from src.grid.grid_types import VehicleState, GridCell, GridMap
from src.tracking.kalman_tracker import KalmanTrackerManager


@pytest.fixture
def grid_engine():
    """Provides a fresh PolarGridEngine instance."""
    return PolarGridEngine()


@pytest.fixture
def tracker():
    """Provides a fresh KalmanTrackerManager with default settings."""
    return KalmanTrackerManager(dt_s=0.1)


@pytest.fixture
def vehicle_state():
    """Provides a default ego vehicle state (8 m/s, straight)."""
    return VehicleState(speed_mps=8.0, steering_angle_rad=0.0)


@pytest.fixture
def synthetic_scan():
    """Generates a simple synthetic LiDAR scan (N, 5) with known geometry.
    
    Returns a point cloud with:
    - Road surface points (class 0) at z ~ -1.5m within |y| < 4m
    - Sidewalk points (class 1) at z ~ -1.3m at |y| ~ 5-6m
    - Building points (class 2) at z ~ 0..2m at |y| > 8m
    - A moving vehicle (class 3) at (10, 2, -0.5)
    """
    rng = np.random.default_rng(42)
    
    # Road surface
    n_road = 200
    road_x = rng.uniform(2.0, 40.0, n_road)
    road_y = rng.uniform(-3.5, 3.5, n_road)
    road_z = np.full(n_road, -1.5) + rng.normal(0, 0.02, n_road)
    road_i = rng.uniform(0.3, 0.8, n_road)
    road_c = np.zeros(n_road)

    # Sidewalk
    n_side = 50
    side_x = rng.uniform(2.0, 30.0, n_side)
    side_y = rng.choice([-5.5, 5.5], n_side) + rng.normal(0, 0.3, n_side)
    side_z = np.full(n_side, -1.3) + rng.normal(0, 0.02, n_side)
    side_i = rng.uniform(0.2, 0.5, n_side)
    side_c = np.ones(n_side)

    # Building
    n_bldg = 30
    bldg_x = rng.uniform(5.0, 25.0, n_bldg)
    bldg_y = rng.choice([-10.0, 10.0], n_bldg)
    bldg_z = rng.uniform(-1.0, 2.0, n_bldg)
    bldg_i = rng.uniform(0.1, 0.3, n_bldg)
    bldg_c = np.full(n_bldg, 2.0)

    # Vehicle
    n_veh = 20
    veh_x = np.full(n_veh, 10.0) + rng.normal(0, 0.3, n_veh)
    veh_y = np.full(n_veh, 2.0) + rng.normal(0, 0.2, n_veh)
    veh_z = rng.uniform(-1.0, 0.0, n_veh)
    veh_i = rng.uniform(0.5, 0.9, n_veh)
    veh_c = np.full(n_veh, 3.0)

    scan = np.column_stack([
        np.concatenate([road_x, side_x, bldg_x, veh_x]),
        np.concatenate([road_y, side_y, bldg_y, veh_y]),
        np.concatenate([road_z, side_z, bldg_z, veh_z]),
        np.concatenate([road_i, side_i, bldg_i, veh_i]),
        np.concatenate([road_c, side_c, bldg_c, veh_c]),
    ]).astype(np.float32)

    return scan
