"""
Core data structures for the FoveaMap 2.5D polar grid.

Defines GridCell (one column in the variable-resolution grid),
VehicleState (ego-vehicle kinematics for dynamic foveation),
and the GridMap type alias.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GridCell:
    """Represents one cell in a variable-resolution 2.5D polar grid.

    Each cell stores the ground-level elevation, an optional overhead
    obstacle layer (bottom/top), a semantic class label, and bookkeeping
    metadata used by the grid engine.

    Semantic classes (4-Class Taxonomy):
        0 = Drivable Terrain
        1 = Non-Drivable Terrain
        2 = Static Obstacle
        3 = Dynamic Object
       -1 = Unknown / Unclassified
    """

    elevation_ground: float
    elevation_obstacle_bottom: Optional[float] = None
    elevation_obstacle_top: Optional[float] = None
    semantic_class: int = -1
    point_count: int = 0
    confidence: float = 1.0
    last_updated_frame: int = 0
    resolution_tier: float = 0.0


@dataclass
class VehicleState:
    """Ego-vehicle kinematic state used for dynamic foveation calculations.

    Attributes:
        speed_mps: Forward speed in metres per second.
        steering_angle_rad: Steering angle in radians.
            Positive = right turn, negative = left turn.
    """

    speed_mps: float = 0.0
    steering_angle_rad: float = 0.0


# Sparse grid stored as a dict keyed by (ring_idx, angle_idx) tuples.
GridMap = dict
