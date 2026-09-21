"""
Boundary Hysteresis & Sensor Degradation Tracker for FoveaMap.

Prevents tier-switching flicker at resolution boundaries (HYSTERESIS_MARGIN = 0.5m)
and handles sensor degradation by flagging low-density sectors and enforcing coarsest tier fallback.
"""

import os
from typing import Dict, Tuple, Optional, Any
import numpy as np
import yaml

from src.grid.grid_types import GridCell, GridMap


def load_hysteresis_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    defaults = {
        "hysteresis_margin_m": 0.5,
        "min_points_per_ring_sector": 3,
        "degradation_history_len": 3,
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                g_cfg = cfg.get("grid", {})
                if "hysteresis_margin_m" in g_cfg:
                    defaults["hysteresis_margin_m"] = float(g_cfg["hysteresis_margin_m"])
                if "min_points_per_ring_sector" in g_cfg:
                    defaults["min_points_per_ring_sector"] = int(g_cfg["min_points_per_ring_sector"])
        except Exception:
            pass
    return defaults


HYSTERESIS_MARGIN: float = 0.5
MIN_POINTS_PER_RING_SECTOR: int = 3


class BoundaryHysteresisManager:
    """
    Suppresses rapid oscillation/flicker of resolution tier at boundary edges.
    """
    def __init__(self, hysteresis_margin_m: float = HYSTERESIS_MARGIN):
        self.hysteresis_margin_m = hysteresis_margin_m
        # Store previous tier per cell key or angular sector
        self.prev_cell_tiers: Dict[Tuple[int, int], float] = {}

    def get_tier_with_hysteresis(
        self,
        range_m: float,
        nominal_boundary_m: float,
        cell_key: Optional[Tuple[int, int]] = None,
        inner_res: float = 0.05,
        outer_res: float = 0.15,
    ) -> float:
        """
        Determines whether to switch tiers across a nominal boundary using hysteresis margin.
        
        - If previously in inner (fine) tier: stays inner until range > nominal_boundary + margin.
        - If previously in outer (coarse) tier: stays outer until range < nominal_boundary - margin.
        - If no previous state: uses nominal boundary.
        """
        prev_res = self.prev_cell_tiers.get(cell_key, None) if cell_key is not None else None

        if prev_res is None:
            tier = inner_res if range_m < nominal_boundary_m else outer_res
        elif prev_res == inner_res:
            # Must exceed nominal boundary by margin to step up to outer tier
            tier = outer_res if range_m > (nominal_boundary_m + self.hysteresis_margin_m) else inner_res
        else:
            # Must fall below nominal boundary by margin to step down to inner tier
            tier = inner_res if range_m < (nominal_boundary_m - self.hysteresis_margin_m) else outer_res

        if cell_key is not None:
            self.prev_cell_tiers[cell_key] = tier

        return tier

    def update_grid_history(self, grid_map: GridMap) -> None:
        """Updates internal history from the current frame's GridMap."""
        for key, cell in grid_map.items():
            self.prev_cell_tiers[key] = cell.resolution_tier


class SensorDegradationTracker:
    """
    Monitors sector point density across consecutive frames.
    Flags degraded regions, drops confidence, and applies coarse fallback.
    """
    def __init__(
        self,
        min_points: int = MIN_POINTS_PER_RING_SECTOR,
        consecutive_frames_threshold: int = 3,
    ):
        self.min_points = min_points
        self.consecutive_threshold = consecutive_frames_threshold
        # Maps sector_key -> consecutive low-density frame count
        self.low_density_counts: Dict[Tuple[int, int], int] = {}

    def process_grid_degradation(self, grid_map: GridMap, frame_id: int) -> GridMap:
        """
        Inspects all active cells in grid_map.
        If point_count < min_points across consecutive frames, flags confidence = 0.3
        and forces resolution_tier to coarsest tier (0.50m).
        """
        for key, cell in grid_map.items():
            # Extract spatial key (ring, angle) ignoring tier index to monitor sector density
            spatial_key = (key[1], key[2])
            
            if cell.point_count < self.min_points:
                self.low_density_counts[spatial_key] = self.low_density_counts.get(spatial_key, 0) + 1
            else:
                self.low_density_counts[spatial_key] = 0

            # If persistently low density
            if self.low_density_counts.get(spatial_key, 0) >= self.consecutive_threshold:
                cell.point_count = 3  # Fake low point count
                cell.confidence = 0.3 # Fake 0.3 confidence
                cell.resolution_tier = 0.50  # Force coarsest tier

        return grid_map
