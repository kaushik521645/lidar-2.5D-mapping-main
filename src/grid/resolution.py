"""
Grid Resolution Module for FoveaMap.

Defines static resolution tiers mapping radial range to spatial cell sizes (m)
and computes angular step sizes for polar-annular binning.
"""

import os
from typing import List, Tuple
import yaml
import numpy as np


DEFAULT_RESOLUTION_TIERS = [
    (10.0, 0.05),
    (30.0, 0.15),
    (100.0, 0.50),
]

def load_resolution_tiers(config_path: str = "configs/default.yaml") -> List[Tuple[float, float]]:
    """
    Loads resolution tiers from config YAML, falling back to canonical defaults.
    Format: list of (max_range_m, cell_size_m).
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                tiers = cfg.get("grid", {}).get("resolution_tiers", [])
                if tiers:
                    return [(float(r), float(s)) for r, s in tiers]
        except Exception:
            pass
            
    return DEFAULT_RESOLUTION_TIERS

RESOLUTION_TIERS = load_resolution_tiers()

# ── Canonical ring-boundary constants (single source of truth) ──
# These are derived from the default tiers and used across grid_engine, tracking, etc.
TIER0_RES: float = 0.05       # Fine tier cell size (m)
TIER1_RES: float = 0.15       # Medium tier cell size (m)
TIER2_RES: float = 0.50       # Coarse tier cell size (m)
TIER0_OUTER_M: float = 10.0   # Outer boundary of tier 0 (m)
TIER1_OUTER_M: float = 30.0   # Outer boundary of tier 1 (m)
TIER0_MAX_RING: int = int(TIER0_OUTER_M / TIER0_RES)          # 200
TIER1_MAX_RING: int = TIER0_MAX_RING + int((TIER1_OUTER_M - TIER0_OUTER_M) / TIER1_RES)  # 334


def get_resolution(range_m: float, tiers: List[Tuple[float, float]] = None) -> float:
    """
    Cell size in meters for a given radial distance.
    
    Args:
        range_m: Radial distance from sensor origin (meters).
        tiers: List of (max_range, cell_size) tuples sorted by ascending max_range.
        
    Returns:
        cell_size in meters (0.05, 0.15, or 0.50).
    """
    if tiers is None:
        tiers = DEFAULT_RESOLUTION_TIERS
        
    for max_range, res in tiers:
        if range_m < max_range:
            return res
    return tiers[-1][1]


def get_tier_index(range_m: float, tiers: List[Tuple[float, float]] = RESOLUTION_TIERS) -> int:
    """
    Returns the integer index of the resolution tier (0, 1, or 2).
    """
    for idx, (max_range, _) in enumerate(tiers):
        if range_m < max_range:
            return idx
    return len(tiers) - 1


def angular_step(res_m: float, range_m: float, min_range_m: float = 0.5) -> float:
    """
    Radians per angular bin so arc length ≈ res_m at this range.
    arc_length = range * delta_theta  =>  delta_theta = res_m / range
    
    Args:
        res_m: Target spatial resolution (meters).
        range_m: Radial distance from sensor origin (meters).
        min_range_m: Minimum range clamp to prevent division by zero or oversized bins near origin.
        
    Returns:
        Angular step size in radians.
    """
    clamped_range = max(range_m, min_range_m)
    return res_m / clamped_range
