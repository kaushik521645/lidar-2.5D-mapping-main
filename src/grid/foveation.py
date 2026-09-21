"""
Dynamic Foveation Module for FoveaMap.

Calculates the kinematic-adaptive high-resolution zone around the ego-vehicle.
R_fine(theta) = R_base * stretch(speed) * shear(theta, steering)
"""

import numpy as np
import yaml
import os
from typing import Union
from src.grid.grid_types import VehicleState

# Module-level defaults
BASE_FINE_RADIUS_M = 10.0
MAX_STRETCH = 2.5
STRETCH_SPEED_REF_MPS = 20.0
SHEAR_STRENGTH = 0.6

# Aliases for backwards compatibility with grid_engine.py
BASE_FINE_RADIUS = BASE_FINE_RADIUS_M


def init_foveation_from_config(config_path: str = "configs/default.yaml") -> None:
    """Loads foveation settings from YAML, keeping backwards compatibility."""
    global BASE_FINE_RADIUS_M, MAX_STRETCH, STRETCH_SPEED_REF_MPS, SHEAR_STRENGTH
    global BASE_FINE_RADIUS
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                fov = cfg.get("foveation", {})
                if fov:
                    BASE_FINE_RADIUS_M = float(fov.get("base_fine_radius_m", BASE_FINE_RADIUS_M))
                    MAX_STRETCH = float(fov.get("max_stretch", MAX_STRETCH))
                    STRETCH_SPEED_REF_MPS = float(fov.get("stretch_speed_ref_mps", STRETCH_SPEED_REF_MPS))
                    SHEAR_STRENGTH = float(fov.get("shear_strength", SHEAR_STRENGTH))
                    BASE_FINE_RADIUS = BASE_FINE_RADIUS_M
        except Exception:
            pass


def stretch_factor(
    speed_mps: Union[float, np.ndarray],
    max_stretch: float = MAX_STRETCH,
    speed_ref: float = STRETCH_SPEED_REF_MPS
) -> Union[float, np.ndarray]:
    """Linear ramp from 1.0 at speed 0 to max_stretch at speed_ref."""
    if speed_ref == 0:
        return np.ones_like(speed_mps) * max_stretch
        
    s = speed_mps / speed_ref
    return 1.0 + (max_stretch - 1.0) * np.clip(s, 0.0, 1.0)


def attention_modifier(
    theta_rad: Union[float, np.ndarray],
    active_tracks: list,
) -> Union[float, np.ndarray]:
    """Calculates Gaussian radius spikes pointing towards dynamic active tracks."""
    if not active_tracks:
        return np.zeros_like(theta_rad) if isinstance(theta_rad, np.ndarray) else 0.0
        
    r_attn = np.zeros_like(theta_rad) if isinstance(theta_rad, np.ndarray) else 0.0
    
    for track in active_tracks:
        # Ignore static or unknown objects (assuming class 3 is dynamic)
        if hasattr(track, 'class_id') and track.class_id != 3:
            continue
            
        tx, ty = track.position_xy
        tr = np.hypot(tx, ty)
        ttheta = np.arctan2(ty, tx)
        
        # Sigma based on angular size of object
        width = max(track.bbox_size_xy) if hasattr(track, 'bbox_size_xy') else 4.0
        sigma = max(width / max(tr, 1.0), 0.1) # min ~5 deg
        
        diff = theta_rad - ttheta
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        # Spike height covers the track + 5m buffer
        spike_height = tr + 5.0
        spike = spike_height * np.exp(-0.5 * (diff / sigma)**2)
        
        if isinstance(theta_rad, np.ndarray):
            r_attn = np.maximum(r_attn, spike)
        else:
            r_attn = max(r_attn, spike)
            
    return r_attn


def fine_radius_at_angle(
    theta_rad: Union[float, np.ndarray],
    state: VehicleState,
    base_radius: float = BASE_FINE_RADIUS_M,
    max_stretch: float = MAX_STRETCH,
    speed_ref: float = STRETCH_SPEED_REF_MPS,
    active_tracks: list = None
) -> Union[float, np.ndarray]:
    """
    Computes a Piecewise Elliptical foveation zone (Option 1) combined with 
    Obstacle-Aware Attention spikes (Option 2).
    """
    stretch = stretch_factor(state.speed_mps, max_stretch, speed_ref)
    
    # Ellipse parameters
    a = base_radius * stretch
    b = base_radius
    
    # Angle relative to steering direction
    d_theta = theta_rad - state.steering_angle_rad
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    
    # Calculate True Ellipse in polar coordinates
    denominator = np.sqrt((b * np.cos(d_theta))**2 + (a * np.sin(d_theta))**2)
    # Prevent division by zero
    if isinstance(denominator, np.ndarray):
        denominator = np.where(denominator == 0, 1e-6, denominator)
    else:
        denominator = 1e-6 if denominator == 0 else denominator
        
    r_ellipse = (a * b) / denominator
    
    # Piecewise: front half is ellipse, back half is base circle
    if isinstance(theta_rad, np.ndarray):
        is_forward = np.abs(d_theta) < (np.pi / 2)
        r_fine = np.where(is_forward, r_ellipse, b)
    else:
        is_forward = abs(d_theta) < (np.pi / 2)
        r_fine = r_ellipse if is_forward else b
        
    # Apply attention modifier if tracks are provided
    if active_tracks:
        r_attn = attention_modifier(theta_rad, active_tracks)
        if isinstance(r_fine, np.ndarray):
            r_fine = np.maximum(r_fine, r_attn)
        else:
            r_fine = max(r_fine, r_attn)
            
    return r_fine
