"""
Motion-Adaptive Dynamic Foveation Module for FoveaMap.

Calculates the kinematic-adaptive high-resolution zone around the ego-vehicle
AND dynamically adapts high-acuity foveal lobes to detect and track moving obstacles
along their predicted motion trajectories.
"""

import numpy as np
import yaml
import os
from typing import Union, List, Dict, Tuple, Optional, Any
from src.grid.grid_types import VehicleState

# Module-level defaults
BASE_FINE_RADIUS_M = 10.0
MAX_STRETCH = 2.5
STRETCH_SPEED_REF_MPS = 20.0
SHEAR_STRENGTH = 0.6

# Motion-Adaptive Foveation Defaults
MOTION_FOVEATION_ENABLED = True
COLLISION_FOCUS_ONLY = True
MOTION_SPEED_THRESHOLD_MPS = 0.8
MOTION_LEAD_TIME_S = 1.0
MOTION_BUFFER_M = 4.0
MOTION_ACUITY_BOOST = 1.3
MAX_THREAT_DISTANCE_M = 35.0
CORRIDOR_HALF_WIDTH_M = 3.2

# Aliases for backwards compatibility with grid_engine.py
BASE_FINE_RADIUS = BASE_FINE_RADIUS_M


def init_foveation_from_config(config_path: str = "configs/default.yaml") -> None:
    """Loads foveation settings from YAML, keeping backwards compatibility."""
    global BASE_FINE_RADIUS_M, MAX_STRETCH, STRETCH_SPEED_REF_MPS, SHEAR_STRENGTH
    global MOTION_FOVEATION_ENABLED, COLLISION_FOCUS_ONLY, MOTION_SPEED_THRESHOLD_MPS
    global MOTION_LEAD_TIME_S, MOTION_BUFFER_M, MOTION_ACUITY_BOOST, MAX_THREAT_DISTANCE_M, CORRIDOR_HALF_WIDTH_M
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
                    MOTION_FOVEATION_ENABLED = bool(fov.get("motion_foveation_enabled", MOTION_FOVEATION_ENABLED))
                    COLLISION_FOCUS_ONLY = bool(fov.get("collision_focus_only", COLLISION_FOCUS_ONLY))
                    MOTION_SPEED_THRESHOLD_MPS = float(fov.get("motion_speed_threshold_mps", MOTION_SPEED_THRESHOLD_MPS))
                    MOTION_LEAD_TIME_S = float(fov.get("motion_lead_time_s", MOTION_LEAD_TIME_S))
                    MOTION_BUFFER_M = float(fov.get("motion_buffer_m", MOTION_BUFFER_M))
                    MOTION_ACUITY_BOOST = float(fov.get("motion_acuity_boost", MOTION_ACUITY_BOOST))
                    MAX_THREAT_DISTANCE_M = float(fov.get("max_threat_distance_m", MAX_THREAT_DISTANCE_M))
                    CORRIDOR_HALF_WIDTH_M = float(fov.get("corridor_half_width_m", CORRIDOR_HALF_WIDTH_M))
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


def is_oncoming_or_collision_threat(
    pos: Tuple[float, float],
    vel: Optional[Tuple[float, float]],
    corridor_half_width_m: float = CORRIDOR_HALF_WIDTH_M,
    max_threat_distance_m: float = MAX_THREAT_DISTANCE_M,
    min_closing_speed_mps: float = 0.5,
) -> Tuple[bool, float, float]:
    """
    Evaluates whether an obstacle is strictly an oncoming or straight-on collision threat.
    
    Rejects:
      - Moving objects heading away from ego (e.g., leading vehicles pulling away)
      - Off-path sidewalk objects / pedestrians outside the forward road corridor
      - Objects beyond the maximum threat horizon (> 35m)
      
    Accepts:
      - Oncoming vehicles ahead heading towards ego (closing in)
      - Stationary or slow obstacles directly blocking the forward travel lane
      - Cross-traffic entering the ego corridor with short Time-To-Collision (TTC)
      
    Returns:
      (is_threat: bool, reach_m: float, target_bearing_rad: float)
    """
    x, y = float(pos[0]), float(pos[1])
    r = float(np.hypot(x, y))
    
    # 1. Ignore objects outside threat horizon
    if r > max_threat_distance_m or r < 0.5:
        return False, 0.0, 0.0
        
    vx = float(vel[0]) if vel is not None else 0.0
    vy = float(vel[1]) if vel is not None else 0.0
    speed = float(np.hypot(vx, vy))
    
    # Range rate / closing speed (-dr/dt): positive means distance is closing
    closing_speed = -(x * vx + y * vy) / max(r, 0.1)
    
    # Case A: Oncoming Traffic Ahead within roadway limits (|y| <= 5.5m) closing in
    if x > 0.0 and abs(y) <= 5.5 and (closing_speed >= min_closing_speed_mps or vx <= -min_closing_speed_mps):
        reach = min(r + 5.0, max_threat_distance_m)
        bearing = float(np.arctan2(y, x))
        return True, reach, bearing
        
    # Case B: Straight-On Obstacle in Direct Forward Travel Lane (|y| <= corridor)
    if x > 0.0 and abs(y) <= corridor_half_width_m:
        # If moving away, it is not an immediate collision threat
        if closing_speed < -0.5 or vx > 0.5:
            return False, 0.0, 0.0
        # Stationary or slow hazard in our direct lane within 25m
        if r <= 25.0:
            reach = min(r + 5.0, max_threat_distance_m)
            bearing = float(np.arctan2(y, x))
            return True, reach, bearing
            
    # Case C: Crossing Traffic Entering Lane (imminent collision course)
    if speed >= 0.8 and abs(y) > corridor_half_width_m and abs(y) <= 10.0:
        # Moving towards road centerline
        if (y > 0.0 and vy < -0.5) or (y < 0.0 and vy > 0.5):
            t_enter = (abs(y) - corridor_half_width_m) / max(abs(vy), 0.1)
            future_x = x + vx * t_enter
            if 0.0 <= future_x <= 25.0 and t_enter <= 3.0:
                future_r = float(np.hypot(future_x, 0.0))
                reach = min(future_r + 4.0, max_threat_distance_m)
                bearing = float(np.arctan2(y, x))
                return True, reach, bearing
                
    return False, 0.0, 0.0


def detect_moving_objects(
    tracks: list,
    speed_threshold_mps: float = MOTION_SPEED_THRESHOLD_MPS,
    collision_focus_only: bool = COLLISION_FOCUS_ONLY,
) -> List[Dict[str, Any]]:
    """
    Detects moving objects from active tracks by evaluating Kalman-estimated velocities.
    """
    if not tracks:
        return []
        
    results = []
    for track in tracks:
        if isinstance(track, dict):
            cls_id = track.get('class_id', 3)
            pos = track.get('position_xy', (0.0, 0.0))
            vel = track.get('velocity_xy', (0.0, 0.0))
            pred = track.get('predicted_next_xy', None)
        else:
            cls_id = getattr(track, 'class_id', 3)
            pos = getattr(track, 'position_xy', (0.0, 0.0))
            vel = getattr(track, 'velocity_xy', (0.0, 0.0))
            pred = getattr(track, 'predicted_next_xy', None)

        if cls_id != 3:
            continue

        speed = float(np.hypot(vel[0], vel[1])) if vel is not None else 0.0

        if pred is None and vel is not None:
            pred = (pos[0] + vel[0] * 0.1, pos[1] + vel[1] * 0.1)
        elif pred is None:
            pred = pos

        heading = float(np.arctan2(vel[1], vel[0])) if speed > 1e-3 else 0.0
        is_moving = speed >= speed_threshold_mps
        
        is_threat, threat_r, _ = is_oncoming_or_collision_threat(pos, vel)

        results.append({
            "track": track,
            "position_xy": (float(pos[0]), float(pos[1])),
            "velocity_xy": (float(vel[0]) if vel else 0.0, float(vel[1]) if vel else 0.0),
            "speed_mps": speed,
            "heading_rad": heading,
            "predicted_next_xy": (float(pred[0]), float(pred[1])),
            "is_moving": is_moving,
            "is_collision_threat": is_threat,
            "threat_reach_m": threat_r,
        })
    return results


def attention_modifier(
    theta_rad: Union[float, np.ndarray],
    active_tracks: list,
    speed_threshold_mps: float = MOTION_SPEED_THRESHOLD_MPS,
    lead_time_s: float = MOTION_LEAD_TIME_S,
    buffer_m: float = MOTION_BUFFER_M,
    boost: float = MOTION_ACUITY_BOOST,
    collision_focus_only: bool = COLLISION_FOCUS_ONLY,
    corridor_half_width_m: float = CORRIDOR_HALF_WIDTH_M,
    max_threat_distance_m: float = MAX_THREAT_DISTANCE_M,
) -> Union[float, np.ndarray]:
    """
    Calculates dynamic foveation radius spikes pointing towards active tracks.
    When collision_focus_only is True, ONLY oncoming vehicles and direct straight-on
    collision hazards receive dynamic foveal expansion, avoiding distraction by receding
    or far-away objects.
    """
    if not active_tracks:
        return np.zeros_like(theta_rad) if isinstance(theta_rad, np.ndarray) else 0.0
        
    r_attn = np.zeros_like(theta_rad) if isinstance(theta_rad, np.ndarray) else 0.0
    
    for track in active_tracks:
        if isinstance(track, dict):
            cls_id = track.get('class_id', 3)
            pos = track.get('position_xy', (0.0, 0.0))
            vel = track.get('velocity_xy', None)
            bbox = track.get('bbox_size_xy', (4.0, 2.0))
        else:
            cls_id = getattr(track, 'class_id', 3)
            pos = getattr(track, 'position_xy', (0.0, 0.0))
            vel = getattr(track, 'velocity_xy', None)
            bbox = getattr(track, 'bbox_size_xy', (4.0, 2.0))

        if cls_id != 3:
            continue

        tx, ty = float(pos[0]), float(pos[1])
        tr = float(np.hypot(tx, ty))
        
        # Threat evaluation
        if collision_focus_only:
            is_threat, reach, bearing = is_oncoming_or_collision_threat(
                (tx, ty), vel,
                corridor_half_width_m=corridor_half_width_m,
                max_threat_distance_m=max_threat_distance_m,
            )
            if not is_threat:
                continue
                
            # Smooth, natural angular spread (not a thin needle)
            sigma = max(0.35, (max(bbox) if bbox else 4.0) / max(tr, 1.0))
            diff = theta_rad - bearing
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            spike = reach * np.exp(-0.5 * (diff / sigma)**2)
        else:
            ttheta = float(np.arctan2(ty, tx))
            has_velocity = vel is not None and (abs(vel[0]) > 1e-3 or abs(vel[1]) > 1e-3)
            width = max(bbox) if bbox else 4.0
            sigma = max(width / max(tr, 1.0), 0.2)
            
            if has_velocity:
                speed = float(np.hypot(vel[0], vel[1]))
                pred_x = tx + vel[0] * lead_time_s
                pred_y = ty + vel[1] * lead_time_s
                pred_r = float(np.hypot(pred_x, pred_y))
                pred_theta = float(np.arctan2(pred_y, pred_x))
                reach = min(max(tr, pred_r) + buffer_m, max_threat_distance_m)
                diff = theta_rad - pred_theta
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                spike = reach * np.exp(-0.5 * (diff / sigma)**2)
            else:
                diff = theta_rad - ttheta
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                spike_height = min(tr + 5.0, max_threat_distance_m)
                spike = spike_height * np.exp(-0.5 * (diff / sigma)**2)
        
        if isinstance(theta_rad, np.ndarray):
            r_attn = np.maximum(r_attn, spike)
        else:
            r_attn = max(r_attn, float(spike))
            
    return r_attn


def fine_radius_at_angle(
    theta_rad: Union[float, np.ndarray],
    state: VehicleState,
    base_radius: float = BASE_FINE_RADIUS_M,
    max_stretch: float = MAX_STRETCH,
    speed_ref: float = STRETCH_SPEED_REF_MPS,
    active_tracks: list = None,
    motion_foveation_enabled: Optional[bool] = None,
    motion_speed_threshold_mps: Optional[float] = None,
    motion_lead_time_s: Optional[float] = None,
    collision_focus_only: Optional[bool] = None,
    corridor_half_width_m: Optional[float] = None,
    max_threat_distance_m: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Computes a Piecewise Elliptical foveation zone (ego-kinematics) combined with 
    Motion-Adaptive Dynamic Obstacle Attention lobes.
    """
    if motion_foveation_enabled is None:
        motion_foveation_enabled = MOTION_FOVEATION_ENABLED
    if motion_speed_threshold_mps is None:
        motion_speed_threshold_mps = MOTION_SPEED_THRESHOLD_MPS
    if motion_lead_time_s is None:
        motion_lead_time_s = MOTION_LEAD_TIME_S
    if collision_focus_only is None:
        collision_focus_only = COLLISION_FOCUS_ONLY
    if corridor_half_width_m is None:
        corridor_half_width_m = CORRIDOR_HALF_WIDTH_M
    if max_threat_distance_m is None:
        max_threat_distance_m = MAX_THREAT_DISTANCE_M

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
        
    # Apply attention / motion modifier if tracks are provided
    if active_tracks and motion_foveation_enabled:
        r_attn = attention_modifier(
            theta_rad,
            active_tracks,
            speed_threshold_mps=motion_speed_threshold_mps,
            lead_time_s=motion_lead_time_s,
            collision_focus_only=collision_focus_only,
            corridor_half_width_m=corridor_half_width_m,
            max_threat_distance_m=max_threat_distance_m,
        )
        if isinstance(r_fine, np.ndarray):
            r_fine = np.maximum(r_fine, r_attn)
        else:
            r_fine = max(r_fine, r_attn)
            
    return r_fine


