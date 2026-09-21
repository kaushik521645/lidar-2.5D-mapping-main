"""
Procedural Synthetic Scenario Generator for FoveaMap.

Generates 4 distinct, rich autonomous driving scenarios with ground truth labels
and ego vehicle dynamics:
  1. Urban Intersection: Cross-traffic, pedestrians, curbs, sidewalks, buildings.
  2. Highway Cruise: High-speed driving (22 m/s), forward foveation elongation, lane dividers.
  3. Pothole & Rough Terrain Alley: Drivable asphalt with non-drivable potholes and gravel verges.
  4. Bridge Overpass: Drivable underpass with overhead bridge ceiling and structural pillars.
"""

from typing import List, Dict, Tuple, Any
import numpy as np

from src.grid.grid_types import VehicleState


class ScenarioGenerator:
    """
    Generates procedural scenarios as sequences of (PointCloud, VehicleState, Metadata).
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_urban_intersection(self, num_frames: int = 20) -> List[Tuple[np.ndarray, VehicleState, Dict[str, Any]]]:
        """
        Scenario 1: Urban Intersection with crossing traffic, pedestrians, and turning ego.
        """
        rng = np.random.default_rng(self.seed)
        frames = []

        for f in range(num_frames):
            t = f * 0.1
            # Ego vehicle slowing down and turning right
            speed = max(3.0, 10.0 - 0.35 * f)
            steer = min(0.35, 0.02 * f)  # Right turn
            v_state = VehicleState(speed_mps=float(speed), steering_angle_rad=float(steer))

            points = []
            labels = []

            # 1. Road Surface (Class 0: Drivable) - Grid cross intersection
            # Longitudinal road: x in [-15, 45], y in [-5, 5]
            n_road1 = 4000
            r1_x = rng.uniform(-15.0, 45.0, n_road1)
            r1_y = rng.uniform(-5.0, 5.0, n_road1)
            r1_z = rng.normal(-1.50, 0.02, n_road1)
            points.append(np.column_stack([r1_x, r1_y, r1_z, rng.uniform(0.2, 0.4, n_road1)]))
            labels.append(np.full(n_road1, 0, dtype=np.int32))

            # Cross road: x in [10, 25], y in [-35, 35]
            n_road2 = 3000
            r2_x = rng.uniform(10.0, 25.0, n_road2)
            r2_y = rng.uniform(-35.0, 35.0, n_road2)
            r2_z = rng.normal(-1.50, 0.02, n_road2)
            points.append(np.column_stack([r2_x, r2_y, r2_z, rng.uniform(0.2, 0.4, n_road2)]))
            labels.append(np.full(n_road2, 0, dtype=np.int32))

            # 2. Sidewalks & Curbs (Class 1: Non-Drivable)
            # Corner sidewalks
            n_sw = 2000
            sw_x = rng.uniform(-15.0, 8.0, n_sw)
            sw_y = rng.uniform(5.5, 12.0, n_sw)
            sw_z = rng.normal(-1.35, 0.02, n_sw)  # 15cm curb height
            points.append(np.column_stack([sw_x, sw_y, sw_z, rng.uniform(0.3, 0.5, n_sw)]))
            labels.append(np.full(n_sw, 1, dtype=np.int32))

            # 3. Static Obstacles: Buildings & Traffic Poles (Class 2)
            # Buildings at corners
            n_bldg = 2000
            b_x = rng.uniform(-15.0, 8.0, n_bldg)
            b_y = rng.uniform(12.0, 14.0, n_bldg)
            b_z = rng.uniform(-1.35, 2.5, n_bldg)
            points.append(np.column_stack([b_x, b_y, b_z, rng.uniform(0.6, 0.9, n_bldg)]))
            labels.append(np.full(n_bldg, 2, dtype=np.int32))

            # Traffic Signal Poles
            for px, py in [(9.0, 5.5), (9.0, -5.5), (26.0, 5.5), (26.0, -5.5)]:
                n_p = 60
                pz = rng.uniform(-1.35, 2.2, n_p)
                p_x = px + rng.normal(0, 0.04, n_p)
                p_y = py + rng.normal(0, 0.04, n_p)
                points.append(np.column_stack([p_x, p_y, pz, rng.uniform(0.8, 1.0, n_p)]))
                labels.append(np.full(n_p, 2, dtype=np.int32))

            # 4. Dynamic Objects (Class 3)
            # Cross traffic vehicle moving left-to-right (x = 18.0, y = -25.0 + 12.0 * t)
            c_x_center = 18.0
            c_y_center = -25.0 + 12.0 * t
            n_c = 350
            c_x = c_x_center + rng.uniform(-1.0, 1.0, n_c)
            c_y = c_y_center + rng.uniform(-2.2, 2.2, n_c)
            c_z = rng.uniform(-1.4, 0.2, n_c)
            points.append(np.column_stack([c_x, c_y, c_z, rng.uniform(0.5, 0.8, n_c)]))
            labels.append(np.full(n_c, 3, dtype=np.int32))

            # Pedestrian crossing crosswalk (x = 9.5, y = 4.5 - 1.0 * t)
            ped_x_center = 9.5
            ped_y_center = 4.5 - 1.0 * t
            n_ped = 100
            ped_x = ped_x_center + rng.uniform(-0.25, 0.25, n_ped)
            ped_y = ped_y_center + rng.uniform(-0.25, 0.25, n_ped)
            ped_z = rng.uniform(-1.35, 0.35, n_ped)
            points.append(np.column_stack([ped_x, ped_y, ped_z, rng.uniform(0.4, 0.7, n_ped)]))
            labels.append(np.full(n_ped, 3, dtype=np.int32))

            pts_arr = np.vstack(points).astype(np.float32)
            lbl_arr = np.concatenate(labels).astype(np.int32)
            cloud = np.column_stack([pts_arr[:, :4], lbl_arr.astype(np.float32)])

            meta = {
                "scenario_name": "Urban Intersection",
                "frame_id": f,
                "timestamp_s": t,
                "description": "Cross-traffic vehicle and pedestrian crossing ahead with ego right turn.",
            }
            frames.append((cloud, v_state, meta))

        return frames

    def generate_highway_cruise(self, num_frames: int = 20) -> List[Tuple[np.ndarray, VehicleState, Dict[str, Any]]]:
        """
        Scenario 2: Highway Cruise at 22 m/s (~80 km/h) demonstrating forward foveation elongation.
        """
        rng = np.random.default_rng(self.seed + 1)
        frames = []

        for f in range(num_frames):
            t = f * 0.1
            # High forward speed, straight trajectory
            speed = 22.0
            steer = 0.0
            v_state = VehicleState(speed_mps=speed, steering_angle_rad=steer)

            points = []
            labels = []

            # 1. 3-Lane Highway Surface (Class 0: Drivable)
            # x in [-20, 85], y in [-6.5, 6.5], z = -1.5m
            n_road = 6000
            rx = rng.uniform(-20.0, 85.0, n_road)
            ry = rng.uniform(-6.5, 6.5, n_road)
            rz = rng.normal(-1.50, 0.015, n_road)
            points.append(np.column_stack([rx, ry, rz, rng.uniform(0.2, 0.4, n_road)]))
            labels.append(np.full(n_road, 0, dtype=np.int32))

            # 2. Highway Guardrails / Barriers (Class 2: Static Obstacle)
            # Left and right steel guardrails at y = -7.0 and y = 7.0
            n_guard = 1500
            gx_left = rng.uniform(-20.0, 85.0, n_guard // 2)
            gy_left = np.full(n_guard // 2, 7.0) + rng.normal(0, 0.03, n_guard // 2)
            gz_left = rng.uniform(-1.4, -0.6, n_guard // 2)

            gx_right = rng.uniform(-20.0, 85.0, n_guard // 2)
            gy_right = np.full(n_guard // 2, -7.0) + rng.normal(0, 0.03, n_guard // 2)
            gz_right = rng.uniform(-1.4, -0.6, n_guard // 2)

            gx = np.concatenate([gx_left, gx_right])
            gy = np.concatenate([gy_left, gy_right])
            gz = np.concatenate([gz_left, gz_right])
            points.append(np.column_stack([gx, gy, gz, rng.uniform(0.7, 0.9, n_guard)]))
            labels.append(np.full(n_guard, 2, dtype=np.int32))

            # 3. Grass Verge outside guardrails (Class 1: Non-Drivable)
            n_verge = 1200
            vx = rng.uniform(-20.0, 85.0, n_verge)
            vy = np.where(rng.uniform(0, 1, n_verge) > 0.5, rng.uniform(7.2, 12.0, n_verge), rng.uniform(-12.0, -7.2, n_verge))
            vz = rng.normal(-1.3, 0.05, n_verge)
            points.append(np.column_stack([vx, vy, vz, rng.uniform(0.1, 0.3, n_verge)]))
            labels.append(np.full(n_verge, 1, dtype=np.int32))

            # 4. Lead Vehicles (Class 3: Dynamic Object)
            # Vehicle 1: Center lane ahead at x = 35.0 + 20.0 * t, y = 0.0
            v1_x_center = 35.0 + 2.0 * t
            v1_y_center = 0.0
            n_v1 = 400
            v1_x = v1_x_center + rng.uniform(-2.2, 2.2, n_v1)
            v1_y = v1_y_center + rng.uniform(-0.9, 0.9, n_v1)
            v1_z = rng.uniform(-1.4, 0.1, n_v1)
            points.append(np.column_stack([v1_x, v1_y, v1_z, rng.uniform(0.5, 0.8, n_v1)]))
            labels.append(np.full(n_v1, 3, dtype=np.int32))

            # Vehicle 2: Left lane overtaking at x = 20.0 + 5.0 * t, y = 4.0
            v2_x_center = 20.0 + 5.0 * t
            v2_y_center = 4.0
            n_v2 = 350
            v2_x = v2_x_center + rng.uniform(-2.2, 2.2, n_v2)
            v2_y = v2_y_center + rng.uniform(-0.9, 0.9, n_v2)
            v2_z = rng.uniform(-1.4, 0.1, n_v2)
            points.append(np.column_stack([v2_x, v2_y, v2_z, rng.uniform(0.5, 0.8, n_v2)]))
            labels.append(np.full(n_v2, 3, dtype=np.int32))

            pts_arr = np.vstack(points).astype(np.float32)
            lbl_arr = np.concatenate(labels).astype(np.int32)
            cloud = np.column_stack([pts_arr[:, :4], lbl_arr.astype(np.float32)])

            meta = {
                "scenario_name": "Highway Cruise",
                "frame_id": f,
                "timestamp_s": t,
                "description": "High-speed cruising at 22 m/s with forward foveation stretch (up to 25m fine zone).",
            }
            frames.append((cloud, v_state, meta))

        return frames

    def generate_pothole_alley(self, num_frames: int = 20) -> List[Tuple[np.ndarray, VehicleState, Dict[str, Any]]]:
        """
        Scenario 3: Pothole & Rough Terrain Alley (Testing Req 1: Drivable vs Non-Drivable terrain).
        """
        rng = np.random.default_rng(self.seed + 2)
        frames = []

        # Fixed pothole centers in world frame
        potholes = [(12.0, 1.0, 1.2), (25.0, -1.2, 1.5), (38.0, 0.5, 1.0)]

        for f in range(num_frames):
            t = f * 0.1
            v_state = VehicleState(speed_mps=6.0, steering_angle_rad=0.0)

            points = []
            labels = []

            # 1. Main Asphalt Surface (Class 0: Drivable)
            n_road = 5000
            rx = rng.uniform(-10.0, 50.0, n_road)
            ry = rng.uniform(-3.5, 3.5, n_road)
            rz = rng.normal(-1.50, 0.02, n_road)

            # Check if points fall inside potholes (depression with depth -0.15m and classified as Class 1)
            is_pothole = np.zeros(n_road, dtype=bool)
            for px, py, pr in potholes:
                dist = np.hypot(rx - px, ry - py)
                inside = dist <= pr
                rz[inside] -= 0.18  # Depressed pothole bed
                is_pothole |= inside

            road_pts = np.column_stack([rx, ry, rz, rng.uniform(0.2, 0.4, n_road)])
            road_lbls = np.where(is_pothole, 1, 0).astype(np.int32)  # Potholes = Non-Drivable Terrain (Class 1)
            points.append(road_pts)
            labels.append(road_lbls)

            # 2. Gravel and Broken Edges (Class 1: Non-Drivable)
            n_edge = 1500
            ex = rng.uniform(-10.0, 50.0, n_edge)
            ey = np.where(rng.uniform(0, 1, n_edge) > 0.5, rng.uniform(3.5, 5.5, n_edge), rng.uniform(-5.5, -3.5, n_edge))
            ez = rng.normal(-1.42, 0.06, n_edge)
            points.append(np.column_stack([ex, ey, ez, rng.uniform(0.1, 0.3, n_edge)]))
            labels.append(np.full(n_edge, 1, dtype=np.int32))

            # 3. Alley Brick Walls (Class 2: Static Obstacle)
            n_wall = 2000
            wx = rng.uniform(-10.0, 50.0, n_wall)
            wy = np.where(rng.uniform(0, 1, n_wall) > 0.5, rng.uniform(5.5, 6.0, n_wall), rng.uniform(-6.0, -5.5, n_wall))
            wz = rng.uniform(-1.4, 2.5, n_wall)
            points.append(np.column_stack([wx, wy, wz, rng.uniform(0.6, 0.9, n_wall)]))
            labels.append(np.full(n_wall, 2, dtype=np.int32))

            pts_arr = np.vstack(points).astype(np.float32)
            lbl_arr = np.concatenate(labels).astype(np.int32)
            cloud = np.column_stack([pts_arr[:, :4], lbl_arr.astype(np.float32)])

            meta = {
                "scenario_name": "Pothole & Rough Terrain Alley",
                "frame_id": f,
                "timestamp_s": t,
                "description": "Distinguishes smooth drivable road (Class 0) from hazardous potholes and broken verges (Class 1).",
            }
            frames.append((cloud, v_state, meta))

        return frames

    def generate_bridge_overpass(self, num_frames: int = 20) -> List[Tuple[np.ndarray, VehicleState, Dict[str, Any]]]:
        """
        Scenario 4: Bridge Overpass (Testing Multi-Layer Elevation & Overhangs).
        """
        rng = np.random.default_rng(self.seed + 3)
        frames = []

        # Bridge deck extends from x = 18.0 to x = 28.0 across all y
        bridge_x_start = 18.0
        bridge_x_end = 28.0

        for f in range(num_frames):
            t = f * 0.1
            v_state = VehicleState(speed_mps=8.0, steering_angle_rad=0.0)

            points = []
            labels = []

            # 1. Drivable Road Surface passing under bridge (Class 0)
            n_road = 5000
            rx = rng.uniform(-10.0, 50.0, n_road)
            ry = rng.uniform(-4.5, 4.5, n_road)
            rz = rng.normal(-1.50, 0.02, n_road)
            points.append(np.column_stack([rx, ry, rz, rng.uniform(0.2, 0.4, n_road)]))
            labels.append(np.full(n_road, 0, dtype=np.int32))

            # 2. Bridge Overpass Deck (Class 2: Static Obstacle)
            # Located at x in [18, 28], y in [-15, 15], z in [1.8, 3.2] (Clearance ~3.3m above road)
            n_bridge = 2500
            bx = rng.uniform(bridge_x_start, bridge_x_end, n_bridge)
            by = rng.uniform(-15.0, 15.0, n_bridge)
            bz = rng.uniform(1.8, 3.2, n_bridge)  # Overhang ceiling
            points.append(np.column_stack([bx, by, bz, rng.uniform(0.7, 0.9, n_bridge)]))
            labels.append(np.full(n_bridge, 2, dtype=np.int32))

            # 3. Bridge Concrete Support Pillars (Class 2: Static Obstacle)
            # Pillars at (23.0, 5.0) and (23.0, -5.0) spanning z in [-1.5, 2.5]
            for px, py in [(23.0, 5.2), (23.0, -5.2)]:
                n_pil = 150
                pil_x = px + rng.uniform(-0.6, 0.6, n_pil)
                pil_y = py + rng.uniform(-0.6, 0.6, n_pil)
                pil_z = rng.uniform(-1.5, 2.5, n_pil)
                points.append(np.column_stack([pil_x, pil_y, pil_z, rng.uniform(0.8, 1.0, n_pil)]))
                labels.append(np.full(n_pil, 2, dtype=np.int32))

            # 4. Sidewalks along underpass (Class 1)
            n_sw = 1200
            sx = rng.uniform(-10.0, 50.0, n_sw)
            sy = np.where(rng.uniform(0, 1, n_sw) > 0.5, rng.uniform(4.5, 7.0, n_sw), rng.uniform(-7.0, -4.5, n_sw))
            sz = rng.normal(-1.35, 0.02, n_sw)
            points.append(np.column_stack([sx, sy, sz, rng.uniform(0.3, 0.5, n_sw)]))
            labels.append(np.full(n_sw, 1, dtype=np.int32))

            pts_arr = np.vstack(points).astype(np.float32)
            lbl_arr = np.concatenate(labels).astype(np.int32)
            cloud = np.column_stack([pts_arr[:, :4], lbl_arr.astype(np.float32)])

            meta = {
                "scenario_name": "Bridge Overpass",
                "frame_id": f,
                "timestamp_s": t,
                "description": "Multi-layer underpass: ground surface (z=-1.5m) and bridge deck ceiling (z=1.8 to 3.2m).",
            }
            frames.append((cloud, v_state, meta))

        return frames

    def get_all_scenarios(self) -> Dict[str, List[Tuple[np.ndarray, VehicleState, Dict[str, Any]]]]:
        """Returns dictionary of all 4 synthetic scenario sequences."""
        return {
            "urban_intersection": self.generate_urban_intersection(),
            "highway_cruise": self.generate_highway_cruise(),
            "pothole_alley": self.generate_pothole_alley(),
            "bridge_overpass": self.generate_bridge_overpass(),
        }
