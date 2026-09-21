"""
Synthetic KITTI-Format Procedural Point Cloud Generator.
Generates 30 procedural sequential LiDAR frames with velodyne (.bin) and ground-truth labels (.label)
in standard SemanticKITTI binary format for testing, training, and benchmarking.
"""

import os
import numpy as np


def generate_kitti_sample_dataset(output_dir: str = "data/synthetic_kitti_like", num_frames: int = 30) -> None:
    """
    Generates a 30-frame procedural synthetic LiDAR sequence formatted like SemanticKITTI.
    """
    velo_dir = os.path.join(output_dir, "velodyne")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(velo_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    np.random.seed(42)

    for frame_idx in range(num_frames):
        t = frame_idx * 0.1  # 10 Hz
        points_list = []
        labels_list = []

        # 1. Road Surface (Class 40 = road -> Drivable)
        # Dimensions: x in [-15, 60], y in [-4, 4], z ~ -1.5m with gentle slope
        num_road = 6000
        road_x = np.random.uniform(-10.0, 55.0, num_road)
        road_y = np.random.uniform(-4.0, 4.0, num_road)
        road_z = -1.50 + 0.005 * road_x + np.random.normal(0.0, 0.02, num_road)
        road_i = np.random.uniform(0.1, 0.3, num_road)
        road_label = np.full(num_road, 40, dtype=np.uint32)

        points_list.append(np.column_stack([road_x, road_y, road_z, road_i]))
        labels_list.append(road_label)

        # 2. Sidewalk / Curbs (Class 48 = sidewalk -> Non-Drivable)
        # Left and right sidewalks: y in [-7, -4] and [4, 7], z ~ -1.35m (curb height +0.15m)
        num_sw = 3000
        sw_left_x = np.random.uniform(-10.0, 55.0, num_sw // 2)
        sw_left_y = np.random.uniform(4.0, 7.5, num_sw // 2)
        sw_left_z = -1.35 + 0.005 * sw_left_x + np.random.normal(0.0, 0.02, num_sw // 2)
        sw_left_i = np.random.uniform(0.2, 0.4, num_sw // 2)

        sw_right_x = np.random.uniform(-10.0, 55.0, num_sw // 2)
        sw_right_y = np.random.uniform(-7.5, -4.0, num_sw // 2)
        sw_right_z = -1.35 + 0.005 * sw_right_x + np.random.normal(0.0, 0.02, num_sw // 2)
        sw_right_i = np.random.uniform(0.2, 0.4, num_sw // 2)

        sw_x = np.concatenate([sw_left_x, sw_right_x])
        sw_y = np.concatenate([sw_left_y, sw_right_y])
        sw_z = np.concatenate([sw_left_z, sw_right_z])
        sw_i = np.concatenate([sw_left_i, sw_right_i])
        sw_label = np.full(num_sw, 48, dtype=np.uint32)

        points_list.append(np.column_stack([sw_x, sw_y, sw_z, sw_i]))
        labels_list.append(sw_label)

        # 3. Static Obstacles: Buildings / Walls (Class 50) and Poles (Class 80)
        # Buildings along y = 8.0 and y = -8.0
        num_bldg = 2500
        bldg_left_x = np.random.uniform(-5.0, 50.0, num_bldg // 2)
        bldg_left_y = np.random.uniform(8.0, 8.5, num_bldg // 2)
        bldg_left_z = np.random.uniform(-1.3, 2.5, num_bldg // 2)
        bldg_left_i = np.random.uniform(0.5, 0.9, num_bldg // 2)

        bldg_right_x = np.random.uniform(-5.0, 50.0, num_bldg // 2)
        bldg_right_y = np.random.uniform(-8.5, -8.0, num_bldg // 2)
        bldg_right_z = np.random.uniform(-1.3, 2.5, num_bldg // 2)
        bldg_right_i = np.random.uniform(0.5, 0.9, num_bldg // 2)

        bldg_x = np.concatenate([bldg_left_x, bldg_right_x])
        bldg_y = np.concatenate([bldg_left_y, bldg_right_y])
        bldg_z = np.concatenate([bldg_left_z, bldg_right_z])
        bldg_i = np.concatenate([bldg_left_i, bldg_right_i])
        bldg_label = np.full(num_bldg, 50, dtype=np.uint32)

        points_list.append(np.column_stack([bldg_x, bldg_y, bldg_z, bldg_i]))
        labels_list.append(bldg_label)

        # Poles along sidewalk edges (Class 80)
        pole_positions = [(10.0, 4.2), (25.0, 4.2), (40.0, 4.2), (15.0, -4.2), (30.0, -4.2)]
        for px, py in pole_positions:
            num_p = 80
            pz = np.random.uniform(-1.35, 1.5, num_p)
            p_x = px + np.random.normal(0.0, 0.05, num_p)
            p_y = py + np.random.normal(0.0, 0.05, num_p)
            p_i = np.random.uniform(0.7, 1.0, num_p)
            p_label = np.full(num_p, 80, dtype=np.uint32)
            points_list.append(np.column_stack([p_x, p_y, pz, p_i]))
            labels_list.append(p_label)

        # 4. Dynamic Objects (Class 10 = car, Class 30 = person)
        # Car 1: Leading vehicle moving forward at 8 m/s (x = 15.0 + 8.0 * t, y = 1.8)
        car1_x_center = 15.0 + 8.0 * t
        car1_y_center = 1.8
        num_car1 = 400
        c1_x = car1_x_center + np.random.uniform(-2.0, 2.0, num_car1)
        c1_y = car1_y_center + np.random.uniform(-0.9, 0.9, num_car1)
        c1_z = np.random.uniform(-1.4, 0.2, num_car1)
        c1_i = np.random.uniform(0.4, 0.8, num_car1)
        c1_label = np.full(num_car1, 10, dtype=np.uint32) | (1 << 16)  # Instance 1

        points_list.append(np.column_stack([c1_x, c1_y, c1_z, c1_i]))
        labels_list.append(c1_label)

        # Car 2: Oncoming vehicle (x = 45.0 - 10.0 * t, y = -1.8)
        car2_x_center = 45.0 - 10.0 * t
        car2_y_center = -1.8
        num_car2 = 350
        c2_x = car2_x_center + np.random.uniform(-2.0, 2.0, num_car2)
        c2_y = car2_y_center + np.random.uniform(-0.9, 0.9, num_car2)
        c2_z = np.random.uniform(-1.4, 0.2, num_car2)
        c2_i = np.random.uniform(0.4, 0.8, num_car2)
        c2_label = np.full(num_car2, 10, dtype=np.uint32) | (2 << 16)  # Instance 2

        points_list.append(np.column_stack([c2_x, c2_y, c2_z, c2_i]))
        labels_list.append(c2_label)

        # Pedestrian: Walking across sidewalk (x = 8.0, y = 5.5 - 0.8 * t)
        ped_x_center = 8.0
        ped_y_center = 5.5 - 0.8 * t
        num_ped = 120
        ped_x = ped_x_center + np.random.uniform(-0.3, 0.3, num_ped)
        ped_y = ped_y_center + np.random.uniform(-0.3, 0.3, num_ped)
        ped_z = np.random.uniform(-1.35, 0.4, num_ped)
        ped_i = np.random.uniform(0.3, 0.7, num_ped)
        ped_label = np.full(num_ped, 30, dtype=np.uint32) | (3 << 16)  # Instance 3

        points_list.append(np.column_stack([ped_x, ped_y, ped_z, ped_i]))
        labels_list.append(ped_label)

        # Combine and save
        all_points = np.vstack(points_list).astype(np.float32)
        all_labels = np.concatenate(labels_list).astype(np.uint32)

        bin_file = os.path.join(velo_dir, f"{frame_idx:06d}.bin")
        label_file = os.path.join(labels_dir, f"{frame_idx:06d}.label")

        all_points.tofile(bin_file)
        all_labels.tofile(label_file)

    print(f"[INFO] Successfully generated {num_frames} SemanticKITTI sample frames in {output_dir}")


if __name__ == "__main__":
    generate_kitti_sample_dataset()
