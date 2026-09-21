"""
Loader module for LiDAR point clouds.
Supports loading SemanticKITTI .bin and .label files, as well as
generating deterministic synthetic scans for testing without real data.
"""

import os
import numpy as np

def load_kitti_bin(bin_path: str) -> np.ndarray:
    """
    Reads a SemanticKITTI .bin file containing (x, y, z, intensity) as flat float32.
    Reshapes and returns as an (N, 4) numpy array.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"LiDAR bin file not found: {bin_path}")
        
    scan = np.fromfile(bin_path, dtype=np.float32)
    return scan.reshape((-1, 4))


def load_kitti_labels(label_path: str) -> np.ndarray:
    """
    Reads a SemanticKITTI .label file containing uint32 labels per point.
    Extracts the semantic class ID from the lower 16 bits and returns as an int32 array.
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"LiDAR label file not found: {label_path}")
        
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_labels = (raw_labels & 0xFFFF).astype(np.int32)
    return semantic_labels


def generate_synthetic_scan(seed: int = 0) -> np.ndarray:
    """
    Generates a deterministic synthetic point cloud for testing.
    Returns array of shape (N, 5): x, y, z, intensity, class_id.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Ground plane: ~4000 points scattered -60 to 60m
    num_ground = 4000
    ground_x = rng.uniform(-60, 60, num_ground)
    ground_y = rng.uniform(-60, 60, num_ground)
    ground_z = rng.normal(0.0, 0.05, num_ground)
    ground_int = rng.uniform(0.2, 0.4, num_ground)
    ground_cls = np.full(num_ground, 40)
    ground = np.column_stack([ground_x, ground_y, ground_z, ground_int, ground_cls])
    
    # 2. Vertical pole at (5, 2) from z=0 to z=3m
    num_pole = 50
    pole_x = np.full(num_pole, 5.0) + rng.normal(0, 0.02, num_pole)
    pole_y = np.full(num_pole, 2.0) + rng.normal(0, 0.02, num_pole)
    pole_z = np.linspace(0, 3.0, num_pole)
    pole_int = np.full(num_pole, 0.8)
    pole_cls = np.full(num_pole, 80)
    pole = np.column_stack([pole_x, pole_y, pole_z, pole_int, pole_cls])
    
    # 3. Pedestrian at (8, -3) from z=0 to z=1.8m with xy jitter
    num_ped = 100
    ped_x = np.full(num_ped, 8.0) + rng.normal(0, 0.2, num_ped)
    ped_y = np.full(num_ped, -3.0) + rng.normal(0, 0.2, num_ped)
    ped_z = rng.uniform(0, 1.8, num_ped)
    ped_int = rng.uniform(0.1, 0.3, num_ped)
    ped_cls = np.full(num_ped, 30)
    ped = np.column_stack([ped_x, ped_y, ped_z, ped_int, ped_cls])
    
    # 4. Curb ridge at x=12, z=0.15, spanning y=-10 to 10
    num_curb = 200
    curb_x = np.full(num_curb, 12.0) + rng.normal(0, 0.05, num_curb)
    curb_y = np.linspace(-10, 10, num_curb)
    curb_z = np.full(num_curb, 0.15) + rng.normal(0, 0.02, num_curb)
    curb_int = np.full(num_curb, 0.5)
    curb_cls = np.full(num_curb, 40)
    curb = np.column_stack([curb_x, curb_y, curb_z, curb_int, curb_cls])
    
    # 5. Overhang at x=20, z around 2.3m, plus matching ground points at z=0
    num_overhang = 80
    overhang_y = np.linspace(-1.5, 1.5, num_overhang)
    overhang_x = np.full(num_overhang, 20.0) + rng.normal(0, 0.05, num_overhang)
    
    overhang_sign_z = rng.normal(2.3, 0.1, num_overhang)
    overhang_sign_int = np.full(num_overhang, 0.9)
    overhang_sign_cls = np.full(num_overhang, 81)
    overhang_sign = np.column_stack([overhang_x, overhang_y, overhang_sign_z, overhang_sign_int, overhang_sign_cls])
    
    overhang_gnd_z = rng.normal(0.0, 0.02, num_overhang)
    overhang_gnd_int = np.full(num_overhang, 0.3)
    overhang_gnd_cls = np.full(num_overhang, 40)
    overhang_gnd = np.column_stack([overhang_x, overhang_y, overhang_gnd_z, overhang_gnd_int, overhang_gnd_cls])
    
    # 6. Far-field car cluster around (70, 5), z=0 to 1.5m
    num_car = 300
    car_x = rng.normal(70.0, 1.0, num_car)
    car_y = rng.normal(5.0, 0.8, num_car)
    car_z = rng.uniform(0.0, 1.5, num_car)
    car_int = rng.uniform(0.4, 0.7, num_car)
    car_cls = np.full(num_car, 10)
    car = np.column_stack([car_x, car_y, car_z, car_int, car_cls])
    
    # Combine everything
    points = np.vstack([
        ground, pole, ped, curb, overhang_sign, overhang_gnd, car
    ])
    
    # Shuffle the points to simulate real unsorted LiDAR data
    rng.shuffle(points)
    
    return points.astype(np.float32)
