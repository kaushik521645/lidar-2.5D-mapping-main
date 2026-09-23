"""
Unit tests verifying integration of the genuine KITTI Odometry Velodyne dataset.

Tests loading directly from data_odometry_velodyne.zip, verifying genuine sensor
signatures, sequence selection, and end-to-end 2.5D polar grid projection.
"""

import os
import pytest
import numpy as np

from src.ingestion.kitti_loader import KITTILoader
from src.perception.heuristic_fallback import heuristic_segment_points
from src.grid.grid_engine import PolarGridEngine
from src.grid.grid_types import VehicleState


@pytest.fixture(scope="module")
def real_kitti_loader():
    """Provides a KITTILoader pointed at the provided dataset archive."""
    loader = KITTILoader("data_odometry_velodyne.zip", sequence="00")
    yield loader
    loader.close()


def test_real_kitti_dataset_discovery(real_kitti_loader):
    """Verify that the dataset is found and sequence 00 has >4,000 frames."""
    assert os.path.exists(real_kitti_loader.dataset_path), "Dataset path should exist"
    assert os.path.isdir(real_kitti_loader.dataset_path) or real_kitti_loader.is_zip
    assert len(real_kitti_loader) > 4000, f"Expected >4000 frames in sequence 00, got {len(real_kitti_loader)}"


def test_real_kitti_sequences_list(real_kitti_loader):
    """Verify all 22 sequences (00 to 21) are detected in the dataset archive."""
    seqs = real_kitti_loader.list_sequences()
    assert len(seqs) >= 22, f"Expected >=22 sequences, found: {seqs}"
    assert "00" in seqs and "21" in seqs


def test_real_kitti_physical_signature(real_kitti_loader):
    """Verify that genuine physical LiDAR signature is detected (non-uniform beam returns)."""
    assert not real_kitti_loader.is_synthetic_signature, (
        "Dataset should NOT be flagged as synthetic stand-in; genuine physical LiDAR scans vary in size"
    )


def test_real_kitti_scan_dimensions_and_types(real_kitti_loader):
    """Verify that loaded scan has shape (N, 5) with float32 and reasonable LiDAR point counts."""
    scan = real_kitti_loader[0]
    assert isinstance(scan, np.ndarray)
    assert scan.ndim == 2
    assert scan.shape[1] == 5
    assert scan.dtype == np.float32
    assert scan.shape[0] > 100000, f"Expected >100,000 points in raw HDL-64E scan, got {scan.shape[0]}"


def test_real_kitti_grid_projection_pipeline(real_kitti_loader):
    """Verify end-to-end projection of a real KITTI scan into the 2.5D PolarGridEngine."""
    scan = real_kitti_loader[0]
    points_xyz = scan[:, :3]

    # Segment real LiDAR points using robust heuristic
    classes = heuristic_segment_points(points_xyz)
    assert len(classes) == len(points_xyz)
    assert np.all((classes >= 0) & (classes <= 3))

    points_5d = np.column_stack([scan[:, :4], classes])

    engine = PolarGridEngine()
    v_state = VehicleState(speed_mps=10.0, steering_angle_rad=0.0)
    grid_map = engine.project_to_grid(
        points=points_5d,
        vehicle_state=v_state,
        frame_id=0,
        roof_height_m=2.5,
        use_foveation=True
    )

    assert len(grid_map) > 1000, f"Grid map should contain thousands of active polar cells, got {len(grid_map)}"

    # Verify polar cell elevations and classes
    for cell in list(grid_map.values())[:100]:
        assert cell.elevation_ground is not None
        assert cell.semantic_class in (0, 1, 2, 3)


def test_real_kitti_sequence_switching(real_kitti_loader):
    """Verify switching across arbitrary sequences in the dataset."""
    test_seqs = ["01", "04", "10", "21"]
    for s in test_seqs:
        real_kitti_loader.set_sequence(s)
        assert real_kitti_loader.sequence == s
        assert len(real_kitti_loader) > 100, f"Expected >100 scans for sequence {s}, got {len(real_kitti_loader)}"
        scan = real_kitti_loader[0]
        assert scan.shape[1] == 5
        assert scan.shape[0] > 100000


def test_server_api_sequences_and_scenarios():
    """Verify FastAPI endpoints return all sequences."""
    from fastapi.testclient import TestClient
    from src.api.server import app

    client = TestClient(app)
    res = client.get("/api/sequences")
    assert res.status_code == 200
    seqs = res.json()
    assert len(seqs) >= 22
    assert any(s["sequence"] == "00" for s in seqs)
    assert any(s["sequence"] == "21" for s in seqs)

    # Scenarios should include all sequences and is_precomputed metadata
    s_res = client.get("/api/scenarios")
    assert s_res.status_code == 200
    scenarios = s_res.json()
    assert any(s["id"] == "kitti_seq_00" for s in scenarios)
    assert any(s["id"] == "kitti_seq_04" for s in scenarios)
    assert "is_precomputed" in scenarios[0]
    assert isinstance(scenarios[0]["is_precomputed"], bool)

    # Test foveation contour endpoint
    f_res = client.get("/api/foveation/contour?speed_mps=12.0&steering_angle_rad=0.3&shear_strength=0.6")
    assert f_res.status_code == 200
    contour_data = f_res.json()
    assert "polyline" in contour_data
    assert len(contour_data["polyline"]) == 120

    # Test WebSocket auto-loop control command
    with client.websocket_connect("/ws/grid") as ws:
        ws.send_json({"command": "set_auto_loop", "enabled": True})
        ws.send_json({"command": "next_scenario"})
        ws.send_json({"command": "prev_scenario"})


def test_frame2_not_mass_labeled_dynamic(real_kitti_loader):
    """
    Regression test for Step 2 mass misclassification:
    Verify that in KITTI Sequence 00 frame 2, points are NOT mass-labeled as
    dynamic (Class 3). The majority must be terrain (0/1) and static obstacles (2),
    with Class 3 < 5% of all points, and compact vehicle-sized clusters cleanly
    passing the 200-cell / 14m clustering filter.
    """
    from src.tracking.kalman_tracker import cluster_dynamic_detections

    real_kitti_loader.set_sequence("00")
    scan2 = real_kitti_loader[2]
    pts = scan2[:, :3]
    labels = heuristic_segment_points(pts)

    tot = len(pts)
    c0 = np.sum(labels == 0)
    c1 = np.sum(labels == 1)
    c2 = np.sum(labels == 2)
    c3 = np.sum(labels == 3)

    # Class 3 must be a small fraction of scene, not mass-labeled (> 50%)
    c3_pct = c3 / tot * 100.0
    assert c3_pct < 5.0, f"Frame 2 dynamic points should be < 5%, got {c3_pct:.1f}%"
    assert (c0 + c1) > c3, "Terrain (Class 0 + 1) should exceed dynamic objects"
    assert c2 > c3, "Static obstacles (Class 2) should exceed dynamic objects"

    # End-to-end grid projection + clustering: verify clusters pass filter
    engine = PolarGridEngine()
    points_5d = np.column_stack([pts, np.zeros(tot), labels])
    grid_map = engine.project_to_grid(points_5d, vehicle_state=VehicleState(10.0, 0.0), frame_id=2)

    raw_c3 = [
        engine.get_cell_spatial_center(k[1], k[2], cell.resolution_tier)[:2] + (3,)
        for k, cell in grid_map.items()
        if cell.semantic_class == 3
    ]
    clusters = cluster_dynamic_detections(raw_c3, cluster_dist_m=2.5)
    assert len(clusters) >= 5, (
        f"Vehicle-scale clusters must pass the 200-cell/14m filter, got {len(clusters)}"
    )


def test_real_moving_vehicle_detected_as_dynamic(real_kitti_loader):
    """
    Regression test ensuring Fix 2 does NOT overcorrect into zero dynamic detections.
    Tests frame 10 of Sequence 00 where moving traffic is present, verifying that
    genuine dynamic obstacles are still detected.
    """
    real_kitti_loader.set_sequence("00")
    scan10 = real_kitti_loader[10]
    pts10 = scan10[:, :3]
    labels10 = heuristic_segment_points(pts10)

    tot10 = len(pts10)
    c3_10 = np.sum(labels10 == 3)

    assert c3_10 > 500, f"Expected > 500 dynamic points in frame 10 traffic, got {c3_10}"
    assert c3_10 / tot10 < 0.10, f"Dynamic points should remain < 10% of total scan, got {c3_10/tot10*100:.1f}%"


