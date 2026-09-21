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
    """Verify that the zip archive is found and sequence 00 has >4,000 frames."""
    assert os.path.exists(real_kitti_loader.dataset_path), "Dataset archive path should exist"
    assert real_kitti_loader.is_zip, "Should identify archive as a valid zip file"
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

    # Scenarios should include all sequences
    s_res = client.get("/api/scenarios")
    assert s_res.status_code == 200
    scenarios = s_res.json()
    assert any(s["id"] == "kitti_seq_00" for s in scenarios)
    assert any(s["id"] == "kitti_seq_04" for s in scenarios)
