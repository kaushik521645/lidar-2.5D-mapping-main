"""
Tests for Kalman Tracker.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking.kalman_tracker import update_tracks


def test_constant_velocity_tracking():
    """
    A single object moving in a straight line at constant velocity across 5 synthetic frames.
    The tracker's predicted_next_xy should stay within a small tolerance of the true next position.
    """
    dt = 0.1
    # Constant velocity: vx = 2.0 m/s, vy = 1.0 m/s
    vx, vy = 2.0, 1.0
    
    tracks = []
    
    for frame in range(5):
        true_x = frame * vx * dt
        true_y = frame * vy * dt
        
        # Single detection
        detections = [(true_x, true_y, 2)]
        
        tracks = update_tracks(tracks, detections, dt)
        
        assert len(tracks) == 1
        track = tracks[0]
        assert track.track_id == 1
        
        # Check prediction for the NEXT frame after observing enough to establish velocity
        if frame >= 2:
            next_true_x = (frame + 1) * vx * dt
            next_true_y = (frame + 1) * vy * dt
            
            # The filter should start converging its prediction to the true next position
            pred_x, pred_y = track.predicted_next_xy
            assert pred_x == pytest.approx(next_true_x, abs=0.15)
            assert pred_y == pytest.approx(next_true_y, abs=0.15)


def test_occlusion_reassociation():
    """
    An object disappears for 2 frames (occlusion) then reappears nearby.
    It should be re-associated to the SAME track_id, not create a new one, 
    as long as it's within MAX_COAST_FRAMES.
    """
    dt = 0.1
    tracks = []
    
    # Frame 0: object appears
    tracks = update_tracks(tracks, [(0.0, 0.0, 2)], dt)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    
    # Frame 1: object moves slightly
    tracks = update_tracks(tracks, [(0.1, 0.1, 2)], dt)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    
    # Frame 2: OCCLUDED (no detections)
    tracks = update_tracks(tracks, [], dt)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].frames_since_seen == 1
    
    # Frame 3: OCCLUDED (no detections)
    tracks = update_tracks(tracks, [], dt)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].frames_since_seen == 2
    
    # Frame 4: REAPPEARS nearby, following its trajectory approximately
    tracks = update_tracks(tracks, [(0.4, 0.4, 2)], dt)
    assert len(tracks) == 1
    # It must retain the exact same track_id
    assert tracks[0].track_id == 1
    # Frames since seen should reset to 0 upon successful match
    assert tracks[0].frames_since_seen == 0


def test_track_id_stable_across_n_frames_highway_speed():
    """
    Regression test for SIH2026 Track ID Churn bug.

    Previously MAX_ASSOC_DIST_M = 3.0m was too tight for highway speed.
    At 22 m/s and dt=0.1s a vehicle moves 2.2m per frame, leaving only 0.8m
    margin before the gate closed.  Any centroid jitter from the grid projection
    (coarser cells in mid-tier at 0.15m resolution) could exceed the gate and
    spawn a new track ID.

    This test drives a vehicle at 22 m/s across 20 frames with small Gaussian
    measurement noise and asserts:
    - The track ID never changes once CONFIRMED (frame >= 2).
    - The ID at the end equals the ID captured at first confirmation.
    """
    dt = 0.1
    speed = 22.0  # m/s — highway cruise scenario
    noise_std = 0.12  # ~12 cm centroid noise from polar grid binning

    tracks = []
    confirmed_id = None

    for frame in range(20):
        true_x = frame * speed * dt
        true_y = 0.5  # slight lateral offset (not centred)

        meas_x = true_x + np.random.normal(0.0, noise_std)
        meas_y = true_y + np.random.normal(0.0, noise_std)
        detections = [(meas_x, meas_y, 3)]

        tracks = update_tracks(tracks, detections, dt)

        assert len(tracks) == 1, (
            f"Frame {frame}: expected 1 track, got {len(tracks)}. "
            "A new track was spuriously spawned (ID churn)."
        )

        if frame >= 2:  # Track is CONFIRMED by now
            current_id = tracks[0].track_id
            if confirmed_id is None:
                confirmed_id = current_id
            assert current_id == confirmed_id, (
                f"Frame {frame}: Track ID changed from {confirmed_id} to {current_id}. "
                "This is the ID churn bug."
            )


def test_track_id_stable_urban_speed_with_occlusion():
    """
    Regression test for Track ID Churn with partial occlusion at urban speed.

    Object moves at 10 m/s (urban), disappears for 2 frames (occluded), then
    reappears.  ID must be the SAME after occlusion.
    """
    dt = 0.1
    speed = 10.0
    noise_std = 0.08

    tracks = []
    confirmed_id = None
    true_x = 5.0

    for frame in range(15):
        # Frames 5-6: occluded (no detections)
        if frame in (5, 6):
            detections = []
        else:
            meas_x = true_x + np.random.normal(0.0, noise_std)
            meas_y = 1.0 + np.random.normal(0.0, noise_std)
            detections = [(meas_x, meas_y, 3)]

        tracks = update_tracks(tracks, detections, dt)
        true_x += speed * dt

        if frame >= 2 and len(tracks) > 0:
            current_id = tracks[0].track_id
            if confirmed_id is None:
                confirmed_id = current_id

        if frame >= 7 and len(tracks) > 0:
            # After re-emergence from occlusion, ID must not have changed
            assert tracks[0].track_id == confirmed_id, (
                f"Frame {frame}: ID changed to {tracks[0].track_id} after "
                f"occlusion (was {confirmed_id})."
            )
