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
