"""
FoveaMap Tracking and Anti-Ghosting Subsystem.
"""

from src.tracking.kalman_tracker import (
    TrackedObject,
    KalmanTrackerManager,
    erase_vacated_footprints,
    cluster_dynamic_detections,
)

__all__ = [
    "TrackedObject",
    "KalmanTrackerManager",
    "erase_vacated_footprints",
    "cluster_dynamic_detections",
]
