"""
Unit tests for Metrics and Evaluation subsystem.
"""

import sys
import os
import pytest
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.evaluate import (
    compute_miou,
    compute_miou_by_distance,
    profile_pipeline_stage,
)


def test_compute_miou_perfect_and_partial():
    """Validates mIoU calculation with perfect and partial overlap."""
    gt = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    # Perfect predictions
    pred_perfect = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    class_ious_p = compute_miou(pred_perfect, gt, num_classes=4)
    assert class_ious_p["mean"] == pytest.approx(1.0)
    assert class_ious_p[0] == pytest.approx(1.0)

    # Partial predictions: 1 mistake in class 0 and 1 in class 1
    pred_partial = np.array([0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    class_ious_part = compute_miou(pred_partial, gt, num_classes=4)
    # Class 0: TP=1, FP=0, FN=1 -> IoU = 1/2 = 0.5
    # Class 1: TP=2, FP=1, FN=0 -> IoU = 2/3 = 0.6667
    # Class 2: TP=2, FP=0, FN=0 -> IoU = 1.0
    # Class 3: TP=2, FP=0, FN=0 -> IoU = 1.0
    assert class_ious_part[0] == pytest.approx(0.5)
    assert class_ious_part[1] == pytest.approx(2.0 / 3.0)
    assert class_ious_part[2] == pytest.approx(1.0)
    assert class_ious_part[3] == pytest.approx(1.0)
    assert class_ious_part["mean"] == pytest.approx((0.5 + 2.0 / 3.0 + 1.0 + 1.0) / 4.0)


def test_compute_miou_by_distance():
    """
    Validates mIoU computed across distance buckets (0-10m, 10-30m, 30-100m).
    """
    points_xyz = np.array([
        [5.0, 0.0, 0.0],   # range 5 (0-10m)
        [15.0, 0.0, 0.0],  # range 15 (10-30m)
        [40.0, 0.0, 0.0],  # range 40 (30-100m)
        [50.0, 50.0, 0.0], # range ~70 (30-100m)
    ])
    gt = np.array([1, 1, 2, 0])
    preds = np.array([1, 2, 2, 0]) # mistake on point index 1
    
    seg_report = compute_miou_by_distance(points_xyz, preds, gt)

    assert "0-10m" in seg_report
    assert "10-30m" in seg_report
    assert "30-100m" in seg_report

    # 0-10m: perfect
    assert seg_report["0-10m"][1] == 1.0
    
    # 10-30m: completely wrong
    assert seg_report["10-30m"][1] == 0.0
    
    # 30-100m: perfect
    assert seg_report["30-100m"][2] == 1.0
    assert seg_report["30-100m"][0] == 1.0


def test_profile_pipeline_stage():
    """Validates the profiling function."""
    def dummy_func(sleep_time):
        time.sleep(sleep_time)
        return True
        
    stats = profile_pipeline_stage(dummy_func, 0.01, n_runs=3)
    assert "mean_ms" in stats
    assert "min_ms" in stats
    assert "max_ms" in stats
    assert "fps_equivalent" in stats
    
    assert stats["mean_ms"] >= 10.0
    assert stats["fps_equivalent"] > 0 and stats["fps_equivalent"] < 150.0


def test_evaluator_module_and_run_full_evaluation():
    """Validates that src.metrics.evaluator can be imported and executed successfully."""
    from src.metrics.evaluator import run_full_evaluation, evaluate_segmentation, evaluate_memory

    # Test individual evaluators
    seg = evaluate_segmentation(num_frames=2)
    assert "overall" in seg
    assert "mean" in seg["overall"]

    mem = evaluate_memory(sequence="00")
    assert "memory_savings_pct" in mem
    assert mem["memory_savings_pct"] > 80.0

    # Test full evaluation report generation
    results = run_full_evaluation(sequence="00", num_frames=2, verbose=False)
    assert "segmentation" in results
    assert "performance" in results
    assert "memory" in results

