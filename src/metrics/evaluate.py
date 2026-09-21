"""
Independent evaluation metrics and profiling utilities for the LiDAR mapping pipeline.
Provides lightweight mIoU calculation and timing capabilities without pipeline dependencies.
"""

import time
import numpy as np


def compute_miou(predicted_classes: np.ndarray, ground_truth_classes: np.ndarray, num_classes: int = 3) -> dict:
    """
    Computes mean Intersection-over-Union (mIoU) per class.
    IoU = true_positives / (true_positives + false_positives + false_negatives)
    Ignores -1 labels (unknown/ignored) in ground truth.
    
    Returns:
        dict: {class_id: iou, ..., 'mean': average_iou}
    """
    results = {}
    
    # Ignore -1 masks (e.g. unmapped raw classes or explicitly ignored points)
    valid_mask = ground_truth_classes != -1
    preds = predicted_classes[valid_mask]
    gts = ground_truth_classes[valid_mask]

    ious = []
    for c in range(num_classes):
        # True Positives: pred is c AND gt is c
        tp = np.sum((preds == c) & (gts == c))
        # False Positives: pred is c BUT gt is not c
        fp = np.sum((preds == c) & (gts != c))
        # False Negatives: pred is not c BUT gt is c
        fn = np.sum((preds != c) & (gts == c))

        denominator = tp + fp + fn
        if denominator > 0:
            iou = float(tp) / float(denominator)
        else:
            iou = 0.0  # Or NaN, assuming 0 for unrepresented classes in union
            
        results[c] = iou
        ious.append(iou)
        
    results["mean"] = float(np.mean(ious)) if ious else 0.0
    return results


def compute_miou_by_distance(
    points_xyz: np.ndarray, 
    predicted_classes: np.ndarray, 
    ground_truth_classes: np.ndarray, 
    distance_buckets: list = None
) -> dict:
    """
    Computes mIoU bucketed by 2D radial distance from the origin.
    
    Args:
        points_xyz: (N, 3) or (N, 2) array of coordinates.
        predicted_classes: (N,) array of predicted classes.
        ground_truth_classes: (N,) array of true classes.
        distance_buckets: List of (min_range, max_range) tuples.
    
    Returns:
        dict: { "min-max": miou_dict, ... }
    """
    if distance_buckets is None:
        distance_buckets = [(0, 10), (10, 30), (30, 100)]
        
    results = {}
    if len(points_xyz) == 0:
        return results
        
    # Calculate 2D range
    ranges = np.hypot(points_xyz[:, 0], points_xyz[:, 1])
    
    for min_r, max_r in distance_buckets:
        mask = (ranges >= min_r) & (ranges < max_r)
        
        sub_preds = predicted_classes[mask]
        sub_gts = ground_truth_classes[mask]
        
        bucket_label = f"{min_r}-{max_r}m"
        results[bucket_label] = compute_miou(sub_preds, sub_gts)
        
    return results


def profile_pipeline_stage(func, *args, n_runs=10, **kwargs) -> dict:
    """
    Profiles the execution time of a given function over n_runs.
    
    Returns:
        dict: {"mean_ms": float, "min_ms": float, "max_ms": float, "fps_equivalent": float}
    """
    if n_runs <= 0:
        return {}
        
    times_ms = []
    
    # Optional warmup
    if n_runs > 1:
        func(*args, **kwargs)
        
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        
        times_ms.append((end - start) * 1000.0)
        
    mean_ms = float(np.mean(times_ms))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))
    
    fps_equivalent = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    
    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps_equivalent": fps_equivalent
    }


def run_full_evaluation(*args, **kwargs):
    """
    Convenience wrapper to run full FoveaMap pipeline benchmark evaluation.
    Re-exports evaluator.run_full_evaluation.
    """
    from src.metrics.evaluator import run_full_evaluation as _eval_impl
    return _eval_impl(*args, **kwargs)

