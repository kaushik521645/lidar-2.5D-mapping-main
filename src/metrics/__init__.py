"""
FoveaMap Evaluation & Metrics Subsystem.
"""

from src.metrics.evaluate import (
    compute_miou,
    compute_miou_by_distance,
    profile_pipeline_stage,
)
from src.metrics.evaluator import (
    run_full_evaluation,
    evaluate_segmentation,
    benchmark_pipeline,
    evaluate_memory,
)

__all__ = [
    "compute_miou",
    "compute_miou_by_distance",
    "profile_pipeline_stage",
    "run_full_evaluation",
    "evaluate_segmentation",
    "benchmark_pipeline",
    "evaluate_memory",
]

