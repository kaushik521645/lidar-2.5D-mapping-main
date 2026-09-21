"""
Semantic class mapping module.
Loads taxonomy definitions from YAML config and applies vectorized class collapsing.
"""

import os
import yaml
import numpy as np
from typing import Dict, Optional


def load_class_map(config_path: Optional[str] = None) -> Dict[int, int]:
    """
    Loads the class mapping dictionary from the specified YAML config.
    If no config_path is provided, defaults to configs/default.yaml relative to the project root.
    Returns a dictionary mapping raw class IDs to collapsed class IDs.
    """
    if config_path is None:
        # File is at src/grid/class_map.py, two levels up is the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "configs", "default.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    raw_map = cfg.get("class_map", {})
    
    # Ensure keys and values are integers
    class_map = {int(k): int(v) for k, v in raw_map.items()}
    return class_map


def collapse_classes(raw_class_ids: np.ndarray, class_map: Dict[int, int]) -> np.ndarray:
    """
    Collapses an array of raw semantic class IDs into a new taxonomy using the provided class_map.
    Unmapped IDs default to -1 (unknown/ignore).
    Uses vectorized numpy operations for performance.
    """
    collapsed = np.full(raw_class_ids.shape, -1, dtype=np.int32)
    
    for raw_id, new_id in class_map.items():
        # Vectorized assignment mask
        collapsed[raw_class_ids == raw_id] = new_id
        
    return collapsed
