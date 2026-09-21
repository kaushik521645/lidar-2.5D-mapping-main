"""
Semantic Class Mapping for FoveaMap.

Collapses raw 28-class SemanticKITTI labels into the 4-class taxonomy:
  0: Drivable Terrain (Road, parking, lane markings)
  1: Non-Drivable Terrain (Sidewalk, terrain, vegetation, curbs)
  2: Static Obstacle (Buildings, poles, traffic signs, walls)
  3: Dynamic Object (Cars, pedestrians, bicyclists, trucks)
 -1: Unknown / Unlabeled

Mappings are dynamically loaded from configs/default.yaml.
"""

import os
from typing import Dict, Any
import numpy as np
import yaml


def load_class_mapping(config_path: str = "configs/default.yaml") -> Dict[int, int]:
    """Loads SemanticKITTI -> 4-class map dictionary from YAML config."""
    mapping = {
        0: -1, 1: -1,
        10: 3, 11: 3, 13: 3, 15: 3, 16: 3, 18: 3, 20: 3, 30: 3, 31: 3, 32: 3,
        40: 0, 44: 0, 48: 1, 49: 0, 50: 2, 51: 2, 52: 2, 60: 0,
        70: 1, 71: 2, 72: 1, 80: 2, 81: 2, 99: 1,
        252: 3, 253: 3, 254: 3, 255: 3, 256: 3, 257: 3, 258: 3, 259: 3,
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                m = cfg.get("perception", {}).get("semantickitti_to_4class", {})
                if m:
                    return {int(k): int(v) for k, v in m.items()}
        except Exception:
            pass
    return mapping


SEMANTICKITTI_TO_4CLASS: Dict[int, int] = load_class_mapping()

# Build fast 65536-entry lookup array for vectorized uint16 / uint32 label remapping
_LOOKUP_TABLE = np.full(65536, -1, dtype=np.int32)
for raw_id, target_class in SEMANTICKITTI_TO_4CLASS.items():
    if 0 <= raw_id < 65536:
        _LOOKUP_TABLE[raw_id] = target_class


def map_semantickitti_to_4class(labels: np.ndarray) -> np.ndarray:
    """
    Vectorized mapping of SemanticKITTI raw labels to 4-class taxonomy.
    
    Args:
        labels: (N,) integer array of SemanticKITTI raw IDs (with or without instance IDs).
        
    Returns:
        (N,) int32 array where values are in {-1, 0, 1, 2, 3}.
    """
    # Extract lower 16 bits for semantic class ID
    labels_int = (labels.astype(np.int64) & 0xFFFF)
    # Mask out-of-range labels safely
    valid_mask = (labels_int >= 0) & (labels_int < 65536)
    out = np.full(labels_int.shape, -1, dtype=np.int32)
    out[valid_mask] = _LOOKUP_TABLE[labels_int[valid_mask]]
    return out
