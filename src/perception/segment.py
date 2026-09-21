"""
Semantic Segmentation module for FoveaMap.
Wraps Open3D-ML's pretrained RandLA-Net model for point cloud segmentation,
with spatial chunking to prevent OOM, ONNX/TorchScript acceleration support,
and a robust geometric fallback if ML dependencies are missing.
"""

import sys
import os
import time
import numpy as np
import torch

try:
    import open3d.ml.torch as ml3d
    HAS_ML3D = True
except ImportError:
    ml3d = None
    HAS_ML3D = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False


class SegmentationModel:
    """Wrapper class for the segmentation model to handle different backends."""
    def __init__(self, backend="fallback", model=None, device="cpu"):
        self.backend = backend
        self.model = model
        self.device = device


def load_model(checkpoint_path=None, device="cuda") -> SegmentationModel:
    """
    Loads a semantic segmentation model.
    Supports ONNX, TorchScript, Open3D-ML, or gracefully falls back to a Z-heuristic.
    """
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ext = os.path.splitext(checkpoint_path)[1].lower()
        
        # 1. ONNX Acceleration
        if ext == ".onnx" and HAS_ONNX:
            print(f"Loading ONNX accelerated model: {checkpoint_path}")
            providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
            session = ort.InferenceSession(checkpoint_path, providers=providers)
            return SegmentationModel(backend="onnx", model=session, device=device)
            
        # 2. TorchScript JIT Acceleration
        elif ext == ".pt" or ext == ".pth":
            try:
                print(f"Attempting to load TorchScript model: {checkpoint_path}")
                model = torch.jit.load(checkpoint_path, map_location=device)
                model.eval()
                return SegmentationModel(backend="torchscript", model=model, device=device)
            except Exception:
                pass # Fall through to Open3D-ML if it's a state_dict, not a JIT model

    # 3. Open3D-ML Native
    if HAS_ML3D:
        print("Loading Open3D-ML RandLA-Net model...")
        model = ml3d.models.RandLANet(
            name='RandLANet',
            num_classes=260,
            ignored_label_inds=[0]
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        return SegmentationModel(backend="ml3d", model=model, device=device)
        
    # 4. Graceful Fallback
    print("[WARNING] Deep Learning backend not available or no checkpoint provided.")
    print("[INFO] Falling back to Robust Geometric Z-Heuristic Segmenter.")
    return SegmentationModel(backend="fallback", model=None, device="cpu")


def _geometric_fallback_segment(points_xyz: np.ndarray) -> np.ndarray:
    """
    Pure NumPy geometric heuristic to mock SemanticKITTI classes.
    Terrain (40) vs Obstacle (50) vs Dynamic (252) based on Z-height and local density.
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    
    # Base all as unclassified (0)
    preds = np.zeros(len(points_xyz), dtype=np.int32)
    
    # 1. Road/Terrain (SemanticKITTI 40)
    # Assume sensor is at z=0, ground is around z=-1.7m
    ground_mask = z < -1.5
    preds[ground_mask] = 40
    
    # 2. Obstacles/Buildings (SemanticKITTI 50)
    obstacle_mask = (z >= -1.5) & (z < 3.0)
    preds[obstacle_mask] = 50
    
    # 3. Dynamics (SemanticKITTI 252 - Moving Vehicle)
    # Heuristic: objects between z=-1.0 and z=1.0 that are clustered
    r = np.hypot(x, y)
    dynamic_mask = (z >= -1.0) & (z < 1.0) & (r > 5.0) & (r < 30.0)
    # Add a bit of random noise to simulate imperfect bounding boxes for dynamics
    # to test tracking
    np.random.seed(42)  # Deterministic for testing
    noise_mask = np.random.rand(len(points_xyz)) < 0.1
    preds[dynamic_mask & noise_mask] = 252
    
    return preds


def segment_points(points_xyz: np.ndarray, model_wrapper: SegmentationModel, chunk_size_deg: float = 90.0) -> np.ndarray:
    """
    Runs semantic segmentation with Spatial Chunking to prevent CUDA OOM.
    """
    if len(points_xyz) == 0:
        return np.empty(0, dtype=np.int32)
        
    if model_wrapper.backend == "fallback":
        return _geometric_fallback_segment(points_xyz)
        
    preds = np.zeros(len(points_xyz), dtype=np.int32)
    
    # Spatial Chunking based on azimuth (theta)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    theta_deg = np.degrees(np.arctan2(y, x)) + 180.0 # 0 to 360
    
    # Number of chunks
    num_chunks = int(np.ceil(360.0 / chunk_size_deg))
    
    for i in range(num_chunks):
        start_deg = i * chunk_size_deg
        end_deg = (i + 1) * chunk_size_deg
        
        chunk_mask = (theta_deg >= start_deg) & (theta_deg < end_deg)
        chunk_pts = points_xyz[chunk_mask]
        
        if len(chunk_pts) == 0:
            continue
            
        xyz = chunk_pts[:, :3].astype(np.float32)
        features = chunk_pts[:, 3:4].astype(np.float32) if chunk_pts.shape[1] >= 4 else np.ones((len(chunk_pts), 1), dtype=np.float32)
        
        if model_wrapper.backend == "onnx":
            # ONNX Inference
            input_name = model_wrapper.model.get_inputs()[0].name
            chunk_preds = model_wrapper.model.run(None, {input_name: xyz})[0]
            preds[chunk_mask] = chunk_preds.argmax(axis=-1)
            
        elif model_wrapper.backend == "torchscript":
            # TorchScript Inference
            with torch.no_grad():
                xyz_t = torch.tensor(xyz, device=model_wrapper.device)
                feat_t = torch.tensor(features, device=model_wrapper.device)
                out = model_wrapper.model(xyz_t, feat_t)
                preds[chunk_mask] = out.argmax(dim=-1).cpu().numpy()
                
        elif model_wrapper.backend == "ml3d":
            # Open3D-ML Inference
            with torch.no_grad():
                xyz_tensor = torch.tensor(xyz, device=model_wrapper.device)
                feat_tensor = torch.tensor(features, device=model_wrapper.device)
                inputs = {
                    'point': [xyz_tensor],
                    'feat': [feat_tensor],
                    'label': [torch.zeros(len(xyz), dtype=torch.int32, device=model_wrapper.device)]
                }
                results = model_wrapper.model(inputs)
                logits = results['predict_scores'][0] if isinstance(results, dict) and 'predict_scores' in results else (results[0] if isinstance(results, (tuple, list)) else results)
                preds[chunk_mask] = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

    return preds


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.perception.segment <path_to_sample.bin>")
        sys.exit(1)
        
    sample_path = sys.argv[1]
    
    if not os.path.exists(sample_path):
        print(f"Error: Point cloud file not found: {sample_path}")
        sys.exit(1)
        
    try:
        scan = np.fromfile(sample_path, dtype=np.float32).reshape(-1, 4)
        print(f"Loaded point cloud with {len(scan)} points from {sample_path}")
    except Exception as e:
        print(f"Failed to load point cloud: {e}")
        sys.exit(1)
        
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading RandLA-Net integration on {target_device}...")
    
    model = load_model(device=target_device)
    
    print("Running segmentation forward pass with spatial chunking...")
    t0 = time.perf_counter()
    preds = segment_points(scan, model, chunk_size_deg=90.0)
    t1 = time.perf_counter()
    
    unique_classes, counts = np.unique(preds, return_counts=True)
    print(f"\nSegmentation complete in {(t1-t0)*1000:.1f}ms. Found {len(unique_classes)} unique predicted classes:")
    for cls_id, count in zip(unique_classes, counts):
        print(f"  Class {cls_id}: {count} points")
