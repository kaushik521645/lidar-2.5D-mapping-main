"""
FastAPI Server & WebSocket Streaming Service for FoveaMap.

Serves REST endpoints, WebSocket streaming, and static Three.js dashboard on port 8080.
Supports all 22 KITTI Velodyne odometry sequences and procedural scenario playback.
"""

import os
import re
import sys
import json
import time
import asyncio
import zlib
from typing import Dict, Any, Optional, List
import yaml
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from src.ingestion.kitti_loader import KITTILoader
from src.perception.segment import segment_points
from src.grid.grid_engine import PolarGridEngine
from src.grid.grid_types import VehicleState
from src.grid.foveation import fine_radius_at_angle, BASE_FINE_RADIUS, MAX_STRETCH, SHEAR_STRENGTH
from src.tracking.kalman_tracker import KalmanTrackerManager, erase_vacated_footprints
from src.synthetic.scenarios import ScenarioGenerator
from src.api.export_precomputed import (
    process_sequence_to_json,
    compute_memory_metrics,
    export_all_precomputed,
    export_kitti_sequence,
    export_synthetic_kitti_like,
)

app = FastAPI(title="FoveaMap Perception API", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cache_control_header(request: Request, call_next):
    """Prevents stale browser caching during development and real-time updates."""
    response = await call_next(request)
    if request.url.path.startswith("/api/") or request.url.path.endswith((".html", ".js", ".css", ".json")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


CONFIG_PATH = "configs/default.yaml"
PRECOMPUTED_DIR = "data/precomputed"

_scenario_gen = ScenarioGenerator()
_grid_engine = PolarGridEngine()
_cached_kitti_seqs: Optional[List[Dict[str, Any]]] = None


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def get_all_kitti_sequences() -> List[Dict[str, Any]]:
    """Discovers and caches the list of all available KITTI sequences from the dataset."""
    global _cached_kitti_seqs
    if _cached_kitti_seqs is not None:
        return _cached_kitti_seqs

    loader = KITTILoader()
    seq_ids = loader.list_sequences()
    result = []
    for s in seq_ids:
        loader.set_sequence(s)
        cnt = len(loader)
        cache_p = os.path.join(PRECOMPUTED_DIR, f"kitti_seq_{s}.json")
        is_precomp = os.path.exists(cache_p) or (s == "00" and os.path.exists(os.path.join(PRECOMPUTED_DIR, "kitti_odometry.json")))
        result.append({
            "id": f"kitti_seq_{s}",
            "sequence": s,
            "name": f"KITTI Sequence {s} ({cnt:,} scans)",
            "description": f"Real Velodyne HDL-64E LiDAR capture from KITTI odometry sequence {s} ({cnt:,} raw scans).",
            "type": "real",
            "scans_count": cnt,
            "is_precomputed": is_precomp,
        })
    _cached_kitti_seqs = result
    return result


@app.get("/health")
async def health_check():
    """Health check endpoint required by Section 7.3."""
    return {"status": "ok", "timestamp": time.time(), "service": "foveamap-api"}


@app.get("/api/config")
async def get_configuration():
    """Returns central pipeline configuration and class color mappings."""
    return JSONResponse(content=load_config())


@app.get("/api/sequences")
async def list_sequences():
    """Returns list of all available KITTI sequences with scan counts."""
    return JSONResponse(content=get_all_kitti_sequences())


@app.get("/api/scenarios")
async def list_scenarios():
    """Returns list of available demo scenarios including all discovered KITTI sequences."""
    kitti_seqs = get_all_kitti_sequences()
    benchmarks_and_synthetic = [
        {
            "id": "synthetic_kitti_like",
            "name": "SemanticKITTI Benchmark (Ground-Truth Labeled)",
            "description": "30-frame sequential LiDAR dataset in SemanticKITTI format with ground-truth semantic point annotations.",
            "type": "semantickitti",
            "scans_count": 30,
            "is_precomputed": os.path.exists(os.path.join(PRECOMPUTED_DIR, "synthetic_kitti_like.json")),
        },
        {
            "id": "urban_intersection",
            "name": "Urban Intersection",
            "description": "Dense cross-traffic, crossing pedestrian, roadside curbs, and turning ego vehicle.",
            "type": "synthetic",
            "is_precomputed": os.path.exists(os.path.join(PRECOMPUTED_DIR, "urban_intersection.json")),
        },
        {
            "id": "highway_cruise",
            "name": "Highway Cruise (22 m/s)",
            "description": "High-speed cruising with active forward foveation elongation up to 25m.",
            "type": "synthetic",
            "is_precomputed": os.path.exists(os.path.join(PRECOMPUTED_DIR, "highway_cruise.json")),
        },
        {
            "id": "pothole_alley",
            "name": "Pothole & Rough Terrain Alley",
            "description": "Distinguishes smooth drivable road (Class 0) from hazardous non-drivable potholes (Class 1).",
            "type": "synthetic",
            "is_precomputed": os.path.exists(os.path.join(PRECOMPUTED_DIR, "pothole_alley.json")),
        },
        {
            "id": "bridge_overpass",
            "name": "Bridge Overpass",
            "description": "Multi-layer underpass: ground surface (z=-1.5m) and bridge deck ceiling (z=1.8 to 3.2m).",
            "type": "synthetic",
            "is_precomputed": os.path.exists(os.path.join(PRECOMPUTED_DIR, "bridge_overpass.json")),
        },
    ]
    return JSONResponse(content=kitti_seqs + benchmarks_and_synthetic)


@app.get("/api/frames/{scenario_id}")
async def get_scenario_frames(scenario_id: str):
    """Retrieves precomputed frame sequence for the requested scenario or sequence."""
    # 1. SemanticKITTI Ground-Truth Benchmark sequence
    if scenario_id in ("synthetic_kitti_like", "semantickitti", "semantic_kitti"):
        cache_file = os.path.join(PRECOMPUTED_DIR, "synthetic_kitti_like.json")
        if not os.path.exists(cache_file) or os.path.getsize(cache_file) > 100 * 1024 * 1024:
            export_synthetic_kitti_like(output_dir=PRECOMPUTED_DIR)
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            return JSONResponse(content=data)

    # 2. Genuine KITTI Velodyne Odometry Sequences
    seq_match = re.search(r"(?:kitti_seq_|kitti_odometry_|kitti_|seq_)?(\d{1,2})", scenario_id, re.IGNORECASE)

    if scenario_id in ("kitti_odometry", "kitti_odometry_velodyne", "kitti_sample"):
        target_id = "kitti_odometry"
        seq_str = "00"
        is_kitti = True
    elif seq_match and ("kitti" in scenario_id or "seq" in scenario_id or scenario_id.isdigit()):
        seq_str = seq_match.group(1).zfill(2)
        target_id = f"kitti_seq_{seq_str}"
        is_kitti = True
        if seq_str == "00" and os.path.exists(os.path.join(PRECOMPUTED_DIR, "kitti_odometry.json")):
            target_id = "kitti_odometry"
    else:
        target_id = scenario_id
        is_kitti = False
        seq_str = "00"

    cache_file = os.path.join(PRECOMPUTED_DIR, f"{target_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)

    alt_cache = os.path.join(PRECOMPUTED_DIR, f"{scenario_id}.json")
    if os.path.exists(alt_cache):
        with open(alt_cache, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)

    # Generate on the fly for requested sequence or synthetic scenario
    if is_kitti:
        export_kitti_sequence(sequence=seq_str, output_dir=PRECOMPUTED_DIR, max_frames=25)
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            return JSONResponse(content=data)
        alt_seq_cache = os.path.join(PRECOMPUTED_DIR, f"kitti_seq_{seq_str}.json")
        if os.path.exists(alt_seq_cache):
            with open(alt_seq_cache, "r") as f:
                data = json.load(f)
            return JSONResponse(content=data)
    else:
        scenarios = _scenario_gen.get_all_scenarios()
        if target_id in scenarios:
            frames = process_sequence_to_json(scenarios[target_id], target_id, cache_file)
            return JSONResponse(content=frames)

    raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")


@app.websocket("/ws/grid")
async def websocket_grid_stream(websocket: WebSocket):
    """
    Real-time WebSocket streaming endpoint conforming to Section 7.3.
    Streams 2.5D grid frames and handles live client parameter adjustments.
    """
    await websocket.accept()

    # Default to kitti_seq_00 or kitti_odometry
    current_scenario = "kitti_seq_00" if os.path.exists(os.path.join(PRECOMPUTED_DIR, "kitti_seq_00.json")) else "kitti_odometry"

    current_frame_idx = 0
    is_playing = True
    auto_loop_all = True
    speed_mps = 10.0
    steering_rad = 0.0
    playback_fps = 10.0
    user_override = False

    # Backpressure: bounded queue prevents unbounded growth for slow clients
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=2)

    def load_scenario_frames(s_id: str) -> List[Dict[str, Any]]:
        seq_match = re.search(r"(?:kitti_seq_|kitti_odometry_|kitti_|seq_)?(\d{1,2})", s_id, re.IGNORECASE)
        if s_id in ("kitti_odometry", "kitti_sample"):
            resolved = "kitti_odometry"
            seq_num = "00"
            is_k = True
        elif seq_match and ("kitti" in s_id or "seq" in s_id or s_id.isdigit()):
            seq_num = seq_match.group(1).zfill(2)
            resolved = f"kitti_seq_{seq_num}"
            is_k = True
            if seq_num == "00" and os.path.exists(os.path.join(PRECOMPUTED_DIR, "kitti_odometry.json")):
                resolved = "kitti_odometry"
        else:
            resolved = s_id
            is_k = False
            seq_num = "00"

        f_path = os.path.join(PRECOMPUTED_DIR, f"{resolved}.json")
        if not os.path.exists(f_path):
            f_path = os.path.join(PRECOMPUTED_DIR, f"{s_id}.json")
        if os.path.exists(f_path):
            try:
                with open(f_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        if is_k:
            out_p = export_kitti_sequence(sequence=seq_num, output_dir=PRECOMPUTED_DIR, max_frames=25)
            if os.path.exists(out_p):
                try:
                    with open(out_p, "r") as f:
                        return json.load(f)
                except Exception:
                    pass
        return []

    frames = load_scenario_frames(current_scenario)
    if not frames:
        export_all_precomputed()
        frames = load_scenario_frames(current_scenario)

    try:
        while True:
            # Check for incoming client messages (non-blocking)
            try:
                msg_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                msg = json.loads(msg_text)
                cmd = msg.get("command")

                if cmd == "set_scenario":
                    scenario_id = msg.get("scenario_id", "kitti_seq_00")
                    if scenario_id != current_scenario:
                        current_scenario = scenario_id
                        current_frame_idx = 0
                        user_override = False
                        new_frames = load_scenario_frames(current_scenario)
                        if new_frames:
                            frames = new_frames

                elif cmd == "set_auto_loop":
                    auto_loop_all = bool(msg.get("enabled", True))

                elif cmd == "next_scenario":
                    all_seq_ids = [s["id"] for s in get_all_kitti_sequences()]
                    if current_scenario in all_seq_ids:
                        idx = all_seq_ids.index(current_scenario)
                        current_scenario = all_seq_ids[(idx + 1) % len(all_seq_ids)]
                    elif all_seq_ids:
                        current_scenario = all_seq_ids[0]
                    current_frame_idx = 0
                    new_frames = load_scenario_frames(current_scenario)
                    if new_frames:
                        frames = new_frames

                elif cmd == "prev_scenario":
                    all_seq_ids = [s["id"] for s in get_all_kitti_sequences()]
                    if current_scenario in all_seq_ids:
                        idx = all_seq_ids.index(current_scenario)
                        current_scenario = all_seq_ids[(idx - 1 + len(all_seq_ids)) % len(all_seq_ids)]
                    elif all_seq_ids:
                        current_scenario = all_seq_ids[0]
                    current_frame_idx = 0
                    new_frames = load_scenario_frames(current_scenario)
                    if new_frames:
                        frames = new_frames

                elif cmd == "set_params":
                    speed_mps = float(msg.get("speed_mps", speed_mps))
                    steering_rad = float(msg.get("steering_angle_rad", steering_rad))
                    user_override = True

                elif cmd == "play":
                    is_playing = True
                elif cmd == "pause":
                    is_playing = False
                elif cmd == "step":
                    is_playing = False
                    current_frame_idx = (current_frame_idx + 1) % max(len(frames), 1)
                elif cmd == "seek":
                    current_frame_idx = int(msg.get("frame_id", 0)) % max(len(frames), 1)

            except asyncio.TimeoutError:
                pass

            if frames and len(frames) > 0:
                frame_data = dict(frames[current_frame_idx])
                frame_data["scenario_id"] = current_scenario

                # Allow live speed/steering override
                if user_override:
                    frame_data["vehicle_state"]["speed_mps"] = speed_mps
                    frame_data["vehicle_state"]["steering_angle_rad"] = steering_rad

                # Enqueue with drop-oldest backpressure
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await frame_queue.put(frame_data)

                # Send all queued frames
                while not frame_queue.empty():
                    queued_frame = await frame_queue.get()
                    compressed = zlib.compress(json.dumps(queued_frame).encode('utf-8'))
                    await websocket.send_bytes(compressed)

                if is_playing:
                    if auto_loop_all and current_frame_idx >= len(frames) - 1:
                        # Automatically advance to next KITTI sequence
                        all_seq_ids = [s["id"] for s in get_all_kitti_sequences()]
                        if current_scenario in all_seq_ids:
                            idx = all_seq_ids.index(current_scenario)
                            current_scenario = all_seq_ids[(idx + 1) % len(all_seq_ids)]
                        elif all_seq_ids:
                            current_scenario = all_seq_ids[0]
                        current_frame_idx = 0
                        new_frames = load_scenario_frames(current_scenario)
                        if new_frames:
                            frames = new_frames
                    else:
                        current_frame_idx = (current_frame_idx + 1) % len(frames)

            await asyncio.sleep(1.0 / playback_fps)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Disconnected: {e}")


# Mount dashboard directory for static frontend hosting
DASHBOARD_DIR = os.path.abspath("dashboard")
if os.path.exists(DASHBOARD_DIR):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = int(server_cfg.get("port", 8080))
    print(f"[INFO] Starting FoveaMap Unified Server on http://{host}:{port}", flush=True)
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)
