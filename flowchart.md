# FoveaMap: Architecture & Pipeline Flowcharts

A comprehensive visual guide to the **FoveaMap** 2.5D LiDAR perception and mapping system. These flowcharts illustrate how raw 3D laser points transform into an adaptive, memory-efficient 2.5D elevation grid with real-time tracking and 3D web streaming.

---

## 1. High-Level System Architecture

This master diagram shows the end-to-end flow from physical sensors/datasets to the user's interactive 3D browser screen.

```mermaid
flowchart TD
    subgraph S1["1. Raw Data Input"]
        A1["KITTI Odometry Velodyne HDL-64E"] --> INGEST["Loader Engine (Zero-Copy np.memmap)"]
        A2["SemanticKITTI Labeled Benchmark"] --> INGEST
        A3["Procedural Driving Scenarios"] --> INGEST
    end

    subgraph S2["2. Perception & Segmentation"]
        INGEST --> PERCEPT{"AI or Heuristic Mode?"}
        PERCEPT -->|"AI Mode"| DL["RandLA-Net Deep Learning Model"]
        PERCEPT -->|"Heuristic Mode"| RANSAC["Geometric RANSAC Ground Fit + Height Slicing"]
        DL --> CLMAP["Class Map Collapser (4 Core Classes)"]
        RANSAC --> CLMAP
    end

    subgraph S3["3. Motion-Adaptive Dynamic Foveation"]
        CLMAP --> FOV["Foveation Engine"]
        VEHICLE["Vehicle Kinematics (Speed vx, Steering delta)"] --> FOV
        FOV --> FOV_CALC["Compute Unified Acuity Field (Ego-Kinematic + Motion Lobes)"]
    end

    subgraph S4["4. 2.5D Polar Grid Engine"]
        FOV_CALC --> POLAR["Polar Coordinate Projection (r, theta)"]
        POLAR --> TIERS["Multi-Tier Range Discretization (Tiers 0, 1, 2)"]
        TIERS --> ELEV["Multi-Layer Elevation Profiling (Ground, Overhangs, Clearance)"]
        ELEV --> HYST["Boundary Hysteresis Filter (Anti-Flicker)"]
    end

    subgraph S5["5. Multi-Object Tracking & Anti-Ghosting"]
        HYST --> CLUST["DBSCAN Dynamic Object Clustering"]
        CLUST --> HUNGARIAN["Hungarian Data Association (scipy linear_sum_assignment)"]
        HUNGARIAN --> KALMAN["Constant-Velocity Kalman Filter (Joseph-Form)"]
        KALMAN --> GHOST["Active Footprint Erasure (Clear Vacated Static Cells)"]
        KALMAN -.->|"Active Tracks & Velocities (Closed Feedback Loop)"| FOV
    end

    subgraph S6["6. Backend API & Live Streaming"]
        GHOST --> SERVE["FastAPI Server (run_demo.py)"]
        SERVE --> ZLIB["Payload Compression (zlib.compress Level 6)"]
        ZLIB --> WS["Binary WebSocket Stream (ws://localhost:8080/ws/grid)"]
    end

    subgraph S7["7. Interactive 3D Web Dashboard"]
        WS --> CLIENT["Browser WebSocket Client"]
        CLIENT --> PAKO["pako.inflate (Decompress Binary Buffer)"]
        PAKO --> THREE["Three.js / Deck.gl 3D Renderer (Polar Grid, Pillars, Boxes)"]
        PAKO --> HUD["Telemetry HUD, FPS Counter, Radar Scanner Modal"]
    end
```

---

## 2. Point Cloud Ingestion & Dataset Selection

How FoveaMap discovers, verifies, and memory-maps point cloud datasets without blocking CPU execution.

```mermaid
flowchart TD
    START(["Launch run_demo.py"]) --> DISCOVER["Scan Workspace Directories"]
    DISCOVER --> CHK_ZIP{"Is data_odometry_velodyne.zip present?"}

    CHK_ZIP -->|"Yes"| DETECT_REAL["Inspect Real KITTI Odometry Sequences (00 to 21)"]
    CHK_ZIP -->|"No"| USE_PROCEDURAL["Use Procedural Scenarios Fallback"]

    DETECT_REAL --> DETECT_SEMANTIC["Verify SemanticKITTI Benchmark in data/synthetic_kitti_like"]
    USE_PROCEDURAL --> DETECT_SEMANTIC

    DETECT_SEMANTIC --> MMAP["Open Selected Scan via np.memmap(mode='r')"]
    MMAP --> ZERO_COPY["Zero-Copy Direct Virtual Memory Page Access"]
    ZERO_COPY --> RESHAPE["Reshape to (N, 4): [x, y, z, reflectivity]"]
    RESHAPE --> OUT_POINTS(["Ready Point Cloud Tensor"])
```

### Simple Explanation:
1. **Auto-Discovery**: The launcher checks if the full KITTI Velodyne odometry dataset (sequences 00–21) is available.
2. **Zero-Copy Memory Mapping**: Instead of reading whole files into RAM, it uses operating system page caching (`np.memmap`), enabling instant scan loading without disk lag.
3. **Dual Dataset Operation**: Real physical sensor scans and ground-truth benchmark sequences coexist seamlessly.

---

## 3. Perception & Semantic Segmentation Pipeline

How 3D laser points are classified into drivable roads, static obstacles, and moving traffic.

```mermaid
flowchart TD
    RAW_POINTS(["Raw 3D Points: (N, 4)"]) --> MODE_DECISION{"Segmentation Method"}

    subgraph HEURISTIC["Deterministic Geometric Path (Fast / CPU)"]
        MODE_DECISION -->|"Geometric RANSAC"| RANSAC_FIT["Fit Ground Plane via RANSAC Iterations"]
        RANSAC_FIT --> ROAD_DIST["Distance to Plane <= 0.20m AND |y| <= Road Width"]
        ROAD_DIST -->|"True"| CLS_ROAD["Class 0: Traversable Ground / Road"]
        ROAD_DIST -->|"False"| HT_CHECK{"Height above ground > 0.3m?"}
        HT_CHECK -->|"Within Road Corridor & Dynamic Profile"| CLS_DYN["Class 2: Dynamic Obstacle (Vehicle / Pedestrian)"]
        HT_CHECK -->|"Off-Road / High Elevation"| CLS_STAT["Class 1: Static Obstacle (Building / Tree / Pole)"]
    end

    subgraph AI_DEEP["Deep Learning Path (GPU / Open3D-ML)"]
        MODE_DECISION -->|"AI Model"| RANDLA["RandLA-Net / LightweightPointNet"]
        RANDLA --> RAW_LABELS["Raw SemanticKITTI Class IDs (30+ classes)"]
        RAW_LABELS --> CONFIG_MAP["Read configs/default.yaml Mapping Table"]
        CONFIG_MAP --> COLLAPSE["Collapse to 4 Core Classes"]
    end

    CLS_ROAD --> MERGED_LABELS(["Labeled Point Cloud with Semantic IDs [0, 1, 2, 3]"])
    CLS_DYN --> MERGED_LABELS
    CLS_STAT --> MERGED_LABELS
    COLLAPSE --> MERGED_LABELS
```

### Simple Explanation:
- **Heuristic RANSAC**: Highly robust fallback that calculates the road plane mathematically. Anything near the plane is drivable ground; elevated objects inside driving corridors are flagged as vehicles or pedestrians.
- **Deep Learning (RandLA-Net)**: Uses neural networks to classify every point.
- **Class Map Collapsing**: Condenses 30+ messy raw classes into 4 actionable categories:
  - **Class 0**: Drivable Terrain
  - **Class 1**: Static Structures
  - **Class 2**: Dynamic Moving Obstacles
  - **Class 3**: Free Space / Ignored

---

## 4. Motion-Adaptive Dynamic Foveation Mechanics

How the system dynamically adjusts its high-resolution perception zone in real time by fusing **Ego-Kinematic Stretch** with **Moving Object Trajectory Look-Ahead**.

```mermaid
flowchart TD
    subgraph KINEMATIC["Branch A: Ego-Vehicle Kinematics"]
        SPEED["Ego Speed vx (m/s)"] --> SPEED_STRETCH{"Is vx > 0?"}
        STEER["Steering Angle delta (rad)"] --> STEER_SHEAR["Steer Look-Ahead: Major Axis aligns with delta"]
        SPEED_STRETCH -->|"Accelerating"| ELONGATE["Elongate Forward Semi-Major Axis: a(vx) = a0 * stretch(vx)"]
        SPEED_STRETCH -->|"Stationary"| DEFAULT_AXIS["Maintain Default Semi-Major Axis: a = a0 (10m)"]
        ELONGATE --> ASSEMBLE_ELLIPSE["Assemble Kinematic Ellipse R_kinematic(theta)"]
        DEFAULT_AXIS --> ASSEMBLE_ELLIPSE
        STEER_SHEAR --> ASSEMBLE_ELLIPSE
    end

    subgraph MOTION["Branch B: Moving Object Motion Detection & Trajectory Prediction"]
        TRACKS["Tracked Objects (Kalman Filter)"] --> VEL_CHECK{"Velocity ||v|| >= v_threshold (0.8 m/s)?"}
        VEL_CHECK -->|"Moving Object"| PREDICT["Trajectory Extrapolation: p_future = p + v * t_lead"]
        VEL_CHECK -->|"Stationary / Parked"| BASE_ATTN["Standard Safety Buffer: r_attn = r_obj + 5.0m"]
        PREDICT --> CORRIDOR["Form Motion Corridor covering current position AND predicted future path"]
        CORRIDOR --> ACUITY_BOOST["Scale Acuity & Reach: Gain = 1.0 + boost * min(speed/15, 1.0)"]
        ACUITY_BOOST --> ASSEMBLE_LOBE["Assemble Directional Dynamic Foveal Lobe R_motion(theta)"]
        BASE_ATTN --> ASSEMBLE_LOBE
    end

    ASSEMBLE_ELLIPSE --> FUSE["Dual-Mode Acuity Fusion: R_fine(theta) = max(R_kinematic(theta), R_motion(theta))"]
    ASSEMBLE_LOBE --> FUSE

    FUSE --> TEST_CELL["For each Polar Grid Point (r, theta): Is r < R_fine(theta)?"]
    TEST_CELL -->|"Inside Dynamic Fovea"| TIER0["Tier 0: Fine Resolution (dr = 0.05m / 5cm Acuity)"]
    TEST_CELL -->|"Outside Dynamic Fovea"| PERIPHERY["Peripheral Range Tiers (Tier 1: 0.15m, Tier 2: 0.50m)"]

    TIER0 --> FINAL_GRID(["Adaptive Variable-Resolution Polar Grid"])
    PERIPHERY --> FINAL_GRID
```

### Motion-Adaptive Foveation Explained:
- **Ego-Kinematic Foveation**: As the ego vehicle accelerates down a highway, the forward foveal zone elongates ahead ($v_{\text{ref}} = 20\text{ m/s}$, up to $2.5\times$ stretch). Turning the wheel bends the ellipse into the turn.
- **Motion Detection & Filtering**: Rather than treating stationary parked cars as dynamic threats, the tracker estimates velocity vectors ($\vec{v} = (v_x, v_y)$). Only obstacles moving above the threshold ($v \ge 0.8\text{ m/s}$) trigger active motion foveation.
- **Predictive Trajectory Corridor**: For moving vehicles, crossing pedestrians, and cyclists, the foveation engine calculates look-ahead positions ($\vec{p}_{\text{future}} = \vec{p} + \vec{v} \cdot \Delta t_{\text{lead}}$). Dynamic foveal lobes stretch ahead of the obstacle along its direction of motion.
- **High-Acuity Allocation**: Points on and ahead of the moving object are allocated into Tier 0 ($5\text{cm}$ grid cells), ensuring high-density geometric capture for collision prevention regardless of their azimuth relative to the car.

---

## 5. 2.5D Polar Grid & Multi-Layer Elevation Engine

How 3D point clouds are compressed into a multi-layer polar grid, capturing clearance under bridges and tunnels.

```mermaid
flowchart TD
    IN_PTS(["3D Point (x, y, z) + Semantic Label"]) --> CART_TO_POLAR["Convert to Polar: r = sqrt(x^2 + y^2), theta = atan2(y, x)"]
    
    CART_TO_POLAR --> TIER_LOOKUP{"Which Radial Tier?"}
    TIER_LOOKUP -->|"0m to 10m"| T0["Tier 0: Near Field (dr = 0.05m, dtheta = 0.5 deg)"]
    TIER_LOOKUP -->|"10m to 30m"| T1["Tier 1: Mid Corridor (dr = 0.15m, dtheta = 1.0 deg)"]
    TIER_LOOKUP -->|"30m to 75m"| T2["Tier 2: Far Field (dr = 0.50m, dtheta = 2.0 deg)"]

    T0 --> CELL_BIN["Bin into GridCell(ring_idx, azimuth_idx)"]
    T1 --> CELL_BIN
    T2 --> CELL_BIN

    CELL_BIN --> SORT_Z["Sort Returns in Cell by Elevation z"]
    SORT_Z --> DETECT_GAP{"Is vertical gap delta_z > 1.8m?"}

    DETECT_GAP -->|"Yes (Overhang / Bridge)"| OVERHANG["Store Multi-Layer Elevation Profile"]
    OVERHANG --> STORE_ML["z_ground = Lower Points; z_obstacle_bottom = Bridge Underside; z_obstacle_top = Bridge Deck"]

    DETECT_GAP -->|"No (Continuous Surface)"| SINGLE_LAYER["Store Standard 2.5D Profile"]
    SINGLE_LAYER --> STORE_SL["z_ground = min(z); z_top = max(z); z_obstacle_bottom = z_ground"]

    STORE_ML --> VOTE["Majority Semantic Class Voting + Confidence Weighting"]
    STORE_SL --> VOTE

    VOTE --> HYST_CHECK["Apply Boundary Hysteresis (Prevent Flickering State)"]
    HYST_CHECK --> GRID_OUT(["Compressed 2.5D Polar Grid Map (88.9% Memory Savings)"])
```

### Simple Explanation:
- **Polar Rings**: Matches how LiDAR spins. Fine rings close to the car; wider rings farther away.
- **Multi-Layer Elevation**: If a bridge is overhead, a normal 2D map marks it as a solid wall. FoveaMap detects the empty vertical gap between the road ($z_{\text{ground}}$) and the bridge underside ($z_{\text{obstacle\_bottom}}$), recognizing the road is drivable underneath.

---

## 6. Multi-Object Kalman Tracking & Anti-Ghosting Erasure

How moving vehicles and pedestrians are tracked without leaving lingering ghost footprints on the static map.

```mermaid
flowchart TD
    DYN_PTS(["Dynamic Points (Class 2)"]) --> CLUSTER["Euclidean Centroid Clustering (DBSCAN)"]
    CLUSTER --> NEW_DETECTIONS["New Detections: Centroids [x, y] & Bounding Boxes"]

    EXISTING_TRACKS["Existing Kalman Tracks: State [x, y, vx, vy]"] --> PREDICT["Predict Step: x_pred = F * x, P_pred = F * P * F^T + Q"]
    
    PREDICT --> COST_MATRIX["Build Distance & Similarity Cost Matrix"]
    NEW_DETECTIONS --> COST_MATRIX

    COST_MATRIX --> HUNGARIAN["Hungarian Optimal Assignment (scipy.optimize.linear_sum_assignment)"]

    HUNGARIAN --> MATCH_CHECK{"Matched to Existing Track?"}

    MATCH_CHECK -->|"Matched"| UPDATE_KF["Joseph-Form Kalman Covariance Update: P = (I - KH)P_pred(I - KH)^T + KRK^T"]
    MATCH_CHECK -->|"Unmatched Detection"| SPAWN["Spawn New Track (ID = next_id)"]
    MATCH_CHECK -->|"Unmatched Track"| AGE["Increment Missed Frames; Prune if missed > max_age"]

    UPDATE_KF --> DISPLACEMENT["Calculate Velocity Displacement Vector d = v * dt"]
    DISPLACEMENT --> VACATED_CELLS["Identify Grid Cells Occupied at t-1 but Vacated at t"]
    VACATED_CELLS --> ERASE["Active Footprint Erasure: Reset Vacated Cells to Drivable Ground"]
    
    ERASE --> CLEAN_MAP(["Clean Static Map (Zero Ghost Footprints) + Active 3D Bounding Boxes"])
    SPAWN --> CLEAN_MAP
    AGE --> CLEAN_MAP
```

### Simple Explanation:
- **Hungarian Matching**: Ensures two crossing vehicles do not swap IDs.
- **Joseph-Form Kalman Filter**: Mathematically guarantees stable velocity tracking without numerical errors.
- **Active Footprint Erasure (Anti-Ghosting)**: Moving trucks leave behind empty space where they used to be. The tracker clears those cells immediately so the car does not brake for "ghost" obstacles.

---

## 7. Real-Time Streaming & Dashboard Playback Architecture

How the FastAPI backend compresses data and streams it to the web browser with zero-lag background prefetching.

```mermaid
flowchart TD
    subgraph BACKEND["FastAPI Python Backend (server.py)"]
        TIMER["Frame Clock (up to 30 FPS)"] --> READ_FRAME["Assemble Polar Grid Frame + Active Tracks"]
        READ_FRAME --> JSON_SERIAL["Convert Frame to JSON String"]
        JSON_SERIAL --> ZLIB_COMPRESS["zlib.compress(level=6) Binary Buffer"]
        ZLIB_COMPRESS --> WS_DISPATCH["Send via WebSocket: websocket.send_bytes()"]
    end

    subgraph NETWORK["High-Speed WebSocket Tunnel"]
        WS_DISPATCH -->|"Compressed Binary Frames (~200 KB)"| WS_RECV["Browser onmessage Event"]
    end

    subgraph FRONTEND["Dashboard Web Client (Three.js + JS Engine)"]
        WS_RECV --> ARRAY_BUF["Read ArrayBuffer Payload"]
        ARRAY_BUF --> PAKO_INFLATE["Decompress via pako.inflate()"]
        PAKO_INFLATE --> PARSE_JSON["JSON.parse()"]

        PARSE_JSON --> RENDER_GRID["Render Polar Grid Rings & Elevation Columns (WebGL)"]
        PARSE_JSON --> RENDER_TRACKS["Render Dynamic Bounding Boxes & Velocity Arrows"]
        PARSE_JSON --> UPDATE_METRICS["Update FPS, Memory Savings & Latency Meters"]

        PARSE_JSON --> PLAY_PROGRESS{"Playback Position in Sequence?"}
        PLAY_PROGRESS -->|"< 70%"| NORMAL_LOOP["Continue Playback"]
        PLAY_PROGRESS -->|">= 70%"| PREFETCH["Silently Prefetch Next Sequence via HTTP Chunked Stream"]
        
        PLAY_PROGRESS -->|"Last Frame Reached"| AUTO_LOOP{"Is Auto-Loop ON?"}
        AUTO_LOOP -->|"Yes (Enabled)"| NEXT_SEQ["Smoothly Advance to Next Sequence (0ms Latency)"]
        AUTO_LOOP -->|"No"| REPEAT["Replay Current Sequence from Frame 0"]
    end
```

### Simple Explanation:
1. **Binary Compression (`zlib` + `pako`)**: Shrinks frame data by ~88%, keeping streams fast even on slow connections.
2. **WebGL Rendering**: Three.js draws the polar grid, 3D elevation pillars, and dynamic tracking boxes.
3. **Background Prefetching**: When a sequence is 70% finished, the browser fetches the next one in the background, allowing seamless continuous auto-looping without buffering delays.

---

## 8. Summary Table of Pipeline Components

| Pipeline Stage | Primary File | Key Input | Key Output | Performance Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion** | [`kitti_loader.py`](file:///c:/Users/kaush/Downloads/lidar-2.5D-mapping-main/src/ingestion/kitti_loader.py) | `.bin` Velodyne scans | `(N, 4)` NumPy array | Zero-copy page cached read |
| **Perception** | [`heuristic_fallback.py`](file:///c:/Users/kaush/Downloads/lidar-2.5D-mapping-main/src/perception/heuristic_fallback.py) / [`segment.py`](file:///c:/Users/kaush/Downloads/lidar-2.5D-mapping-main/src/perception/segment.py) | `(N, 4)` point array | Semantic class IDs `[0, 1, 2, 3]` | mIoU $\ge 0.4238$ |
| **Foveation** | [`foveation.py`](file:///c:/Users/kaush/Downloads/lidar-2.5D-mapping-main/src/grid/foveation.py) | Vehicle $v_x$, $\delta$ | Fovea ellipse parameters | Dynamic speed stretch & look-ahead shear |
| **Grid Engine** | [`grid_engine.py`](file:///c:/Users/kaush/Downloads/lidar-2.5D-mapping-main/src/grid/grid_engine.py) | Points + Labels | `GridCell` dictionary | **88.9% memory savings** vs 3D voxels |
| **Tracking** | [`kalman_tracker.py`](file:///c:/Users/kaush/Downloads/lidar-2.5D-mapping-main/src/tracking/kalman_tracker.py) | Dynamic clusters | Filtered tracks + cleared footprints | Hungarian association + anti-ghosting |
| **Streaming** | [`server.py`](file:///c:/Users/kaush/Downloads/lidar-2.5D-mapping-main/src/api/server.py) | Frame dictionary | Binary WebSocket frames | 88.2% bandwidth compression ratio |
| **Dashboard** | [`dashboard/js/app.js`](file:///c:/Users/kaush/Downloads/lidar-2.5D-mapping-main/dashboard/js/app.js) | Binary frames | Interactive 3D WebGL visualization | Smooth 30 FPS playback & continuous auto-loop |
