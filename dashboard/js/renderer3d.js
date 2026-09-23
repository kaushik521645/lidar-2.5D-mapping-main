/**
 * Three.js 2.5D Polar Grid & Multi-Layer Scene Renderer for FoveaMap
 */

class FoveaRenderer3D {
  constructor(canvasContainerId) {
    this.container = document.getElementById(canvasContainerId);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0e17);
    this.scene.fog = new THREE.FogExp2(0x0a0e17, 0.012);

    // Camera setup
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(55, aspect, 0.1, 500);
    this.camera.position.set(-25, 0, 28);
    this.camera.up.set(0, 0, 1); // Z-up LiDAR coordinate frame

    // WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // OrbitControls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(15, 0, 0);
    this.controls.maxPolarAngle = Math.PI / 2 + 0.1;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
    dirLight.position.set(20, 30, 40);
    this.scene.add(dirLight);

    // Groups
    this.cellsGroup = new THREE.Group();
    this.tracksGroup = new THREE.Group();
    this.scene.add(this.cellsGroup);
    this.scene.add(this.tracksGroup);

    // Polar Grid and Foveation Contour Overlay
    this.polarOverlay = new PolarGridOverlay(this.scene);

    // Color Palette
    this.classColors = {
      0: new THREE.Color(0x22c55e),  // Drivable Terrain
      1: new THREE.Color(0xf59e0b),  // Non-Drivable Terrain
      2: new THREE.Color(0xef4444),  // Static Obstacle
      3: new THREE.Color(0x3b82f6),  // Dynamic Object
      '-1': new THREE.Color(0x64748b), // Unknown
    };

    // Shared Geometries & Materials
    this.baseBoxGeo = new THREE.BoxGeometry(1, 1, 1);
    this.boxMaterials = {};
    for (const [cls, col] of Object.entries(this.classColors)) {
      this.boxMaterials[cls] = new THREE.MeshLambertMaterial({
        color: col,
        transparent: true,
        opacity: 0.88,
      });
    }

    // Overhang material
    this.overhangMaterial = new THREE.MeshLambertMaterial({
      color: 0xff4444,
      transparent: true,
      opacity: 0.92,
    });

    // High-Performance InstancedMesh Pools
    this.initialCapacity = 4096;
    this.dummy = new THREE.Object3D();
    this.instancedMeshes = {};
    const allMeshKeys = ['0', '1', '2', '3', '-1', 'overhang'];
    for (const k of allMeshKeys) {
      const mat = k === 'overhang' ? this.overhangMaterial : (this.boxMaterials[k] || this.boxMaterials['-1']);
      const im = new THREE.InstancedMesh(this.baseBoxGeo, mat, this.initialCapacity);
      im.count = 0;
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.cellsGroup.add(im);
      this.instancedMeshes[k] = im;
    }

    // Resize Handler
    window.addEventListener('resize', () => this.onResize());

    this.animate();
  }

  ensureCapacity(key, neededCount) {
    let im = this.instancedMeshes[key];
    if (im.instanceMatrix.count < neededCount) {
      this.cellsGroup.remove(im);
      im.dispose();
      const newCap = Math.max(neededCount * 2, im.instanceMatrix.count * 2);
      const mat = key === 'overhang' ? this.overhangMaterial : (this.boxMaterials[key] || this.boxMaterials['-1']);
      im = new THREE.InstancedMesh(this.baseBoxGeo, mat, newCap);
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.cellsGroup.add(im);
      this.instancedMeshes[key] = im;
    }
    return im;
  }

  onResize() {
    if (!this.container) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  setCameraView(viewMode) {
    if (viewMode === 'topdown') {
      this.camera.position.set(15, 0, 65);
      this.controls.target.set(15, 0, 0);
    } else if (viewMode === 'perspective') {
      this.camera.position.set(-25, 0, 28);
      this.controls.target.set(15, 0, 0);
    } else if (viewMode === 'cockpit') {
      this.camera.position.set(-3, 0, 1.5);
      this.controls.target.set(25, 0, 0);
    }
    this.controls.update();
  }

  clearScene() {
    for (const im of Object.values(this.instancedMeshes)) {
      im.count = 0;
      im.instanceMatrix.needsUpdate = true;
    }
    while (this.tracksGroup.children.length > 0) {
      const obj = this.tracksGroup.children.pop();
      if (obj.geometry) obj.geometry.dispose();
    }
  }

  renderFrame(frameData) {
    if (!frameData || !frameData.cells) {
      this.clearScene();
      return;
    }

    const cells = frameData.cells;
    const v_state = frameData.vehicle_state || { speed_mps: 0, steering_angle_rad: 0 };
    const fovea_cfg = frameData.foveation_params || { base_fine_radius_m: 10, max_stretch: 2.5, shear_strength: 0.6 };

    // Update Foveation Contour: Prefer backend polyline for single source of truth
    if (frameData.fovea_polyline && frameData.fovea_polyline.length > 0) {
      this.polarOverlay.setContourPoints(frameData.fovea_polyline);
    } else {
      this.polarOverlay.updateFoveationContour(
        v_state.speed_mps,
        v_state.steering_angle_rad,
        fovea_cfg.base_fine_radius_m,
        fovea_cfg.max_stretch,
        frameData.tracks || [],
        fovea_cfg.motion_foveation_enabled !== false,
        fovea_cfg.motion_speed_threshold_mps || 0.8,
        fovea_cfg.motion_lead_time_s || 1.0,
        fovea_cfg.collision_focus_only !== false,
        fovea_cfg.corridor_half_width_m || 3.2,
        fovea_cfg.max_threat_distance_m || 35.0,
        fovea_cfg.shear_strength || 0.6
      );
    }

    // Count cells needed per instanced mesh class
    const counts = { '0': 0, '1': 0, '2': 0, '3': 0, '-1': 0, 'overhang': 0 };
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      const clsKey = c.semantic_class !== undefined && this.instancedMeshes[String(c.semantic_class)] ? String(c.semantic_class) : '-1';
      counts[clsKey]++;
      if (c.elevation_obstacle_bottom !== null && c.elevation_obstacle_top !== null) {
        counts['overhang']++;
      }
    }

    // Ensure capacity for all classes
    for (const [k, cnt] of Object.entries(counts)) {
      this.ensureCapacity(k, cnt);
    }

    // Update instance transforms
    const indices = { '0': 0, '1': 0, '2': 0, '3': 0, '-1': 0, 'overhang': 0 };
    for (let i = 0; i < cells.length; i++) {
      const cell = cells[i];
      const x = cell.x || 0;
      const y = cell.y || 0;
      const res = cell.resolution_tier || 0.05;
      const cls = cell.semantic_class;
      const clsKey = cls !== undefined && this.instancedMeshes[String(cls)] ? String(cls) : '-1';
      const z_ground = cell.elevation_ground !== undefined ? cell.elevation_ground : -1.5;

      // Ground slab
      const slabHeight = cls === 2 ? 1.5 : (cls === 3 ? 1.2 : 0.12);
      this.dummy.scale.set(res * 0.95, res * 0.95, slabHeight);
      this.dummy.position.set(x, y, z_ground + slabHeight / 2.0);
      this.dummy.updateMatrix();

      const im = this.instancedMeshes[clsKey];
      im.setMatrixAt(indices[clsKey]++, this.dummy.matrix);

      // Overhang slab
      if (cell.elevation_obstacle_bottom !== null && cell.elevation_obstacle_top !== null) {
        const obs_bottom = cell.elevation_obstacle_bottom;
        const obs_top = cell.elevation_obstacle_top;
        const obs_height = Math.max(0.2, obs_top - obs_bottom);

        this.dummy.scale.set(res * 0.95, res * 0.95, obs_height);
        this.dummy.position.set(x, y, obs_bottom + obs_height / 2.0);
        this.dummy.updateMatrix();

        const ohIm = this.instancedMeshes['overhang'];
        ohIm.setMatrixAt(indices['overhang']++, this.dummy.matrix);
      }
    }

    // Apply updated instance counts and notify GPU
    for (const [k, im] of Object.entries(this.instancedMeshes)) {
      im.count = indices[k];
      im.instanceMatrix.needsUpdate = true;
    }

    // Render Tracked Objects (Bounding Box + Velocity Vector)
    while (this.tracksGroup.children.length > 0) {
      const obj = this.tracksGroup.children.pop();
      if (obj.geometry) obj.geometry.dispose();
    }

    // Render Tracked Objects (Bounding Box + Velocity Vector)
    if (frameData.tracks) {
      frameData.tracks.forEach((track) => {
        const [tx, ty] = track.position_xy;
        const [vx, vy] = track.velocity_xy;
        const [bl, bw] = track.bbox_size_xy || [3.5, 1.8];

        // Wireframe box
        const bboxGeo = new THREE.BoxGeometry(bl, bw, 1.6);
        const bboxMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 2 });
        const wireframe = new THREE.LineSegments(new THREE.EdgesGeometry(bboxGeo), bboxMat);
        wireframe.position.set(tx, ty, -0.6);
        this.tracksGroup.add(wireframe);

        // Velocity vector arrow
        const v_len = Math.hypot(vx, vy);
        if (v_len > 0.5) {
          const dir = new THREE.Vector3(vx, vy, 0).normalize();
          const origin = new THREE.Vector3(tx, ty, 0.4);
          const arrowHelper = new THREE.ArrowHelper(dir, origin, Math.min(v_len * 0.8, 8.0), 0x06b6d4, 0.8, 0.4);
          this.tracksGroup.add(arrowHelper);
        }
      });
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

window.FoveaRenderer3D = FoveaRenderer3D;
