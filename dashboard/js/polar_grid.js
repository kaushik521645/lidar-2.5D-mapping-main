/**
 * FoveaMap Polar Grid & Dynamic Foveation Contour Overlay for Three.js
 */

class PolarGridOverlay {
  constructor(scene) {
    this.scene = scene;
    this.gridGroup = new THREE.Group();
    this.scene.add(this.gridGroup);

    this.foveaLine = null;
    this.foveaMesh = null;
    // EMA smoothing: retain previous buffer positions to lerp between frames.
    // alpha=0 → instant snap; alpha=1 → never updates. 0.35 gives a ~3-frame lag.
    this._foveaSmoothAlpha = 0.35;
    this._prevFoveaPositions = null;
    this.initStaticRings();
    this.initFoveationMesh();
  }

  initStaticRings() {
    // Static concentric guide rings at 10m, 30m, 50m, 80m, 100m
    const ringRadii = [10.0, 30.0, 50.0, 80.0, 100.0];
    const ringMaterial = new THREE.LineBasicMaterial({
      color: 0x334155,
      transparent: true,
      opacity: 0.4,
      linewidth: 1,
    });

    ringRadii.forEach((r) => {
      const circleGeo = new THREE.BufferGeometry();
      const points = [];
      const segments = 128;
      for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI * 2;
        points.push(new THREE.Vector3(r * Math.cos(theta), r * Math.sin(theta), -1.55));
      }
      circleGeo.setFromPoints(points);
      const line = new THREE.Line(circleGeo, ringMaterial);
      this.gridGroup.add(line);
    });

    // Cross axes (X and Y forward/lateral)
    const axesGeo = new THREE.BufferGeometry();
    const axisPts = [
      new THREE.Vector3(-100, 0, -1.55),
      new THREE.Vector3(100, 0, -1.55),
      new THREE.Vector3(0, -100, -1.55),
      new THREE.Vector3(0, 100, -1.55),
    ];
    axesGeo.setFromPoints(axisPts);
    const axesLine = new THREE.LineSegments(axesGeo, ringMaterial);
    this.gridGroup.add(axesLine);

    // Ego vehicle marker (Green/Cyan Arrow at origin)
    const arrowGeo = new THREE.BufferGeometry();
    const arrowPts = [
      new THREE.Vector3(2.0, 0.0, -1.45),
      new THREE.Vector3(-1.0, -0.8, -1.45),
      new THREE.Vector3(-0.5, 0.0, -1.45),
      new THREE.Vector3(-1.0, 0.8, -1.45),
      new THREE.Vector3(2.0, 0.0, -1.45),
    ];
    arrowGeo.setFromPoints(arrowPts);
    const arrowMat = new THREE.LineBasicMaterial({ color: 0x06b6d4, linewidth: 2 });
    const arrow = new THREE.Line(arrowGeo, arrowMat);
    this.gridGroup.add(arrow);
  }

  initFoveationMesh() {
    // Dynamic Foveation boundary contour line
    const segments = 120;
    const points = [];
    for (let i = 0; i <= segments; i++) {
      points.push(new THREE.Vector3(0, 0, -1.48));
    }
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.9,
      linewidth: 2,
    });
    this.foveaLine = new THREE.Line(lineGeo, lineMat);
    this.gridGroup.add(this.foveaLine);
  }

  setContourPoints(polyline) {
    if (!this.foveaLine || !Array.isArray(polyline) || polyline.length === 0) return;
    const positions = this.foveaLine.geometry.attributes.position.array;
    const maxPts = positions.length / 3;
    const n = Math.min(polyline.length, maxPts);
    const alpha = this._foveaSmoothAlpha;
    const prev = this._prevFoveaPositions;
    const usePrev = prev !== null && prev.length === positions.length;

    for (let i = 0; i < n; i++) {
      const tx = polyline[i][0];
      const ty = polyline[i][1];
      if (usePrev) {
        positions[i * 3]     = prev[i * 3]     * alpha + tx * (1 - alpha);
        positions[i * 3 + 1] = prev[i * 3 + 1] * alpha + ty * (1 - alpha);
      } else {
        positions[i * 3]     = tx;
        positions[i * 3 + 1] = ty;
      }
      positions[i * 3 + 2] = -1.48;
    }
    // Close the loop back to the first point
    if (n < maxPts) {
      positions[n * 3]     = positions[0];
      positions[n * 3 + 1] = positions[1];
      positions[n * 3 + 2] = -1.48;
    }

    // Persist smoothed positions for next frame
    if (!this._prevFoveaPositions) {
      this._prevFoveaPositions = new Float32Array(positions.length);
    }
    this._prevFoveaPositions.set(positions);

    this.foveaLine.geometry.attributes.position.needsUpdate = true;
  }

  updateFoveationContour(
    speed_mps,
    steering_angle_rad,
    base_radius = 10.0,
    max_stretch = 2.5,
    activeTracks = [],
    motionFoveationEnabled = true,
    motionSpeedThreshold = 0.8,
    motionLeadTime = 1.0,
    collisionFocusOnly = true,
    corridorHalfWidthM = 3.2,
    maxThreatDistanceM = 35.0,
    shear_strength = 0.6
  ) {
    if (!this.foveaLine) return;

    const segments = 120;
    const positions = this.foveaLine.geometry.attributes.position.array;

    // Calculate speed stretch
    const speed_ref = 20.0;
    const ratio = Math.min(Math.max(0.0, speed_mps) / speed_ref, 1.0);
    const stretch = 1.0 + (max_stretch - 1.0) * ratio;

    const a = base_radius * stretch;
    const b = base_radius;
    const tau = 2 * Math.PI;

    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2 - Math.PI;

      // Angle relative to steering direction with shear sensitivity and correct wrap
      let d_theta = theta - (steering_angle_rad * shear_strength);
      d_theta = ((d_theta % tau) + tau) % tau;
      if (d_theta > Math.PI) d_theta -= tau;

      const is_forward = Math.abs(d_theta) < (Math.PI / 2.0);

      // True Ellipse in polar coordinates
      const denominator = Math.sqrt(Math.pow(b * Math.cos(d_theta), 2) + Math.pow(a * Math.sin(d_theta), 2));
      const r_ellipse = denominator > 1e-6 ? (a * b) / denominator : b;

      // Piecewise: front half is ellipse, back half is base circle
      let r = is_forward ? r_ellipse : b;

      // Dynamic collision-focused foveation lobes
      if (motionFoveationEnabled && activeTracks && activeTracks.length > 0) {
        let r_attn = 0;
        activeTracks.forEach(track => {
          if (track.class_id !== 3) return; // Only dynamic obstacles
          const tx = track.position_xy[0];
          const ty = track.position_xy[1];
          const tr = Math.hypot(tx, ty);
          const ttheta = Math.atan2(ty, tx);

          const vel = track.velocity_xy || [0, 0];
          const vx = vel[0] || 0;
          const vy = vel[1] || 0;
          const speed = Math.hypot(vx, vy);
          const width = track.bbox_size_xy ? Math.max(track.bbox_size_xy[0], track.bbox_size_xy[1]) : 4.0;

          if (collisionFocusOnly) {
            // Strictly focus only on oncoming / straight-on collision threats
            if (tr > maxThreatDistanceM || tr < 0.5) return;

            const closing_speed = -(tx * vx + ty * vy) / Math.max(tr, 0.1);
            let is_threat = false;
            let reach = 0.0;
            let bearing = ttheta;

            // Case A: Oncoming Traffic Ahead within roadway limits (|y| <= 5.5m) closing in
            if (tx > 0.0 && Math.abs(ty) <= 5.5 && (closing_speed >= 0.5 || vx <= -0.5)) {
              is_threat = true;
              reach = Math.min(tr + 5.0, maxThreatDistanceM);
              bearing = Math.atan2(ty, tx);
            }
            // Case B: Straight-On Obstacle in Direct Forward Travel Lane (|y| <= corridorHalfWidthM)
            else if (tx > 0.0 && Math.abs(ty) <= corridorHalfWidthM) {
              // Ignore if moving away
              if (closing_speed >= -0.5 && vx <= 0.5 && tr <= 25.0) {
                is_threat = true;
                reach = Math.min(tr + 5.0, maxThreatDistanceM);
                bearing = Math.atan2(ty, tx);
              }
            }
            // Case C: Crossing Traffic entering ego corridor on collision course
            else if (speed >= 0.8 && Math.abs(ty) > corridorHalfWidthM && Math.abs(ty) <= 10.0) {
              if ((ty > 0.0 && vy < -0.5) || (ty < 0.0 && vy > 0.5)) {
                const t_enter = (Math.abs(ty) - corridorHalfWidthM) / Math.max(Math.abs(vy), 0.1);
                const future_x = tx + vx * t_enter;
                if (future_x >= 0.0 && future_x <= 25.0 && t_enter <= 3.0) {
                  is_threat = true;
                  const future_r = Math.hypot(future_x, 0.0);
                  reach = Math.min(future_r + 4.0, maxThreatDistanceM);
                  bearing = Math.atan2(ty, tx);
                }
              }
            }

            if (!is_threat) return;

            // Smooth Gaussian lobe with minimum 0.35 rad angular spread (eliminates needle spear)
            const sigma = Math.max(0.35, width / Math.max(tr, 1.0));
            let diff = theta - bearing;
            diff = ((diff + Math.PI) % (2 * Math.PI)) - Math.PI;
            if (diff < -Math.PI) diff += 2 * Math.PI;

            const spike = reach * Math.exp(-0.5 * Math.pow(diff / sigma, 2));
            r_attn = Math.max(r_attn, spike);
          } else {
            // General motion lookahead (fallback)
            const sigma = Math.max(width / Math.max(tr, 1.0), 0.2);
            if (speed >= motionSpeedThreshold) {
              const pred_x = tx + vx * motionLeadTime;
              const pred_y = ty + vy * motionLeadTime;
              const pred_r = Math.hypot(pred_x, pred_y);
              const pred_theta = Math.atan2(pred_y, pred_x);
              const reach = Math.min(Math.max(tr, pred_r) + 4.0, maxThreatDistanceM);
              let diff = theta - pred_theta;
              diff = ((diff + Math.PI) % (2 * Math.PI)) - Math.PI;
              if (diff < -Math.PI) diff += 2 * Math.PI;
              const spike = reach * Math.exp(-0.5 * Math.pow(diff / sigma, 2));
              r_attn = Math.max(r_attn, spike);
            } else {
              let diff = theta - ttheta;
              diff = ((diff + Math.PI) % (2 * Math.PI)) - Math.PI;
              if (diff < -Math.PI) diff += 2 * Math.PI;
              const spike = Math.min(tr + 5.0, maxThreatDistanceM) * Math.exp(-0.5 * Math.pow(diff / sigma, 2));
              r_attn = Math.max(r_attn, spike);
            }
          }
        });
        r = Math.max(r, r_attn);
      }

      const x = r * Math.cos(theta);
      const y = r * Math.sin(theta);
      const z = -1.48;

      const alpha = this._foveaSmoothAlpha;
      const prev = this._prevFoveaPositions;
      if (prev !== null && prev.length === positions.length) {
        positions[i * 3]     = prev[i * 3]     * alpha + x * (1 - alpha);
        positions[i * 3 + 1] = prev[i * 3 + 1] * alpha + y * (1 - alpha);
      } else {
        positions[i * 3]     = x;
        positions[i * 3 + 1] = y;
      }
      positions[i * 3 + 2] = z;
    }

    // Persist smoothed positions for next frame
    if (!this._prevFoveaPositions) {
      this._prevFoveaPositions = new Float32Array(positions.length);
    }
    this._prevFoveaPositions.set(positions);

    this.foveaLine.geometry.attributes.position.needsUpdate = true;
  }

}

window.PolarGridOverlay = PolarGridOverlay;
