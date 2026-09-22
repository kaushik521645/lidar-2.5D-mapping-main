/**
 * FoveaMap Main Dashboard Application Controller
 * Handles 3D rendering, WebSocket/Precomputed playback, dynamic foveation,
 * and unified progress/loading states across all user interactions.
 */

document.addEventListener('DOMContentLoaded', async () => {
  const renderer = new FoveaRenderer3D('canvas3d');
  const metricsHUD = new MetricsHUDController();
  const scenarioMgr = new ScenarioManager();
  const loader = new DashboardLoadingController();

  let isPlaying = true;
  let isLiveWS = false;
  let ws = null;
  let playbackInterval = null;
  let fpsRate = 10;

  // Live Foveation Overrides
  let overrideParams = {
    speed_mps: 8.0,
    steering_angle_rad: 0.0,
    base_fine_radius_m: 10.0,
    max_stretch: 2.5,
    shear_strength: 0.6,
    motion_foveation_enabled: true,
    collision_focus_only: true,
    motion_speed_threshold_mps: 0.8,
    motion_lead_time_s: 1.0,
  };
  let lastRenderedFrame = null;

  // Auto-Loop Controls & State
  let autoLoopAll = true;
  let loopScope = 'all_kitti'; // 'all_kitti', 'ready_only', 'all_scenes'
  let isTransitioning = false;

  // DOM Elements
  const selectScenario = document.getElementById('select-scenario');
  const selectMode = document.getElementById('select-mode');
  const btnPlay = document.getElementById('btn-play');
  const btnStep = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');
  const sliderScrub = document.getElementById('slider-scrub');
  const lblFrame = document.getElementById('lbl-frame-counter');
  const lblScenarioDesc = document.getElementById('lbl-scenario-desc');

  // Auto-Loop Elements
  const toggleAutoLoop = document.getElementById('toggle-auto-loop');
  const selectLoopScope = document.getElementById('select-loop-scope');
  const btnPrevSeq = document.getElementById('btn-prev-seq');
  const btnNextSeq = document.getElementById('btn-next-seq');
  const badgeAutoLoop = document.getElementById('badge-autoloop');
  const lblSeqPos = document.getElementById('lbl-sequence-pos');

  // Sliders
  const sliderSpeed = document.getElementById('slider-speed');
  const valSpeed = document.getElementById('val-speed');
  const sliderSteering = document.getElementById('slider-steering');
  const valSteering = document.getElementById('val-steering');
  const sliderMaxStretch = document.getElementById('slider-max-stretch');
  const valMaxStretch = document.getElementById('val-max-stretch');
  const sliderBaseRadius = document.getElementById('slider-base-radius');
  const valBaseRadius = document.getElementById('val-base-radius');

  // Motion-Adaptive Foveation Elements
  const chkMotionFoveation = document.getElementById('chk-motion-foveation');
  const chkCollisionFocus = document.getElementById('chk-collision-focus');
  const sliderMotionThresh = document.getElementById('slider-motion-thresh');
  const valMotionThresh = document.getElementById('val-motion-thresh');
  const sliderLeadTime = document.getElementById('slider-lead-time');
  const valLeadTime = document.getElementById('val-lead-time');
  const hudMotionStatus = document.getElementById('hud-motion-status');

  // Camera Buttons
  document.getElementById('btn-view-perspective')?.addEventListener('click', () => {
    renderer.setCameraView('perspective');
    loader.toast("Camera: 3D Perspective View");
  });
  document.getElementById('btn-view-topdown')?.addEventListener('click', () => {
    renderer.setCameraView('topdown');
    loader.toast("Camera: Bird's Eye 2D Top-Down View");
  });
  document.getElementById('btn-view-cockpit')?.addEventListener('click', () => {
    renderer.setCameraView('cockpit');
    loader.toast("Camera: Ego-Vehicle Cockpit View");
  });

  // Slider Track Fill logic
  function updateSliderFill(slider) {
    if (!slider) return;
    const min = parseFloat(slider.min) || 0;
    const max = parseFloat(slider.max) || 100;
    const val = parseFloat(slider.value) || 0;
    const percent = ((val - min) / (max - min)) * 100;
    slider.style.setProperty('--fill-percent', `${percent}%`);
  }

  function updateAllFills() {
    [sliderSpeed, sliderSteering, sliderMaxStretch, sliderBaseRadius, sliderMotionThresh, sliderLeadTime, sliderScrub].forEach(updateSliderFill);
  }

  // Helper: Get active playlist options based on loopScope
  function getPlaylistSequences() {
    if (!selectScenario) return [];
    const allOptions = Array.from(selectScenario.options);

    if (loopScope === 'all_kitti') {
      const kitti = allOptions.filter(opt => opt.value.startsWith('kitti_seq_') || opt.value === 'kitti_odometry');
      return kitti.length > 0 ? kitti : allOptions;
    } else if (loopScope === 'ready_only') {
      const readyIds = new Set([
        'kitti_seq_00', 'kitti_odometry', 'kitti_seq_01', 'kitti_seq_02',
        'kitti_seq_03', 'kitti_seq_04', 'kitti_seq_08', 'kitti_seq_21',
        'synthetic_kitti_like', 'urban_intersection', 'highway_cruise',
        'pothole_alley', 'bridge_overpass'
      ]);
      const ready = allOptions.filter(opt => readyIds.has(opt.value) || scenarioMgr.hasCached(opt.value));
      return ready.length > 0 ? ready : allOptions;
    } else {
      return allOptions;
    }
  }

  // Helper: Update HUD badge and sequence position label
  function updateSequenceBadge() {
    const playlist = getPlaylistSequences();
    const currentVal = selectScenario?.value || scenarioMgr.currentScenarioId;
    const idx = playlist.findIndex(opt => opt.value === currentVal);
    if (lblSeqPos && idx >= 0) {
      lblSeqPos.textContent = `Seq ${idx + 1} / ${playlist.length}`;
    }
    if (badgeAutoLoop) {
      if (autoLoopAll) {
        badgeAutoLoop.className = 'badge-status looping';
        badgeAutoLoop.textContent = '🔁 AUTO-LOOP ON';
      } else {
        badgeAutoLoop.className = 'badge-status ready';
        badgeAutoLoop.textContent = 'SINGLE REPEAT';
      }
    }
  }

  // Automatic & Manual Sequence Advance Handler
  async function advanceToNextSequence(direction = 1, userInitiated = false) {
    if (isTransitioning) return;
    const playlist = getPlaylistSequences();
    if (playlist.length === 0) return;

    const currentVal = selectScenario?.value || scenarioMgr.currentScenarioId;
    let currIdx = playlist.findIndex(opt => opt.value === currentVal);
    if (currIdx < 0) currIdx = 0;

    const nextIdx = (currIdx + direction + playlist.length) % playlist.length;
    const nextOption = playlist[nextIdx];
    const nextScenarioId = nextOption.value;
    const nextName = nextOption.textContent.trim();

    if (selectScenario) {
      selectScenario.value = nextScenarioId;
    }

    isTransitioning = true;
    updateSequenceBadge();

    if (userInitiated) {
      loader.toast(`${direction > 0 ? '⏭ Next' : '⏮ Prev'}: ${nextName}`);
    } else {
      loader.toast(`🔁 Auto-Loop: Advancing to ${nextName}`);
    }

    try {
      await switchScenario(nextScenarioId);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: 'set_scenario', scenario_id: nextScenarioId }));
      }
    } finally {
      isTransitioning = false;
      updateSequenceBadge();
    }
  }

  // Auto-Loop Toggle & Scope Listeners
  toggleAutoLoop?.addEventListener('change', (e) => {
    autoLoopAll = e.target.checked;
    updateSequenceBadge();
    loader.toast(autoLoopAll ? '🔁 Auto-Loop: All Sequences ON' : '⏸ Auto-Loop: Single Sequence Repeat');
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ command: 'set_auto_loop', enabled: autoLoopAll }));
    }
  });

  selectLoopScope?.addEventListener('change', (e) => {
    loopScope = e.target.value;
    updateSequenceBadge();
    const playlist = getPlaylistSequences();
    loader.toast(`Loop Playlist: ${playlist.length} sequences active`);
  });

  btnPrevSeq?.addEventListener('click', () => {
    advanceToNextSequence(-1, true);
  });

  btnNextSeq?.addEventListener('click', () => {
    advanceToNextSequence(1, true);
  });

  // Slider Event Listeners (Live Dynamic Foveation updates)
  let sliderDebounce = null;
  function onSliderChange() {
    overrideParams.speed_mps = parseFloat(sliderSpeed.value);
    valSpeed.textContent = `${overrideParams.speed_mps.toFixed(1)} m/s`;

    const steerDeg = parseFloat(sliderSteering.value);
    overrideParams.steering_angle_rad = (steerDeg * Math.PI) / 180.0;
    valSteering.textContent = `${steerDeg > 0 ? '+' : ''}${steerDeg.toFixed(0)}°`;

    overrideParams.max_stretch = parseFloat(sliderMaxStretch.value);
    valMaxStretch.textContent = `${overrideParams.max_stretch.toFixed(1)}x`;

    overrideParams.base_fine_radius_m = parseFloat(sliderBaseRadius.value);
    valBaseRadius.textContent = `${overrideParams.base_fine_radius_m.toFixed(1)} m`;

    if (chkMotionFoveation) {
      overrideParams.motion_foveation_enabled = chkMotionFoveation.checked;
    }
    if (chkCollisionFocus) {
      overrideParams.collision_focus_only = chkCollisionFocus.checked;
    }
    if (sliderMotionThresh && valMotionThresh) {
      overrideParams.motion_speed_threshold_mps = parseFloat(sliderMotionThresh.value);
      valMotionThresh.textContent = `${overrideParams.motion_speed_threshold_mps.toFixed(1)} m/s`;
    }
    if (sliderLeadTime && valLeadTime) {
      overrideParams.motion_lead_time_s = parseFloat(sliderLeadTime.value);
      valLeadTime.textContent = `${overrideParams.motion_lead_time_s.toFixed(1)} s`;
    }

    const currentTracks = lastRenderedFrame ? (lastRenderedFrame.tracks || []) : [];

    // Immediately update 3D contour overlay with live tracks and motion parameters
    renderer.polarOverlay.updateFoveationContour(
      overrideParams.speed_mps,
      overrideParams.steering_angle_rad,
      overrideParams.base_fine_radius_m,
      overrideParams.max_stretch,
      currentTracks,
      overrideParams.motion_foveation_enabled,
      overrideParams.motion_speed_threshold_mps,
      overrideParams.motion_lead_time_s,
      overrideParams.collision_focus_only !== false,
      3.2,
      35.0
    );

    // If WebSocket is active, send live params
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        command: 'set_params',
        speed_mps: overrideParams.speed_mps,
        steering_angle_rad: overrideParams.steering_angle_rad,
        motion_foveation_enabled: overrideParams.motion_foveation_enabled,
        collision_focus_only: overrideParams.collision_focus_only,
      }));
    }
    updateAllFills();

    // Debounced toast feedback for slider adjustments
    clearTimeout(sliderDebounce);
    sliderDebounce = setTimeout(() => {
      loader.toast(`Foveation: Speed ${overrideParams.speed_mps.toFixed(1)} m/s | Steer ${steerDeg}° | Collision Threat Only: ${overrideParams.collision_focus_only ? 'ON' : 'OFF'}`);
    }, 600);
  }

  [sliderSpeed, sliderSteering, sliderMaxStretch, sliderBaseRadius, sliderMotionThresh, sliderLeadTime].forEach((s) => {
    s?.addEventListener('input', onSliderChange);
  });
  chkMotionFoveation?.addEventListener('change', onSliderChange);
  chkCollisionFocus?.addEventListener('change', onSliderChange);

  // Scenario / Dataset Sequence Change Handler with Full Progress Bar
  async function switchScenario(scenarioId) {
    const selectedOption = selectScenario?.options[selectScenario.selectedIndex];
    const sceneName = selectedOption ? selectedOption.textContent.trim() : scenarioId;
    const isFast = scenarioMgr.hasCached(scenarioId);

    // Start prominent progress indicator & modal
    loader.start({
      title: "Loading Dataset Sequence",
      subtitle: sceneName,
      isFast: isFast,
    });

    if (lblScenarioDesc) {
      lblScenarioDesc.textContent = `Loading & processing sequence: ${sceneName}...`;
    }

    try {
      const frames = await scenarioMgr.loadScenario(scenarioId, (pct, statusText) => {
        loader.setProgress(pct, statusText);
      });

      if (sliderScrub) {
        sliderScrub.max = Math.max(0, frames.length - 1);
        sliderScrub.value = 0;
        updateSliderFill(sliderScrub);
      }

      const currentFrame = scenarioMgr.getCurrentFrame();
      if (currentFrame) {
        if (lblScenarioDesc) {
          lblScenarioDesc.textContent = currentFrame.description || `Loaded ${frames.length} frames for ${sceneName}.`;
        }
        renderCurrentFrame(currentFrame);
      }

      updateSequenceBadge();
      loader.complete(`Ready (${frames.length} frames)`);
      if (!isFast) {
        loader.toast(`Loaded: ${sceneName} (${frames.length} frames)`);
      }
    } catch (err) {
      console.error("[App] Failed to load scenario:", err);
      loader.error(`Failed to load ${sceneName}`);
    }
  }

  selectScenario?.addEventListener('change', (e) => {
    switchScenario(e.target.value);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ command: 'set_scenario', scenario_id: e.target.value }));
    }
  });

  // Playback Control Handlers with Loading / Toast Feedback
  btnPlay?.addEventListener('click', () => {
    isPlaying = !isPlaying;
    btnPlay.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
    loader.toast(isPlaying ? '▶ Playback Resumed' : '⏸ Playback Paused');
  });

  btnStep?.addEventListener('click', async () => {
    isPlaying = false;
    if (btnPlay) btnPlay.textContent = '▶ Play';

    if (scenarioMgr.isLastFrame() && autoLoopAll) {
      await advanceToNextSequence(1, true);
      return;
    }

    const f = scenarioMgr.nextFrame();
    if (f) {
      renderCurrentFrame(f);
      loader.toast(`Stepped to Frame ${scenarioMgr.currentFrameIdx + 1}`);
    }
  });

  btnReset?.addEventListener('click', () => {
    const f = scenarioMgr.setFrame(0);
    if (f) {
      renderCurrentFrame(f);
      loader.toast('Reset playback to Frame 1');
    }
  });

  sliderScrub?.addEventListener('input', (e) => {
    isPlaying = false;
    if (btnPlay) btnPlay.textContent = '▶ Play';
    const frameIdx = parseInt(e.target.value);
    const f = scenarioMgr.setFrame(frameIdx);
    updateSliderFill(sliderScrub);
    if (f) renderCurrentFrame(f);
  });

  function renderCurrentFrame(frameData) {
    if (!frameData) return;
    lastRenderedFrame = frameData;

    // Apply live foveation slider overrides if adjusted by user
    frameData.foveation_params = {
      base_fine_radius_m: overrideParams.base_fine_radius_m,
      max_stretch: overrideParams.max_stretch,
      motion_foveation_enabled: overrideParams.motion_foveation_enabled,
      collision_focus_only: overrideParams.collision_focus_only !== false,
      motion_speed_threshold_mps: overrideParams.motion_speed_threshold_mps,
      motion_lead_time_s: overrideParams.motion_lead_time_s,
      corridor_half_width_m: 3.2,
      max_threat_distance_m: 35.0,
    };

    // Update Motion Status HUD Badge
    if (hudMotionStatus) {
      if (!overrideParams.motion_foveation_enabled) {
        hudMotionStatus.textContent = 'DISABLED';
        hudMotionStatus.style.color = '#ef4444';
      } else if (overrideParams.collision_focus_only) {
        const threats = (frameData.tracks || []).filter(t => {
          const tx = t.position_xy[0];
          const ty = t.position_xy[1];
          const tr = Math.hypot(tx, ty);
          if (tr > 35.0 || tr < 0.5) return false;
          const vel = t.velocity_xy || [0, 0];
          const vx = vel[0] || 0;
          const vy = vel[1] || 0;
          const closing_speed = -(tx * vx + ty * vy) / Math.max(tr, 0.1);
          if (tx > 0.0 && Math.abs(ty) <= 5.5 && (closing_speed >= 0.5 || vx <= -0.5)) return true;
          if (tx > 0.0 && Math.abs(ty) <= 3.2 && closing_speed >= -0.5 && vx <= 0.5 && tr <= 25.0) return true;
          return false;
        });
        if (threats.length > 0) {
          hudMotionStatus.textContent = `ONCOMING THREAT (${threats.length} Tracked)`;
          hudMotionStatus.style.color = '#38bdf8';
        } else {
          hudMotionStatus.textContent = 'CORRIDOR CLEAR';
          hudMotionStatus.style.color = '#4ade80';
        }
      } else {
        const movingTracks = (frameData.tracks || []).filter(t => {
          const vel = t.velocity_xy || [0, 0];
          return Math.hypot(vel[0], vel[1]) >= overrideParams.motion_speed_threshold_mps;
        });
        if (movingTracks.length > 0) {
          hudMotionStatus.textContent = `LOCKED (${movingTracks.length} Moving)`;
          hudMotionStatus.style.color = '#38bdf8';
        } else {
          hudMotionStatus.textContent = 'ACTIVE (Tracking)';
          hudMotionStatus.style.color = '#4ade80';
        }
      }
    }

    // Update 3D viewport
    renderer.renderFrame(frameData);

    // Update Telemetry HUD
    metricsHUD.updateMetrics(frameData);

    // Update Scrub bar and label
    if (sliderScrub && !isLiveWS) {
      sliderScrub.value = scenarioMgr.currentFrameIdx;
      updateSliderFill(sliderScrub);
    }
    if (lblFrame) {
      const total = scenarioMgr.frames.length || 30;
      lblFrame.textContent = `Frame ${scenarioMgr.currentFrameIdx + 1} / ${total}`;
    }
  }

  // Precomputed Playback Loop (10 Hz) with Auto-Loop across Sequences
  function startPlaybackLoop() {
    if (playbackInterval) clearInterval(playbackInterval);
    playbackInterval = setInterval(async () => {
      if (isPlaying && !isLiveWS && !isTransitioning) {
        const frameCount = scenarioMgr.getFrameCount();
        const currIdx = scenarioMgr.currentFrameIdx;

        // Prefetch the upcoming sequence ahead of time when current sequence is 70% complete
        if (autoLoopAll && frameCount > 3 && currIdx >= Math.floor(frameCount * 0.7)) {
          const playlist = getPlaylistSequences();
          const currentVal = selectScenario?.value || scenarioMgr.currentScenarioId;
          const cIdx = playlist.findIndex(opt => opt.value === currentVal);
          if (cIdx >= 0) {
            const nextOpt = playlist[(cIdx + 1) % playlist.length];
            scenarioMgr.prefetchScenario(nextOpt.value);
          }
        }

        // When current sequence reaches its last frame, automatically advance to next sequence
        if (scenarioMgr.isLastFrame()) {
          if (autoLoopAll) {
            await advanceToNextSequence(1, false);
            return;
          }
        }

        const frame = scenarioMgr.nextFrame();
        if (frame) renderCurrentFrame(frame);
      }
    }, 1000 / fpsRate);
  }

  // WebSocket Live Mode with Connection Progress
  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/grid`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log('[WS] Connected to FoveaMap live stream.');
      document.getElementById('ws-status-dot')?.style.setProperty('background-color', '#22c55e');
      loader.toast("WebSocket Stream: Connected & Live");
    };

    ws.onmessage = (event) => {
      if (isLiveWS) {
        try {
          let dataStr;
          if (event.data instanceof ArrayBuffer) {
            dataStr = pako.inflate(event.data, { to: 'string' });
          } else {
            dataStr = event.data;
          }
          const frameData = JSON.parse(dataStr);

          // Sync sequence in UI if WebSocket stream auto-advanced sequence
          if (frameData.scenario_id && selectScenario && selectScenario.value !== frameData.scenario_id) {
            selectScenario.value = frameData.scenario_id;
            updateSequenceBadge();
          }

          renderCurrentFrame(frameData);
        } catch (err) {
          console.error("[WS] Error decoding message:", err);
        }
      }
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected. Reconnecting in 3s...');
      document.getElementById('ws-status-dot')?.style.setProperty('background-color', '#ef4444');
      setTimeout(initWebSocket, 3000);
    };
  }

  selectMode?.addEventListener('change', (e) => {
    isLiveWS = e.target.value === 'live_ws';
    const modeName = isLiveWS ? "Live WebSocket Stream (Real-Time)" : "Precomputed JSON (Deterministic)";
    loader.start({
      title: "Switching Stream Mode",
      subtitle: modeName,
      isFast: true,
    });
    setTimeout(() => {
      loader.complete("Mode Active");
      loader.toast(`Stream Mode: ${isLiveWS ? 'Live WebSocket (30 FPS)' : 'Precomputed Replay'}`);
    }, 450);

    if (isLiveWS && !ws) initWebSocket();
  });

  // Initial Load with Progress Bar
  const initialScenario = selectScenario?.value || 'kitti_seq_00';
  await switchScenario(initialScenario);
  updateSequenceBadge();
  updateAllFills();
  startPlaybackLoop();
  initWebSocket();
});

