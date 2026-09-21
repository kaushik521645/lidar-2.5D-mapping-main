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
  };

  // DOM Elements
  const selectScenario = document.getElementById('select-scenario');
  const selectMode = document.getElementById('select-mode');
  const btnPlay = document.getElementById('btn-play');
  const btnStep = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');
  const sliderScrub = document.getElementById('slider-scrub');
  const lblFrame = document.getElementById('lbl-frame-counter');
  const lblScenarioDesc = document.getElementById('lbl-scenario-desc');

  // Sliders
  const sliderSpeed = document.getElementById('slider-speed');
  const valSpeed = document.getElementById('val-speed');
  const sliderSteering = document.getElementById('slider-steering');
  const valSteering = document.getElementById('val-steering');
  const sliderMaxStretch = document.getElementById('slider-max-stretch');
  const valMaxStretch = document.getElementById('val-max-stretch');
  const sliderBaseRadius = document.getElementById('slider-base-radius');
  const valBaseRadius = document.getElementById('val-base-radius');

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
    [sliderSpeed, sliderSteering, sliderMaxStretch, sliderBaseRadius, sliderScrub].forEach(updateSliderFill);
  }

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

    // Immediately update 3D contour overlay
    renderer.polarOverlay.updateFoveationContour(
      overrideParams.speed_mps,
      overrideParams.steering_angle_rad,
      overrideParams.base_fine_radius_m,
      overrideParams.max_stretch,
      []
    );

    // If WebSocket is active, send live params
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        command: 'set_params',
        speed_mps: overrideParams.speed_mps,
        steering_angle_rad: overrideParams.steering_angle_rad,
      }));
    }
    updateAllFills();

    // Debounced toast feedback for slider adjustments
    clearTimeout(sliderDebounce);
    sliderDebounce = setTimeout(() => {
      loader.toast(`Foveation: Speed ${overrideParams.speed_mps.toFixed(1)} m/s | Steer ${steerDeg}°`);
    }, 600);
  }

  [sliderSpeed, sliderSteering, sliderMaxStretch, sliderBaseRadius].forEach((s) => {
    s?.addEventListener('input', onSliderChange);
  });

  // Scenario / Dataset Sequence Change Handler with Full Progress Bar
  async function switchScenario(scenarioId) {
    const selectedOption = selectScenario?.options[selectScenario.selectedIndex];
    const sceneName = selectedOption ? selectedOption.textContent.trim() : scenarioId;

    // Start prominent progress indicator & modal
    loader.start({
      title: "Loading Dataset Sequence",
      subtitle: sceneName,
      isFast: false,
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
      }

      const currentFrame = scenarioMgr.getCurrentFrame();
      if (currentFrame) {
        if (lblScenarioDesc) {
          lblScenarioDesc.textContent = currentFrame.description || `Loaded ${frames.length} frames for ${sceneName}.`;
        }
        renderCurrentFrame(currentFrame);
      }

      loader.complete(`Ready (${frames.length} frames)`);
      loader.toast(`Loaded: ${sceneName} (${frames.length} frames)`);
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

  btnStep?.addEventListener('click', () => {
    isPlaying = false;
    if (btnPlay) btnPlay.textContent = '▶ Play';
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

    // Apply live foveation slider overrides if adjusted by user
    frameData.foveation_params = {
      base_fine_radius_m: overrideParams.base_fine_radius_m,
      max_stretch: overrideParams.max_stretch,
    };

    // Update 3D viewport
    renderer.renderFrame(frameData);

    // Update Telemetry HUD
    metricsHUD.updateMetrics(frameData);

    // Update Scrub bar and label
    if (sliderScrub && !isLiveWS) {
      sliderScrub.value = scenarioMgr.currentFrameIdx;
    }
    if (lblFrame) {
      const total = scenarioMgr.frames.length || 30;
      lblFrame.textContent = `Frame ${scenarioMgr.currentFrameIdx + 1} / ${total}`;
    }
  }

  // Precomputed Playback Loop (10 Hz)
  function startPlaybackLoop() {
    if (playbackInterval) clearInterval(playbackInterval);
    playbackInterval = setInterval(() => {
      if (isPlaying && !isLiveWS) {
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
  updateAllFills();
  startPlaybackLoop();
  initWebSocket();
});
