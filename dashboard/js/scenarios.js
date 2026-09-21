/**
 * Scenario Management & Data Fetching for FoveaMap Dashboard
 */

class ScenarioManager {
  constructor() {
    this.currentScenarioId = 'kitti_odometry';
    this.frames = [];
    this.currentFrameIdx = 0;
  }

  async loadScenario(scenarioId, onProgress = null) {
    this.currentScenarioId = scenarioId;
    this.currentFrameIdx = 0;

    if (onProgress) onProgress(8, "Connecting to perception API server...");

    try {
      const response = await fetch(`/api/frames/${scenarioId}`);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const contentLength = response.headers.get('content-length');
      const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;

      if (response.body && totalBytes > 0) {
        const reader = response.body.getReader();
        let receivedBytes = 0;
        const chunks = [];
        const decoder = new TextDecoder("utf-8");

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          receivedBytes += value.length;
          const pct = Math.min(88, 10 + Math.round((receivedBytes / totalBytes) * 75));
          const mb = (receivedBytes / (1024 * 1024)).toFixed(1);
          const totalMb = (totalBytes / (1024 * 1024)).toFixed(1);
          if (onProgress) onProgress(pct, `Streaming LiDAR data (${mb} / ${totalMb} MB)...`);
        }

        if (onProgress) onProgress(90, "Reconstructing 2.5D polar grid structures...");
        const allChunks = new Uint8Array(receivedBytes);
        let position = 0;
        for (const chunk of chunks) {
          allChunks.set(chunk, position);
          position += chunk.length;
        }
        const jsonText = decoder.decode(allChunks);
        if (onProgress) onProgress(95, "Parsing frame sequences and active tracks...");
        this.frames = JSON.parse(jsonText);
      } else {
        if (onProgress) onProgress(50, "Receiving 2.5D polar grid payload...");
        this.frames = await response.json();
      }

      if (onProgress) onProgress(100, `Ready (${this.frames.length} frames initialized)`);
      return this.frames;
    } catch (err) {
      console.warn(`[ScenarioManager] Could not fetch from server (${err}). Using fallback procedural data.`);
      return [];
    }
  }

  getCurrentFrame() {
    if (!this.frames || this.frames.length === 0) return null;
    return this.frames[this.currentFrameIdx];
  }

  nextFrame() {
    if (!this.frames || this.frames.length === 0) return null;
    this.currentFrameIdx = (this.currentFrameIdx + 1) % this.frames.length;
    return this.frames[this.currentFrameIdx];
  }

  setFrame(idx) {
    if (!this.frames || this.frames.length === 0) return null;
    this.currentFrameIdx = Math.max(0, Math.min(idx, this.frames.length - 1));
    return this.frames[this.currentFrameIdx];
  }
}

window.ScenarioManager = ScenarioManager;
