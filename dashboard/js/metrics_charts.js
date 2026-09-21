/**
 * HUD Telemetry & Metrics Charts Controller for FoveaMap Dashboard
 */

class MetricsHUDController {
  constructor() {
    this.fpsElem = document.getElementById('val-fps');
    this.latencyElem = document.getElementById('val-latency');
    this.memSavingsElem = document.getElementById('val-mem-savings');
    this.memFoveaElem = document.getElementById('val-mem-fovea');
    this.memUniformElem = document.getElementById('val-mem-uniform');
    this.memBarFillElem = document.getElementById('mem-bar-fill');
    this.activeCellsElem = document.getElementById('val-active-cells');
    this.ghostingErasedElem = document.getElementById('val-ghosting-erased');
    this.tracksTableBody = document.getElementById('tracks-table-body');
    this.totalGhostingErased = 0;
  }

  updateMetrics(frameData) {
    if (!frameData || !frameData.metrics) return;

    const m = frameData.metrics;

    // Latency & FPS
    if (this.fpsElem) this.fpsElem.textContent = (m.fps || 0).toFixed(1);
    if (this.latencyElem) this.latencyElem.textContent = (m.latency_ms || 0).toFixed(1);

    // Memory Savings
    const savingsPct = m.memory_savings_pct || 0;
    if (this.memSavingsElem) this.memSavingsElem.textContent = `${savingsPct.toFixed(1)}%`;

    const foveaKB = (m.memory_bytes_foveated / 1024.0).toFixed(1);
    const uniformKB = (m.memory_bytes_uniform_baseline / 1024.0).toFixed(1);

    if (this.memFoveaElem) this.memFoveaElem.textContent = `${foveaKB} KB`;
    if (this.memUniformElem) this.memUniformElem.textContent = `${uniformKB} KB`;

    if (this.memBarFillElem) {
      this.memBarFillElem.style.width = `${Math.min(100, Math.max(5, savingsPct))}%`;
    }

    if (this.activeCellsElem) {
      this.activeCellsElem.textContent = m.active_cells_count || 0;
    }

    // Ghosting Counter
    if (m.ghosting_cells_erased) {
      this.totalGhostingErased += m.ghosting_cells_erased;
    }
    if (this.ghostingErasedElem) {
      this.ghostingErasedElem.textContent = this.totalGhostingErased;
    }

    // Active Tracks Table
    if (this.tracksTableBody) {
      const tracks = frameData.tracks || [];
      if (tracks.length === 0) {
        this.tracksTableBody.innerHTML = '<tr><td colspan="4" style="color:var(--text-muted);text-align:center;">No active tracks</td></tr>';
      } else {
        this.tracksTableBody.innerHTML = tracks
          .map((t) => {
            const [x, y] = t.position_xy;
            const [vx, vy] = t.velocity_xy;
            const speed = Math.hypot(vx, vy).toFixed(1);
            return `
              <tr>
                <td>#${t.track_id}</td>
                <td>(${x.toFixed(1)}, ${y.toFixed(1)})</td>
                <td>${speed} m/s</td>
                <td><span style="color:#3b82f6;">Dynamic</span></td>
              </tr>
            `;
          })
          .join('');
      }
    }
  }
}

window.MetricsHUDController = MetricsHUDController;
