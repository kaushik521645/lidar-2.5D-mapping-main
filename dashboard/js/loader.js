/**
 * FoveaMap Dashboard Progress & Loading State Manager
 * 
 * Manages the global progress bar, viewport 3D loading overlay,
 * pipeline stage checklist, and real-time toast notifications.
 */

class DashboardLoadingController {
  constructor() {
    this.globalBar = document.getElementById('global-progress-bar');
    this.viewportOverlay = document.getElementById('viewport-loader-overlay');
    this.viewportStrip = document.getElementById('viewport-progress-strip');
    this.viewportFill = document.getElementById('viewport-progress-fill');
    this.viewportStatus = document.getElementById('viewport-progress-status');
    this.viewportPct = document.getElementById('viewport-progress-pct');

    this.loaderTitle = document.getElementById('loader-title');
    this.loaderSubtitle = document.getElementById('loader-subtitle');
    this.loaderStatusText = document.getElementById('loader-status-text');
    this.loaderPercentText = document.getElementById('loader-percent-text');
    this.loaderProgressFill = document.getElementById('loader-progress-fill');
    this.stageItems = document.querySelectorAll('.loader-stage-item');
    this.toastContainer = document.getElementById('hud-toast-container');
    
    // Sidebar mini progress elements
    this.sidebarProgressFill = document.getElementById('sidebar-progress-fill');
    this.sidebarBadge = document.getElementById('badge-scenario-status');

    this.currentPercent = 0;
    this.animInterval = null;
    this.stages = [
      "Reading Velodyne binary scans (.bin)",
      "Estimating RANSAC ground surface",
      "Binning multi-resolution 2.5D polar grid",
      "Updating Kalman multi-object tracker",
    ];
  }

  /**
   * Starts a loading sequence with progressive feedback.
   */
  start({ title = "Processing LiDAR Data", subtitle = "Initializing...", isFast = false } = {}) {
    if (this.animInterval) clearInterval(this.animInterval);

    this.currentPercent = 10;
    this._applyPercent(10);

    if (this.loaderTitle) this.loaderTitle.textContent = title;
    if (this.loaderSubtitle) this.loaderSubtitle.textContent = subtitle;
    if (this.loaderStatusText) this.loaderStatusText.textContent = this.stages[0];
    if (this.viewportStatus) this.viewportStatus.textContent = title.toUpperCase();

    // Show overlays and strips
    if (this.globalBar) {
      this.globalBar.classList.add('active', 'shimmer');
    }
    if (this.viewportStrip) {
      this.viewportStrip.classList.add('active');
    }
    if (this.viewportOverlay) {
      this.viewportOverlay.classList.add('active');
    }
    if (this.sidebarBadge) {
      this.sidebarBadge.className = 'badge-status loading';
      this.sidebarBadge.textContent = 'PROCESSING';
    }

    this._setStage(0);

    // Smooth simulated ticker to provide continuous visual feedback while async task resolves
    const targetCap = isFast ? 90 : 88;
    const stepDuration = isFast ? 40 : 120;

    this.animInterval = setInterval(() => {
      if (this.currentPercent < targetCap) {
        const increment = Math.max(0.6, (targetCap - this.currentPercent) * 0.09);
        this.currentPercent = Math.min(targetCap, this.currentPercent + increment);
        this._applyPercent(this.currentPercent);

        // Update stage indicators based on percent thresholds
        if (this.currentPercent >= 70) {
          this._setStage(3);
        } else if (this.currentPercent >= 45) {
          this._setStage(2);
        } else if (this.currentPercent >= 20) {
          this._setStage(1);
        }
      }
    }, stepDuration);
  }

  /**
   * Updates progress explicitly.
   */
  setProgress(percent, statusText = null, stageIdx = null) {
    this.currentPercent = Math.min(100, Math.max(0, percent));
    this._applyPercent(this.currentPercent);

    if (statusText) {
      if (this.loaderStatusText) this.loaderStatusText.textContent = statusText;
      if (this.viewportStatus) this.viewportStatus.textContent = statusText.toUpperCase();
    }
    if (stageIdx !== null) {
      this._setStage(stageIdx);
    }
  }

  /**
   * Completes the loading process and plays a smooth exit transition.
   */
  complete(successMessage = "Ready") {
    if (this.animInterval) {
      clearInterval(this.animInterval);
      this.animInterval = null;
    }

    this.currentPercent = 100;
    this._applyPercent(100);
    this._setStage(4); // All done

    if (this.loaderStatusText) {
      this.loaderStatusText.textContent = successMessage;
    }
    if (this.viewportStatus) {
      this.viewportStatus.textContent = successMessage.toUpperCase();
    }

    if (this.sidebarBadge) {
      this.sidebarBadge.className = 'badge-status ready';
      this.sidebarBadge.textContent = 'READY';
    }

    setTimeout(() => {
      if (this.viewportOverlay) {
        this.viewportOverlay.classList.remove('active');
      }
      if (this.viewportStrip) {
        this.viewportStrip.classList.remove('active');
      }
      if (this.globalBar) {
        this.globalBar.classList.remove('shimmer');
        setTimeout(() => {
          this.globalBar.classList.remove('active');
          this._applyPercent(0);
        }, 350);
      }
    }, 320);
  }

  /**
   * Handles errors gracefully with visual status feedback.
   */
  error(errorMessage = "Failed to load") {
    if (this.animInterval) {
      clearInterval(this.animInterval);
      this.animInterval = null;
    }
    if (this.sidebarBadge) {
      this.sidebarBadge.className = 'badge-status error';
      this.sidebarBadge.textContent = 'ERROR';
    }
    if (this.loaderStatusText) {
      this.loaderStatusText.textContent = errorMessage;
    }
    this.toast(`⚠️ ${errorMessage}`, 3500);
    setTimeout(() => {
      if (this.viewportOverlay) this.viewportOverlay.classList.remove('active');
      if (this.viewportStrip) this.viewportStrip.classList.remove('active');
      if (this.globalBar) this.globalBar.classList.remove('active');
    }, 1200);
  }

  _applyPercent(val) {
    const rounded = Math.round(val);
    if (this.globalBar) this.globalBar.style.width = `${rounded}%`;
    if (this.loaderProgressFill) this.loaderProgressFill.style.width = `${rounded}%`;
    if (this.sidebarProgressFill) this.sidebarProgressFill.style.width = `${rounded}%`;
    if (this.viewportFill) this.viewportFill.style.width = `${rounded}%`;
    if (this.loaderPercentText) this.loaderPercentText.textContent = `${rounded}%`;
    if (this.viewportPct) this.viewportPct.textContent = `${rounded}%`;
  }

  _setStage(activeIdx) {
    this.stageItems = document.querySelectorAll('.loader-stage-item');
    this.stageItems.forEach((item, idx) => {
      const icon = item.querySelector('.stage-icon');
      if (idx < activeIdx) {
        item.className = 'loader-stage-item done';
        if (icon) icon.textContent = '✓';
      } else if (idx === activeIdx) {
        item.className = 'loader-stage-item active';
        if (icon) icon.textContent = '⚙';
        if (this.loaderStatusText) this.loaderStatusText.textContent = this.stages[idx] || "Processing...";
      } else {
        item.className = 'loader-stage-item';
        if (icon) icon.textContent = `${idx + 1}`;
      }
    });
  }

  /**
   * Shows a floating HUD toast pill in the top-right of the 3D viewport.
   */
  toast(message, duration = 2500) {
    if (!this.toastContainer) return;

    const toast = document.createElement('div');
    toast.className = 'hud-toast';
    toast.innerHTML = `
      <div class="hud-toast-icon"></div>
      <span>${message}</span>
    `;

    this.toastContainer.appendChild(toast);

    requestAnimationFrame(() => {
      toast.classList.add('show');
    });

    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 350);
    }, duration);
  }
}

window.DashboardLoadingController = DashboardLoadingController;
