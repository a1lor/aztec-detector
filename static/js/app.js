/* ============================================================
   AztecVision — Frontend Application
   ============================================================ */

const API_BASE = window.location.origin;

// Colour palette for bounding boxes (cycling)
const PALETTE = [
  '#e8b84b', '#e8624b', '#4be8b8', '#b44be8',
  '#4b8be8', '#e84b8b', '#8be84b', '#f0d840',
  '#40d0f0', '#f04880', '#80f048', '#d060ff',
  '#ff8040', '#40e880', '#4080ff', '#ff4060',
  '#ffa040', '#40ffc0', '#c040ff', '#ff6060',
];

// ============================================================
// State
// ============================================================

let currentImage      = null;   // HTMLImageElement
let currentDetections = [];     // API detections array
let currentResponse   = null;   // full API response
let currentLayout     = null;   // { scale, offsetX, offsetY, dispW, dispH, imgW, imgH }
let offscreenCanvas   = null;   // cached image-only render
let canvasHoverId     = -1;     // det.id under mouse
let listHoverId       = -1;     // det.id highlighted from list
let errorTimeout      = null;

// ============================================================
// DOM references
// ============================================================

const uploadView     = document.getElementById('upload-view');
const analysisView   = document.getElementById('analysis-view');
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const mainCanvas     = document.getElementById('main-canvas');
const ctx            = mainCanvas.getContext('2d');
const zoomPanel      = document.getElementById('zoom-panel');
const zoomCanvas     = document.getElementById('zoom-canvas');
const zoomCtx        = zoomCanvas.getContext('2d');
const loadingOverlay = document.getElementById('loading-overlay');
const statusDot      = document.getElementById('status-dot');
const statusText     = document.getElementById('status-text');
const confSlider     = document.getElementById('conf-slider');
const confValueEl    = document.getElementById('conf-value');
const detectionList  = document.getElementById('detection-list');
const statDetections = document.getElementById('stat-detections');
const statTime       = document.getElementById('stat-time');
const detCountBadge  = document.getElementById('det-count-badge');
const errorToast     = document.getElementById('error-toast');
const errorMsg       = document.getElementById('error-msg');

// ============================================================
// Health Check
// ============================================================

async function checkHealth() {
  setStatus('loading', 'Connexion…');
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();

    if (data.model_loaded) {
      setStatus('online', 'Modèle prêt');
    } else {
      setStatus('loading', 'Chargement modèle…');
      setTimeout(checkHealth, 3000);
    }
  } catch {
    setStatus('offline', 'Serveur hors ligne');
    setTimeout(checkHealth, 6000);
  }
}

function setStatus(cls, text) {
  statusDot.className = 'status-dot ' + cls;
  statusText.textContent = text;
}

// ============================================================
// Confidence Slider
// ============================================================

confSlider.addEventListener('input', () => {
  confValueEl.textContent = confSlider.value + '%';
});

// ============================================================
// Drag & Drop + File Input
// ============================================================

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', e => {
  if (!dropZone.contains(e.relatedTarget)) {
    dropZone.classList.remove('drag-over');
  }
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

dropZone.addEventListener('click', e => {
  if (e.target.id === 'browse-btn' || e.target.closest('#browse-btn')) return;
  fileInput.click();
});

document.getElementById('browse-btn').addEventListener('click', e => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) handleFile(file);
  fileInput.value = '';
});

// ============================================================
// Main File Handler
// ============================================================

async function handleFile(file) {
  const ALLOWED_MIME = ['image/png','image/jpeg','image/bmp','image/tiff','image/webp','image/gif'];
  const ALLOWED_EXT  = /\.(png|jpg|jpeg|bmp|tif|tiff|webp)$/i;

  if (!ALLOWED_MIME.includes(file.type) && !ALLOWED_EXT.test(file.name)) {
    showError('Format non supporté. Utilisez PNG, JPG, JPEG, BMP, TIFF ou WEBP.');
    return;
  }

  document.getElementById('loading-filename').textContent = file.name;
  loadingOverlay.classList.remove('hidden');

  try {
    // Load image locally (for canvas & zoom)
    const img = await loadImage(file);
    currentImage = img;

    // Call predict API
    const conf   = confSlider.value / 100;
    const tiling = document.getElementById('tiling-toggle').checked;
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/predict?conf=${conf}&iou=0.7&tiling=${tiling}`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Erreur inconnue' }));
      throw new Error(err.detail || `Erreur API (${res.status})`);
    }

    const data = await res.json();
    currentResponse   = data;
    currentDetections = data.detections;

    loadingOverlay.classList.add('hidden');
    showAnalysis(img, data);

  } catch (e) {
    loadingOverlay.classList.add('hidden');
    showError(e.message || "Erreur lors de l'analyse");
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload  = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Impossible de charger l'image")); };
    img.src = url;
  });
}

// ============================================================
// Analysis View
// ============================================================

function showAnalysis(img, data) {
  uploadView.classList.remove('active');
  analysisView.classList.add('active');

  statDetections.textContent = data.num_detections;
  statTime.textContent       = data.inference_time_ms;
  detCountBadge.textContent  = data.num_detections;

  // Tiling badge
  const badge = document.getElementById('tiling-badge');
  if (data.tiling_used) {
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }

  resetHoverState();
  setupCanvas(img);
  buildDetectionList(data.detections);
}

function resetToUpload() {
  analysisView.classList.remove('active');
  uploadView.classList.add('active');

  currentImage      = null;
  currentDetections = [];
  currentResponse   = null;
  currentLayout     = null;
  offscreenCanvas   = null;
  canvasHoverId     = -1;
  listHoverId       = -1;

  zoomPanel.classList.add('hidden');
  detectionList.innerHTML = '';
}

document.getElementById('new-btn').addEventListener('click', resetToUpload);

// ============================================================
// Canvas Setup & Rendering
// ============================================================

function setupCanvas(img) {
  const wrapper = document.getElementById('canvas-wrapper');
  const W = wrapper.clientWidth;
  const H = wrapper.clientHeight;

  const imgW = img.naturalWidth;
  const imgH = img.naturalHeight;

  const scale   = Math.min(W / imgW, H / imgH);
  const dispW   = Math.round(imgW * scale);
  const dispH   = Math.round(imgH * scale);
  const offsetX = Math.round((W - dispW) / 2);
  const offsetY = Math.round((H - dispH) / 2);

  currentLayout = { scale, offsetX, offsetY, dispW, dispH, imgW, imgH };

  mainCanvas.width  = W;
  mainCanvas.height = H;

  // Build offscreen cache (image layer only)
  offscreenCanvas        = document.createElement('canvas');
  offscreenCanvas.width  = W;
  offscreenCanvas.height = H;
  const offCtx = offscreenCanvas.getContext('2d');
  offCtx.fillStyle = '#050408';
  offCtx.fillRect(0, 0, W, H);
  offCtx.drawImage(img, offsetX, offsetY, dispW, dispH);

  renderCanvas();
}

function renderCanvas() {
  if (!offscreenCanvas) return;

  // Restore cached image
  ctx.drawImage(offscreenCanvas, 0, 0);

  const l = currentLayout;
  const activeId = canvasHoverId !== -1 ? canvasHoverId : listHoverId;
  const hasActive = activeId !== -1;

  currentDetections.forEach((det, i) => {
    const color    = PALETTE[i % PALETTE.length];
    const isActive = det.id === activeId;
    const { x1, y1, x2, y2 } = det.bbox;

    const sx1 = x1 * l.scale + l.offsetX;
    const sy1 = y1 * l.scale + l.offsetY;
    const sw  = (x2 - x1) * l.scale;
    const sh  = (y2 - y1) * l.scale;

    // Dim inactive when something is hovered
    ctx.globalAlpha = hasActive && !isActive ? 0.25 : 1;

    // Glow on active
    ctx.shadowBlur  = isActive ? 10 : 0;
    ctx.shadowColor = color;

    // Bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth   = isActive ? 2.5 : 1.5;
    ctx.strokeRect(sx1, sy1, sw, sh);

    // Fill tint on active
    if (isActive) {
      ctx.fillStyle = color + '1a';
      ctx.fillRect(sx1, sy1, sw, sh);
    }

    ctx.shadowBlur = 0;

    // Label
    const label    = det.class_name;
    const fontSize = Math.max(10, Math.min(12, sw * 0.07));
    ctx.font       = `500 ${fontSize}px Inter, system-ui, sans-serif`;
    const tw       = ctx.measureText(label).width;
    const lx       = sx1;
    const ly       = sy1 - 5;

    if (ly > fontSize + 2) {
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly - fontSize - 1, tw + 8, fontSize + 5);
      ctx.fillStyle = '#000';
      ctx.globalAlpha = hasActive && !isActive ? 0.25 : 1;
      ctx.fillText(label, lx + 4, ly + 1);
    }
  });

  ctx.globalAlpha = 1;
}

// ============================================================
// Canvas Mouse Events
// ============================================================

mainCanvas.addEventListener('mousemove', onCanvasMove);
mainCanvas.addEventListener('mouseleave', onCanvasLeave);

function onCanvasMove(e) {
  const rect = mainCanvas.getBoundingClientRect();
  const mx   = (e.clientX - rect.left) * (mainCanvas.width  / rect.width);
  const my   = (e.clientY - rect.top)  * (mainCanvas.height / rect.height);

  const det = getDetectionAt(mx, my);

  if (det) {
    if (det.id !== canvasHoverId) {
      canvasHoverId = det.id;
      renderCanvas();
      mainCanvas.style.cursor = 'pointer';
    }
    showZoomPanel(det, e.clientX, e.clientY);
    highlightListItem(det.id);
  } else {
    if (canvasHoverId !== -1) {
      canvasHoverId = -1;
      renderCanvas();
      mainCanvas.style.cursor = 'crosshair';
    }
    zoomPanel.classList.add('hidden');
    highlightListItem(-1);
  }
}

function onCanvasLeave() {
  if (canvasHoverId !== -1) {
    canvasHoverId = -1;
    renderCanvas();
    mainCanvas.style.cursor = 'crosshair';
  }
  zoomPanel.classList.add('hidden');
  highlightListItem(-1);
}

function getDetectionAt(mx, my) {
  const l = currentLayout;
  const imgX = (mx - l.offsetX) / l.scale;
  const imgY = (my - l.offsetY) / l.scale;

  if (imgX < 0 || imgX > l.imgW || imgY < 0 || imgY > l.imgH) return null;

  // Return last matched (topmost visually)
  let found = null;
  for (const det of currentDetections) {
    const { x1, y1, x2, y2 } = det.bbox;
    if (imgX >= x1 && imgX <= x2 && imgY >= y1 && imgY <= y2) {
      found = det;
    }
  }
  return found;
}

function resetHoverState() {
  canvasHoverId = -1;
  listHoverId   = -1;
}

// ============================================================
// Zoom & Enhance Panel
// ============================================================

function showZoomPanel(det, clientX, clientY) {
  const wrapper     = document.getElementById('canvas-wrapper');
  const wrapperRect = wrapper.getBoundingClientRect();

  const { x1, y1, x2, y2 } = det.bbox;
  const srcW = Math.max(x2 - x1, 1);
  const srcH = Math.max(y2 - y1, 1);

  // Compute zoom canvas size (max 220px on longest side)
  const MAX = 220;
  const ratio = srcW / srcH;
  let zW, zH;
  if (ratio >= 1) {
    zW = MAX;
    zH = Math.round(MAX / ratio);
  } else {
    zH = MAX;
    zW = Math.round(MAX * ratio);
  }
  zW = Math.max(zW, 60);
  zH = Math.max(zH, 60);

  zoomCanvas.width  = zW;
  zoomCanvas.height = zH;

  // Draw magnified region (pixelated for glyph detail)
  zoomCtx.imageSmoothingEnabled = false;
  zoomCtx.drawImage(currentImage, x1, y1, srcW, srcH, 0, 0, zW, zH);

  // Update info
  document.getElementById('zoom-class-name').textContent = det.class_name;
  const confPct = Math.round(det.confidence * 100);
  document.getElementById('zoom-conf-fill').style.width  = confPct + '%';
  document.getElementById('zoom-conf-value').textContent = confPct + '%';
  document.getElementById('zoom-bbox-info').innerHTML =
    `Position : ${Math.round(x1)}, ${Math.round(y1)} → ${Math.round(x2)}, ${Math.round(y2)}<br>` +
    `Taille : ${Math.round(srcW)} × ${Math.round(srcH)} px`;

  // Position tooltip (avoid viewport overflow)
  const localX  = clientX - wrapperRect.left;
  const localY  = clientY - wrapperRect.top;
  const panelW  = zW + 24;
  const panelH  = zH + 115; // canvas + header + info approx

  let pLeft = localX + 18;
  let pTop  = localY - panelH / 2;

  if (pLeft + panelW > wrapperRect.width  - 8) pLeft = localX - panelW - 18;
  if (pLeft < 8) pLeft = 8;
  if (pTop  < 8) pTop  = 8;
  if (pTop + panelH > wrapperRect.height - 8) pTop = wrapperRect.height - panelH - 8;

  zoomPanel.style.left = pLeft + 'px';
  zoomPanel.style.top  = pTop  + 'px';
  zoomPanel.classList.remove('hidden');
}

// ============================================================
// Detection List
// ============================================================

function buildDetectionList(detections) {
  detectionList.innerHTML = '';

  if (!detections.length) {
    detectionList.innerHTML =
      '<div class="det-empty">Aucune détection.<br>Essayez de réduire le seuil de confiance.</div>';
    return;
  }

  const fragment = document.createDocumentFragment();

  detections.forEach((det, i) => {
    const color   = PALETTE[i % PALETTE.length];
    const confPct = Math.round(det.confidence * 100);

    const item = document.createElement('div');
    item.className  = 'det-item';
    item.dataset.id = det.id;
    item.innerHTML  = `
      <div class="det-color-dot" style="background:${color}"></div>
      <div class="det-info">
        <div class="det-name" title="${det.class_name}">${det.class_name}</div>
        <div class="det-bar-row">
          <div class="det-bar">
            <div class="det-bar-fill" style="width:${confPct}%;background:${color}"></div>
          </div>
          <span class="det-conf">${confPct}%</span>
        </div>
      </div>
    `;

    item.addEventListener('mouseenter', () => {
      listHoverId = det.id;
      renderCanvas();
    });

    item.addEventListener('mouseleave', () => {
      listHoverId = -1;
      renderCanvas();
    });

    fragment.appendChild(item);
  });

  detectionList.appendChild(fragment);
}

function highlightListItem(detId) {
  detectionList.querySelectorAll('.det-item').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.id) === detId);
  });
  if (detId !== -1) {
    const el = detectionList.querySelector(`[data-id="${detId}"]`);
    if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

// ============================================================
// Export
// ============================================================

document.getElementById('export-btn').addEventListener('click', () => {
  if (!currentResponse) return;
  const json = JSON.stringify(currentResponse, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `aztecvision_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// ============================================================
// Error Toast
// ============================================================

function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.classList.remove('hidden');
  clearTimeout(errorTimeout);
  errorTimeout = setTimeout(() => errorToast.classList.add('hidden'), 7000);
}

document.getElementById('toast-close').addEventListener('click', () => {
  errorToast.classList.add('hidden');
});

// ============================================================
// Window Resize
// ============================================================

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (currentImage && analysisView.classList.contains('active')) {
      zoomPanel.classList.add('hidden');
      resetHoverState();
      setupCanvas(currentImage);
    }
  }, 150);
});

// ============================================================
// Init
// ============================================================

checkHealth();
