// Formatting and drawing helpers shared by the pages.
export const f = (v, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(d));
export const pct = v => (v === null || v === undefined ? '–' : (100 * v).toFixed(1) + '%');
export const int = v => (v ?? 0).toLocaleString('en');

// Grey-scale image (percentile-clipped) into a canvas of the image's own size.
export function drawGrey(canvas, data, w, h, lowPct = 0.5, highPct = 99.8) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d'), s = Float64Array.from(data).sort();
  const lo = s[Math.floor((s.length - 1) * lowPct / 100)], hi = s[Math.floor((s.length - 1) * highPct / 100)];
  const im = ctx.createImageData(w, h);
  for (let i = 0; i < data.length; i++) {
    const v = 255 * Math.max(0, Math.min(1, (data[i] - lo) / ((hi - lo) || 1)));
    im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v; im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  return ctx;
}

// A scale bar (absolute-positioned in a .stage) for a field of view fovNm wide: a round length near a fifth of it.
export function scaleBar(fovNm) {
  const target = fovNm / 5, steps = [50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000];
  const len = steps.reduce((b, s) => (Math.abs(s - target) < Math.abs(b - target) ? s : b), steps[0]);
  const label = len >= 1000 ? `${len / 1000} µm` : `${len} nm`;
  return `<div class="scalebar" style="width:${(100 * len / fovNm).toFixed(2)}%" aria-label="scale bar ${label}"><i></i>${label}</div>`;
}

// Upscale a small canvas onto a display canvas (nearest neighbour) and draw circles at (x, y) camera-pixel centres.
export function drawMarkers(display, src, scale, marks) {
  display.width = src.width * scale; display.height = src.height * scale;
  const g = display.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.drawImage(src, 0, 0, display.width, display.height);
  for (const m of marks) {
    g.strokeStyle = m.c; g.lineWidth = m.lw || 1.6;
    g.beginPath(); g.arc((m.x + 0.5) * scale, (m.y + 0.5) * scale, m.r * scale, 0, 2 * Math.PI); g.stroke();
  }
}
