// Rendering, ground-truth evaluation and thunderSTORM-format export, ported from visualization.GaussianRenderer,
// evaluation.PerformanceEvaluator.evaluate and thunderstorm_format.to_thunderstorm_columns (tested in
// ts_engine.test.mjs). Localisations are columns { frame, x, y, ... } with x / y in nm (pixel-centre origin).
import { img } from './ts_engine.js';

// GaussianRenderer.render with image_size = (width, height): each localisation adds an unnormalised Gaussian
// (peak 1) of sigma sigmaNm (or its own uncertainty when sigmaNm === 'computed'), cut at 3 sigma.
export function renderGaussian(locs, { pixelSize = 10, sigmaNm = 20, width, height }) {
  const out = img(width, height);
  const n = locs.x.length;
  for (let i = 0; i < n; i++) {
    if (locs.x[i] === null || locs.y[i] === null) continue;
    const x = locs.x[i] / pixelSize, y = locs.y[i] / pixelSize;
    const s = (sigmaNm === 'computed' ? locs.uncertainty[i] : sigmaNm) / pixelSize;
    if (!Number.isFinite(s) || s <= 0) continue; // e.g. failed fits (uncertainty inf) in 'computed' mode
    const r = Math.trunc(3 * s);
    const x0 = Math.trunc(x) - r, x1 = Math.trunc(x) + r + 1, y0 = Math.trunc(y) - r, y1 = Math.trunc(y) + r + 1;
    const xa = Math.max(0, x0), xb = Math.min(width, x1), ya = Math.max(0, y0), yb = Math.min(height, y1);
    if (xa >= xb || ya >= yb) continue;
    const k = 1 / (2 * s * s);
    for (let yy = ya; yy < yb; yy++) for (let xx = xa; xx < xb; xx++) {
      out.d[yy * width + xx] += Math.exp(-((xx - x) ** 2 + (yy - y) ** 2) * k);
    }
  }
  return out;
}

// PerformanceEvaluator.evaluate for one frame (nearest-neighbour matching within tolerance, not one-to-one).
export function evaluateFrame(det, gt, tolerance = 100) {
  const nd = det.x.length, ng = gt.x.length;
  if (nd === 0) return { n_true_positive: 0, n_false_positive: 0, n_false_negative: ng, recall: 0, precision: 0,
    f1_score: 0, rmse_x: null, rmse_y: null, jaccard: 0 };
  const nearestOf = (ax, ay, bx, by) => {
    let best = Infinity, bi = -1;
    for (let j = 0; j < bx.length; j++) { const d = Math.hypot(ax - bx[j], ay - by[j]); if (d < best) { best = d; bi = j; } }
    return [best, bi];
  };
  let tp = 0, matchedDet = 0, ex = 0, ey = 0;
  for (let i = 0; i < ng; i++) {
    const [d, j] = nearestOf(gt.x[i], gt.y[i], det.x, det.y);
    if (d <= tolerance) { tp++; ex += (det.x[j] - gt.x[i]) ** 2; ey += (det.y[j] - gt.y[i]) ** 2; }
  }
  for (let j = 0; j < nd; j++) if (nearestOf(det.x[j], det.y[j], gt.x, gt.y)[0] <= tolerance) matchedDet++;
  const fn = ng - tp, fp = nd - matchedDet, recall = ng ? tp / ng : 0, precision = tp / nd;
  return { n_true_positive: tp, n_false_positive: fp, n_false_negative: fn, recall, precision,
    f1_score: precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0,
    rmse_x: tp ? Math.sqrt(ex / tp) : null, rmse_y: tp ? Math.sqrt(ey / tp) : null,
    rmse: tp ? Math.sqrt((ex + ey) / tp) : null, jaccard: tp / (tp + fp + fn) };
}

// All frames of a movie against its ground truth (columns frame/x/y), summed as web/demo.py evaluate() does.
export function evaluateMovie(locs, truth, nFrames, tolerance = 100) {
  const byFrame = cols => { const m = new Map(); cols.frame.forEach((f, i) => {
    if (cols.x[i] === null) return; if (!m.has(f)) m.set(f, { x: [], y: [] }); m.get(f).x.push(cols.x[i]); m.get(f).y.push(cols.y[i]); }); return m; };
  const D = byFrame(locs), G = byFrame(truth);
  let tp = 0, fp = 0, fn = 0, nDet = 0, nGt = 0, se = 0;
  for (let f = 0; f < nFrames; f++) {
    const d = D.get(f) || { x: [], y: [] }, g = G.get(f);
    nDet += d.x.length;
    if (!g) { fp += d.x.length; continue; }
    nGt += g.x.length;
    const m = evaluateFrame(d, g, tolerance);
    tp += m.n_true_positive; fp += m.n_false_positive; fn += m.n_false_negative;
    if (m.n_true_positive) se += m.rmse ** 2 * m.n_true_positive;
  }
  return { tp, fp, fn, n_det: nDet, n_gt: nGt, recall: nGt ? tp / nGt : 0, precision: nDet ? tp / nDet : 0,
    rmse_nm: tp ? Math.sqrt(se / tp) : null, tolerance_nm: tolerance };
}

const PHOTON_FITTERS = ['gaussian_mle'];
const allclose = (a, b) => a.every((v, i) => (v === null && b[i] === null) || (v !== null && b[i] !== null && Math.abs(v - b[i]) <= 1e-8 + 1e-5 * Math.abs(b[i])));

// thunderstorm_format.to_thunderstorm_columns: 1-based frame, corner-origin nm, sigma in nm, photons.
export function toThunderstormColumns(locs, { pixelSize, photonsPerAdu = 1, baseline = 0, fitterType = 'gaussian_lsq' }) {
  const n = locs.x.length, half = pixelSize / 2, ph = !PHOTON_FITTERS.includes(fitterType);
  const nn = (v, f) => (v === null || v === undefined ? null : f(v));
  const sx = locs.sigma_x, sy = locs.sigma_y;
  const cols = { id: Array.from({ length: n }, (_, i) => i + 1), frame: locs.frame.map(f => f + 1),
    'x [nm]': locs.x.map(v => nn(v, a => a + half)), 'y [nm]': locs.y.map(v => nn(v, a => a + half)) };
  if (allclose(sx, sy)) cols['sigma [nm]'] = sx.map(v => nn(v, a => a * pixelSize));
  else { cols['sigma1 [nm]'] = sx.map(v => nn(v, a => a * pixelSize)); cols['sigma2 [nm]'] = sy.map(v => nn(v, a => a * pixelSize)); }
  cols['intensity [photon]'] = locs.intensity.map(v => nn(v, a => (ph ? a * photonsPerAdu : a)));
  cols['offset [photon]'] = locs.background.map(v => nn(v, a => (ph ? (a - baseline) * photonsPerAdu : a)));
  cols['uncertainty [nm]'] = locs.uncertainty.map(v => v);
  return cols;
}

// CSV text with a quoted header row as thunderSTORM writes it; missing / infinite values are written as inf.
export function toCsv(cols) {
  const keys = Object.keys(cols), n = cols[keys[0]].length, lines = [keys.map(k => `"${k}"`).join(',')];
  for (let i = 0; i < n; i++) lines.push(keys.map(k => (cols[k][i] === null ? 'inf' : String(cols[k][i]))).join(','));
  return lines.join('\n') + '\n';
}

// 'hot'-like colour map with a percentile clip, written into ImageData.
export function toImageData(im, ctx, clipPct = 99.5, gamma = 1) {
  const vals = Float64Array.from(im.d).sort(), hi = vals[Math.min(vals.length - 1, Math.floor(vals.length * clipPct / 100))] || 1;
  const lo = vals[0] < 0 ? vals[Math.floor(vals.length * 0.01)] : 0;
  const out = ctx.createImageData(im.w, im.h);
  for (let i = 0; i < im.d.length; i++) {
    const t = Math.max(0, Math.min(1, (im.d[i] - lo) / (hi - lo || 1))) ** gamma;
    out.data[i * 4] = 255 * Math.min(1, t * 2.6); out.data[i * 4 + 1] = 255 * Math.min(1, Math.max(0, t * 2.6 - 1));
    out.data[i * 4 + 2] = 255 * Math.max(0, t * 3 - 2); out.data[i * 4 + 3] = 255;
  }
  return out;
}
