// In-browser port of the ThunderSTORM Python pipeline (filters.py, detection.py, fitting.py, pipeline.py).
// Pure functions, no DOM. Images are { w, h, d: Float64Array } in row-major order (d[row * w + col]).
// Units follow the Python pipeline: x / y / uncertainty in nm (pixel-centre origin), sigma in camera pixels.
// ts_engine.test.mjs checks every step against values computed by the Python code (tests/fixtures/engine_cases.json).

export const img = (w, h, d = new Float64Array(w * h)) => ({ w, h, d });

// ---- boundary modes (scipy.ndimage) -----------------------------------------------------------------------------
const nearest = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);
function reflect(i, n) { // 'reflect': d c b a | a b c d | d c b a
  const p = 2 * n, m = ((i % p) + p) % p;
  return m < n ? m : p - 1 - m;
}

// ndimage.correlate1d with an odd-length kernel centred on the pixel (origin 0).
export function correlate1d(im, k, axis, mode = nearest) {
  const { w, h, d } = im, out = new Float64Array(w * h), r = (k.length - 1) / 2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < k.length; j++) {
      if (k[j] === 0) continue;
      s += k[j] * (axis === 1 ? d[y * w + mode(x + j - r, w)] : d[mode(y + j - r, h) * w + x]);
    }
    out[y * w + x] = s;
  }
  return img(w, h, out);
}

// ---- filters.py ---------------------------------------------------------------------------------------------------
// WaveletFilter.apply: a trous B-spline wavelet, edges padded by repeating the border pixel; returns the last level.
export function wavelet(im, scale = 2, order = 3) {
  const h = order === 1 ? [1 / 4, 1 / 2, 1 / 4] : [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16];
  if (!(scale >= 1)) throw new Error('wavelet scale must be at least 1');
  let approx = im, wave = null;
  for (let j = 0; j < scale; j++) {
    const step = 2 ** j, k = new Array(h.length + (h.length - 1) * (step - 1)).fill(0);
    h.forEach((v, i) => { k[i * step] = v; });
    const next = correlate1d(correlate1d(approx, k, 1), k, 0);
    wave = img(im.w, im.h, approx.d.map((v, i) => v - next.d[i]));
    approx = next;
  }
  return wave;
}

// ndimage.gaussian_filter (truncate 4.0, mode 'reflect'), axis 0 then axis 1.
export function gaussian(im, sigma) {
  if (!(sigma > 0)) return img(im.w, im.h, Float64Array.from(im.d));
  const r = Math.trunc(4 * sigma + 0.5), k = [];
  for (let i = -r; i <= r; i++) k.push(Math.exp(-0.5 * i * i / (sigma * sigma)));
  const s = k.reduce((a, b) => a + b, 0);
  const kn = k.map(v => v / s);
  return correlate1d(correlate1d(im, kn, 0, reflect), kn, 1, reflect);
}

export const FILTERS = {
  wavelet: (im, p) => wavelet(im, p.scale ?? 2, p.order ?? 3),
  gaussian: (im, p) => gaussian(im, p.sigma ?? 1.6),
  dog: (im, p) => { const a = gaussian(im, p.sigma1 ?? 1.0), b = gaussian(im, p.sigma2 ?? 1.6);
    return img(im.w, im.h, a.d.map((v, i) => v - b.d[i])); },
  none: im => img(im.w, im.h, Float64Array.from(im.d)),
};

// ---- threshold expressions (filters.compute_threshold_expression) ---------------------------------------------------
const sorted = a => Float64Array.from(a).sort();
const FUNCS = {
  mean: a => a.reduce((s, v) => s + v, 0) / a.length,
  std: a => { const m = FUNCS.mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); },
  median: a => { const s = sorted(a), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; },
  max: a => a.reduce((m, v) => (v > m ? v : m), -Infinity),
  min: a => a.reduce((m, v) => (v < m ? v : m), Infinity),
};

function tokenize(s) {
  const out = [], re = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?)|(\*\*|[-+*/()]))/y;
  let m;
  re.lastIndex = 0;
  while (re.lastIndex < s.length) {
    if (/^\s*$/.test(s.slice(re.lastIndex))) break;
    const at = re.lastIndex;
    if (!(m = re.exec(s))) throw new Error(`unexpected '${s.slice(at).trim()[0]}'`);
    out.push(m[1] ? { t: 'num', v: +m[1] } : m[2] ? { t: 'id', v: m[2] } : { t: 'op', v: m[3] });
  }
  return out;
}

const binop = (a, b, f) => {
  if (typeof a === 'number' && typeof b === 'number') return f(a, b);
  const n = typeof a === 'number' ? b.length : a.length, o = new Float64Array(n);
  for (let i = 0; i < n; i++) o[i] = f(typeof a === 'number' ? a : a[i], typeof b === 'number' ? b : b[i]);
  return o;
};

// Evaluate a threshold expression. vars(name) returns a Float64Array (I1, F1, Wave.F1, Wave.F2) or undefined.
export function evalExpression(expr, vars) {
  if (typeof expr === 'number') return expr;
  const tk = tokenize(String(expr));
  let i = 0;
  const peek = () => tk[i], eat = v => { if (!tk[i] || tk[i].v !== v) throw new Error(`expected '${v}'`); i++; };
  const atom = () => {
    const t = tk[i++];
    if (!t) throw new Error('unexpected end');
    if (t.t === 'num') return t.v;
    if (t.t === 'op' && t.v === '(') { const v = sum(); eat(')'); return v; }
    if (t.t === 'op' && (t.v === '-' || t.v === '+')) { const v = power(); return t.v === '-' ? binop(0, v, (a, b) => a - b) : v; }
    if (t.t === 'id') {
      if (FUNCS[t.v] && peek() && peek().v === '(') {
        i++; const v = sum(); eat(')');
        if (typeof v === 'number') return v;
        return FUNCS[t.v](v);
      }
      const v = vars(t.v);
      if (v === undefined) throw new Error(`name '${t.v}' is not defined`);
      return v;
    }
    throw new Error(`unexpected '${t.v}'`);
  };
  const power = () => { const b = atom(); if (peek() && peek().v === '**') { i++; return binop(b, unary(), Math.pow); } return b; };
  const unary = () => (peek() && peek().t === 'op' && (peek().v === '-' || peek().v === '+')) ? atom() : power();
  const prod = () => {
    let v = unary();
    while (peek() && (peek().v === '*' || peek().v === '/')) {
      const op = tk[i++].v, r = unary();
      v = binop(v, r, op === '*' ? (a, b) => a * b : (a, b) => a / b);
    }
    return v;
  };
  const sum = () => {
    let v = prod();
    while (peek() && (peek().v === '+' || peek().v === '-')) {
      const op = tk[i++].v, r = prod();
      v = binop(v, r, op === '+' ? (a, b) => a + b : (a, b) => a - b);
    }
    return v;
  };
  try {
    const v = sum();
    if (i < tk.length) throw new Error(`unexpected '${tk[i].v}'`);
    if (typeof v !== 'number' || Number.isNaN(v)) throw new Error('the expression must reduce to a single number');
    return v;
  } catch (e) {
    throw new Error(`Invalid threshold expression '${expr}': ${e.message}`);
  }
}

export function threshold(raw, filtered, expr) {
  const memo = {};
  const vars = name => {
    if (name === 'I1') return raw.d;
    if (name === 'F1') return filtered.d;
    const m = /^Wave\.F([12])$/.exec(name);
    if (m) return (memo[name] ??= wavelet(raw, +m[1], 3).d);
    return undefined;
  };
  return evalExpression(expr, vars);
}

// ---- detection.py ---------------------------------------------------------------------------------------------------
// LocalMaximumDetector: skimage peak_local_max(min_distance, threshold_abs, exclude_border=True). Returns [[row, col]].
export function localMaximum(im, thr, minDistance = 1) {
  const { w, h, d } = im, r = minDistance;
  let mx = d;
  if (r >= 1) { // maximum_filter, square footprint 2r+1, mode 'nearest' (separable)
    const maxAlong = (src, axis) => {
      const o = new Float64Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let m = -Infinity;
        for (let j = -r; j <= r; j++) {
          const v = axis === 1 ? src[y * w + nearest(x + j, w)] : src[nearest(y + j, h) * w + x];
          if (v > m) m = v;
        }
        o[y * w + x] = m;
      }
      return o;
    };
    mx = maxAlong(maxAlong(d, 1), 0);
  }
  const isMax = Array.from(d, (v, i) => r < 1 || v === mx[i]);
  if (r >= 1 && isMax.every(Boolean)) return [];
  const peaks = [];
  for (let y = r; y < h - r; y++) for (let x = r; x < w - r; x++) {
    const i = y * w + x;
    if (isMax[i] && d[i] > thr) peaks.push([y, x, d[i]]);
  }
  peaks.sort((a, b) => b[2] - a[2]); // stable: ties keep row-major order
  if (r > 1) { // skimage ensure_spacing, Chebyshev distance, reject closer than min_distance
    const rejected = new Set();
    for (let a = 0; a < peaks.length; a++) {
      if (rejected.has(a)) continue;
      for (let b = 0; b < peaks.length; b++) {
        if (b !== a && Math.max(Math.abs(peaks[a][0] - peaks[b][0]), Math.abs(peaks[a][1] - peaks[b][1])) < r) rejected.add(b);
      }
    }
    return peaks.filter((_, a) => !rejected.has(a)).map(p => [p[0], p[1]]);
  }
  return peaks.map(p => [p[0], p[1]]);
}

// NonMaximumSuppression: grey_dilation with a cross (connectivity 1) or 3x3 (2) footprint; pixels equal to the
// dilation and above threshold, in row-major order.
export function nonMaxSuppression(im, thr, connectivity = 2) {
  const { w, h, d } = im, out = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = -Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (connectivity < 2 && dx && dy) continue;
      const v = d[reflect(y + dy, h) * w + reflect(x + dx, w)];
      if (v > m) m = v;
    }
    if (d[y * w + x] === m && d[y * w + x] > thr) out.push([y, x]);
  }
  return out;
}

export const DETECTORS = {
  local_maximum: (im, thr, p) => localMaximum(im, thr, p.min_distance ?? 1),
  non_maximum_suppression: (im, thr, p) => nonMaxSuppression(im, thr, p.connectivity ?? 2),
};

