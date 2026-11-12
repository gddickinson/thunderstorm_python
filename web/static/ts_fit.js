// Gaussian least-squares PSF fitting (fitting.GaussianLSQFitter) and the per-frame pipeline (pipeline.ThunderSTORM).
// Python fits with scipy curve_fit, i.e. MINPACK lmdif; ts_minpack.js is a port of lmdif, so both take the same steps.
import { DETECTORS, FILTERS, img, threshold } from './ts_engine.js';
import { lmdif } from './ts_minpack.js';

// erf as scipy.special.erf computes it (Cephes ndtr.c, via scipy's xsf headers): same rational approximations.
const P = [2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0, 4.86371970985681366614E1,
  1.96520832956077098242E2, 5.26445194995477358631E2, 9.34528527171957607540E2, 1.02755188689515710272E3, 5.57535335369399327526E2];
const Q = [1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2, 9.75708501743205489753E2,
  1.82390916687909736289E3, 2.24633760818710981792E3, 1.65666309194161350182E3, 5.57535340817727675546E2];
const R = [5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0, 6.16021097993053585195E0,
  7.40974269950448939160E0, 2.97886665372100240670E0];
const S = [2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1, 1.70814450747565897222E1,
  9.60896809063285878198E0, 3.36907645100081516050E0];
const T = [9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3, 7.00332514112805075473E3, 5.55923013010394962768E4];
const U = [3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3, 2.26290000613890934246E4, 4.92673942608635921086E4];
const polevl = (x, c) => { let a = c[0]; for (let i = 1; i < c.length; i++) a = a * x + c[i]; return a; };
const p1evl = (x, c) => { let a = x + c[0]; for (let i = 1; i < c.length; i++) a = a * x + c[i]; return a; };
const MAXLOG = 7.09782712893383996843E2;

export function erfc(a) {
  const x = Math.abs(a);
  if (x < 1) return 1 - erf(a);
  let z = -a * a;
  if (z < -MAXLOG) return a < 0 ? 2 : 0;
  z = Math.exp(z);
  const [p, q] = x < 8 ? [polevl(x, P), p1evl(x, Q)] : [polevl(x, R), p1evl(x, S)];
  let y = (z * p) / q;
  if (a < 0) y = 2 - y;
  return y !== 0 ? y : (a < 0 ? 2 : 0);
}

export function erf(x) {
  if (Number.isNaN(x)) return NaN;
  if (x < 0) return -erf(-x);
  if (Math.abs(x) > 1) return 1 - erfc(x);
  const z = x * x;
  return x * polevl(z, T) / p1evl(z, U);
}

// fitting_base.integrated_gaussian_2d / gaussian_2d, same operation order as NumPy, model - data as curve_fit does.
function residuals(xs, ys, data, integrated) {
  const n = xs.length;
  return p => {
    const [x0, y0, A, sx, sy, bg] = p, out = new Float64Array(n);
    if (integrated) {
      const nx = Math.SQRT2 * sx, ny = Math.SQRT2 * sy;
      for (let i = 0; i < n; i++) {
        const ix = 0.5 * (erf((xs[i] + 0.5 - x0) / nx) - erf((xs[i] - 0.5 - x0) / nx));
        const iy = 0.5 * (erf((ys[i] + 0.5 - y0) / ny) - erf((ys[i] - 0.5 - y0) / ny));
        out[i] = (A * ix * iy + bg) - data[i];
      }
    } else {
      for (let i = 0; i < n; i++) {
        const g = A * Math.exp(-(((xs[i] - x0) ** 2) / (2 * sx ** 2) + ((ys[i] - y0) ** 2) / (2 * sy ** 2)));
        out[i] = (g + bg) - data[i];
      }
    }
    return out;
  };
}

// scipy curve_fit(maxfev=1000): MINPACK lmdif; info 1-4 is success, anything else raises RuntimeError in Python.
export function curveFit(xs, ys, data, p0, integrated) {
  const r = lmdif(residuals(xs, ys, data, integrated), p0, { maxfev: 1000 });
  const ok = r.info >= 1 && r.info <= 4 && Array.from(r.x).every(Number.isFinite);
  return { p: Array.from(r.x), S: r.fvec.reduce((s, v) => s + v * v, 0), ok, info: r.info, nfev: r.nfev };
}

const median = a => { const s = Float64Array.from(a).sort(), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

// uncertainty.localization_uncertainty, least-squares form (Mortensen et al. 2010, doi:10.1038/nmeth.1447):
// var = F (sa^2/N) (16/9 + 8 pi sa^2 b^2 / (N a^2)), sa^2 = s^2 + a^2/12, b^2 = background photons per pixel.
export function mortensenLSQ(sigmaPx, photons, bgPhotons, pixelSize, excess = 1) {
  if (!Number.isFinite(photons) || photons <= 0) return Infinity;
  const a = pixelSize, b2 = Math.max(bgPhotons, 0), sa2 = (sigmaPx * a) ** 2 + a * a / 12.0;
  const v = sa2 / photons * (16.0 / 9.0 + 8.0 * Math.PI * sa2 * b2 / (photons * a * a));
  return Math.sqrt(excess * v);
}

// BaseFitter.counts_to_photons and the EMCCD excess-noise factor.
export const camera = ({ photonsPerAdu = 1, baseline = 0, emGain = 1 } = {}) => ({
  photons: (c, sub = false) => (sub ? c - baseline : c) * photonsPerAdu / emGain,
  excess: emGain > 1 ? 2.0 : 1.0,
});

// GaussianLSQFitter.fit: positions [[row, col]] -> localisations in camera pixels.
export function fitLSQ(raw, positions, { fitRadius = 3, integrated = true, initialSigma = 1.3, pixelSize = 100, photonsPerAdu = 1, baseline = 0, emGain = 1 } = {}) {
  const out = [], cam = camera({ photonsPerAdu, baseline, emGain });
  for (const pos of positions) {
    const row = Math.trunc(pos[0]), col = Math.trunc(pos[1]);
    const r0 = Math.max(0, row - fitRadius), r1 = Math.min(raw.h, row + fitRadius + 1);
    const c0 = Math.max(0, col - fitRadius), c1 = Math.min(raw.w, col + fitRadius + 1);
    const xs = [], ys = [], data = [];
    for (let y = r0; y < r1; y++) for (let x = c0; x < c1; x++) { xs.push(x - c0); ys.push(y - r0); data.push(raw.d[y * raw.w + x]); }
    const bg0 = median(data), p0 = [col - c0, row - r0, Math.max(...data) - bg0, initialSigma, initialSigma, bg0];
    const f = curveFit(xs, ys, data, p0, integrated);
    if (!f.ok) {
      out.push({ x: col, y: row, intensity: 0, background: 0, sigma_x: initialSigma, sigma_y: initialSigma,
        uncertainty: Infinity, chi_squared: Infinity, failed: true });
      continue;
    }
    const [x0, y0, A, sx, sy, bg] = f.p;
    out.push({ x: x0 + c0, y: y0 + r0, intensity: A, background: bg, sigma_x: Math.abs(sx), sigma_y: Math.abs(sy),
      uncertainty: mortensenLSQ(Math.abs(sx), cam.photons(A), cam.photons(bg, true), pixelSize, cam.excess), chi_squared: f.S });
  }
  return out;
}

export const LOC_KEYS = ['frame', 'x', 'y', 'sigma_x', 'sigma_y', 'intensity', 'background', 'uncertainty', 'chi_squared'];

// pipeline.ThunderSTORM.analyze_frame for the parts ported to JS. params has the /api/info "defaults" shape.
export function analyzeFrame(raw, frame, params, trace = null) {
  const ft = params.filter_type, dt = params.detector_type;
  if (!FILTERS[ft]) throw new Error(`The ${ft} filter runs only in the local app (Python).`);
  if (!DETECTORS[dt]) throw new Error(`The ${dt} detector runs only in the local app (Python).`);
  if (params.fitter_type !== 'gaussian_lsq') throw new Error(`The ${params.fitter_type} fitter runs only in the local app (Python).`);
  const filtered = FILTERS[ft](raw, params.filter_params || {});
  const thr = threshold(raw, filtered, params.threshold_expression);
  const pos = DETECTORS[dt](filtered, thr, params.detector_params || {});
  if (trace) trace.push({ frame, threshold: thr, positions: pos });
  const fp = params.fitter_params || {};
  const fits = fitLSQ(raw, pos, { fitRadius: params.fit_radius, integrated: fp.integrated !== false,
    initialSigma: fp.initial_sigma ?? 1.3, pixelSize: params.pixel_size, photonsPerAdu: params.photons_per_adu,
    baseline: params.baseline ?? 0, emGain: params.em_gain ?? 1 });
  return fits.map(f => ({ ...f, frame, x: f.x * params.pixel_size, y: f.y * params.pixel_size }));
}

// Analyse a whole movie ({ w, h, n, frame(i) -> Float64Array }); returns columns like the Python API result.
export function analyzeMovie(movie, params, onProgress = null) {
  const cols = Object.fromEntries(LOC_KEYS.map(k => [k, []]));
  for (let f = 0; f < movie.n; f++) {
    for (const l of analyzeFrame(img(movie.w, movie.h, movie.frame(f)), f, params)) {
      for (const k of LOC_KEYS) cols[k].push(Number.isFinite(l[k]) ? l[k] : null);
    }
    if (onProgress) onProgress(f + 1, movie.n);
  }
  return { engine: 'js', params, pixel_size_nm: params.pixel_size, shape: [movie.n, movie.h, movie.w],
    n: cols.x.length, locs: cols,
    units: { x: 'nm', y: 'nm', uncertainty: 'nm', sigma_x: 'px', sigma_y: 'px', origin: 'pixel centre (pixel (0,0) centre at 0 nm)' } };
}
