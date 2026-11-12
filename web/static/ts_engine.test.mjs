// node web/static/ts_engine.test.mjs — the JS pipeline against the Python pipeline (tests/fixtures/engine_cases.json,
// written by scripts/make_engine_fixture.py). Exits non-zero on any mismatch; prints the worst differences.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { FILTERS, evalExpression, img, threshold } from './ts_engine.js';
import { analyzeFrame, erf, LOC_KEYS, mortensenLSQ } from './ts_fit.js';
import { evaluateFrame, renderGaussian, toThunderstormColumns } from './ts_render.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, '..', '..', 'tests', 'fixtures', 'engine_cases.json'), 'utf8'));
let failures = 0;
const check = (ok, msg) => { if (!ok) { failures++; console.error('FAIL', msg); } };
const close = (a, b, atol, rtol = 0) => (a === null || b === null) ? a === b : Math.abs(a - b) <= atol + rtol * Math.abs(b);

// erf against known values (Abramowitz & Stegun table / scipy)
for (const [x, v] of [[0.5, 0.5204998778130465], [1, 0.8427007929497149], [2.5, 0.9995930479825550], [3.5, 0.9999992569016276], [-1.2, -0.9103139782296353]]) {
  check(close(erf(x), v, 2e-15), `erf(${x}) = ${erf(x)} vs ${v}`);
}
check(evalExpression('2*(3+4)-1/2', () => undefined) === 13.5, 'arithmetic');
check(evalExpression('-2**2', () => undefined) === -4, 'unary minus binds looser than **');
// hand-computed Mortensen least-squares case (see tests/test_uncertainty.py): 6.2793 nm, x sqrt(2) with EM gain
check(close(mortensenLSQ(1.3, 1000, 10, 100), 6.2793, 1e-4), 'Mortensen LSQ');
check(close(mortensenLSQ(1.3, 1000, 10, 100, 2), 6.2793 * Math.SQRT2, 1e-4), 'Mortensen LSQ, EMCCD');

// Tolerances: positions in nm, the rest relative. Both sides run MINPACK lmdif with the same number of function
// evaluations; rounding differences (finite-difference Jacobian) leave ~1e-8 relative differences in the parameters.
const TOL = { x: [0.02, 0], y: [0.02, 0], sigma_x: [1e-6, 1e-3], sigma_y: [1e-6, 1e-3], intensity: [1e-6, 1e-3],
  background: [1e-3, 1e-3], uncertainty: [1e-6, 1e-3], chi_squared: [1e-6, 1e-6], frame: [0, 0] };
const worst = Object.fromEntries(LOC_KEYS.map(k => [k, 0]));

for (const c of fx.cases) {
  const [n, h, w] = c.shape, p = c.params;
  const frames = [];
  for (let f = 0; f < n; f++) frames.push(img(w, h, Float64Array.from(c.movie.slice(f * h * w, (f + 1) * h * w))));
  const js = [];
  frames.forEach((raw, f) => {
    const filtered = FILTERS[p.filter_type](raw, p.filter_params);
    const t = c.trace[f];
    check(close(filtered.d.reduce((s, v) => s + v, 0), t.filtered_sum, 1e-8, 1e-10), `${c.name} f${f} filtered sum`);
    check(close(filtered.d[0], t.filtered_00, 1e-10, 1e-10), `${c.name} f${f} filtered[0,0] ${filtered.d[0]} vs ${t.filtered_00}`);
    check(close(filtered.d[Math.floor(h / 2) * w + Math.floor(w / 3)], t.filtered_mid, 1e-10, 1e-10), `${c.name} f${f} filtered mid`);
    check(close(threshold(raw, filtered, p.threshold_expression), t.threshold, 1e-10, 1e-12), `${c.name} f${f} threshold`);
    const trace = [];
    js.push(...analyzeFrame(raw, f, p, trace));
    check(JSON.stringify(trace[0].positions) === JSON.stringify(t.positions), `${c.name} f${f} detections ${JSON.stringify(trace[0].positions)} vs ${JSON.stringify(t.positions)}`);
  });
  check(js.length === c.locs.x.length, `${c.name}: ${js.length} vs ${c.locs.x.length} localisations`);
  js.forEach((l, i) => {
    for (const k of LOC_KEYS) {
      const a = Number.isFinite(l[k]) ? l[k] : null, b = c.locs[k][i];
      if (a !== null && b !== null) worst[k] = Math.max(worst[k], Math.abs(a - b) / (k === 'x' || k === 'y' || k === 'frame' ? 1 : Math.max(1, Math.abs(b))));
      check(close(a, b, ...TOL[k]), `${c.name} loc ${i} ${k}: ${a} vs ${b}`);
    }
  });
}

// Gaussian rendering (visualization.GaussianRenderer)
const ref = fx.cases[0].locs, R = fx.render;
const im = renderGaussian(ref, { pixelSize: R.pixel_size, sigmaNm: R.sigma_nm, width: R.image_size[0], height: R.image_size[1] });
check(close(im.d.reduce((s, v) => s + v, 0), R.sum, 1e-8, 1e-12), 'render sum');
check(im.d.indexOf(Math.max(...im.d)) === R.argmax, 'render argmax');
for (const [r, row] of Object.entries(R.rows)) row.forEach((v, c) => check(close(im.d[+r * im.w + c], v, 1e-12, 1e-12), `render [${r},${c}]`));

// Ground-truth evaluation (evaluation.PerformanceEvaluator.evaluate), per frame
fx.evaluate.truth.forEach((gt, i) => {
  const k = ref.frame.map((f, j) => (f === gt.frame ? j : -1)).filter(j => j >= 0);
  const m = evaluateFrame({ x: k.map(j => ref.x[j]), y: k.map(j => ref.y[j]) }, gt, 100);
  for (const [key, v] of Object.entries(fx.evaluate.metrics[i])) check(close(m[key] ?? null, v, 1e-9, 1e-9), `evaluate frame ${gt.frame} ${key}: ${m[key]} vs ${v}`);
});

// thunderSTORM-format export (thunderstorm_format.to_thunderstorm_columns)
const E = fx.export, cols = toThunderstormColumns(ref, { pixelSize: E.pixel_size, photonsPerAdu: E.photons_per_adu, baseline: E.baseline, emGain: E.em_gain });
check(JSON.stringify(Object.keys(cols)) === JSON.stringify(Object.keys(E.columns)), `export columns ${Object.keys(cols)}`);
for (const [k, vals] of Object.entries(E.columns)) vals.forEach((v, i) => check(close(cols[k][i], v, 1e-9, 1e-12), `export ${k}[${i}]`));

console.log('worst |JS - Python| (x, y in nm; others relative):', Object.fromEntries(Object.entries(worst).map(([k, v]) => [k, +v.toExponential(2)])));
if (failures) { console.error(`${failures} failure(s)`); process.exit(1); }
console.log(`ok: ${fx.cases.length} pipeline cases (${fx.cases.reduce((s, c) => s + c.locs.x.length, 0)} localisations), render, evaluation, export`);
