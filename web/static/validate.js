// Validation: (1) the current demo result against the simulation's ground truth; (2) the browser port against the
// Python pipeline on the whole demo movie, localisation by localisation (Python result exported at build time).
import { $, api, app, esc, route } from './core.js';
import { f, int, pct } from './fmt.js';
import { ensureResult, getMovie } from './state.js';
import { analyzeMovie } from './ts_fit.js';
import { evaluateMovie } from './ts_render.js';

const evalTable = (ev, label) => `<tr><th scope="row">${label}</th><td>${int(ev.n_det)}</td><td>${int(ev.tp)}</td><td>${int(ev.fp)}</td>
  <td>${int(ev.fn)}</td><td>${pct(ev.recall)}</td><td>${pct(ev.precision)}</td><td>${f(ev.rmse_nm, 2)}</td></tr>`;

route('validate', async () => {
  app().innerHTML = '<p class="mute">Loading...</p>';
  const [r, py, mv] = await Promise.all([ensureResult(), api('/demo/result'), getMovie()]);
  app().innerHTML = `<p class="kicker">Checks</p><h1>Validation</h1>
  <p class="lead">The demo movie is simulated, so every true emitter position is known. And the browser runs a port of the
    Python code, so the two can be compared one localisation at a time.</p>
  <h2>Against the ground truth</h2>
  <p>${int(mv.truth.x.length)} simulated emissions in ${mv.n} frames. <code>PerformanceEvaluator.evaluate</code> per frame
    (nearest neighbour within ${mv.tolerance_nm} nm), counts summed over frames, RMSE pooled over matched pairs.</p>
  <div class="scroll"><table><thead><tr><th scope="col">Run</th><th scope="col">localisations</th><th scope="col">true pos.</th><th scope="col">false pos.</th>
    <th scope="col">missed</th><th scope="col">recall</th><th scope="col">precision</th><th scope="col">RMSE [nm]</th></tr></thead><tbody>
    ${evalTable(py.evaluation, 'Python pipeline, default settings')}
    ${r.source === 'demo' && r.evaluation ? evalTable(r.evaluation, `Your current run (${r.engine === 'python' ? 'Python' : 'JS'}, threshold <code>${esc(r.params.threshold_expression)}</code>)`) : ''}
  </tbody></table></div>
  <p class="small mute">Typical causes: two molecules on at the same time within about a PSF width of each other are fitted as one
    (one emitter missed, and the merged fit may land more than ${mv.tolerance_nm} nm from both), and dim emissions fall below the threshold.</p>
  <h2>Browser port against Python</h2>
  <p>The Python result below was computed by <code>ThunderSTORM.analyze_stack</code> with the default settings when this page was built
    (or by the local server). The same settings are now run by the JavaScript port in your browser.</p>
  <div id="cmp" class="panel"><p class="mute">Running the JS port on ${mv.n} frames...</p></div>`;
  await new Promise(res => setTimeout(res, 30));
  const t0 = performance.now();
  const js = analyzeMovie(mv, py.params);
  const secs = (performance.now() - t0) / 1000;
  js.evaluation = evaluateMovie(js.locs, mv.truth, mv.n, mv.tolerance_nm);
  const P = py.locs, J = js.locs, same = J.x.length === P.x.length && J.frame.every((v, i) => v === P.frame[i]);
  let rows = '', d = [];
  if (same) {
    d = J.x.map((x, i) => Math.hypot(x - P.x[i], J.y[i] - P.y[i]));
    const sorted = [...d].sort((a, b) => a - b), off = d.map((v, i) => [v, i]).filter(([v]) => v > 0.01);
    rows = `<dl class="kv">
      <dt>localisations</dt><dd>${int(J.x.length)} JS · ${int(P.x.length)} Python (same frames, same order)</dd>
      <dt>within 0.01 nm</dt><dd>${int(d.length - off.length)} of ${int(d.length)} (${pct((d.length - off.length) / d.length)})</dd>
      <dt>median |Δposition|</dt><dd>${sorted[sorted.length >> 1].toExponential(1)} nm</dd>
      <dt>JS run time</dt><dd>${secs.toFixed(2)} s in this browser</dd></dl>
      ${off.length ? `<h3>Fits that end at different points</h3><p class="small">Both sides run MINPACK's Levenberg-Marquardt with a
        finite-difference Jacobian; last-digit rounding differences grow in ill-conditioned fits (here mostly a collapsing PSF width or
        two molecules fitted as one) and the optimiser stops elsewhere.</p>
        <div class="scroll"><table class="tbl"><thead><tr><th>#</th><th>frame</th><th>|Δ| [nm]</th><th>σx JS / Py [px]</th><th>amplitude JS / Py</th></tr></thead><tbody>
        ${off.map(([v, i]) => `<tr><td>${i}</td><td>${J.frame[i] + 1}</td><td>${f(v, 3)}</td><td>${f(J.sigma_x[i], 3)} / ${f(P.sigma_x[i], 3)}</td>
          <td>${f(J.intensity[i], 0)} / ${f(P.intensity[i], 0)}</td></tr>`).join('')}</tbody></table></div>` : ''}`;
  } else {
    rows = `<p class="err">Different localisation lists: ${int(J.x.length)} in JS, ${int(P.x.length)} in Python.</p>`;
  }
  $('#cmp').innerHTML = `${rows}<div class="scroll"><table><thead><tr><th scope="col">Run</th><th scope="col">localisations</th><th scope="col">true pos.</th>
    <th scope="col">false pos.</th><th scope="col">missed</th><th scope="col">recall</th><th scope="col">precision</th><th scope="col">RMSE [nm]</th></tr></thead>
    <tbody>${evalTable(py.evaluation, 'Python')}${evalTable(js.evaluation, 'JS port')}</tbody></table></div>
    <p class="small mute">The repo's tests (<code>ts_engine.test.mjs</code>) also check the filtered images, thresholds and detections
    exactly, and every fitted value, for four filter / detector / fitter combinations.</p>`;
});
