// Localisation table (sortable, paged, filter by frame) and CSV export in thunderSTORM's column format
// (port of thunderstorm_format.to_thunderstorm_columns).
import { $, app, download, esc, route } from './core.js';
import { f, int } from './fmt.js';
import { ensureResult, engineBadge, sourceBadge } from './state.js';
import { toCsv, toThunderstormColumns } from './ts_render.js';

const COLS = [['frame', 'frame', 0], ['x', 'x [nm]', 2], ['y', 'y [nm]', 2], ['sigma_x', 'σx [px]', 3], ['sigma_y', 'σy [px]', 3],
  ['intensity', 'amplitude', 1], ['background', 'background', 2], ['uncertainty', 'uncertainty [nm]', 2], ['chi_squared', 'χ² (SSR)', 1]];
const PAGE = 50;

// Candidates whose PSF fit failed: not localisations, listed with the reason (pipeline.get_rejected()).
function rejectedTable(r) {
  const R = r.rejected || { frame: [], x: [], y: [], reason: [] }, n = R.frame.length;
  if (!n) return '<p class="mute">None: every detected candidate gave a valid fit.</p>';
  const rows = R.frame.slice(0, 200).map((fr, i) => `<tr><td>${fr + 1}</td><td>${f(R.x[i], 1)}</td><td>${f(R.y[i], 1)}</td><td>${esc(R.reason[i])}</td></tr>`);
  return `<p class="small">${int(n)} candidate${n === 1 ? '' : 's'} whose fit failed. They are left out of the table above and of the CSV;
    x / y are the candidate pixel centre in nm.${n > 200 ? ' First 200 shown.' : ''}</p>
    <div class="scroll"><table class="tbl" aria-label="Rejected candidates"><thead><tr><th scope="col">frame</th><th scope="col">x [nm]</th>
    <th scope="col">y [nm]</th><th scope="col">reason</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
}

route('table', async () => {
  app().innerHTML = '<p class="mute">Analysing the demo movie with the default settings...</p>';
  const r = await ensureResult(), L = r.locs, p = r.params, n = L.x.length;
  const photonFitter = p.fitter_type === 'gaussian_mle';
  app().innerHTML = `<p class="kicker">Results</p><h1>Localisations</h1>
  <p class="lead">${int(n)} localisations. Positions and uncertainty in nm (pixel size ${f(r.pixel_size_nm, r.pixel_size_nm % 1 ? 2 : 0)} nm,
    ${esc(r.pixel_size_source || 'user')}); PSF sigma in camera pixels; amplitude and background in ${photonFitter ? 'photons' : 'camera counts (ADU)'}
    as the ${esc(p.fitter_type)} fitter returns them.</p>
  <p>${engineBadge(r)} ${sourceBadge(r)}</p>
  <div class="panel"><h3>Export</h3>
    <p class="small">thunderSTORM column format: <code>id, frame, x [nm], y [nm], sigma [nm]</code> (or <code>sigma1/sigma2 [nm]</code> when the
      fit gave different x / y widths), <code>intensity [photon], offset [photon], uncertainty [nm]</code>. Frames are 1-based and x / y use
      thunderSTORM's corner origin (x + pixel/2), so the file opens in thunderSTORM's results table. Counts are converted with
      ${p.photons_per_adu} photons/ADU, a ${p.baseline} ADU baseline and EM gain ${p.em_gain ?? 1}${photonFitter ? ' (the MLE fitter already works in photons)' : ''};
      thunderSTORM's bkgstd column is not written because this package does not estimate it.</p>
    <button type="button" id="csv">Download thunderSTORM CSV</button></div>
  <div class="row"><div><label for="f0">From frame</label><input id="f0" type="number" min="1" value="1"></div>
    <div><label for="f1">To frame</label><input id="f1" type="number" min="1" value="${r.shape[0]}"></div></div>
  <div class="pager"><button type="button" class="alt" id="pp">Previous</button><span id="pi" class="small mute"></span><button type="button" class="alt" id="pn">Next</button></div>
  <div class="scroll"><table class="tbl" aria-label="Localisations"><thead><tr>${COLS.map(([k, label]) => `<th scope="col"><button type="button" data-k="${k}">${label}</button></th>`).join('')}</tr></thead><tbody id="tb"></tbody></table></div>
  <h2>Rejected candidates</h2>
  ${rejectedTable(r)}`;

  let sortKey = 'frame', dir = 1, page = 0;
  const rows = () => {
    const a = +$('#f0').value - 1, b = +$('#f1').value - 1, idx = [];
    for (let i = 0; i < n; i++) if (L.frame[i] >= a && L.frame[i] <= b) idx.push(i);
    const v = i => (L[sortKey][i] === null ? Infinity : L[sortKey][i]);
    return idx.sort((i, j) => dir * (v(i) - v(j)) || i - j);
  };
  const draw = () => {
    const idx = rows(), pages = Math.max(1, Math.ceil(idx.length / PAGE));
    page = Math.min(page, pages - 1);
    $('#tb').innerHTML = idx.slice(page * PAGE, (page + 1) * PAGE).map(i => `<tr>${COLS.map(([k, , d]) =>
      `<td>${k === 'frame' ? L.frame[i] + 1 : f(L[k][i], d)}</td>`).join('')}</tr>`).join('');
    $('#pi').textContent = `${int(idx.length)} rows · page ${page + 1} of ${pages} · sorted by ${sortKey} ${dir > 0 ? '↑' : '↓'} (frames 1-based)`;
  };
  app().querySelectorAll('th button').forEach(b => b.addEventListener('click', () => {
    dir = sortKey === b.dataset.k ? -dir : 1; sortKey = b.dataset.k; draw();
  }));
  ['f0', 'f1'].forEach(id => $('#' + id).addEventListener('change', () => { page = 0; draw(); }));
  $('#pp').addEventListener('click', () => { page = Math.max(0, page - 1); draw(); });
  $('#pn').addEventListener('click', () => { page++; draw(); });
  $('#csv').addEventListener('click', () => {
    const cols = toThunderstormColumns(L, { pixelSize: r.pixel_size_nm, photonsPerAdu: p.photons_per_adu, baseline: p.baseline, fitterType: p.fitter_type, emGain: p.em_gain ?? 1 });
    const stem = r.source === 'demo' ? 'demo_simulated' : String(r.source).replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_');
    download(`${stem}_${r.pixel_size_nm}nm_thunderstorm.csv`, toCsv(cols), 'text/csv');
  });
  draw();
});
