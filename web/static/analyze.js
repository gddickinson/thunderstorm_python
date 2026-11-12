// Analyse: data source, filter / threshold / detector / fitter and camera settings -> run (Python server, or the
// JS port on the static site) -> summary and a frame viewer with every detection marked.
import { $, app, esc, isStatic, list, route } from './core.js';
import { drawGrey, drawMarkers, f, int, pct } from './fmt.js';
import { engineBadge, getFrame, getInfo, getMovie, getParams, getResult, run, setParams, sourceBadge } from './state.js';

const KINDS = [['filter', 'Image filter'], ['detector', 'Detector'], ['fitter', 'PSF fitter']];

function paramInputs(kind, opt, values) {
  return list(opt.params, p => {
    const id = `${kind}-${p.name}`, v = values?.[p.name] ?? p.default;
    if (p.type === 'bool') return `<label class="chk" for="${id}"><input type="checkbox" id="${id}" data-kind="${kind}" data-p="${p.name}" data-t="bool" ${v ? 'checked' : ''}> ${esc(p.name)}</label>`;
    const step = p.type === 'int' ? '1' : 'any';
    return `<label for="${id}">${esc(p.name)} <span class="unit">${p.name.includes('sigma') ? 'px' : p.name.includes('size') || p.name.includes('distance') || p.name === 'spacing' ? 'px' : ''}</span></label>
      <input id="${id}" type="${p.type === 'str' ? 'text' : 'number'}" step="${step}" data-kind="${kind}" data-p="${p.name}" data-t="${p.type}" value="${esc(v)}">`;
  });
}

route('analyze', async () => {
  const [info, p0] = await Promise.all([getInfo(), getParams()]);
  const opts = info.options, cam = info.demo.camera, st = isStatic();
  const sel = (kind, cur) => list(opts[kind], o => `<option value="${o.id}" ${o.id === cur ? 'selected' : ''} ${st && !o.js ? 'disabled' : ''}>${esc(o.id)}${st && !o.js ? ' (local app)' : ''}</option>`);
  app().innerHTML = `<p class="kicker">Run the pipeline</p><h1>Analyse a movie</h1>
  <p class="lead">Each frame is filtered, thresholded, searched for local maxima and every candidate is fitted with a 2D
    Gaussian (<code>ThunderSTORM.analyze_stack</code>). Defaults come from the package itself.</p>
  <div class="side">
  <form class="params panel" id="pf" novalidate>
    <fieldset class="src"><legend>Data</legend>
      <label><input type="radio" name="source" value="demo" checked> <span>Simulated demo movie<br><small class="mute">${info.demo.shape[0]} frames, ${info.demo.shape[2]}×${info.demo.shape[1]} px, ${cam.pixel_size} nm/px</small></span></label>
      <label><input type="radio" name="source" value="upload" ${st ? 'disabled' : ''}> <span>Your TIFF stack${st ? ' <small class="mute">(local app)</small>' : ''}</span></label>
      <div id="upwrap" class="hide"><label for="file">TIFF file (2D image or frames × rows × columns)</label>
        <input id="file" type="file" accept=".tif,.tiff,image/tiff"></div>
    </fieldset>
    ${list(KINDS, ([k, label]) => `<fieldset><legend>${label}</legend>
      <label for="${k}_type">${label}</label><select id="${k}_type">${sel(k, p0[k + '_type'])}</select>
      <div id="${k}-params"></div>
      ${k === 'filter' ? `<label for="thr">Threshold expression</label><input id="thr" value="${esc(p0.threshold_expression)}">
        <p class="hint">Numbers and + − * / ** ( ) with std, mean, median, max, min of I1 (raw frame), F1 (filtered frame),
        Wave.F1 / Wave.F2 (wavelet levels 1 / 2 of the raw frame).</p>` : ''}
      ${k === 'fitter' ? `<label for="fit_radius">Fit radius <span class="unit">px</span></label><input id="fit_radius" type="number" min="1" max="10" step="1" value="${p0.fit_radius}">` : ''}
    </fieldset>`)}
    <fieldset id="camfs"><legend>Camera</legend>
      <label for="pixel_size">Pixel size <span class="unit">nm per camera pixel</span></label><input id="pixel_size" type="number" min="1" step="any" value="${p0.pixel_size}">
      <label class="chk" for="use_file_px"><input type="checkbox" id="use_file_px" ${p0.use_file_pixel_size ? 'checked' : ''}> use the TIFF's own pixel size if it records one</label>
      <label for="photons_per_adu">Photons per ADU</label><input id="photons_per_adu" type="number" min="0" step="any" value="${p0.photons_per_adu}">
      <label for="baseline">Baseline <span class="unit">ADU</span></label><input id="baseline" type="number" min="0" step="any" value="${p0.baseline}">
      <p class="hint" id="camhint"></p>
    </fieldset>
    <button type="submit" id="runbtn">Run analysis</button><button type="button" class="alt" id="reset">Defaults</button>
    <p id="msg" class="small" role="status"></p>
  </form>
  <div id="out"></div></div>`;

  const form = $('#pf');
  const source = () => form.querySelector('input[name=source]:checked').value;
  const drawParams = (k, values) => { const o = opts[k].find(x => x.id === $(`#${k}_type`).value); $(`#${k}-params`).innerHTML = paramInputs(k, o, values); };
  KINDS.forEach(([k]) => { drawParams(k, p0[k + '_params']); $(`#${k}_type`).addEventListener('change', () => drawParams(k)); });
  const syncSource = () => {
    const demo = source() === 'demo';
    $('#upwrap').classList.toggle('hide', demo);
    ['pixel_size', 'photons_per_adu', 'baseline', 'use_file_px'].forEach(id => { $('#' + id).disabled = demo; });
    $('#camhint').textContent = demo ? `The demo is in photon counts with no camera offset: ${cam.pixel_size} nm/px, 1 photon per ADU, baseline 0.`
      : 'Positions are converted to nm with this pixel size; the value used (and whether it came from the file) is shown with the result.';
    if (demo) { $('#pixel_size').value = cam.pixel_size; $('#photons_per_adu').value = cam.photons_per_adu; $('#baseline').value = cam.baseline; }
    else { $('#pixel_size').value = info.defaults.pixel_size; $('#photons_per_adu').value = info.defaults.photons_per_adu; $('#baseline').value = info.defaults.baseline; }
  };
  form.querySelectorAll('input[name=source]').forEach(r => r.addEventListener('change', syncSource));
  syncSource();

  const collect = () => {
    const p = { threshold_expression: $('#thr').value.trim(), fit_radius: +$('#fit_radius').value,
      pixel_size: +$('#pixel_size').value, photons_per_adu: +$('#photons_per_adu').value, baseline: +$('#baseline').value,
      use_file_pixel_size: $('#use_file_px').checked };
    for (const [k] of KINDS) {
      p[k + '_type'] = $(`#${k}_type`).value; p[k + '_params'] = {};
      form.querySelectorAll(`[data-kind=${k}]`).forEach(el => {
        const t = el.dataset.t;
        p[k + '_params'][el.dataset.p] = t === 'bool' ? el.checked : t === 'str' ? el.value : +el.value;
      });
    }
    if (!(p.pixel_size > 0)) throw new Error('Pixel size must be a positive number of nm.');
    if (!Number.isInteger(p.fit_radius) || p.fit_radius < 1 || p.fit_radius > 10) throw new Error('Fit radius must be a whole number from 1 to 10.');
    if (!p.threshold_expression) throw new Error('Enter a threshold expression or a number.');
    return p;
  };

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = $('#runbtn'), msg = $('#msg');
    try {
      const p = collect(), file = source() === 'upload' ? $('#file').files[0] : null;
      if (source() === 'upload' && !file) throw new Error('Choose a TIFF file first.');
      setParams(p);
      btn.disabled = true; msg.className = 'small'; msg.textContent = file ? `Uploading and analysing ${file.name}...` : 'Analysing the demo movie...';
      await new Promise(r => setTimeout(r, 30)); // let the status paint before the browser-side run
      const r = await run(p, file);
      msg.textContent = `Done: ${int(r.n)} localisations in ${r.seconds.toFixed(1)} s.`;
      await show(r);
    } catch (err) { msg.className = 'small err'; msg.textContent = err.message; } finally { btn.disabled = false; }
  });
  $('#reset').addEventListener('click', () => { setParams(null); location.hash = '#/analyze?r=' + Date.now(); });
  const prev = getResult();
  if (prev) await show(prev);
  else $('#out').innerHTML = `<div class="panel"><p>Choose the settings and press <b>Run analysis</b>.
    ${st ? 'On this static site the demo movie is analysed in your browser (about a second).' : ''}</p></div>`;
});

async function show(r) {
  const [n, h, w] = r.shape, px = r.pixel_size_nm, ev = r.evaluation;
  const truth = r.source === 'demo' ? (await getMovie()).truth : null;
  $('#out').innerHTML = `<div class="panel">
    <p>${engineBadge(r)} ${sourceBadge(r)}</p>
    <div class="tiles"><div class="tile"><b>${int(r.n)}</b><span>localisations</span></div>
      <div class="tile"><b>${n}</b><span>frames (${w}×${h} px)</span></div>
      <div class="tile"><b>${f(px, px % 1 ? 2 : 0)} nm</b><span>pixel size (${esc(r.pixel_size_source || 'user')})</span></div>
      ${ev ? `<div class="tile"><b>${pct(ev.recall)}</b><span>recall vs ground truth</span></div>
      <div class="tile"><b>${f(ev.rmse_nm, 1)} nm</b><span>RMSE of matched positions</span></div>` : ''}</div>
    ${r.file_pixel_size_nm && r.pixel_size_source !== 'file metadata' ? `<p class="small warnc">The file records ${f(r.file_pixel_size_nm, 2)} nm/px; your value was used instead.</p>` : ''}
    ${r.source !== 'demo' && !r.file_pixel_size_nm ? '<p class="small mute">The file does not record a pixel size; the value you entered was used.</p>' : ''}
    <p class="small"><a href="#/image">Super-resolution image</a> · <a href="#/table">Localisation table and CSV</a> · <a href="#/validate">Validation</a></p></div>
  <div class="panel viewer"><h3>Frames</h3>
    <div class="stage"><canvas id="fv" role="img" aria-label="Camera frame with detections marked"></canvas></div>
    <div class="legend"><span style="--c:#ffb02e">fitted localisation</span>${truth ? '<span style="--c:#7fd4ff">ground truth (simulation)</span>' : ''}</div>
    <label for="fs">Frame <b id="fn">1</b> of ${n} · <span id="fc"></span></label>
    <input type="range" id="fs" min="0" max="${n - 1}" value="0">
    <button type="button" class="alt" id="fprev" aria-label="Previous frame">◀</button><button type="button" class="alt" id="fnext" aria-label="Next frame">▶</button>
    <p class="caption">Raw camera frame (grey, contrast-stretched). Circles are 1.5 px in radius around each position.</p></div>`;
  const off = document.createElement('canvas'), disp = $('#fv'), scale = Math.max(2, Math.min(10, Math.floor(640 / w)));
  const draw = async i => {
    const fr = await getFrame(r, i);
    drawGrey(off, fr, w, h);
    const marks = [];
    if (truth) truth.frame.forEach((t, j) => { if (t === i) marks.push({ x: truth.x[j] / px, y: truth.y[j] / px, r: 2.2, c: '#7fd4ff', lw: 1.2 }); });
    let k = 0;
    r.locs.frame.forEach((t, j) => { if (t === i && r.locs.x[j] !== null) { k++; marks.push({ x: r.locs.x[j] / px, y: r.locs.y[j] / px, r: 1.5, c: '#ffb02e' }); } });
    drawMarkers(disp, off, scale, marks);
    $('#fn').textContent = i + 1;
    $('#fc').textContent = `${k} localisation${k === 1 ? '' : 's'}`;
  };
  const sl = $('#fs');
  sl.addEventListener('input', () => draw(+sl.value));
  $('#fprev').addEventListener('click', () => { sl.value = Math.max(0, +sl.value - 1); draw(+sl.value); });
  $('#fnext').addEventListener('click', () => { sl.value = Math.min(n - 1, +sl.value + 1); draw(+sl.value); });
  await draw(0);
}
