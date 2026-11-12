// Super-resolution image: Gaussian rendering of the current localisations (port of visualization.GaussianRenderer),
// beside the diffraction-limited view of the same field. Rendering always runs in the browser.
import { $, app, download, route } from './core.js';
import { drawGrey, f, int, scaleBar } from './fmt.js';
import { ensureResult, engineBadge, getFrame, getMovie, meanImage, sourceBadge } from './state.js';
import { renderGaussian, toImageData } from './ts_render.js';

const MAX_PIXELS = 9e6;

route('image', async () => {
  app().innerHTML = '<p class="mute">Analysing the demo movie with the default settings...</p>';
  const r = await ensureResult();
  const [n, h, w] = r.shape, px = r.pixel_size_nm, fovW = w * px, fovH = h * px;
  app().innerHTML = `<p class="kicker">Rendering</p><h1>Super-resolution image</h1>
  <p class="lead">Every localisation is drawn as a small 2D Gaussian (<code>GaussianRenderer</code>) on a grid much finer than
    the camera's ${f(px, px % 1 ? 2 : 0)} nm pixels. ${int(r.n)} localisations from ${n} frames.</p>
  <p>${engineBadge(r)} ${sourceBadge(r)}</p>
  <div class="panel"><div class="row">
    <div><label for="rpx">Render pixel <span class="mute">nm</span></label><select id="rpx">${[5, 10, 20, 25, 50].map(v => `<option ${v === 10 ? 'selected' : ''}>${v}</option>`).join('')}</select></div>
    <div><label for="sig">Gaussian sigma</label><select id="sig"><option value="20" selected>20 nm (fixed)</option><option value="10">10 nm (fixed)</option>
      <option value="40">40 nm (fixed)</option><option value="computed">each localisation's uncertainty</option></select></div>
    <div><label for="clip">Saturate the brightest <b id="clipv">0.5</b>%</label><input type="range" id="clip" min="0" max="5" step="0.1" value="0.5"></div>
  </div><p id="note" class="small mute"></p></div>
  <div class="two">
    <figure class="panel" style="margin:0"><h3>Super-resolution</h3><div class="stage sr"><canvas id="sr" class="smooth" role="img" aria-label="Super-resolution rendering"></canvas><span id="sb1"></span></div>
      <figcaption class="caption" id="cap"></figcaption><button type="button" class="alt" id="png">Download PNG</button></figure>
    <figure class="panel" style="margin:0"><h3 id="wtitle">Diffraction-limited</h3><div class="stage"><canvas id="wf" class="smooth" role="img" aria-label="Diffraction-limited view"></canvas><span id="sb2"></span></div>
      <figcaption class="caption" id="wcap"></figcaption></figure>
  </div>`;

  // Diffraction-limited view: mean of all frames (demo) or the first frame (uploads, kept on the server).
  const wf = $('#wf');
  if (r.source === 'demo') { drawGrey(wf, meanImage(await getMovie()), w, h); $('#wcap').textContent = `Mean of all ${n} frames at ${px} nm per camera pixel.`; }
  else { drawGrey(wf, await getFrame(r, 0), w, h); $('#wtitle').textContent = 'First frame'; $('#wcap').textContent = `Frame 1 at ${f(px, 2)} nm per camera pixel.`; }
  wf.style.imageRendering = 'pixelated';
  $('#sb2').innerHTML = scaleBar(fovW);

  const draw = () => {
    let rp = +$('#rpx').value;
    const minRp = Math.ceil(Math.sqrt(fovW * fovH / MAX_PIXELS));
    $('#note').textContent = rp < minRp ? `A ${rp} nm grid would be too large for this field of view; using ${minRp} nm.` : '';
    rp = Math.max(rp, minRp);
    const sv = $('#sig').value, sigmaNm = sv === 'computed' ? 'computed' : +sv;
    const W = Math.ceil(fovW / rp), H = Math.ceil(fovH / rp);
    const im = renderGaussian(r.locs, { pixelSize: rp, sigmaNm, width: W, height: H });
    const c = $('#sr'); c.width = W; c.height = H;
    const ctx = c.getContext('2d');
    ctx.putImageData(toImageData(im, ctx, 100 - +$('#clip').value), 0, 0);
    $('#sb1').innerHTML = scaleBar(fovW);
    $('#cap').textContent = `${W}×${H} px at ${rp} nm/px, sigma ${sigmaNm === 'computed' ? 'from each uncertainty' : sigmaNm + ' nm'}`
      + `; positions in nm (pixel-centre origin), the grid starts at the centre of camera pixel (0, 0).`;
  };
  ['rpx', 'sig'].forEach(id => $('#' + id).addEventListener('change', draw));
  $('#clip').addEventListener('input', () => { $('#clipv').textContent = $('#clip').value; draw(); });
  $('#png').addEventListener('click', () => $('#sr').toBlob(b => download(`thunderstorm_sr_${$('#rpx').value}nm.png`, b)));
  draw();
});
