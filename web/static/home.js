// Home: hero, real numbers from the demo movie and the pipeline defaults, and cards to the pages.
import { app, isStatic, route } from './core.js';
import { int } from './fmt.js';
import { getInfo } from './state.js';

route('home', async () => {
  const info = await getInfo(), d = info.defaults, s = info.demo.sim, [n, h, w] = info.demo.shape;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Single-molecule localisation microscopy · PALM / STORM</p>
    <h1 class="glow">ThunderSTORM Python</h1>
    <p class="lead">A Python implementation of the thunderSTORM ImageJ plugin: filter each frame, detect candidate
      molecules, fit a Gaussian PSF to each one for a sub-pixel position in nm, then render the super-resolution image.
      Run the whole pipeline here on a simulated SMLM movie made by the package's own simulator${isStatic() ? '' : ', or on your own TIFF stack'}.</p>
    <a class="btn" href="#/analyze">Run the analysis</a><a class="btn alt" href="#/image">Super-resolution image</a><a class="btn alt" href="#/method">How it works</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${n}</b><span>frames in the demo movie</span></div>
    <div class="tile"><b>${w}×${h}</b><span>camera pixels per frame</span></div>
    <div class="tile"><b>${s.pixel_size} nm</b><span>camera pixel size (stated, not assumed)</span></div>
    <div class="tile"><b>${int(info.demo.n_emitters)}</b><span>simulated emissions (ground truth)</span></div>
    <div class="tile"><b>${info.options.filter.length} / ${info.options.fitter.length}</b><span>filters / PSF fitters in the package</span></div>
  </div>
  ${isStatic() ? `<div class="callout">This is the static site. The pipeline runs <b>in your browser</b>: a JavaScript port of the
    package's filter, detector and least-squares Gaussian fitter (including scipy's MINPACK Levenberg-Marquardt), checked
    against the Python output in the repo's tests. Your own TIFF files, and the other filters and fitters, need the local
    app (see About).</div>` : `<div class="callout a">Local app: every analysis runs the package's Python pipeline on this computer.
    Upload a TIFF stack on the Analyse page.</div>`}
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/analyze"><b>Analyse</b><span>Choose the filter (default ${d.filter_type}, scale ${d.filter_params.scale}), threshold
      (<code>${d.threshold_expression}</code>), detector and fitter, run, and step through the frames with every detection marked.</span></a>
    <a class="card" href="#/image"><b>Super-resolution image</b><span>Gaussian rendering of the localisations at a chosen nm/pixel, beside the
      diffraction-limited mean of the frames.</span></a>
    <a class="card" href="#/table"><b>Localisations</b><span>Sortable table of positions (nm), PSF width, photons and uncertainty, and a CSV
      export in thunderSTORM's column format.</span></a>
    <a class="card" href="#/validate"><b>Validation</b><span>Recall, precision and RMSE against the simulation's ground truth, and the browser
      port compared localisation by localisation with the Python pipeline.</span></a>
    <a class="card" href="#/method"><b>Method</b><span>Each step of the pipeline with its formulas, units and coordinate conventions.</span></a>
    <a class="card" href="#/about"><b>About</b><span>What this is, how to run the Python package and the web app, and credits.</span></a>
  </div>
  <h2>The demo movie</h2>
  <div class="panel"><p><span class="badge sim">simulated</span> Made by <code>simulation.SMLMSimulator</code> with seed ${s.seed}:
    ${int(s.n_molecules)} molecules placed on a ${s.pattern.replace('_', ' ')} mask, blinking (p<sub>on</sub> ${s.p_on}, p<sub>off</sub> ${s.p_off},
    p<sub>bleach</sub> ${s.p_bleach} per frame), ${s.photons_per_molecule} photons each with a ${s.psf_sigma} nm Gaussian PSF on a background of
    ${s.background_photons} photons/pixel, Poisson noise. It is not real data.</p></div>`;
});
