// Method and About pages. Content follows the package's code and README.
import { app, isStatic, route } from './core.js';
import { getInfo } from './state.js';

route('method', async () => {
  const d = (await getInfo()).defaults;
  app().innerHTML = `<p class="kicker">How it works</p><h1>The pipeline, step by step</h1>
  <p class="lead">What <code>ThunderSTORM.analyze_frame</code> does to every frame, with the defaults of this package
    (<code>pipeline.py</code>, <code>filters.py</code>, <code>detection.py</code>, <code>fitting.py</code>).</p>
  <h2>1. Filtering</h2>
  <p>The default filter (${d.filter_type}, scale ${d.filter_params.scale}, B-spline order ${d.filter_params.order}) is the undecimated
    "à trous" wavelet transform. The frame is smoothed with the cubic B-spline kernel, then again with the same kernel dilated by
    inserting 2<sup>j</sup>−1 zeros between its taps; each wavelet level is the difference of two successive smoothings:</p>
  <p class="eq">V₀ = I · Vⱼ = Vⱼ₋₁ ∗ hⱼ · Fⱼ = Vⱼ₋₁ − Vⱼ · h = [1, 4, 6, 4, 1] / 16</p>
  <p>The filter returns F<sub>${d.filter_params.scale}</sub>, which keeps spot-sized features and removes both pixel noise and slow background.
    Edges are padded by repeating the border pixel. Other filters in the package: Gaussian, difference of Gaussians, lowered Gaussian,
    difference of averaging, median, box and none.</p>
  <h2>2. Threshold</h2>
  <p>The detection threshold is an expression evaluated per frame. The default <code>${d.threshold_expression}</code> is the standard deviation of
    the first wavelet level of the raw frame, a noise estimate. Expressions can use <code>I1</code> (raw frame), <code>F1</code> (filtered frame),
    <code>Wave.F1</code>, <code>Wave.F2</code> and the functions std, mean, median, max and min, e.g. <code>2*std(Wave.F1)</code> or a plain number.</p>
  <h2>3. Detection</h2>
  <p>Candidates are local maxima of the filtered frame above the threshold (<code>skimage.feature.peak_local_max</code>, min_distance
    ${d.detector_params.min_distance}: a pixel must equal the maximum of its 3×3 neighbourhood; the outermost pixel row and column are excluded).
    They are taken brightest first. A non-maximum-suppression detector and a connected-component centroid detector are also available.</p>
  <h2>4. PSF fitting</h2>
  <p>Around each candidate a (2r+1)×(2r+1) window of the <b>raw</b> frame (r = ${d.fit_radius} px) is fitted by least squares
    (<code>scipy.optimize.curve_fit</code>, Levenberg-Marquardt) with a Gaussian integrated over each pixel:</p>
  <p class="eq">model(x, y) = A · Eₓ · E_y + b,  Eₓ = ½ [erf((x + ½ − x₀)/(√2 σₓ)) − erf((x − ½ − x₀)/(√2 σₓ))]</p>
  <p>The free parameters are x₀, y₀, amplitude A, the width σ and background b. Start values: the candidate pixel, σ = ${d.fitter_params.initial_sigma} px,
    b = median of the window, A = maximum − b. With <code>elliptical</code> on, σₓ and σ_y are fitted separately (six parameters);
    by default one width is fitted (five). A candidate whose fit does not converge in 1000 evaluations, gives non-finite values or a
    non-positive amplitude is <b>rejected</b>: it is left out of the localisations and listed separately with the reason
    (<code>ThunderSTORM.get_rejected()</code>). Maximum-likelihood (Poisson), radial-symmetry and centroid fitters are also in the package (local app).</p>
  <h2>5. Localisation uncertainty</h2>
  <p>Mortensen et al. (2010) for a Gaussian fitted by least squares (<code>uncertainty.py</code>; the maximum-likelihood fitter uses the
    paper's MLE expression):</p>
  <p class="eq">σ<sub>xy</sub>² = F · (s<sub>a</sub>²/N) · (16/9 + 8π s<sub>a</sub>² b² / (N a²)),  s<sub>a</sub>² = s² + a²/12</p>
  <p>s = PSF sigma in nm, a = pixel size, N = signal photons = amplitude × photons/ADU ÷ EM gain, b² = background photons per pixel
    = (fitted background − baseline) × photons/ADU ÷ EM gain (the Poisson variance of the background), F = 2 for an EMCCD (EM gain &gt; 1),
    otherwise 1.</p>
  <h2>6. Units and coordinates</h2>
  <ul>
    <li>The camera pixel size (nm) is always explicit: the demo states it, and for a TIFF the app uses the file's own value (OME
      PhysicalSizeX or ImageJ resolution) when there is one, otherwise the value you enter, and reports which.</li>
    <li>x and y are in nm with a <b>pixel-centre origin</b>: the centre of camera pixel (row 0, column 0) is (0, 0) nm, as in the simulator.
      PSF sigma stays in camera pixels; uncertainty is in nm.</li>
    <li>The thunderSTORM CSV export uses thunderSTORM's conventions instead: <b>corner origin</b> (x + pixel/2), <b>1-based frames</b>,
      sigma in nm and intensities in photons.</li>
  </ul>
  <h2>7. Rendering</h2>
  <p><code>GaussianRenderer</code> adds, for every localisation, a Gaussian of peak 1 and a fixed sigma (default 20 nm) or the localisation's own
    uncertainty, on a fine grid (default 10 nm per pixel), cut off at 3 sigma.</p>
  <h2>Reference</h2>
  <p>Mortensen K. I., Churchman L. S., Spudich J. A. &amp; Flyvbjerg H. (2010). Optimized localization analysis for single-molecule
    tracking and super-resolution microscopy. <i>Nature Methods</i> 7(5), 377–381. doi:10.1038/nmeth.1447</p>
  <p>Ovesný M., Křížek P., Borkovec J., Švindrych Z. &amp; Hagen G. M. (2014). ThunderSTORM: a comprehensive ImageJ plug-in for PALM and STORM
    data analysis and super-resolution imaging. <i>Bioinformatics</i> 30(16), 2389–2390.</p>`;
});

route('about', () => {
  app().innerHTML = `<p class="kicker">About</p><h1>ThunderSTORM Python</h1>
  <p class="lead">A Python implementation of thunderSTORM for single-molecule localisation microscopy (SMLM): image filtering, molecule
    detection, PSF fitting (least squares, maximum likelihood, radial symmetry, centroid), drift correction, molecule merging,
    quality filtering, rendering and simulation, behind one <code>ThunderSTORM</code> pipeline class.</p>
  <h2>This web interface</h2>
  <ul>
    <li><b>Static site</b> (GitHub Pages): the pipeline runs in your browser on a simulated movie from the package's own
      <code>SMLMSimulator</code>. The JavaScript port covers the wavelet, Gaussian and DoG filters, threshold expressions, the local-maximum
      and non-maximum-suppression detectors and the least-squares Gaussian fitter, and is tested against the Python output.</li>
    <li><b>Local app</b>: the same pages, but every run is the Python pipeline, with all filters, detectors and fitters, on the demo or on
      your own TIFF stack.</li>
    <li>Not in the web interface: drift correction, molecule merging, density filtering and 3D astigmatism; use the Python API for those.</li>
  </ul>
  ${isStatic() ? '<div class="callout">You are on the static site. Analysing your own TIFF files needs the local app.</div>' : ''}
  <h2>Run it</h2>
  <h3>Python package</h3>
  <pre>git clone https://github.com/gddickinson/thunderstorm_python.git
cd thunderstorm_python
pip install -r requirements.txt tifffile
# then, from the parent folder:
python -c "from thunderstorm_python import ThunderSTORM; help(ThunderSTORM)"</pre>
  <h3>Web app</h3>
  <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8210</pre>
  <p class="small mute">The repository folder must be called <code>thunderstorm_python</code> (it is the package itself).</p>
  <h2>Credits</h2>
  <p>Python implementation by George Dickinson (University of California, Irvine), based on thunderSTORM by Ovesný et al. (2014);
    all credit for the algorithms and methodology goes to the original authors. MIT licence (see <code>LICENSE</code>).</p>`;
});
