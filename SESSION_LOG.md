# Session log

## 2026-11-12: web interface, static site, bug fixes

Branch `web-interface` (not pushed).

**Web interface** (`web/`, `run_web.py`, port 8210) and static GitHub Pages site (`scripts/build_static.py`,
`.github/workflows/pages.yml`). Pages: home, Analyse (parameters + frame viewer), Image (Gaussian SR rendering),
Localisations (table + thunderSTORM-format CSV), Validation (ground truth, JS vs Python), Method, About.
Demo data: a seeded 64x64 px, 150-frame movie from the package's own `SMLMSimulator` (Siemens-star mask), labelled
as simulated. The static site runs a JS port of the pipeline (`web/static/ts_*.js`, including MINPACK lmdif);
`ts_engine.test.mjs` checks filtered images, thresholds and detections exactly and fits within 0.02 nm / 1e-3
relative against `tests/fixtures/engine_cases.json`. On the full demo: 1,492 / 1,499 localisations within 0.01 nm
(median 2e-7 nm); 7 ill-conditioned fits differ. The local app runs the Python pipeline on uploaded TIFFs, taking the
pixel size from OME / ImageJ metadata when present and reporting which value was used.

Also added: `thunderstorm_format.py` (thunderSTORM CSV columns), module-level `FILTER_TYPES` / `DETECTOR_TYPES` /
`FITTER_TYPES`, `fitting.py` split (`fitting_base.py`), INTERFACE.md, CLAUDE.md, .gitignore.

**Bugs fixed** (each with a regression test, separate commits):
1. Package did not import: unused `import pywt` (and `pywt` is not the PyPI name); pandas undeclared; invalid
   build backend in pyproject.toml; flat layout not mapped.
2. Default threshold `std(Wave.F1)` raised NameError (dotted name in `eval`); now level-1 wavelet of the raw frame.
3. Pipeline returned x / y in camera pixels while CSV, renderers, merging, drift and evaluation assume nm.
4. Renderers allocated (width, height) but indexed [row, col]: non-square fields crashed.
5. Simulator applied Poisson noise to the background twice (variance 2x).
6. Wavelet filter zero-padded the edges: ~3/4 of false detections with the default threshold were at the border.
7. `GaussianRenderer(sigma='computed')` crashed on failed fits (int(inf)).
8. README said GPL-3.0; LICENSE / pyproject / CITATION say MIT. Placeholder clone URL.

**Open questions for George** (not changed): Thompson uncertainty uses the fitted background *level* (including
any camera baseline for LSQ) as b, where Thompson / thunderSTORM use the background noise; `connectivity` of
LocalMaximumDetector and `elliptical` of the Gaussian fitters have no effect; `em_gain` is unused; pipeline keeps a
pixel-centre origin (the export converts to thunderSTORM's corner origin); failed fits stay in the output with
infinite uncertainty; postprocessing.py is over 500 lines.

Tests: `pytest tests` 82 passed; `node web/static/ts_engine.test.mjs` ok; screenshots.py --static and --base all ok,
no overflow at 390 px. Nothing ran on the Mac mini.
