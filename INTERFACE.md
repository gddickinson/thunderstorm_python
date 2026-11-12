# INTERFACE.md: navigation map

The repository root **is** the `thunderstorm_python` package (flat layout; `pyproject.toml` maps it), so the folder
must be named `thunderstorm_python` and its parent must be on `sys.path` (tests, `run_web.py` and the scripts do this).

## Package modules (analysis)

| File | Contents |
|---|---|
| `__init__.py` | Re-exports the submodules and `ThunderSTORM`. |
| `pipeline.py` | `ThunderSTORM` (filter -> threshold -> detect -> fit per frame; `analyze_frame`, `analyze_stack`, drift correction, merging, filtering, `render`, `save`), `create_default_pipeline`, `quick_analysis`. Output x / y / uncertainty in nm (pixel-centre origin), sigma in camera pixels. |
| `filters.py` | `WaveletFilter` (a trous B-spline), `GaussianFilter`, `DifferenceOfGaussians`, `LoweredGaussian`, `DifferenceOfAveraging`, `MedianFilter`, `BoxFilter`, `NoFilter`; registry `FILTER_TYPES`, `create_filter`; `compute_threshold_expression` (`std(Wave.F1)` etc.). |
| `detection.py` | `LocalMaximumDetector` (skimage `peak_local_max`), `NonMaximumSuppression`, `CentroidDetector`, `GridDetector`; registry `DETECTOR_TYPES`, `create_detector`; helpers. |
| `fitting_base.py` | `LocalizationResult`, `BaseFitter` (camera parameters), `gaussian_2d`, `integrated_gaussian_2d`. |
| `fitting.py` | `GaussianLSQFitter` (curve_fit + Thompson uncertainty), `GaussianMLEFitter`, `RadialSymmetryFitter`, `CentroidFitter`; registry `FITTER_TYPES`, `create_fitter`; re-exports `fitting_base`. |
| `postprocessing.py` | `DriftCorrector`, `MolecularMerger`, `LocalizationFilter`, `LocalDensityFilter`, duplicate removal, z-stage offset. (573 lines: over the 500-line budget, not yet split.) |
| `visualization.py` | `GaussianRenderer`, `HistogramRenderer`, `AverageShiftedHistogram`, `ScatterRenderer`, `create_renderer`, 3D projection, colormap. Images are (height, width). |
| `simulation.py` | `SMLMSimulator` (positions from a mask, blinking, Gaussian PSF + Poisson noise, `generate_movie`). |
| `evaluation.py` | `PerformanceEvaluator` (recall / precision / RMSE vs ground truth), `create_test_pattern`. |
| `utils.py` | CSV / TIFF / npy I/O, statistics, nearest-neighbour and Ripley's K, ROI, `LocalizationTable`. |
| `thunderstorm_format.py` | `to_thunderstorm_columns`, `save_thunderstorm_csv`: thunderSTORM column layout (1-based frame, corner origin, photons). |
| `docs/examples/examples.py` | Usage demos (not part of the library). |

## Web interface (`web/`)

| File | Contents |
|---|---|
| `web/app.py` | FastAPI app: `/api/info`, `/api/demo/movie`, `/api/demo/result`, `POST /api/analyze` (demo or TIFF upload), `/api/runs/{id}/frame/{i}`; `STATIC_ENDPOINTS`. |
| `web/analysis.py` | `options()` / `defaults()` read from the registries and constructor signatures; `parse_params`, `run_pipeline`, `read_tiff` (pixel size from OME / ImageJ metadata). |
| `web/demo.py` | The seeded demo movie from `SMLMSimulator` (`simulate`, `movie_json`) and `evaluate` against its ground truth. |
| `web/static/index.html`, `style.css` (kit theme), `style_app.css`, `core.js` (kit router / api) | Shell. |
| `web/static/main.js`, `state.js`, `static_api.js`, `fmt.js` | Entry point, shared state (movie decoding, current result, `run`), static-site data layer (runs the JS port), helpers. |
| `web/static/home.js`, `analyze.js`, `image.js`, `table.js`, `validate.js`, `docs.js` | Pages: home, analyse + frame viewer, SR image, localisation table + CSV, validation, method + about. |
| `web/static/ts_engine.js` | JS port: filters (wavelet, Gaussian, DoG, none), threshold-expression parser, local-maximum and NMS detectors. |
| `web/static/ts_fit.js` | JS port: Cephes `erf`, LSQ Gaussian fit, Thompson uncertainty, `analyzeFrame` / `analyzeMovie`. |
| `web/static/ts_minpack.js` | Port of MINPACK `lmdif` (scipy `curve_fit`'s optimiser). |
| `web/static/ts_render.js` | Gaussian rendering, `evaluateFrame` / `evaluateMovie`, thunderSTORM columns and CSV, colour map. |
| `web/static/ts_engine.test.mjs` | Node test of the JS port against `tests/fixtures/engine_cases.json`. |

## Scripts and tests

| File | Contents |
|---|---|
| `run_web.py` | Starts the web app on 127.0.0.1:8210. |
| `scripts/build_static.py` | Exports the static GitHub Pages site (`site/`). |
| `scripts/make_engine_fixture.py` | Writes `tests/fixtures/engine_cases.json` from the Python pipeline. |
| `scripts/screenshots.py` | Playwright smoke test / README screenshots (`docs/img/`). |
| `tests/test_*.py` | Unit tests per module, regression tests (`test_packaging`, `test_threshold`, `test_pipeline_units`, `test_rendering`), `test_thunderstorm_format`, `test_web` (API, static build, node test, fixture current). |

## How the pieces connect

`ThunderSTORM` composes one filter, one detector and one fitter from the registries; `web/analysis.py` builds the
same pipeline from browser parameters; the static site replaces the server with `static_api.js`, which runs
`ts_fit.analyzeMovie` (the JS port) on the demo movie exported by `web/demo.py`.
