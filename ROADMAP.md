# ThunderSTORM Python -- Roadmap

## Current State
A comprehensive Python reimplementation of the thunderSTORM ImageJ plugin for
SMLM data analysis. Well-organized into 10 modules: `filters.py`, `detection.py`,
`fitting.py`, `postprocessing.py`, `visualization.py`, `simulation.py`, `utils.py`,
`pipeline.py`, `examples.py`, and `__init__.py`. The `ThunderSTORM` pipeline class
provides a high-level API matching the original plugin's workflow. Supports wavelet/
Gaussian/DoG filtering, multiple PSF fitters (LSQ, MLE, radial symmetry), drift
correction, molecule merging, and multiple renderers. No tests, no `pyproject.toml`,
no CLI entry point -- library-only usage.

## Short-term Improvements
- [x] Add unit tests for each module: verify filter outputs, detection counts on
      synthetic data, fitting accuracy (RMSE < 1 pixel on noiseless Gaussians)
- [x] Add a `pyproject.toml` for packaging with `pip install -e .`
- [ ] Add a CLI entry point: `python -m thunderstorm_python --input data.tif
      --filter wavelet --fitter gaussian_lsq --output results.csv`
- [ ] Add type hints throughout (especially `pipeline.py`, `fitting.py`)
- [ ] Add input validation: check image dtype, dimensions, NaN values before
      processing
- [ ] Add proper error messages when optional dependencies (librosa, etc.) are
      missing
- [x] Move `examples.py` to a `docs/examples/` directory with Jupyter notebooks

## Feature Enhancements
- [ ] Add CRLB (Cramer-Rao Lower Bound) calculation for theoretical precision
      estimation per localization
- [ ] Implement multi-emitter fitting: detect and fit overlapping PSFs using
      model selection (BIC/AIC)
- [ ] Add GPU acceleration for MLE fitting using CuPy or JAX
- [ ] Implement thunderSTORM's calibration curve fitting for 3D astigmatism
      (currently users must supply their own Z lookup)
- [ ] Add FRC (Fourier Ring Correlation) resolution estimation
- [ ] Support TIFF stack lazy loading for memory-efficient processing of
      large datasets (10k+ frames)
- [ ] Add a minimal GUI (Qt or napari plugin) for interactive parameter tuning
- [ ] Implement the remaining thunderSTORM filters not yet ported (if any):
      verify parity with the original plugin

## Long-term Vision
- [ ] napari plugin: integrate as a napari dock widget for seamless use in the
      Python microscopy ecosystem
- [ ] Distributed processing: split large stacks across multiple cores/machines
      using Dask
- [ ] Deep learning fitter: train a neural network to predict molecule positions
      directly from raw frames (DECODE-style)
- [ ] Live analysis mode: process frames as they arrive from the camera during
      acquisition
- [ ] Benchmarking suite: compare performance against the original Java
      thunderSTORM, SMAP, picasso, and other SMLM packages
- [ ] Publication: write a methods paper comparing this implementation to the
      original on standard SMLM challenge datasets

## Technical Debt
- [ ] Individual module files may be large -- audit `fitting.py` and
      `postprocessing.py` for size and split if over 500 lines
- [x] `examples.py` is a module inside the package but contains usage demos --
      it should not be importable as part of the library
- [x] No `requirements.txt` or dependency specification file
- [ ] The threshold expression parser (e.g., `std(Wave.F1)`) needs robust
      parsing -- current implementation may be fragile with complex expressions
- [ ] Drift correction and molecule merging in `postprocessing.py` may share
      spatial indexing logic -- extract into a shared utility
- [x] No `.gitignore` -- generated output files and `__pycache__` may be tracked
- [x] The `simulation.py` module bundles data generation with performance
      evaluation -- split into `simulation.py` and `evaluation.py`
