# thunderstorm_python

Python reimplementation of the thunderSTORM ImageJ plugin (SMLM analysis) plus a web interface (`web/`).

@INTERFACE.md

- The repo folder is the package: keep it named `thunderstorm_python`; tests import `thunderstorm_python.*`.
- Units: pipeline x / y / uncertainty in nm with a pixel-centre origin; sigma in camera pixels. The camera pixel
  size is always explicit. thunderSTORM CSV export (`thunderstorm_format.py`) uses corner origin and 1-based frames.
- The JS port in `web/static/ts_*.js` must stay in step with the Python code: after changing filters, detection,
  fitting or rendering, run `python scripts/make_engine_fixture.py` and `node web/static/ts_engine.test.mjs`.
- Tests: `python -m pytest -q tests` (0 failures, 0 errors).
