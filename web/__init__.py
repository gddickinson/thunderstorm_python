"""Web interface for ThunderSTORM Python (FastAPI app + static front end).

Contents:
    app.py       routes under /api, STATIC_ENDPOINTS for the static GitHub Pages build
    analysis.py  options/defaults read from the package registries, run_pipeline, TIFF reading (pixel size metadata)
    demo.py      the simulated demo movie (SMLMSimulator) and its ground-truth evaluation
    static/      the front end (ES modules, no build step); ts_engine.js is the in-browser pipeline port
"""
