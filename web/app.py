"""FastAPI web app for ThunderSTORM Python: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8210, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

The analysis is the package's own ThunderSTORM pipeline (web/analysis.py); the demo movie comes from its own
simulator (web/demo.py). On the static site the same steps run in the browser (static/ts_engine.js, node-tested
against this Python code).
"""
import json
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .analysis import AnalysisError, defaults, options, read_tiff, run_pipeline
from .demo import CAMERA, SIM, evaluate, movie_json, simulate

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "ThunderSTORM Python"
MAX_UPLOAD = 1024 * 1024 * 1024

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Recent uploaded stacks (for the frame viewer), local app only; a small LRU.
RUNS: "OrderedDict[str, np.ndarray]" = OrderedDict()
MAX_RUNS = 3


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


def _jsonable(d):
    return json.loads(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


@app.get("/api/info")
def info():
    """Project summary: pipeline defaults, the filter/detector/fitter choices, and the demo movie's settings."""
    movie, truth, _ = simulate()
    return _jsonable({
        "title": TITLE,
        "version": __version__,
        "defaults": defaults(),
        "options": options(),
        "demo": {"sim": SIM, "camera": CAMERA, "shape": list(movie.shape),
                 "n_emitters": int(sum(len(g["x"]) for g in truth))},
    })


@app.get("/api/demo/movie")
def demo_movie():
    """The simulated demo movie (uint16 base64) with its ground truth."""
    return movie_json()


@app.get("/api/demo/result")
def demo_result():
    """The demo movie analysed by the Python pipeline with the defaults and the demo's camera settings."""
    movie, _, _ = simulate()
    res = run_pipeline(movie, {**_jsonable(defaults()), **CAMERA})
    res["evaluation"] = evaluate(res["locs"])
    res["source"] = "demo"
    return res


@app.post("/api/analyze")
async def analyze(params: str = Form(...), file: UploadFile | None = File(None)):
    """Analyse the demo movie or an uploaded TIFF stack with the Python pipeline.

    params is JSON in the shape of /api/info "defaults" (plus use_file_pixel_size). The pixel size actually used, and
    where it came from, is returned with the result."""
    try:
        p = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(400, "params must be JSON")
    try:
        if file is not None and file.filename:
            data = await file.read()
            if len(data) > MAX_UPLOAD:
                raise AnalysisError("file too large for the web app (1 GB)")
            movie, file_px = read_tiff(data)
            source = "user"
            if file_px and p.get("use_file_pixel_size", True):
                p["pixel_size"], source = round(file_px, 6), "file metadata"
            res = run_pipeline(movie, p)
            res["pixel_size_source"] = source
            res["file_pixel_size_nm"] = file_px
            res["source"] = file.filename
            run_id = uuid.uuid4().hex[:12]
            RUNS[run_id] = movie.astype(np.float32)
            while len(RUNS) > MAX_RUNS:
                RUNS.popitem(last=False)
            res["run_id"] = run_id
        else:
            movie, _, _ = simulate()
            res = run_pipeline(movie, {**p, **CAMERA})
            res["evaluation"] = evaluate(res["locs"])
            res["source"] = "demo"
            res["pixel_size_source"] = "simulation"
    except AnalysisError as e:
        raise HTTPException(400, str(e))
    return res


@app.get("/api/runs/{run_id}/frame/{index}")
def run_frame(run_id: str, index: int):
    """One frame of an uploaded stack as little-endian float32 (base64), for the frame viewer."""
    import base64
    if run_id not in RUNS:
        raise HTTPException(404, "run not found (the local app keeps the last few uploads only)")
    movie = RUNS[run_id]
    if not 0 <= index < movie.shape[0]:
        raise HTTPException(404, "frame out of range")
    f = movie[index]
    return {"index": index, "shape": list(f.shape), "dtype": "float32",
            "b64": base64.b64encode(f.astype("<f4").tobytes()).decode("ascii")}


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/demo/movie", "/demo/result"]
