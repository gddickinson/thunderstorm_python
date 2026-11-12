"""Run the repo's ThunderSTORM pipeline for the web app and describe its options.

Everything here calls the package's own code: the filter / detector / fitter choices and their default parameters are
read from FILTER_TYPES, DETECTOR_TYPES and FITTER_TYPES (constructor signatures), the pipeline defaults from
ThunderSTORM.__init__ and analyze_stack. Units: positions and uncertainty in nm, PSF sigma in camera pixels, and the
camera pixel size (nm) is always explicit and returned with the result.
"""
import inspect
import io
import math

import numpy as np

from .. import detection, filters, fitting
from ..pipeline import ThunderSTORM

# Parts the in-browser engine (static/ts_engine.js) implements; the rest need the local app.
JS_SUPPORTED = {"filter": ["wavelet", "gaussian", "dog", "none"],
                "detector": ["local_maximum", "non_maximum_suppression"],
                "fitter": ["gaussian_lsq"]}
# Constructor arguments that have no effect in the current code, so the UI does not offer them (none at present).
INERT = set()
# Text parameters with a fixed set of values (shown as a select).
CHOICES = {("detector", "local_maximum", "connectivity"): list(detection.LocalMaximumDetector.CONNECTIVITIES)}
REGISTRIES = {"filter": filters.FILTER_TYPES, "detector": detection.DETECTOR_TYPES, "fitter": fitting.FITTER_TYPES}
MAX_FRAMES = 5000


class AnalysisError(ValueError):
    """Bad parameters or input data (shown to the user as a 400)."""


def _defaults(func, skip=("self",)):
    return {k: p.default for k, p in inspect.signature(func).parameters.items()
            if k not in skip and p.default is not inspect.Parameter.empty}


def options():
    """Choices and their parameters, read from the package's registries and constructor signatures."""
    out = {}
    for kind, reg in REGISTRIES.items():
        out[kind] = []
        for name, cls in reg.items():
            params = [{"name": k, "default": v, "type": type(v).__name__, "choices": CHOICES.get((kind, name, k))}
                      for k, v in _defaults(cls.__init__).items() if (kind, name, k) not in INERT]
            doc = (inspect.getdoc(cls) or "").split("\n")[0]
            out[kind].append({"id": name, "label": doc or name, "params": params, "js": name in JS_SUPPORTED[kind]})
    return out


def defaults():
    """Pipeline defaults (ThunderSTORM.__init__ + analyze_stack) with each component's own parameter defaults."""
    d = _defaults(ThunderSTORM.__init__)
    d["fit_radius"] = _defaults(ThunderSTORM.analyze_stack)["fit_radius"]
    for kind in REGISTRIES:
        cls = REGISTRIES[kind][d[f"{kind}_type"]]
        d[f"{kind}_params"] = {k: v for k, v in _defaults(cls.__init__).items() if (kind, d[f"{kind}_type"], k) not in INERT}
    return d


def _num(v, name, lo=None, hi=None, integer=False):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise AnalysisError(f"{name} must be a number")
    if not math.isfinite(x) or (lo is not None and x < lo) or (hi is not None and x > hi):
        raise AnalysisError(f"{name} must be between {lo} and {hi}")
    if integer:
        if x != int(x):
            raise AnalysisError(f"{name} must be a whole number")
        return int(x)
    return x


def parse_params(p):
    """Validate a params dict from the browser (same shape as defaults()) into ThunderSTORM kwargs + fit_radius."""
    d = defaults()
    kw = {}
    for kind in REGISTRIES:
        t = str(p.get(f"{kind}_type", d[f"{kind}_type"]))
        if t not in REGISTRIES[kind]:
            raise AnalysisError(f"unknown {kind} '{t}'")
        allowed = {k: v for k, v in _defaults(REGISTRIES[kind][t].__init__).items() if (kind, t, k) not in INERT}
        given = p.get(f"{kind}_params") or {}
        params = {}
        for k, dv in allowed.items():
            v = given.get(k, dv)
            if isinstance(dv, bool):
                params[k] = v in (True, "true", "1", 1, "on")
            elif isinstance(dv, int):
                params[k] = _num(v, k, 0, 50, integer=True)
            elif isinstance(dv, float):
                params[k] = _num(v, k, 0, 1e6)
            else:
                params[k] = str(v)
                if (kind, t, k) in CHOICES and params[k] not in CHOICES[(kind, t, k)]:
                    raise AnalysisError(f"{k} must be one of {CHOICES[(kind, t, k)]}")
        kw[f"{kind}_type"], kw[f"{kind}_params"] = t, params
    thr = p.get("threshold_expression", d["threshold_expression"])
    if isinstance(thr, str) and len(thr) > 120:
        raise AnalysisError("threshold expression is too long")
    kw["threshold_expression"] = thr
    kw["pixel_size"] = _num(p.get("pixel_size", d["pixel_size"]), "pixel size (nm)", 1, 10000)
    kw["photons_per_adu"] = _num(p.get("photons_per_adu", d["photons_per_adu"]), "photons per ADU", 1e-6, 1e6)
    kw["baseline"] = _num(p.get("baseline", d["baseline"]), "baseline (ADU)", 0, 1e6)
    kw["em_gain"] = _num(p.get("em_gain", d["em_gain"]), "EM gain", 1, 1e5)
    fit_radius = _num(p.get("fit_radius", d["fit_radius"]), "fit radius", 1, 10, integer=True)
    return kw, fit_radius


def _clean(a):
    """Array -> list with non-finite values as None (JSON has no inf/nan)."""
    return [float(v) if math.isfinite(v) else None for v in np.asarray(a, float)]


def run_pipeline(movie, params):
    """Analyse a (frames, rows, cols) stack with the real pipeline; returns a JSON-ready result."""
    kw, fit_radius = parse_params(params)
    try:
        pipe = ThunderSTORM(**kw)
        locs = pipe.analyze_stack(np.asarray(movie, float), fit_radius=fit_radius, show_progress=False)
    except ValueError as e:  # e.g. an invalid threshold expression
        raise AnalysisError(str(e))
    keys = ["frame", "x", "y", "sigma_x", "sigma_y", "intensity", "background", "uncertainty", "chi_squared"]
    n = 0 if locs.get("x") is None else len(locs["x"])
    cols = {k: ([] if n == 0 else (_clean(locs[k]) if k != "frame" else [int(v) for v in locs[k]])) for k in keys}
    return {
        "engine": "python",
        "params": {**{k: v for k, v in kw.items()}, "fit_radius": fit_radius},
        "pixel_size_nm": kw["pixel_size"],
        "shape": list(np.asarray(movie).shape),
        "n": n,
        "units": {"x": "nm", "y": "nm", "uncertainty": "nm", "sigma_x": "px", "sigma_y": "px",
                  "origin": "pixel centre (pixel (0,0) centre at 0 nm)"},
        "locs": cols,
    }


def _tiff_pixel_size_nm(tif):
    """Physical pixel size from OME or ImageJ/TIFF resolution metadata, in nm, or None."""
    try:
        if tif.ome_metadata:
            import re
            m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata)
            if m:
                unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', tif.ome_metadata)
                scale = {"nm": 1.0, "µm": 1000.0, "um": 1000.0, "mm": 1e6}.get(unit.group(1) if unit else "µm")
                if scale:
                    return float(m.group(1)) * scale
        page = tif.pages[0]
        ij = tif.imagej_metadata or {}
        res = page.tags.get("XResolution")
        if res and ij.get("unit") in ("micron", "µm", "um", "\\u00B5m"):
            num, den = res.value
            if num:
                return den / num * 1000.0
    except Exception:  # metadata is optional; fall back to the user's value
        return None
    return None


def read_tiff(data: bytes):
    """Read an uploaded TIFF into (frames, rows, cols) float array + the file's pixel size (nm) if it states one."""
    import tifffile
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as tif:
            arr = tif.asarray()
            px = _tiff_pixel_size_nm(tif)
    except Exception as e:
        raise AnalysisError(f"could not read the TIFF: {e}")
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise AnalysisError(f"expected a single-channel 2D image or (frames, rows, cols) stack, got shape {arr.shape}")
    if arr.shape[0] > MAX_FRAMES:
        raise AnalysisError(f"{arr.shape[0]} frames: the web app analyses at most {MAX_FRAMES} (use the Python API)")
    if min(arr.shape[1:]) < 8:
        raise AnalysisError("frames must be at least 8 x 8 pixels")
    return arr.astype(float), px
