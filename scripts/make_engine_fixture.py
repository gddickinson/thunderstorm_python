"""Write tests/fixtures/engine_cases.json: inputs and the Python pipeline's outputs that ts_engine.test.mjs checks.

Cases: the first frames of the demo movie (web/demo.py) with the default settings, plus small simulated frames run
through other filter / detector / threshold / model choices; a Gaussian rendering; a ground-truth evaluation; and a
thunderSTORM-format export. Everything is computed by the package's own code.

Run: python scripts/make_engine_fixture.py   (tests/test_web.py checks the stored file is still current)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from thunderstorm_python import filters  # noqa: E402
from thunderstorm_python.evaluation import PerformanceEvaluator  # noqa: E402
from thunderstorm_python.pipeline import ThunderSTORM  # noqa: E402
from thunderstorm_python.simulation import SMLMSimulator  # noqa: E402
from thunderstorm_python.thunderstorm_format import to_thunderstorm_columns  # noqa: E402
from thunderstorm_python.visualization import GaussianRenderer  # noqa: E402
from thunderstorm_python.web.analysis import defaults  # noqa: E402
from thunderstorm_python.web.demo import CAMERA, simulate  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_cases.json"
KEYS = ["frame", "x", "y", "sigma_x", "sigma_y", "intensity", "background", "uncertainty", "chi_squared"]


def small_movie():
    state = np.random.get_state()
    np.random.seed(5)
    try:
        sim = SMLMSimulator(image_size=(24, 30), pixel_size=100.0, psf_sigma=140.0,
                            photons_per_molecule=1500, background_photons=12)
        pos = sim.generate_molecule_positions(n_molecules=14)
        frames = [sim.render_frame(pos, np.random.rand(14) < 0.4)[0] for _ in range(4)]
    finally:
        np.random.set_state(state)
    return np.array(frames)


def run_case(name, movie, params):
    p = dict(params)
    fit_radius = p.pop("fit_radius")
    kw = {k: v for k, v in p.items() if k != "em_gain"}
    pipe = ThunderSTORM(**kw)
    trace = []
    for i, frame in enumerate(movie):
        filt = pipe.filter.apply(frame)
        thr = filters.compute_threshold_expression(frame, filt, pipe.threshold_expression)
        pos = pipe.detector.detect(filt, thr)
        trace.append({"threshold": float(thr), "positions": np.asarray(pos).astype(int).tolist(),
                      "filtered_sum": float(filt.sum()), "filtered_00": float(filt[0, 0]),
                      "filtered_mid": float(filt[filt.shape[0] // 2, filt.shape[1] // 3])})
    locs = pipe.analyze_stack(movie, fit_radius=fit_radius, show_progress=False)
    cols = {k: [None if not np.isfinite(v) else float(v) for v in np.asarray(locs[k], float)] for k in KEYS}
    return {"name": name, "params": {**params}, "shape": list(movie.shape),
            "movie": np.asarray(movie).astype(int).ravel().tolist(), "trace": trace, "locs": cols}


def build():
    warnings.simplefilter("ignore")
    d = defaults()
    demo_movie = simulate()[0][:12].astype(float)
    small = small_movie()
    base = {**d}
    cases = [
        run_case("demo frames 0-11, defaults", demo_movie, {**base, **CAMERA}),
        run_case("gaussian filter, NMS, 3*std(I1)", small, {**base, "filter_type": "gaussian", "filter_params": {"sigma": 1.2},
                 "detector_type": "non_maximum_suppression", "detector_params": {"connectivity": 2},
                 "threshold_expression": "mean(I1) + 3*std(I1) - mean(I1)", "pixel_size": 100.0, "baseline": 0.0}),
        run_case("dog, 4-neighbourhood min_distance 2, plain Gaussian, radius 4", small, {**base, "filter_type": "dog",
                 "filter_params": {"sigma1": 1.0, "sigma2": 1.6},
                 "detector_params": {"min_distance": 2, "connectivity": "4-neighbourhood"},
                 "fitter_params": {"integrated": False, "initial_sigma": 1.5}, "fit_radius": 4,
                 "threshold_expression": "1.5*std(F1)", "pixel_size": 160.0, "photons_per_adu": 0.5}),
        run_case("wavelet, 4-neighbourhood", small, {**base, "detector_params": {"min_distance": 1, "connectivity": "4-neighbourhood"},
                 "threshold_expression": "0.8*std(Wave.F1)", "pixel_size": 100.0, "baseline": 0.0}),
        run_case("wavelet order 1 scale 3, Wave.F2, NMS cross", small, {**base, "filter_params": {"scale": 3, "order": 1},
                 "detector_type": "non_maximum_suppression", "detector_params": {"connectivity": 1},
                 "threshold_expression": "2*std(Wave.F2)", "pixel_size": 107.0}),
    ]
    ref = cases[0]["locs"]
    locs = {k: np.array([np.inf if v is None else v for v in ref[k]]) for k in KEYS}
    render = GaussianRenderer(sigma=20.0).render(locs, pixel_size=25, image_size=(256, 256))
    _, truth, _ = simulate()
    ev = [{k: (None if not np.isfinite(float(v)) else float(v)) for k, v in
           PerformanceEvaluator().evaluate({"x": locs["x"][locs["frame"] == f], "y": locs["y"][locs["frame"] == f]}, truth[f]).items()}
          for f in range(12) if len(truth[f]["x"])]
    gt12 = [{"frame": f, "x": truth[f]["x"].tolist(), "y": truth[f]["y"].tolist()} for f in range(12) if len(truth[f]["x"])]
    ts = to_thunderstorm_columns(locs, pixel_size=100.0, photons_per_adu=0.5, baseline=3.0)
    return {
        "cases": cases,
        "render": {"sigma_nm": 20.0, "pixel_size": 25, "image_size": [256, 256], "sum": float(render.sum()),
                   "max": float(render.max()), "argmax": int(np.argmax(render)),
                   "rows": {str(r): render[r].tolist() for r in (40, 128, 200)}},
        "evaluate": {"truth": gt12, "metrics": ev},
        "export": {"pixel_size": 100.0, "photons_per_adu": 0.5, "baseline": 3.0,
                   "columns": {k: [None if not np.isfinite(float(v)) else float(v) for v in vals] for k, vals in ts.items()}},
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
