"""The demo movie: a small simulated SMLM acquisition made by the repo's own simulator (simulation.SMLMSimulator).

Molecules are placed on a Siemens-star mask (evaluation.create_test_pattern), blink with
SMLMSimulator.simulate_blinking and are rendered frame by frame with SMLMSimulator.render_frame (Gaussian PSF,
Poisson shot noise). The global NumPy RNG is seeded, so the movie is the same on every build; the RNG state is
restored afterwards. Counts are photons (no camera offset or gain), so the demo is analysed with baseline 0 and
1 photon per ADU.
"""
import base64
from functools import lru_cache

import numpy as np

from ..evaluation import PerformanceEvaluator, create_test_pattern
from ..simulation import SMLMSimulator

SIM = {"seed": 2024, "image_size": 64, "n_frames": 150, "n_molecules": 400, "pattern": "siemens_star",
       "pixel_size": 100.0, "psf_sigma": 150.0, "photons_per_molecule": 1000.0, "background_photons": 10.0,
       "p_on": 0.01, "p_off": 0.3, "p_bleach": 0.01}
# Camera settings that match the simulation (photon counts, no offset).
CAMERA = {"pixel_size": SIM["pixel_size"], "photons_per_adu": 1.0, "baseline": 0.0}
TOLERANCE_NM = PerformanceEvaluator().tolerance


@lru_cache(maxsize=1)
def simulate():
    """Return (movie uint16 (frames, rows, cols), ground truth list of dicts, molecule positions (nm))."""
    s = SIM
    state = np.random.get_state()
    np.random.seed(s["seed"])
    try:
        sim = SMLMSimulator(image_size=(s["image_size"], s["image_size"]), pixel_size=s["pixel_size"],
                            psf_sigma=s["psf_sigma"], photons_per_molecule=s["photons_per_molecule"],
                            background_photons=s["background_photons"])
        mask = create_test_pattern(s["pattern"], size=s["image_size"])
        positions = sim.generate_molecule_positions(n_molecules=s["n_molecules"], mask=mask)
        states = sim.simulate_blinking(s["n_frames"], s["n_molecules"], p_on=s["p_on"], p_off=s["p_off"],
                                       p_bleach=s["p_bleach"])
        frames, truth = [], []
        for f in range(s["n_frames"]):
            img, gt = sim.render_frame(positions, states[f])
            frames.append(img)
            truth.append(gt)
    finally:
        np.random.set_state(state)
    movie = np.array(frames)
    assert movie.max() < 65536 and np.all(movie == np.round(movie))
    return movie.astype(np.uint16), truth, positions


def movie_json():
    """The demo movie for the browser: little-endian uint16 frames (base64) + ground truth + simulation settings."""
    movie, truth, positions = simulate()
    return {
        "title": "Simulated Siemens star (SMLMSimulator)",
        "simulated": True,
        "description": ("Simulated with the repo's own SMLMSimulator: molecules on a Siemens-star mask, blinking "
                        "(simulate_blinking), Gaussian PSF and Poisson shot noise. Not real data."),
        "sim": SIM,
        "camera": CAMERA,
        "shape": list(movie.shape),
        "dtype": "uint16",
        "frames_b64": base64.b64encode(movie.astype("<u2").tobytes()).decode("ascii"),
        "n_molecules": int(len(positions)),
        "truth": {"frame": [i for i, gt in enumerate(truth) for _ in gt["x"]],
                  "x": [round(float(v), 4) for gt in truth for v in gt["x"]],
                  "y": [round(float(v), 4) for gt in truth for v in gt["y"]],
                  "photons": [int(v) for gt in truth for v in gt["photons"]]},
        "tolerance_nm": TOLERANCE_NM,
        "units": {"x": "nm", "y": "nm", "origin": "pixel centre (pixel (0,0) centre at 0 nm)"},
    }


def evaluate(locs):
    """Per-frame PerformanceEvaluator.evaluate against the demo ground truth, summed over frames.

    Counts (TP, FP, FN) are summed; recall = TP / ground-truth emitters, precision = TP / localisations (as in
    evaluate()); RMSE is pooled over all matched pairs (sum of squared errors / TP)."""
    _, truth, _ = simulate()
    ev = PerformanceEvaluator()
    frame = np.asarray(locs["frame"])
    x, y = np.asarray(locs["x"], float), np.asarray(locs["y"], float)
    tp = fp = fn = n_det = n_gt = 0
    se = 0.0
    for i, gt in enumerate(truth):
        k = frame == i
        n_det += int(k.sum())
        n_gt += len(gt["x"])
        if len(gt["x"]) == 0:
            fp += int(k.sum())
            continue
        m = ev.evaluate({"x": x[k], "y": y[k]}, gt)
        tp += int(m["n_true_positive"]); fp += int(m["n_false_positive"]); fn += int(m["n_false_negative"])
        if m["n_true_positive"] > 0:
            se += float(m["rmse"]) ** 2 * int(m["n_true_positive"])
    return {"tp": tp, "fp": fp, "fn": fn, "n_det": n_det, "n_gt": n_gt, "recall": tp / n_gt if n_gt else 0.0,
            "precision": tp / n_det if n_det else 0.0, "rmse_nm": (se / tp) ** 0.5 if tp else None,
            "tolerance_nm": TOLERANCE_NM}
