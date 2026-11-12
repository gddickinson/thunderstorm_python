"""
Regression test: the pipeline reports x/y in nm (as the CSV header 'x [nm]',
the renderers, merging, drift correction and the evaluator all assume), not
in camera pixels.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.pipeline import ThunderSTORM
from thunderstorm_python.simulation import SMLMSimulator
from thunderstorm_python.evaluation import PerformanceEvaluator


def test_positions_are_in_nm_and_match_ground_truth():
    np.random.seed(7)
    sim = SMLMSimulator(image_size=(32, 32), pixel_size=100.0, psf_sigma=130.0,
                        photons_per_molecule=3000, background_photons=10)
    gt_xy = np.array([[1234.0, 1678.0], [2210.0, 820.0]])
    frame, gt = sim.render_frame(gt_xy)
    pipe = ThunderSTORM(pixel_size=100.0, baseline=0.0)
    locs = pipe.analyze_frame(frame)
    assert len(locs['x']) == 2
    order = np.argsort(locs['x'])
    np.testing.assert_allclose(locs['x'][order], np.sort(gt_xy[:, 0]), atol=30)
    np.testing.assert_allclose(
        locs['y'][order], gt_xy[np.argsort(gt_xy[:, 0]), 1], atol=30)
    metrics = PerformanceEvaluator(tolerance=50.0).evaluate(locs, gt)
    assert metrics['recall'] == 1.0 and metrics['precision'] == 1.0


def test_pixel_size_scales_positions():
    np.random.seed(8)
    sim = SMLMSimulator(image_size=(24, 24), pixel_size=100.0, psf_sigma=130.0,
                        photons_per_molecule=3000)
    frame, _ = sim.render_frame(np.array([[1150.0, 1150.0]]))
    a = ThunderSTORM(pixel_size=100.0).analyze_frame(frame)
    b = ThunderSTORM(pixel_size=160.0).analyze_frame(frame)
    np.testing.assert_allclose(b['x'] / 160.0, a['x'] / 100.0, rtol=1e-9)


def test_readme_quick_start_imports():
    """README: `from thunderstorm_python import quick_analysis` used to raise ImportError."""
    from thunderstorm_python import create_default_pipeline, quick_analysis
    from thunderstorm_python.pipeline import quick_analysis as q2
    assert quick_analysis is q2 and create_default_pipeline().pixel_size == 100.0


def test_pixel_centre_origin_documented_and_exact():
    """A spot centred on pixel (row 5, col 7) is reported at (700, 500) nm: pixel-centre origin."""
    import thunderstorm_python.pipeline as pl
    assert 'pixel-centre origin' in pl.__doc__
    yy, xx = np.mgrid[0:15, 0:15]
    img = 10 + 2000 * np.exp(-((xx - 7) ** 2 + (yy - 5) ** 2) / (2 * 1.3 ** 2))
    locs = ThunderSTORM(pixel_size=100.0, threshold_expression=50.0).analyze_frame(img)
    np.testing.assert_allclose([locs['x'][0], locs['y'][0]], [700.0, 500.0], atol=1e-3)
