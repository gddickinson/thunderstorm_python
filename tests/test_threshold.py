"""
Regression tests for threshold expressions such as the default 'std(Wave.F1)'.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.filters import WaveletFilter, compute_threshold_expression
from thunderstorm_python.pipeline import ThunderSTORM


@pytest.fixture
def image():
    rng = np.random.default_rng(3)
    img = rng.poisson(10, (40, 40)).astype(float)
    img[20, 20] += 500
    return img


def test_default_expression_evaluates(image):
    """'std(Wave.F1)' (the pipeline default) used to raise NameError: 'Wave'."""
    filtered = WaveletFilter(scale=2, order=3).apply(image)
    thr = compute_threshold_expression(image, filtered, 'std(Wave.F1)')
    # Wave.F1 is the first wavelet level of the raw image, as documented.
    assert thr == pytest.approx(np.std(WaveletFilter(scale=1, order=3).apply(image)))


def test_compound_expressions(image):
    filtered = WaveletFilter().apply(image)
    f1 = WaveletFilter(scale=1).apply(image)
    assert compute_threshold_expression(image, filtered, '2*std(Wave.F1)') == pytest.approx(2 * np.std(f1))
    assert compute_threshold_expression(image, filtered, 'mean(I1) + 3*std(I1)') == pytest.approx(
        image.mean() + 3 * image.std())
    assert compute_threshold_expression(image, filtered, 7) == 7.0
    with pytest.raises(ValueError):
        compute_threshold_expression(image, filtered, 'std(Nope.F9)')


def test_default_pipeline_runs(image):
    result = ThunderSTORM().analyze_frame(image)
    assert len(result['x']) >= 1
