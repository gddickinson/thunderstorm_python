"""
Regression tests: renderers return (rows, cols) = (height, width) images for
non-square fields of view, with each localization drawn at its own position.
image_size is documented as (width, height).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.visualization import create_renderer

LOCS = {'x': np.array([400.0, 50.0]), 'y': np.array([100.0, 30.0]),
        'uncertainty': np.array([10.0, 10.0])}


@pytest.mark.parametrize('kind', ['gaussian', 'histogram', 'ash', 'scatter'])
def test_non_square_shape_and_peak(kind):
    img = create_renderer(kind).render(LOCS, pixel_size=10, image_size=(60, 20))
    assert img.shape == (20, 60)
    row, col = np.unravel_index(np.argmax(img * (np.arange(60) > 20)), img.shape)
    assert (row, col) in [(10, 40), (9, 39), (10, 39), (9, 40)]


@pytest.mark.parametrize('kind', ['gaussian', 'histogram', 'ash', 'scatter'])
def test_auto_size_wide_field(kind):
    img = create_renderer(kind).render(LOCS, pixel_size=10)
    assert img.shape == (int(100 / 10) + 10, int(400 / 10) + 10)


def test_computed_sigma_skips_failed_fits():
    """Regression: a failed fit has uncertainty inf, and int(3 * inf) raised OverflowError."""
    locs = {'x': np.array([400.0, 200.0]), 'y': np.array([100.0, 100.0]),
            'uncertainty': np.array([10.0, np.inf])}
    img = create_renderer('gaussian', sigma='computed').render(locs, pixel_size=10, image_size=(60, 20))
    assert np.isfinite(img).all() and img[10, 40] == pytest.approx(1.0) and img[10, 20] == 0
