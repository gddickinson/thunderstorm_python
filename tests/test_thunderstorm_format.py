"""
Tests for the thunderSTORM-format export (frame 1-based, corner origin, photons).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.thunderstorm_format import save_thunderstorm_csv, to_thunderstorm_columns

LOCS = {'x': np.array([0.0, 1234.5]), 'y': np.array([50.0, 10.0]), 'frame': np.array([0, 4]),
        'sigma_x': np.array([1.5, 1.2]), 'sigma_y': np.array([1.5, 1.2]),
        'intensity': np.array([1000.0, 500.0]), 'background': np.array([110.0, 120.0]),
        'uncertainty': np.array([10.0, 20.0])}


def test_columns_and_conventions():
    c = to_thunderstorm_columns(LOCS, pixel_size=100.0, photons_per_adu=0.5, baseline=100.0)
    assert list(c) == ['id', 'frame', 'x [nm]', 'y [nm]', 'sigma [nm]', 'intensity [photon]',
                       'offset [photon]', 'uncertainty [nm]']
    np.testing.assert_array_equal(c['frame'], [1, 5])
    np.testing.assert_allclose(c['x [nm]'], [50.0, 1284.5])      # corner origin
    np.testing.assert_allclose(c['sigma [nm]'], [150.0, 120.0])
    np.testing.assert_allclose(c['intensity [photon]'], [500.0, 250.0])
    np.testing.assert_allclose(c['offset [photon]'], [5.0, 10.0])


def test_elliptical_and_mle(tmp_path):
    locs = dict(LOCS, sigma_y=np.array([1.6, 1.1]))
    c = to_thunderstorm_columns(locs, 100.0, photons_per_adu=0.5, baseline=100.0, fitter_type='gaussian_mle')
    assert 'sigma1 [nm]' in c and 'sigma2 [nm]' in c and 'sigma [nm]' not in c
    np.testing.assert_allclose(c['intensity [photon]'], LOCS['intensity'])   # MLE already photons
    out = tmp_path / 'locs.csv'
    save_thunderstorm_csv(locs, out, pixel_size=100.0)
    df = pd.read_csv(out)
    assert df.columns[2] == 'x [nm]' and len(df) == 2
