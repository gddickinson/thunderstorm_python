"""
Localization uncertainty (Mortensen et al. 2010) and the camera conversion
feeding it: hand-computed cases, baseline subtraction and EM gain.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.fitting import GaussianLSQFitter, GaussianMLEFitter
from thunderstorm_python.uncertainty import localization_uncertainty, mle_integral


def test_lsq_hand_computed():
    # s = 1.3 px * 100 nm = 130 nm; sa^2 = 130^2 + 100^2/12 = 17733.33 nm^2; N = 1000; b^2 = 10
    # sa^2/N = 17.7333; 8*pi*sa^2*b^2/(N*a^2) = 8*pi*0.0177333 = 0.445688; 16/9 + 0.445688 = 2.223466
    # var = 17.7333 * 2.223466 = 39.4295 nm^2 -> 6.2793 nm
    assert localization_uncertainty(1.3, 1000, 10, 100) == pytest.approx(6.2793, abs=1e-4)
    assert localization_uncertainty(1.3, 1000, 10, 100, excess_factor=2) == pytest.approx(6.2793 * np.sqrt(2), abs=1e-4)
    assert localization_uncertainty(1.3, 0, 10, 100) == np.inf
    # negative background (fit below baseline) is treated as zero background
    assert localization_uncertainty(1.3, 1000, -5, 100) == pytest.approx(np.sqrt(17.73333 * 16 / 9), abs=1e-4)


def test_mle_hand_computed():
    # no background: tau = 0 -> var = sa^2 / N = 17.7333 -> 4.2111 nm
    assert localization_uncertainty(1.3, 1000, 0, 100, estimator='mle') == pytest.approx(4.2111, abs=1e-4)
    for tau in (0.05, 1.0, 20.0):
        assert mle_integral(tau) == pytest.approx(quad(lambda t: np.log(t) / (1 + t / tau), 0, 1)[0], rel=1e-9)
    # MLE is never worse than least squares
    assert localization_uncertainty(1.3, 1000, 10, 100, 'mle') < localization_uncertainty(1.3, 1000, 10, 100, 'lsq')


def test_fitter_subtracts_baseline_and_divides_em_gain():
    f = GaussianLSQFitter()
    f.set_camera_params(pixel_size=100, photons_per_adu=0.5, baseline=100, em_gain=1)
    # amplitude 2000 counts -> 1000 photons; background 120 counts -> (120 - 100) * 0.5 = 10 photons
    assert f._compute_uncertainty(1.3, 2000, 120, 100) == pytest.approx(6.2793, abs=1e-4)
    f.set_camera_params(em_gain=50)
    # counts / 50 -> N = 20, b^2 = 0.2, and F = 2
    expected = localization_uncertainty(1.3, 20, 0.2, 100, excess_factor=2)
    assert f._compute_uncertainty(1.3, 2000, 120, 100) == pytest.approx(expected)
    with pytest.raises(ValueError):
        f.set_camera_params(em_gain=0.5)


def test_mle_fitter_uses_mle_formula():
    f = GaussianMLEFitter()
    f.set_camera_params(pixel_size=100)
    assert f._compute_mle_uncertainty(1.3, 1000, 0) == pytest.approx(4.2111, abs=1e-4)
