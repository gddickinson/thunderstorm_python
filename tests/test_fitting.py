"""
Tests for PSF fitting module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.fitting import (
    LocalizationResult, GaussianLSQFitter, GaussianMLEFitter, CentroidFitter,
    RadialSymmetryFitter, create_fitter, gaussian_2d
)


@pytest.fixture
def noiseless_gaussian_image():
    """Create a noiseless Gaussian spot at a known position."""
    size = 21
    img = np.zeros((size, size), dtype=float)
    x0, y0 = 10.3, 10.7  # Subpixel position
    sigma = 1.5
    amplitude = 500.0
    background = 10.0

    yy, xx = np.mgrid[0:size, 0:size]
    img = amplitude * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
    img += background
    return img, x0, y0, sigma


class TestLocalizationResult:
    def test_empty_result(self):
        r = LocalizationResult()
        assert len(r) == 0

    def test_add_localization(self):
        r = LocalizationResult()
        r.add_localization(x=10.5, y=20.3, intensity=100, background=5,
                          sigma_x=1.5, sigma_y=1.5)
        assert len(r) == 1
        assert r.x[0] == 10.5

    def test_to_array(self):
        r = LocalizationResult()
        r.add_localization(x=10.5, y=20.3, intensity=100, background=5,
                          sigma_x=1.5)
        data = r.to_array()
        assert 'x' in data
        assert data['x'][0] == 10.5


class TestGaussianLSQFitter:
    def test_fit_noiseless(self, noiseless_gaussian_image):
        img, x0, y0, sigma = noiseless_gaussian_image
        fitter = GaussianLSQFitter(integrated=False, initial_sigma=1.3)
        positions = np.array([[10, 10]])  # Approximate integer position
        result = fitter.fit(img, positions, fit_radius=5)
        assert len(result) == 1
        # RMSE < 1 pixel on noiseless data
        error = np.sqrt((result.x[0] - x0)**2 + (result.y[0] - y0)**2)
        assert error < 1.0, f"Fitting error {error:.3f} exceeds 1 pixel"

    def test_fit_returns_reasonable_sigma(self, noiseless_gaussian_image):
        img, x0, y0, sigma = noiseless_gaussian_image
        fitter = GaussianLSQFitter(integrated=False, initial_sigma=1.3)
        positions = np.array([[10, 10]])
        result = fitter.fit(img, positions, fit_radius=5)
        assert abs(result.sigma_x[0] - sigma) < 0.5


class TestCentroidFitter:
    def test_fit_noiseless(self, noiseless_gaussian_image):
        img, x0, y0, sigma = noiseless_gaussian_image
        fitter = CentroidFitter()
        positions = np.array([[10, 10]])
        result = fitter.fit(img, positions, fit_radius=5)
        assert len(result) == 1
        # Centroid should be within 2 pixels
        error = np.sqrt((result.x[0] - x0)**2 + (result.y[0] - y0)**2)
        assert error < 2.0


class TestCreateFitter:
    def test_create_gaussian_lsq(self):
        f = create_fitter('gaussian_lsq')
        assert isinstance(f, GaussianLSQFitter)

    def test_create_centroid(self):
        f = create_fitter('centroid')
        assert isinstance(f, CentroidFitter)

    def test_create_radial_symmetry(self):
        f = create_fitter('radial_symmetry')
        assert isinstance(f, RadialSymmetryFitter)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            create_fitter('nonexistent')


class TestEllipticalOption:
    """Regression: `elliptical` was ignored; both settings fitted separate sigma_x and sigma_y."""

    @staticmethod
    def _spot(sx, sy):
        yy, xx = np.mgrid[0:15, 0:15]
        return 1000 * np.exp(-((xx - 7.3) ** 2 / (2 * sx ** 2) + (yy - 6.8) ** 2 / (2 * sy ** 2))) + 10

    @pytest.mark.parametrize('cls', [GaussianLSQFitter, GaussianMLEFitter])
    def test_symmetric_fit_reports_one_width(self, cls):
        res = cls(integrated=False, elliptical=False).fit(self._spot(1.2, 2.0), np.array([[7, 7]]), fit_radius=5)
        assert res.sigma_x[0] == res.sigma_y[0]
        assert 1.2 < res.sigma_x[0] < 2.0

    @pytest.mark.parametrize('cls', [GaussianLSQFitter, GaussianMLEFitter])
    def test_elliptical_fit_recovers_both_widths(self, cls):
        res = cls(integrated=False, elliptical=True).fit(self._spot(1.2, 2.0), np.array([[7, 7]]), fit_radius=5)
        assert res.sigma_x[0] == pytest.approx(1.2, abs=0.05) and res.sigma_y[0] == pytest.approx(2.0, abs=0.05)
        assert res.x[0] == pytest.approx(7.3, abs=0.02) and res.y[0] == pytest.approx(6.8, abs=0.02)
