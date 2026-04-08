"""
Tests for image filtering module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.filters import (
    GaussianFilter, WaveletFilter, DifferenceOfGaussians,
    LoweredGaussian, DifferenceOfAveraging, MedianFilter,
    BoxFilter, NoFilter, create_filter
)


@pytest.fixture
def sample_image():
    """Create a simple test image with a bright spot."""
    img = np.zeros((64, 64), dtype=float)
    img[32, 32] = 1000.0  # Bright spot
    # Add some background noise
    rng = np.random.default_rng(42)
    img += rng.normal(10, 2, img.shape)
    return img


class TestGaussianFilter:
    def test_output_shape(self, sample_image):
        f = GaussianFilter(sigma=1.6)
        result = f.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_smoothing_reduces_peak(self, sample_image):
        f = GaussianFilter(sigma=2.0)
        result = f.apply(sample_image)
        # Peak should be lower after smoothing
        assert result.max() < sample_image.max()

    def test_preserves_dtype_float(self, sample_image):
        f = GaussianFilter(sigma=1.6)
        result = f.apply(sample_image)
        assert result.dtype == np.float64


class TestDifferenceOfGaussians:
    def test_output_shape(self, sample_image):
        f = DifferenceOfGaussians(sigma1=1.0, sigma2=1.6)
        result = f.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_bandpass_enhances_spot(self, sample_image):
        f = DifferenceOfGaussians(sigma1=1.0, sigma2=3.0)
        result = f.apply(sample_image)
        # The spot should still be the brightest feature
        peak_loc = np.unravel_index(np.argmax(result), result.shape)
        assert abs(peak_loc[0] - 32) <= 1
        assert abs(peak_loc[1] - 32) <= 1


class TestNoFilter:
    def test_passthrough(self, sample_image):
        f = NoFilter()
        result = f.apply(sample_image)
        np.testing.assert_array_almost_equal(result, sample_image.astype(float))


class TestCreateFilter:
    def test_create_gaussian(self):
        f = create_filter('gaussian', sigma=2.0)
        assert isinstance(f, GaussianFilter)

    def test_create_dog(self):
        f = create_filter('dog')
        assert isinstance(f, DifferenceOfGaussians)

    def test_create_none(self):
        f = create_filter('none')
        assert isinstance(f, NoFilter)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_filter('invalid_filter_name')

    def test_case_insensitive(self):
        f = create_filter('GAUSSIAN')
        assert isinstance(f, GaussianFilter)


class TestMedianFilter:
    def test_output_shape(self, sample_image):
        f = MedianFilter(size=3)
        result = f.apply(sample_image)
        assert result.shape == sample_image.shape


class TestBoxFilter:
    def test_output_shape(self, sample_image):
        f = BoxFilter(size=3)
        result = f.apply(sample_image)
        assert result.shape == sample_image.shape
