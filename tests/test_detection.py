"""
Tests for molecule detection module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.detection import (
    LocalMaximumDetector, NonMaximumSuppression, CentroidDetector,
    create_detector, refine_detections, remove_border_detections
)


@pytest.fixture
def image_with_spots():
    """Create an image with known bright spots."""
    img = np.zeros((64, 64), dtype=float)
    # Place Gaussian spots at known positions
    from scipy.ndimage import gaussian_filter
    for r, c in [(20, 20), (20, 40), (40, 20), (40, 40)]:
        tmp = np.zeros_like(img)
        tmp[r, c] = 1000.0
        img += gaussian_filter(tmp, sigma=1.5)
    img += 10.0  # background
    return img


class TestLocalMaximumDetector:
    def test_detects_spots(self, image_with_spots):
        det = LocalMaximumDetector(min_distance=5)
        positions = det.detect(image_with_spots, threshold=50.0)
        assert len(positions) == 4

    def test_threshold_filters(self, image_with_spots):
        det = LocalMaximumDetector(min_distance=5)
        positions = det.detect(image_with_spots, threshold=1e6)
        assert len(positions) == 0


class TestNonMaximumSuppression:
    def test_detects_spots(self, image_with_spots):
        det = NonMaximumSuppression()
        positions = det.detect(image_with_spots, threshold=50.0)
        assert len(positions) >= 4


class TestCentroidDetector:
    def test_detects_spots(self, image_with_spots):
        det = CentroidDetector(min_area=1)
        positions = det.detect(image_with_spots, threshold=50.0)
        assert len(positions) >= 4


class TestCreateDetector:
    def test_create_local_maximum(self):
        det = create_detector('local_maximum')
        assert isinstance(det, LocalMaximumDetector)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            create_detector('nonexistent')


class TestRefineDetections:
    def test_refine_improves_accuracy(self, image_with_spots):
        # Start with integer positions
        detections = np.array([[20, 20], [20, 40], [40, 20], [40, 40]])
        refined = refine_detections(detections, image_with_spots, radius=3)
        assert refined.shape == detections.shape
        # Refined positions should be close to original
        assert np.allclose(refined, detections, atol=2.0)


class TestRemoveBorderDetections:
    def test_removes_border(self):
        detections = np.array([[2, 2], [30, 30], [62, 62]])
        filtered = remove_border_detections(detections, (64, 64), border=5)
        assert len(filtered) == 1
        assert filtered[0, 0] == 30

    def test_empty_input(self):
        detections = np.array([]).reshape(0, 2)
        filtered = remove_border_detections(detections, (64, 64), border=5)
        assert len(filtered) == 0
