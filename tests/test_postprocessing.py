"""
postprocessing.py was split (drift correction now in drift.py); the public
names stay importable from postprocessing and keep working.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python import drift, postprocessing


def test_reexports():
    assert postprocessing.DriftCorrector is drift.DriftCorrector
    for name in ('MolecularMerger', 'LocalizationFilter', 'LocalDensityFilter', 'DuplicateRemover',
                 'z_stage_offset_correction'):
        assert hasattr(postprocessing, name)


def test_merger_and_filter_run():
    locs = {'x': np.array([100.0, 105.0, 900.0]), 'y': np.array([100.0, 98.0, 900.0]),
            'frame': np.array([0, 1, 5]), 'intensity': np.array([500.0, 600.0, 50.0]),
            'uncertainty': np.array([10.0, 12.0, 30.0])}
    merged = postprocessing.MolecularMerger(max_distance=20, max_frame_gap=1).merge(locs)
    assert len(merged['x']) == 2
    kept = postprocessing.LocalizationFilter(min_intensity=100).filter(locs)
    assert len(kept['x']) == 2
