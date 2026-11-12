"""
Regression: failed fits used to be returned as localizations at the candidate
pixel with zero intensity and infinite uncertainty. They now go to a separate
rejected table with a reason, and the default output holds real fits only.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from thunderstorm_python.fitting import CentroidFitter, GaussianLSQFitter, GaussianMLEFitter
from thunderstorm_python.pipeline import ThunderSTORM


def frame_with_spot_and_dip():
    yy, xx = np.mgrid[0:30, 0:30]
    img = 20.0 + 800 * np.exp(-((xx - 8.2) ** 2 + (yy - 9.6) ** 2) / (2 * 1.4 ** 2))
    img -= 15 * np.exp(-((xx - 21) ** 2 + (yy - 20) ** 2) / (2 * 1.5 ** 2))   # a dip, not a molecule
    return img


@pytest.mark.parametrize('fitter', [GaussianLSQFitter(), GaussianMLEFitter()])
def test_fitters_reject_dip(fitter):
    res = fitter.fit(frame_with_spot_and_dip(), np.array([[10, 8], [20, 21]]))
    assert len(res) == 1 and res.x[0] == pytest.approx(8.2, abs=0.05)
    rej = res.rejected_to_array()
    assert rej['x'].tolist() == [21.0] and rej['y'].tolist() == [20.0]
    assert rej['reason'][0] in ('non-positive fitted intensity', 'fit did not converge') or rej['reason'][0].startswith('fit error')


def test_centroid_rejects_flat_region():
    res = CentroidFitter().fit(np.full((20, 20), 7.0), np.array([[10, 10]]))
    assert len(res) == 0 and res.rejected_to_array()['reason'].tolist() == ['no signal above background']


def test_pipeline_keeps_rejected_separately(monkeypatch):
    pipe = ThunderSTORM(pixel_size=100.0, baseline=0.0)
    monkeypatch.setattr(pipe.detector, 'detect', lambda img, thr: np.array([[10, 8], [20, 21]]))
    locs = pipe.analyze_stack(np.stack([frame_with_spot_and_dip()] * 2), show_progress=False)
    assert len(locs['x']) == 2 and np.all(np.isfinite(locs['uncertainty'])) and np.all(locs['intensity'] > 0)
    rej = pipe.get_rejected()
    assert rej['frame'].tolist() == [0, 1]
    assert rej['x'].tolist() == [2100.0, 2100.0] and rej['y'].tolist() == [2000.0, 2000.0]   # nm, pixel centre
    assert len(rej['reason']) == 2
