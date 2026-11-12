"""
thunderSTORM-format export
==========================

Convert pipeline localizations to the column layout of thunderSTORM's CSV
export, so results can be opened in thunderSTORM / ImageJ or any tool that
reads its tables.

Conventions (stated here because they differ from the pipeline's own dict):

- frame: 1-based (thunderSTORM), the pipeline's frames are 0-based.
- x [nm], y [nm]: corner origin, as in ImageJ / thunderSTORM: the top-left
  corner of the top-left camera pixel is (0, 0), so
  x_thunderstorm = x_pipeline + pixel_size / 2 (the pipeline reports nm with a
  pixel-centre origin).
- sigma [nm] (or sigma1 [nm] / sigma2 [nm] when the fit gave separate x / y
  widths): PSF sigma in camera pixels times pixel_size.
- intensity [photon], offset [photon]: fitted amplitude and background
  converted to photons. Fitters that work in camera counts (gaussian_lsq,
  radial_symmetry, centroid) are converted with (value - baseline) *
  photons_per_adu / em_gain for the offset and value * photons_per_adu /
  em_gain for the intensity; gaussian_mle already works in photons and is
  copied unchanged.
- uncertainty [nm]: the uncertainty computed by this package's fitter.

thunderSTORM's bkgstd column is not written: this package does not estimate
the background standard deviation.
"""

import numpy as np
import pandas as pd

PHOTON_FITTERS = ('gaussian_mle',)


def to_thunderstorm_columns(localizations, pixel_size, photons_per_adu=1.0,
                            baseline=0.0, fitter_type='gaussian_lsq', em_gain=1.0):
    """Return an ordered dict of thunderSTORM columns (numpy arrays).

    Parameters
    ----------
    localizations : dict
        Pipeline output (x, y in nm with pixel-centre origin; frame 0-based;
        sigma_x / sigma_y in pixels; intensity / background in fitter units).
    pixel_size : float
        Camera pixel size in nm.
    photons_per_adu : float
        Photoelectrons per A/D count.
    baseline : float
        Camera offset in counts (subtracted from the background).
    fitter_type : str
        The fitter that produced the localizations (decides photon conversion).
    em_gain : float
        EM gain (counts are divided by it).
    """
    n = len(localizations['x'])
    half = pixel_size / 2.0
    sx = np.asarray(localizations.get('sigma_x', np.full(n, np.nan)), float)
    sy = np.asarray(localizations.get('sigma_y', sx), float)
    intensity = np.asarray(localizations.get('intensity', np.full(n, np.nan)), float)
    background = np.asarray(localizations.get('background', np.full(n, np.nan)), float)
    if fitter_type not in PHOTON_FITTERS:
        intensity = intensity * photons_per_adu / em_gain
        background = (background - baseline) * photons_per_adu / em_gain

    cols = {
        'id': np.arange(1, n + 1),
        'frame': np.asarray(localizations['frame']).astype(int) + 1,
        'x [nm]': np.asarray(localizations['x'], float) + half,
        'y [nm]': np.asarray(localizations['y'], float) + half,
    }
    if np.allclose(sx, sy, equal_nan=True):
        cols['sigma [nm]'] = sx * pixel_size
    else:
        cols['sigma1 [nm]'] = sx * pixel_size
        cols['sigma2 [nm]'] = sy * pixel_size
    cols['intensity [photon]'] = intensity
    cols['offset [photon]'] = background
    cols['uncertainty [nm]'] = np.asarray(
        localizations.get('uncertainty', np.full(n, np.nan)), float)
    return cols


def save_thunderstorm_csv(localizations, filepath, pixel_size, **kwargs):
    """Write localizations as a thunderSTORM-format CSV (see module docstring)."""
    cols = to_thunderstorm_columns(localizations, pixel_size, **kwargs)
    pd.DataFrame(cols).to_csv(filepath, index=False)
