"""
Localization uncertainty
========================

Lateral localization precision of a fitted 2D Gaussian PSF, from

    Mortensen K. I., Churchman L. S., Spudich J. A. & Flyvbjerg H. (2010).
    Optimized localization analysis for single-molecule tracking and
    super-resolution microscopy. Nature Methods 7(5), 377-381.
    doi:10.1038/nmeth.1447

which refines Thompson, Larson & Webb (2002) for least squares and gives the
maximum-likelihood bound:

    least squares:  <(dx)^2> = F (sa^2 / N) (16/9 + 8 pi sa^2 b^2 / (N a^2))
    max likelihood: <(dx)^2> = F (sa^2 / N) / (1 + I(tau)),
                    I(tau) = int_0^1 ln(t) / (1 + t / tau) dt,
                    tau = 2 pi sa^2 b^2 / (N a^2)

with sa^2 = s^2 + a^2/12 (s = PSF standard deviation, a = pixel size, both in
nm), N = signal photons, b^2 = expected background photons per pixel (the
Poisson variance of the background, i.e. the background level in photons
after the camera baseline has been subtracted) and F the excess-noise factor:
F = 2 for an EMCCD (EM gain > 1), F = 1 otherwise (Mortensen et al. 2010).

The result is the standard deviation in nm, per axis.
"""

import numpy as np
from scipy.special import spence

ESTIMATORS = ('lsq', 'mle')


def excess_noise_factor(em_gain):
    """F = 2 with electron multiplication (EM gain > 1), 1 without."""
    return 2.0 if em_gain is not None and em_gain > 1 else 1.0


def mle_integral(tau):
    """I(tau) = int_0^1 ln t / (1 + t/tau) dt = tau * Li2(-1/tau); 0 when tau = 0."""
    if tau <= 0:
        return 0.0
    # scipy's spence(z) is Li2(1 - z), so Li2(-1/tau) = spence(1 + 1/tau)
    return float(tau * spence(1.0 + 1.0 / tau))


def localization_uncertainty(sigma_px, photons, background_photons, pixel_size,
                             estimator='lsq', excess_factor=1.0):
    """Localization uncertainty in nm (Mortensen et al. 2010), np.inf if photons <= 0.

    Parameters
    ----------
    sigma_px : float
        PSF standard deviation in camera pixels.
    photons : float
        Signal photons N.
    background_photons : float
        Background photons per pixel (b^2); negative values are treated as 0.
    pixel_size : float
        Camera pixel size a in nm.
    estimator : str
        'lsq' (least squares) or 'mle' (maximum likelihood).
    excess_factor : float
        F (see excess_noise_factor).
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    if not np.isfinite(photons) or photons <= 0:
        return np.inf
    a = float(pixel_size)
    b2 = max(float(background_photons), 0.0)
    sa2 = (sigma_px * a) ** 2 + a * a / 12.0
    if estimator == 'lsq':
        var = sa2 / photons * (16.0 / 9.0 + 8.0 * np.pi * sa2 * b2 / (photons * a * a))
    else:
        tau = 2.0 * np.pi * sa2 * b2 / (photons * a * a)
        var = sa2 / photons / (1.0 + mle_integral(tau))
    return float(np.sqrt(excess_factor * var))
