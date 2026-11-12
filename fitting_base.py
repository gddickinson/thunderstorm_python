"""
PSF Fitting: shared pieces
==========================

LocalizationResult (result container), BaseFitter (camera parameters) and the
2D Gaussian PSF models used by the fitters in fitting.py. Re-exported from
fitting.py, so `from thunderstorm_python.fitting import gaussian_2d` still works.
"""

import numpy as np
from scipy.special import erf

from .uncertainty import excess_noise_factor

class LocalizationResult:
    """Container for localization results."""
    
    def __init__(self):
        self.x = []  # x position (col)
        self.y = []  # y position (row)
        self.z = []  # z position (if 3D)
        self.intensity = []  # Fitted intensity
        self.background = []  # Fitted background
        self.sigma_x = []  # PSF width in x
        self.sigma_y = []  # PSF width in y (for elliptical)
        self.frame = []  # Frame number
        self.uncertainty = []  # Localization uncertainty
        self.chi_squared = []  # Goodness of fit
        
    def add_localization(self, x, y, intensity, background, sigma_x, 
                        sigma_y=None, frame=None, uncertainty=None, 
                        chi_squared=None, z=None):
        """Add a localization."""
        self.x.append(x)
        self.y.append(y)
        self.intensity.append(intensity)
        self.background.append(background)
        self.sigma_x.append(sigma_x)
        self.sigma_y.append(sigma_y if sigma_y is not None else sigma_x)
        self.frame.append(frame)
        self.uncertainty.append(uncertainty)
        self.chi_squared.append(chi_squared)
        if z is not None:
            self.z.append(z)
            
    def to_array(self):
        """Convert to numpy array."""
        data = {
            'x': np.array(self.x),
            'y': np.array(self.y),
            'intensity': np.array(self.intensity),
            'background': np.array(self.background),
            'sigma_x': np.array(self.sigma_x),
            'sigma_y': np.array(self.sigma_y),
            'frame': np.array(self.frame),
            'uncertainty': np.array(self.uncertainty),
            'chi_squared': np.array(self.chi_squared)
        }
        if self.z:
            data['z'] = np.array(self.z)
        return data
        
    def __len__(self):
        return len(self.x)


class BaseFitter:
    """Base class for PSF fitters."""
    
    def __init__(self, name="BaseFitter"):
        self.name = name
        self.pixel_size = 100.0  # nm/pixel
        self.photons_per_adu = 1.0  # Conversion factor
        self.baseline = 0.0  # Camera baseline offset
        self.em_gain = 1.0  # EM gain for EMCCD
        
    def set_camera_params(self, pixel_size=None, photons_per_adu=None, 
                         baseline=None, em_gain=None):
        """Set camera parameters."""
        if pixel_size is not None:
            self.pixel_size = pixel_size
        if photons_per_adu is not None:
            self.photons_per_adu = photons_per_adu
        if baseline is not None:
            self.baseline = baseline
        if em_gain is not None:
            if em_gain < 1:
                raise ValueError("em_gain must be >= 1 (1 = no electron multiplication)")
            self.em_gain = em_gain

    def counts_to_photons(self, counts, subtract_baseline=False):
        """Camera counts (ADU) -> photons: (counts [- baseline]) * photons_per_adu / em_gain."""
        c = counts - self.baseline if subtract_baseline else counts
        return c * self.photons_per_adu / self.em_gain

    def excess_noise_factor(self):
        """EMCCD excess-noise factor for the uncertainty (2 if em_gain > 1, else 1)."""
        return excess_noise_factor(self.em_gain)
            
    def fit(self, image, positions, fit_radius=3):
        """Fit PSF at given positions.
        
        Parameters
        ----------
        image : ndarray
            Image to fit
        positions : ndarray
            Array of (row, col) approximate positions
        fit_radius : int
            Radius of fitting region
            
        Returns
        -------
        result : LocalizationResult
            Fitted localizations
        """
        raise NotImplementedError


def gaussian_2d(xy, x0, y0, amplitude, sigma_x, sigma_y, background):
    """2D Gaussian function.
    
    Parameters
    ----------
    xy : tuple
        (x, y) coordinates
    x0, y0 : float
        Center position
    amplitude : float
        Peak amplitude
    sigma_x, sigma_y : float
        Standard deviations
    background : float
        Background offset
        
    Returns
    -------
    values : ndarray
        Gaussian values at xy positions
    """
    x, y = xy
    g = amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + 
                            ((y - y0) ** 2) / (2 * sigma_y ** 2)))
    return (g + background).ravel()


def integrated_gaussian_2d(xy, x0, y0, amplitude, sigma_x, sigma_y, background):
    """Integrated 2D Gaussian (accounts for pixel integration).
    
    More accurate for fitting to pixel data.
    """
    x, y = xy
    
    # Pixel boundaries
    x_left = x - 0.5
    x_right = x + 0.5
    y_bottom = y - 0.5
    y_top = y + 0.5
    
    # Integrated Gaussian using error function
    norm_x = np.sqrt(2) * sigma_x
    norm_y = np.sqrt(2) * sigma_y
    
    int_x = 0.5 * (erf((x_right - x0) / norm_x) - erf((x_left - x0) / norm_x))
    int_y = 0.5 * (erf((y_top - y0) / norm_y) - erf((y_bottom - y0) / norm_y))
    
    g = amplitude * int_x * int_y
    return (g + background).ravel()


def psf_model(integrated=True, elliptical=False):
    """Model function for curve_fit / likelihoods.

    elliptical=False: f(xy, x0, y0, amplitude, sigma, background), one width.
    elliptical=True:  f(xy, x0, y0, amplitude, sigma_x, sigma_y, background).
    """
    base = integrated_gaussian_2d if integrated else gaussian_2d
    if elliptical:
        return base

    def symmetric(xy, x0, y0, amplitude, sigma, background):
        return base(xy, x0, y0, amplitude, sigma, sigma, background)
    return symmetric


def initial_params(x0, y0, amplitude, sigma, background, elliptical=False):
    """Start vector matching psf_model(..., elliptical)."""
    if elliptical:
        return [x0, y0, amplitude, sigma, sigma, background]
    return [x0, y0, amplitude, sigma, background]


def split_params(p, elliptical=False):
    """(x0, y0, amplitude, sigma_x, sigma_y, background) from a parameter vector; widths made positive."""
    if elliptical:
        x0, y0, amp, sx, sy, bg = p
    else:
        x0, y0, amp, sx, bg = p
        sy = sx
    return x0, y0, amp, abs(sx), abs(sy), bg


def effective_sigma(sigma_x, sigma_y):
    """Single PSF width for the uncertainty formula: geometric mean sqrt(sigma_x * sigma_y)."""
    return float(np.sqrt(sigma_x * sigma_y))
