"""
PSF Fitting Module
==================

Implements sub-pixel localization methods:
- Gaussian PSF fitting (2D and 3D astigmatism)
- Least Squares (LSQ) fitting
- Weighted Least Squares (WLSQ) fitting  
- Maximum Likelihood Estimation (MLE)
- Radial Symmetry method
- Centroid refinement
- Multiple emitter fitting
"""

import numpy as np
from scipy import optimize, ndimage

from .fitting_base import (  # noqa: F401  (re-exported for backward compatibility)
    LocalizationResult, BaseFitter, gaussian_2d, integrated_gaussian_2d,
    psf_model, initial_params, split_params, effective_sigma,
)
from .uncertainty import localization_uncertainty


class GaussianLSQFitter(BaseFitter):
    """Gaussian fitting using Least Squares.
    
    Parameters
    ----------
    integrated : bool
        Use integrated Gaussian model
    elliptical : bool
        False (default): one PSF width (sigma_x = sigma_y, 5 parameters).
        True: separate sigma_x and sigma_y (6 parameters); the uncertainty
        then uses sqrt(sigma_x * sigma_y).
    initial_sigma : float
        Initial guess for sigma
    """
    
    def __init__(self, integrated=True, elliptical=False, initial_sigma=1.3):
        super().__init__("GaussianLSQ")
        self.integrated = integrated
        self.elliptical = elliptical
        self.initial_sigma = initial_sigma
        
    def fit(self, image, positions, fit_radius=3):
        """Fit Gaussian to positions using LSQ."""
        result = LocalizationResult()
        
        for pos in positions:
            row, col = int(pos[0]), int(pos[1])
            
            # Extract fitting region
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)
            
            region = image[r0:r1, c0:c1]
            
            # Create coordinate grids relative to region
            y_coords, x_coords = np.mgrid[0:region.shape[0], 0:region.shape[1]]
            
            # Initial parameters
            background_init = np.median(region)
            amplitude_init = region.max() - background_init
            x0_init = col - c0
            y0_init = row - r0
            sigma_init = self.initial_sigma
            
            p0 = initial_params(x0_init, y0_init, amplitude_init, sigma_init, background_init, self.elliptical)
            model = psf_model(self.integrated, self.elliptical)
            
            try:
                # Fit
                popt, pcov = optimize.curve_fit(
                    model,
                    (x_coords, y_coords),
                    region.ravel(),
                    p0=p0,
                    maxfev=1000
                )
                
                # Convert back to image coordinates
                x0, y0, amplitude, sigma_x, sigma_y, bg = split_params(popt, self.elliptical)
                x_fit = x0 + c0
                y_fit = y0 + r0
                
                # Compute uncertainty (Mortensen et al., 2010)
                uncertainty = self._compute_uncertainty(
                    effective_sigma(sigma_x, sigma_y), amplitude, bg, self.pixel_size)
                
                # Compute chi-squared
                fitted_data = model((x_coords, y_coords), *popt).reshape(region.shape)
                chi_sq = np.sum((region - fitted_data) ** 2)
                
                result.add_localization(
                    x=x_fit,
                    y=y_fit,
                    intensity=amplitude,
                    background=bg,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    uncertainty=uncertainty,
                    chi_squared=chi_sq
                )
                
            except (RuntimeError, ValueError):
                # Fitting failed, use approximate position
                result.add_localization(
                    x=col,
                    y=row,
                    intensity=0,
                    background=0,
                    sigma_x=self.initial_sigma,
                    sigma_y=self.initial_sigma,
                    uncertainty=np.inf,
                    chi_squared=np.inf
                )
                
        return result
    
    def _compute_uncertainty(self, sigma, intensity, background, pixel_size):
        """Localization uncertainty in nm (Mortensen et al. 2010, least squares; see uncertainty.py).

        intensity and background are the fitted amplitude and offset in camera counts: N = amplitude in
        photons, b^2 = background photons per pixel after subtracting the camera baseline, and the EMCCD
        excess-noise factor applies when em_gain > 1.
        """
        return localization_uncertainty(
            sigma, self.counts_to_photons(intensity), self.counts_to_photons(background, subtract_baseline=True),
            pixel_size, estimator='lsq', excess_factor=self.excess_noise_factor())


class GaussianMLEFitter(BaseFitter):
    """Gaussian fitting using Maximum Likelihood Estimation.
    
    Uses Poisson likelihood for photon-counting noise.
    Based on Mortensen et al., 2010.
    
    Parameters
    ----------
    integrated : bool
        Use integrated Gaussian model
    elliptical : bool
        False (default): one PSF width; True: separate sigma_x and sigma_y
        (the uncertainty then uses sqrt(sigma_x * sigma_y)).
    initial_sigma : float
        Initial guess for sigma
    """
    
    def __init__(self, integrated=True, elliptical=False, initial_sigma=1.3):
        super().__init__("GaussianMLE")
        self.integrated = integrated
        self.elliptical = elliptical
        self.initial_sigma = initial_sigma
        
    def fit(self, image, positions, fit_radius=3):
        """Fit Gaussian using MLE."""
        result = LocalizationResult()
        
        for pos in positions:
            row, col = int(pos[0]), int(pos[1])
            
            # Extract fitting region
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)
            
            region = image[r0:r1, c0:c1].astype(float)
            
            # Convert to photon counts
            region_photons = self.counts_to_photons(region, subtract_baseline=True)
            region_photons = np.maximum(region_photons, 0.1)  # Avoid log(0)
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:region.shape[0], 0:region.shape[1]]
            
            # Initial parameters
            background_init = np.median(region_photons)
            amplitude_init = region_photons.max() - background_init
            x0_init = col - c0
            y0_init = row - r0
            sigma_init = self.initial_sigma
            
            p0 = initial_params(x0_init, y0_init, amplitude_init, sigma_init, background_init, self.elliptical)
            
            # Negative log-likelihood function
            model = psf_model(self.integrated, self.elliptical)
            
            def neg_log_likelihood(params):
                predicted = model((x_coords, y_coords), *params).reshape(region.shape)
                predicted = np.maximum(predicted, 0.1)  # Avoid log(0)
                
                # Poisson log-likelihood: sum(data*log(model) - model)
                nll = -np.sum(region_photons * np.log(predicted) - predicted)
                return nll
            
            try:
                # Minimize negative log-likelihood
                res = optimize.minimize(
                    neg_log_likelihood,
                    p0,
                    method='L-BFGS-B',
                    bounds=[
                        (0, region.shape[1]),  # x0
                        (0, region.shape[0]),  # y0
                        (0, None),  # amplitude
                    ] + [(0.5, 5)] * (2 if self.elliptical else 1) + [  # sigma(s)
                        (0, None)  # background
                    ]
                )
                
                # Convert back to image coordinates
                x0, y0, amplitude, sigma_x, sigma_y, bg = split_params(res.x, self.elliptical)
                x_fit = x0 + c0
                y_fit = y0 + r0
                
                # Compute uncertainty (for MLE with photon noise)
                uncertainty = self._compute_mle_uncertainty(effective_sigma(sigma_x, sigma_y), amplitude, bg)
                
                result.add_localization(
                    x=x_fit,
                    y=y_fit,
                    intensity=amplitude,
                    background=bg,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    uncertainty=uncertainty,
                    chi_squared=res.fun
                )
                
            except (RuntimeError, ValueError):
                # Fitting failed
                result.add_localization(
                    x=col,
                    y=row,
                    intensity=0,
                    background=0,
                    sigma_x=self.initial_sigma,
                    sigma_y=self.initial_sigma,
                    uncertainty=np.inf,
                    chi_squared=np.inf
                )
                
        return result
    
    def _compute_mle_uncertainty(self, sigma, intensity, background):
        """Localization uncertainty in nm (Mortensen et al. 2010, maximum likelihood; see uncertainty.py).
        The MLE fit already works in photons (baseline subtracted, EM gain divided out)."""
        return localization_uncertainty(sigma, intensity, background, self.pixel_size,
                                        estimator='mle', excess_factor=self.excess_noise_factor())


class RadialSymmetryFitter(BaseFitter):
    """Radial symmetry method for fast subpixel localization.
    
    Based on Parthasarathy (2012) - non-iterative, very fast.
    
    Parameters
    ----------
    smoothing : bool
        Apply 3x3 smoothing to gradients
    """
    
    def __init__(self, smoothing=True):
        super().__init__("RadialSymmetry")
        self.smoothing = smoothing
        
    def fit(self, image, positions, fit_radius=3):
        """Fit using radial symmetry."""
        result = LocalizationResult()
        
        for pos in positions:
            row, col = int(pos[0]), int(pos[1])
            
            # Extract fitting region
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)
            
            region = image[r0:r1, c0:c1].astype(float)
            
            # Compute gradients
            gy, gx = np.gradient(region)
            
            # Optional smoothing of gradients
            if self.smoothing:
                gy = ndimage.uniform_filter(gy, 3)
                gx = ndimage.uniform_filter(gx, 3)
            
            # Compute gradient magnitude
            g_mag = np.sqrt(gx**2 + gy**2) + 1e-10
            
            # Normalize gradients
            gx_norm = gx / g_mag
            gy_norm = gy / g_mag
            
            # Create position grids
            y_grid, x_grid = np.mgrid[0:region.shape[0], 0:region.shape[1]]
            
            # Compute weights (gradient magnitude)
            weights = g_mag.ravel()
            
            # Solve for radial center using weighted least squares
            # Each gradient vector defines a line; find intersection
            x_flat = x_grid.ravel()
            y_flat = y_grid.ravel()
            gx_flat = gx_norm.ravel()
            gy_flat = gy_norm.ravel()
            
            # Perpendicular to gradient gives radial direction
            # Point on line: (x_i, y_i)
            # Direction perpendicular to gradient: (-gy, gx)
            
            # Weighted least squares solution
            denom = np.sum(weights * (gx_flat**2 + gy_flat**2))
            
            if denom > 0:
                x0 = np.sum(weights * (gx_flat * (gx_flat * x_flat + gy_flat * y_flat))) / denom
                y0 = np.sum(weights * (gy_flat * (gx_flat * x_flat + gy_flat * y_flat))) / denom
                
                # Convert to image coordinates
                x_fit = x0 + c0
                y_fit = y0 + r0
                
                # Estimate intensity and background
                intensity = region.max() - np.median(region)
                background = np.median(region)
                
                result.add_localization(
                    x=x_fit,
                    y=y_fit,
                    intensity=intensity,
                    background=background,
                    sigma_x=self.initial_sigma if hasattr(self, 'initial_sigma') else 1.3,
                    uncertainty=self.pixel_size  # Rough estimate
                )
            else:
                # Failed - use approximate position
                result.add_localization(
                    x=col,
                    y=row,
                    intensity=0,
                    background=0,
                    sigma_x=1.3,
                    uncertainty=np.inf
                )
                
        return result


class CentroidFitter(BaseFitter):
    """Simple centroid localization (fast but less accurate)."""
    
    def __init__(self):
        super().__init__("Centroid")
        
    def fit(self, image, positions, fit_radius=3):
        """Compute intensity-weighted centroid."""
        result = LocalizationResult()
        
        for pos in positions:
            row, col = int(pos[0]), int(pos[1])
            
            # Extract fitting region
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)
            
            region = image[r0:r1, c0:c1].astype(float)
            
            # Subtract background
            background = np.percentile(region, 25)
            region_bg_sub = region - background
            region_bg_sub = np.maximum(region_bg_sub, 0)
            
            # Compute centroid
            total = region_bg_sub.sum()
            
            if total > 0:
                y_grid, x_grid = np.mgrid[r0:r1, c0:c1]
                x_fit = (x_grid * region_bg_sub).sum() / total
                y_fit = (y_grid * region_bg_sub).sum() / total
                intensity = region.max() - background
            else:
                x_fit = col
                y_fit = row
                intensity = 0
                
            result.add_localization(
                x=x_fit,
                y=y_fit,
                intensity=intensity,
                background=background,
                sigma_x=1.3,
                uncertainty=self.pixel_size
            )
            
        return result



# Registry used by create_fitter (and by the web interface to list the choices).
FITTER_TYPES = {
    'gaussian_lsq': GaussianLSQFitter,
    'gaussian_mle': GaussianMLEFitter,
    'radial_symmetry': RadialSymmetryFitter,
    'centroid': CentroidFitter
}


def create_fitter(fitter_type, **kwargs):
    """Factory function to create fitters.
    
    Parameters
    ----------
    fitter_type : str
        Type of fitter: 'gaussian_lsq', 'gaussian_mle', 
        'radial_symmetry', 'centroid'
    **kwargs : dict
        Fitter-specific parameters
        
    Returns
    -------
    fitter : BaseFitter
        Configured fitter object
    """
    fitter_map = FITTER_TYPES
    
    fitter_type = fitter_type.lower()
    if fitter_type not in fitter_map:
        raise ValueError(f"Unknown fitter type: {fitter_type}")
        
    return fitter_map[fitter_type](**kwargs)
