"""
Simulation Module
=================

Generate synthetic SMLM data for testing and validation:
- Molecule placement from masks or patterns
- PSF simulation (2D and 3D)
- Photon noise and camera noise
- Blinking dynamics
- Performance evaluation
"""

import numpy as np
from scipy import ndimage


class SMLMSimulator:
    """Simulate SMLM data with realistic noise and dynamics.
    
    Parameters
    ----------
    image_size : tuple
        (height, width) of images
    pixel_size : float
        Pixel size in nm
    psf_sigma : float
        PSF standard deviation in nm
    photons_per_molecule : float
        Mean photons per molecule
    background_photons : float
        Background photons per pixel
    """
    
    def __init__(self, image_size=(256, 256), pixel_size=100.0, 
                 psf_sigma=150.0, photons_per_molecule=1000.0,
                 background_photons=10.0):
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.psf_sigma = psf_sigma
        self.photons_per_molecule = photons_per_molecule
        self.background_photons = background_photons
        
    def generate_molecule_positions(self, n_molecules=None, density=None, mask=None):
        """Generate molecule positions.
        
        Parameters
        ----------
        n_molecules : int, optional
            Number of molecules (if mask not used)
        density : float, optional
            Molecules per square micron (if mask used)
        mask : ndarray, optional
            Grayscale mask defining spatial density
            
        Returns
        -------
        positions : ndarray
            Array of (x, y) positions in nm
        """
        if mask is not None:
            # Use mask to define spatial distribution
            mask_norm = mask / mask.sum()
            
            # Number of molecules based on density
            if density is not None:
                area_um2 = (self.image_size[0] * self.pixel_size * 
                           self.image_size[1] * self.pixel_size) / 1e6
                n_molecules = int(density * area_um2)
            elif n_molecules is None:
                n_molecules = 1000
            
            # Sample positions from mask
            flat_mask = mask_norm.ravel()
            indices = np.random.choice(len(flat_mask), size=n_molecules, p=flat_mask)
            
            rows, cols = np.unravel_index(indices, mask.shape)
            
            # Convert to nm with subpixel jitter
            x_positions = (cols + np.random.uniform(-0.5, 0.5, n_molecules)) * self.pixel_size
            y_positions = (rows + np.random.uniform(-0.5, 0.5, n_molecules)) * self.pixel_size
            
        else:
            # Uniform random distribution
            if n_molecules is None:
                n_molecules = 1000
                
            x_positions = np.random.uniform(0, self.image_size[1] * self.pixel_size, n_molecules)
            y_positions = np.random.uniform(0, self.image_size[0] * self.pixel_size, n_molecules)
        
        positions = np.column_stack([x_positions, y_positions])
        return positions
        
    def render_frame(self, molecule_positions, active_molecules=None):
        """Render a single frame with photon noise.
        
        Parameters
        ----------
        molecule_positions : ndarray
            Array of (x, y) positions in nm
        active_molecules : ndarray, optional
            Boolean array indicating which molecules are active
            
        Returns
        -------
        frame : ndarray
            Simulated image with Poisson noise
        ground_truth : dict
            Ground truth positions and parameters
        """
        if active_molecules is None:
            active_molecules = np.ones(len(molecule_positions), dtype=bool)
        
        # Initialize image
        image = np.zeros(self.image_size)
        
        # Add background
        background = np.random.poisson(self.background_photons, self.image_size)
        image += background
        
        # Render each active molecule
        sigma_pixels = self.psf_sigma / self.pixel_size
        ground_truth_positions = []
        
        for i, (x, y) in enumerate(molecule_positions):
            if not active_molecules[i]:
                continue
                
            # Sample photon count
            n_photons = np.random.poisson(self.photons_per_molecule)
            
            # Convert to pixel coordinates
            x_pix = x / self.pixel_size
            y_pix = y / self.pixel_size
            
            # Render Gaussian PSF
            self._add_gaussian_psf(image, x_pix, y_pix, n_photons, sigma_pixels)
            
            ground_truth_positions.append([x, y, n_photons])
        
        # Apply Poisson noise to entire image
        image = np.random.poisson(image)
        
        ground_truth = {
            'x': np.array([p[0] for p in ground_truth_positions]),
            'y': np.array([p[1] for p in ground_truth_positions]),
            'photons': np.array([p[2] for p in ground_truth_positions])
        }
        
        return image.astype(float), ground_truth
        
    def _add_gaussian_psf(self, image, x_pix, y_pix, intensity, sigma):
        """Add Gaussian PSF to image."""
        # Determine region
        radius = int(3 * sigma)
        x0 = int(x_pix) - radius
        x1 = int(x_pix) + radius + 1
        y0 = int(y_pix) - radius
        y1 = int(y_pix) + radius + 1
        
        # Clip to image bounds
        x0_clip = max(0, x0)
        x1_clip = min(image.shape[1], x1)
        y0_clip = max(0, y0)
        y1_clip = min(image.shape[0], y1)
        
        if x0_clip >= x1_clip or y0_clip >= y1_clip:
            return
        
        # Create Gaussian
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gaussian = np.exp(-((xx - x_pix)**2 + (yy - y_pix)**2) / (2 * sigma**2))
        gaussian /= gaussian.sum()
        gaussian *= intensity
        
        # Extract valid region and add
        gaussian_clip = gaussian[y0_clip-y0:y1_clip-y0, x0_clip-x0:x1_clip-x0]
        image[y0_clip:y1_clip, x0_clip:x1_clip] += gaussian_clip
        
    def simulate_blinking(self, n_frames, n_molecules, p_on=0.1, p_off=0.3,
                         p_bleach=0.01):
        """Simulate blinking dynamics.
        
        Parameters
        ----------
        n_frames : int
            Number of frames
        n_molecules : int
            Number of molecules
        p_on : float
            Probability of turning on per frame
        p_off : float
            Probability of turning off per frame
        p_bleach : float
            Probability of irreversible bleaching per frame
            
        Returns
        -------
        states : ndarray
            Boolean array (n_frames, n_molecules) indicating active state
        """
        states = np.zeros((n_frames, n_molecules), dtype=bool)
        bleached = np.zeros(n_molecules, dtype=bool)
        
        # Initial state - all off
        active = np.zeros(n_molecules, dtype=bool)
        
        for frame in range(n_frames):
            # Update states
            for i in range(n_molecules):
                if bleached[i]:
                    continue
                    
                if active[i]:
                    # Molecule is on - can turn off or bleach
                    if np.random.rand() < p_bleach:
                        bleached[i] = True
                        active[i] = False
                    elif np.random.rand() < p_off:
                        active[i] = False
                else:
                    # Molecule is off - can turn on
                    if np.random.rand() < p_on:
                        active[i] = True
            
            states[frame] = active.copy()
        
        return states
        
    def generate_movie(self, n_frames, molecule_positions=None, n_molecules=1000,
                      mask=None, blinking=True):
        """Generate complete SMLM movie.
        
        Parameters
        ----------
        n_frames : int
            Number of frames
        molecule_positions : ndarray, optional
            Pre-defined molecule positions
        n_molecules : int
            Number of molecules (if positions not provided)
        mask : ndarray, optional
            Spatial density mask
        blinking : bool
            Simulate blinking dynamics
            
        Returns
        -------
        movie : ndarray
            Image stack (n_frames, height, width)
        ground_truth : list of dict
            Ground truth for each frame
        """
        # Generate molecule positions
        if molecule_positions is None:
            molecule_positions = self.generate_molecule_positions(
                n_molecules=n_molecules, mask=mask
            )
        else:
            n_molecules = len(molecule_positions)
        
        # Simulate blinking
        if blinking:
            active_states = self.simulate_blinking(n_frames, n_molecules)
        else:
            # Random activation
            active_states = np.random.rand(n_frames, n_molecules) < 0.1
        
        # Generate frames
        movie = np.zeros((n_frames, *self.image_size))
        ground_truth = []
        
        for frame in range(n_frames):
            img, gt = self.render_frame(molecule_positions, active_states[frame])
            movie[frame] = img
            gt['frame'] = frame
            ground_truth.append(gt)
        
        return movie, ground_truth


# Backward-compatible imports from evaluation module
from .evaluation import PerformanceEvaluator, create_test_pattern
