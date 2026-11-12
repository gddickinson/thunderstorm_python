"""
Drift Correction
================

DriftCorrector: sample-drift estimation from fiducial markers or from the
cross-correlation of reconstructed images, and its correction. Re-exported by
postprocessing.py (`from thunderstorm_python.postprocessing import DriftCorrector`
still works). Positions in nm.
"""

import numpy as np
from scipy import ndimage, signal


class DriftCorrector:
    """Correct for sample drift during acquisition.
    
    Supports two methods:
    1. Fiducial marker tracking
    2. Cross-correlation of reconstructed images
    
    Parameters
    ----------
    method : str
        'fiducial' or 'cross_correlation'
    smoothing : float
        Smoothing parameter for drift trajectory
    """
    
    def __init__(self, method='cross_correlation', smoothing=0.25):
        self.method = method
        self.smoothing = smoothing
        self.drift_x = None
        self.drift_y = None
        
    def compute_drift_fiducial(self, localizations, fiducial_region, frames):
        """Compute drift from fiducial markers.
        
        Parameters
        ----------
        localizations : dict
            Localization data with 'x', 'y', 'frame'
        fiducial_region : tuple
            (x_min, x_max, y_min, y_max) defining fiducial region
        frames : ndarray
            Frame numbers
            
        Returns
        -------
        drift_x, drift_y : ndarray
            Drift in x and y for each frame
        """
        x_min, x_max, y_min, y_max = fiducial_region
        
        # Extract fiducial localizations
        mask = ((localizations['x'] >= x_min) & (localizations['x'] <= x_max) &
                (localizations['y'] >= y_min) & (localizations['y'] <= y_max))
        
        fid_x = localizations['x'][mask]
        fid_y = localizations['y'][mask]
        fid_frames = localizations['frame'][mask]
        
        # Compute mean position per frame
        drift_x = np.zeros(len(frames))
        drift_y = np.zeros(len(frames))
        
        for i, frame in enumerate(frames):
            frame_mask = fid_frames == frame
            if np.sum(frame_mask) > 0:
                drift_x[i] = np.mean(fid_x[frame_mask])
                drift_y[i] = np.mean(fid_y[frame_mask])
        
        # Smooth drift trajectory using LOWESS-like smoothing
        drift_x = self._smooth_trajectory(drift_x)
        drift_y = self._smooth_trajectory(drift_y)
        
        # Convert to relative drift (subtract initial position)
        drift_x = drift_x - drift_x[0]
        drift_y = drift_y - drift_y[0]
        
        self.drift_x = drift_x
        self.drift_y = drift_y
        
        return drift_x, drift_y
        
    def compute_drift_xcorr(self, localizations, frames, pixel_size=10, 
                           image_size=None, segment_frames=500):
        """Compute drift using cross-correlation.
        
        Parameters
        ----------
        localizations : dict
            Localization data with 'x', 'y', 'frame'
        frames : ndarray
            Frame numbers
        pixel_size : float
            Pixel size for reconstruction (nm)
        image_size : tuple, optional
            (width, height) of reconstruction
        segment_frames : int
            Number of frames per segment for cross-correlation
            
        Returns
        -------
        drift_x, drift_y : ndarray
            Drift in x and y for each frame
        """
        # Determine image size
        if image_size is None:
            x_max = int(np.max(localizations['x']) / pixel_size) + 1
            y_max = int(np.max(localizations['y']) / pixel_size) + 1
            image_size = (x_max, y_max)
        
        # Split frames into segments
        n_segments = len(frames) // segment_frames + 1
        segment_drift_x = []
        segment_drift_y = []
        segment_centers = []
        
        # Reconstruct first segment as reference
        ref_frames = frames[:segment_frames]
        ref_img = self._reconstruct_image(
            localizations, ref_frames, pixel_size, image_size
        )
        
        # Process each segment
        for i in range(n_segments):
            start_frame = i * segment_frames
            end_frame = min((i + 1) * segment_frames, len(frames))
            
            if start_frame >= len(frames):
                break
                
            seg_frames = frames[start_frame:end_frame]
            seg_img = self._reconstruct_image(
                localizations, seg_frames, pixel_size, image_size
            )
            
            # Cross-correlate with reference
            xcorr = signal.correlate2d(ref_img, seg_img, mode='same')
            
            # Find peak
            peak_y, peak_x = np.unravel_index(np.argmax(xcorr), xcorr.shape)
            
            # Convert to drift (relative to center)
            center_y, center_x = np.array(xcorr.shape) // 2
            drift_x_pix = peak_x - center_x
            drift_y_pix = peak_y - center_y
            
            segment_drift_x.append(drift_x_pix * pixel_size)
            segment_drift_y.append(drift_y_pix * pixel_size)
            segment_centers.append((start_frame + end_frame) / 2)
        
        # Interpolate drift for all frames
        drift_x = np.interp(np.arange(len(frames)), segment_centers, segment_drift_x)
        drift_y = np.interp(np.arange(len(frames)), segment_centers, segment_drift_y)
        
        # Smooth
        drift_x = self._smooth_trajectory(drift_x)
        drift_y = self._smooth_trajectory(drift_y)
        
        self.drift_x = drift_x
        self.drift_y = drift_y
        
        return drift_x, drift_y
        
    def apply_drift_correction(self, localizations):
        """Apply computed drift correction to localizations.
        
        Parameters
        ----------
        localizations : dict
            Localization data with 'x', 'y', 'frame'
            
        Returns
        -------
        corrected : dict
            Drift-corrected localizations
        """
        if self.drift_x is None or self.drift_y is None:
            raise ValueError("Must compute drift first")
        
        corrected = localizations.copy()
        
        # Apply drift correction per frame
        for frame_idx in np.unique(localizations['frame']):
            mask = localizations['frame'] == frame_idx
            if frame_idx < len(self.drift_x):
                corrected['x'][mask] -= self.drift_x[int(frame_idx)]
                corrected['y'][mask] -= self.drift_y[int(frame_idx)]
        
        return corrected
        
    def _reconstruct_image(self, localizations, frames, pixel_size, image_size):
        """Reconstruct super-resolution image from localizations."""
        img = np.zeros(image_size)
        
        for frame in frames:
            mask = localizations['frame'] == frame
            x_coords = (localizations['x'][mask] / pixel_size).astype(int)
            y_coords = (localizations['y'][mask] / pixel_size).astype(int)
            
            # Clip to image bounds
            valid = ((x_coords >= 0) & (x_coords < image_size[0]) &
                    (y_coords >= 0) & (y_coords < image_size[1]))
            
            x_coords = x_coords[valid]
            y_coords = y_coords[valid]
            
            # Accumulate
            for x, y in zip(x_coords, y_coords):
                img[y, x] += 1
        
        return img
        
    def _smooth_trajectory(self, trajectory):
        """Smooth drift trajectory using Gaussian filter."""
        window = int(len(trajectory) * self.smoothing)
        if window < 3:
            window = 3
        sigma = window / 6.0
        
        # Handle NaN values
        valid = ~np.isnan(trajectory)
        if not np.any(valid):
            return trajectory
            
        # Interpolate NaN values
        interp_traj = np.copy(trajectory)
        if not np.all(valid):
            indices = np.arange(len(trajectory))
            interp_traj = np.interp(indices, indices[valid], trajectory[valid])
        
        # Smooth
        smoothed = ndimage.gaussian_filter1d(interp_traj, sigma)
        
        return smoothed
