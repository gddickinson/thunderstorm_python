"""
Post-Processing Module
=====================

Implements post-processing methods:
- Drift correction (fiducial markers and cross-correlation; in drift.py)
- Merging of reappearing molecules
- Duplicate removal
- Filtering based on localization quality
- Local density filtering
- Z-stage offset correction
"""

import numpy as np
from scipy.spatial import cKDTree

from .drift import DriftCorrector  # noqa: F401  (re-exported for backward compatibility)


class MolecularMerger:
    """Merge molecules appearing in subsequent frames.
    
    Accounts for blinking - molecules can disappear for several frames.
    
    Parameters
    ----------
    max_distance : float
        Maximum distance (nm) to consider molecules the same
    max_frame_gap : int
        Maximum frame gap for reappearance
    """
    
    def __init__(self, max_distance=50, max_frame_gap=1):
        self.max_distance = max_distance
        self.max_frame_gap = max_frame_gap
        
    def merge(self, localizations):
        """Merge reappearing molecules.
        
        Parameters
        ----------
        localizations : dict
            Localization data
            
        Returns
        -------
        merged : dict
            Merged localizations
        """
        # Sort by frame
        sort_idx = np.argsort(localizations['frame'])
        
        x = localizations['x'][sort_idx]
        y = localizations['y'][sort_idx]
        frames = localizations['frame'][sort_idx]
        
        # Track which localizations have been merged
        merged_into = -np.ones(len(x), dtype=int)  # -1 means not merged
        
        # Process frame by frame
        unique_frames = np.unique(frames)
        
        for i, frame in enumerate(unique_frames[:-1]):
            # Get localizations in current frame
            current_mask = frames == frame
            current_indices = np.where(current_mask)[0]
            
            # Look ahead up to max_frame_gap
            for gap in range(1, self.max_frame_gap + 1):
                if i + gap >= len(unique_frames):
                    break
                    
                next_frame = unique_frames[i + gap]
                next_mask = frames == next_frame
                next_indices = np.where(next_mask)[0]
                
                # Build KD-tree for current frame
                current_positions = np.column_stack([x[current_indices], y[current_indices]])
                
                if len(current_positions) == 0:
                    continue
                    
                tree = cKDTree(current_positions)
                
                # Query for each next frame localization
                next_positions = np.column_stack([x[next_indices], y[next_indices]])
                
                for j, next_idx in enumerate(next_indices):
                    if merged_into[next_idx] >= 0:
                        # Already merged
                        continue
                        
                    # Find nearest neighbor in current frame
                    dist, idx = tree.query(next_positions[j])
                    
                    if dist <= self.max_distance:
                        # Merge: mark as merged into current localization
                        current_idx = current_indices[idx]
                        merged_into[next_idx] = current_idx
        
        # Create merged localizations
        # Keep only localizations that are either:
        # 1. Not merged (merged_into == -1)
        # 2. Are the target of a merge
        
        keep_mask = merged_into == -1
        
        merged = {}
        for key in localizations:
            if key in ['x', 'y', 'intensity', 'background', 'sigma_x', 'sigma_y', 'frame']:
                merged[key] = localizations[key][sort_idx][keep_mask]
        
        return merged


class LocalizationFilter:
    """Filter localizations based on quality criteria.
    
    Parameters
    ----------
    min_intensity : float, optional
        Minimum intensity
    max_intensity : float, optional
        Maximum intensity  
    max_uncertainty : float, optional
        Maximum localization uncertainty (nm)
    min_sigma : float, optional
        Minimum PSF sigma
    max_sigma : float, optional
        Maximum PSF sigma
    """
    
    def __init__(self, min_intensity=None, max_intensity=None,
                 max_uncertainty=None, min_sigma=None, max_sigma=None):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.max_uncertainty = max_uncertainty
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
    def filter(self, localizations):
        """Apply filters to localizations.
        
        Parameters
        ----------
        localizations : dict
            Localization data
            
        Returns
        -------
        filtered : dict
            Filtered localizations
        """
        mask = np.ones(len(localizations['x']), dtype=bool)
        
        # Apply intensity filters
        if self.min_intensity is not None:
            mask &= localizations['intensity'] >= self.min_intensity
            
        if self.max_intensity is not None:
            mask &= localizations['intensity'] <= self.max_intensity
        
        # Apply uncertainty filter
        if self.max_uncertainty is not None and 'uncertainty' in localizations:
            mask &= localizations['uncertainty'] <= self.max_uncertainty
        
        # Apply sigma filters
        if self.min_sigma is not None:
            mask &= localizations['sigma_x'] >= self.min_sigma
            
        if self.max_sigma is not None:
            mask &= localizations['sigma_x'] <= self.max_sigma
        
        # Apply mask
        filtered = {}
        for key, value in localizations.items():
            if isinstance(value, np.ndarray):
                filtered[key] = value[mask]
            else:
                filtered[key] = value
                
        return filtered


class LocalDensityFilter:
    """Filter based on local density (removes isolated localizations).
    
    Parameters
    ----------
    radius : float
        Search radius (nm)
    min_neighbors : int
        Minimum number of neighbors required
    use_3d : bool
        Use 3D distance (requires 'z' coordinate)
    """
    
    def __init__(self, radius=50, min_neighbors=3, use_3d=False):
        self.radius = radius
        self.min_neighbors = min_neighbors
        self.use_3d = use_3d
        
    def filter(self, localizations):
        """Filter by local density.
        
        Parameters
        ----------
        localizations : dict
            Localization data
            
        Returns
        -------
        filtered : dict
            Filtered localizations
        """
        # Build position array
        if self.use_3d and 'z' in localizations:
            positions = np.column_stack([
                localizations['x'],
                localizations['y'],
                localizations['z']
            ])
        else:
            positions = np.column_stack([
                localizations['x'],
                localizations['y']
            ])
        
        # Build KD-tree
        tree = cKDTree(positions)
        
        # Query for neighbors
        neighbors = tree.query_ball_point(positions, self.radius)
        
        # Count neighbors (excluding self)
        neighbor_counts = np.array([len(n) - 1 for n in neighbors])
        
        # Filter
        mask = neighbor_counts >= self.min_neighbors
        
        filtered = {}
        for key, value in localizations.items():
            if isinstance(value, np.ndarray):
                filtered[key] = value[mask]
            else:
                filtered[key] = value
                
        return filtered


class DuplicateRemover:
    """Remove duplicate localizations from multi-emitter fitting.
    
    Parameters
    ----------
    max_distance : float
        Maximum distance (nm) to consider duplicates
    """
    
    def __init__(self, max_distance=20):
        self.max_distance = max_distance
        
    def remove_duplicates(self, localizations):
        """Remove duplicate localizations within same frame.
        
        Parameters
        ----------
        localizations : dict
            Localization data
            
        Returns
        -------
        unique : dict
            Localizations with duplicates removed
        """
        unique_mask = np.ones(len(localizations['x']), dtype=bool)
        
        # Process frame by frame
        for frame in np.unique(localizations['frame']):
            frame_mask = localizations['frame'] == frame
            frame_indices = np.where(frame_mask)[0]
            
            if len(frame_indices) <= 1:
                continue
            
            # Build KD-tree for frame
            positions = np.column_stack([
                localizations['x'][frame_indices],
                localizations['y'][frame_indices]
            ])
            
            tree = cKDTree(positions)
            
            # Find pairs within max_distance
            pairs = tree.query_pairs(self.max_distance)
            
            # For each pair, keep the brighter one
            for i, j in pairs:
                idx_i = frame_indices[i]
                idx_j = frame_indices[j]
                
                if localizations['intensity'][idx_i] < localizations['intensity'][idx_j]:
                    unique_mask[idx_i] = False
                else:
                    unique_mask[idx_j] = False
        
        # Apply mask
        unique = {}
        for key, value in localizations.items():
            if isinstance(value, np.ndarray):
                unique[key] = value[unique_mask]
            else:
                unique[key] = value
                
        return unique


def z_stage_offset_correction(localizations, z_stage_positions, frame_to_zstage):
    """Correct Z positions for multi-Z-stage acquisition.
    
    Parameters
    ----------
    localizations : dict
        Localization data with 'z' positions
    z_stage_positions : dict
        Mapping of Z-stage index to absolute Z position
    frame_to_zstage : dict
        Mapping of frame number to Z-stage index
        
    Returns
    -------
    corrected : dict
        Z-corrected localizations
    """
    if 'z' not in localizations:
        return localizations
        
    corrected = localizations.copy()
    
    for frame in np.unique(localizations['frame']):
        if frame in frame_to_zstage:
            z_stage_idx = frame_to_zstage[frame]
            if z_stage_idx in z_stage_positions:
                z_offset = z_stage_positions[z_stage_idx]
                mask = localizations['frame'] == frame
                corrected['z'][mask] += z_offset
    
    return corrected
