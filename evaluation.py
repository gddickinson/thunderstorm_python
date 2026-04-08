"""
Performance Evaluation Module
=============================

Evaluate localization algorithm performance:
- Compare detected localizations to ground truth
- Compute precision, recall, F1, RMSE metrics
- Generate test patterns for validation
"""

import numpy as np
from scipy.spatial import cKDTree


class PerformanceEvaluator:
    """Evaluate localization algorithm performance.

    Compare detected localizations to ground truth.

    Parameters
    ----------
    tolerance : float
        Matching tolerance in nm
    """

    def __init__(self, tolerance=100.0):
        self.tolerance = tolerance

    def evaluate(self, detected, ground_truth):
        """Evaluate detection performance.

        Parameters
        ----------
        detected : dict
            Detected localizations with 'x', 'y'
        ground_truth : dict
            Ground truth positions with 'x', 'y'

        Returns
        -------
        metrics : dict
            Performance metrics (recall, precision, F1, RMSE, etc.)
        """
        # Build trees
        if len(detected['x']) == 0:
            return {
                'n_true_positive': 0,
                'n_false_positive': 0,
                'n_false_negative': len(ground_truth['x']),
                'recall': 0.0,
                'precision': 0.0,
                'f1_score': 0.0,
                'rmse_x': np.nan,
                'rmse_y': np.nan,
                'jaccard': 0.0
            }

        det_positions = np.column_stack([detected['x'], detected['y']])
        gt_positions = np.column_stack([ground_truth['x'], ground_truth['y']])

        det_tree = cKDTree(det_positions)
        gt_tree = cKDTree(gt_positions)

        # Find matches (ground truth to detected)
        distances, indices = det_tree.query(gt_positions)
        matches_gt = distances <= self.tolerance

        # Find matches (detected to ground truth)
        distances_det, indices_det = gt_tree.query(det_positions)
        matches_det = distances_det <= self.tolerance

        # Count true positives, false positives, false negatives
        n_true_positive = np.sum(matches_gt)
        n_false_negative = len(gt_positions) - n_true_positive
        n_false_positive = len(det_positions) - np.sum(matches_det)

        # Compute metrics
        recall = n_true_positive / len(gt_positions) if len(gt_positions) > 0 else 0
        precision = n_true_positive / len(det_positions) if len(det_positions) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0)
        jaccard = n_true_positive / (n_true_positive + n_false_positive + n_false_negative)

        # Compute localization error for true positives
        if n_true_positive > 0:
            matched_det_indices = indices[matches_gt]
            matched_gt = gt_positions[matches_gt]
            matched_det = det_positions[matched_det_indices]

            error_x = matched_det[:, 0] - matched_gt[:, 0]
            error_y = matched_det[:, 1] - matched_gt[:, 1]

            rmse_x = np.sqrt(np.mean(error_x**2))
            rmse_y = np.sqrt(np.mean(error_y**2))
            rmse = np.sqrt(np.mean(error_x**2 + error_y**2))
        else:
            rmse_x = np.nan
            rmse_y = np.nan
            rmse = np.nan

        return {
            'n_true_positive': n_true_positive,
            'n_false_positive': n_false_positive,
            'n_false_negative': n_false_negative,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse': rmse,
            'jaccard': jaccard
        }


def create_test_pattern(pattern_type='siemens_star', size=256):
    """Create test patterns for simulation.

    Parameters
    ----------
    pattern_type : str
        'siemens_star', 'grid', 'circle', 'random'
    size : int
        Image size

    Returns
    -------
    mask : ndarray
        Pattern mask (values 0-1)
    """
    if pattern_type == 'siemens_star':
        # Siemens star pattern
        y, x = np.mgrid[-size//2:size//2, -size//2:size//2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        n_spokes = 24
        mask = np.cos(n_spokes * theta) > 0
        mask = mask.astype(float)

        # Fade at center and edge
        mask *= np.clip(r / 20 - 1, 0, 1)
        mask *= np.clip(1 - r / (size // 2), 0, 1)

    elif pattern_type == 'grid':
        # Grid pattern
        mask = np.zeros((size, size))
        spacing = size // 10
        mask[::spacing, :] = 1
        mask[:, ::spacing] = 1

    elif pattern_type == 'circle':
        # Concentric circles
        y, x = np.mgrid[-size//2:size//2, -size//2:size//2]
        r = np.sqrt(x**2 + y**2)

        mask = np.zeros((size, size))
        for radius in range(20, size//2, 20):
            ring = (np.abs(r - radius) < 5).astype(float)
            mask += ring

        mask = np.clip(mask, 0, 1)

    elif pattern_type == 'random':
        # Uniform random
        mask = np.ones((size, size))

    else:
        mask = np.ones((size, size))

    return mask
