"""
Quick-start helpers
===================

create_default_pipeline() and quick_analysis(): a ThunderSTORM pipeline with
the default settings, and a one-call analysis + Gaussian rendering. Also
importable from the package root and from pipeline.py.
"""

from .pipeline import ThunderSTORM


def create_default_pipeline():
    """Create ThunderSTORM pipeline with default settings.
    
    These are the default settings that work well on many datasets.
    
    Returns
    -------
    pipeline : ThunderSTORM
        Configured pipeline
    """
    return ThunderSTORM(
        filter_type='wavelet',
        filter_params={'scale': 2, 'order': 3},
        detector_type='local_maximum',
        detector_params={'connectivity': '8-neighbourhood', 'min_distance': 1},
        fitter_type='gaussian_lsq',
        fitter_params={'integrated': True, 'elliptical': False, 'initial_sigma': 1.3},
        threshold_expression='std(Wave.F1)',
        pixel_size=100.0,  # nm
        photons_per_adu=1.0,
        baseline=100.0,
        em_gain=1.0
    )


def quick_analysis(images, **kwargs):
    """Quick analysis with default parameters.
    
    Parameters
    ----------
    images : ndarray
        Image stack to analyze
    **kwargs : dict
        Additional parameters for pipeline
        
    Returns
    -------
    localizations : dict
        Detected localizations
    rendered : ndarray
        Rendered super-resolution image
    """
    # Create pipeline
    pipeline = create_default_pipeline()
    
    # Update any custom parameters
    for key, value in kwargs.items():
        if hasattr(pipeline, key):
            setattr(pipeline, key, value)
    
    # Analyze
    localizations = pipeline.analyze_stack(images)
    
    # Render
    rendered = pipeline.render(renderer_type='gaussian', pixel_size=10)
    
    return localizations, rendered, pipeline
