"""
Packaging regression tests: the package imports with only its declared
dependencies, and pyproject.toml names a build backend that exists.
"""

import importlib
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))


def test_filters_import_without_pywavelets(monkeypatch):
    """filters.py must not need PyWavelets (the a trous wavelet is pure numpy)."""
    monkeypatch.setitem(sys.modules, 'pywt', None)  # make `import pywt` fail
    for name in [m for m in sys.modules if m.startswith('thunderstorm_python')]:
        monkeypatch.delitem(sys.modules, name)
    mod = importlib.import_module('thunderstorm_python.filters')
    assert hasattr(mod, 'WaveletFilter')


def test_declared_dependencies_cover_imports():
    meta = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    deps = ' '.join(meta['project']['dependencies']).lower()
    assert 'pywt' not in deps            # not a PyPI distribution name
    for needed in ('numpy', 'scipy', 'scikit-image', 'scikit-learn', 'pandas'):
        assert needed in deps, needed


def test_build_backend_exists():
    meta = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    backend = meta['build-system']['build-backend'].split(':')[0]
    importlib.import_module(backend)
