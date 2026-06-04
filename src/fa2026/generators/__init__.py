"""Data generator implementations for CMF.

This module contains the core data generators for Covariance Matrix Fitting (CMF).
"""

from .generator import CMFDataGenerator

# Optional imports - only available if the respective frameworks are installed
try:
    from .generator_torch import CMFTorchDataset
except Exception:
    CMFTorchDataset = None

try:
    from .generator_tf import make_tf_dataset
except Exception:
    make_tf_dataset = None

__all__ = ["CMFDataGenerator", "CMFTorchDataset", "make_tf_dataset"]