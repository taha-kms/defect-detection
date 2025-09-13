"""
Models package: contains wrappers for anomaly detection models
(PaDiM, PatchCore) using PyTorch backbones.
"""

from .padim import PaDiMModel
from .patchcore import PatchCoreModel

__all__ = ["PaDiMModel", "PatchCoreModel"]