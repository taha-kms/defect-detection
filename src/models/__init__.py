"""
Models package: contains wrappers for anomaly detection models
(PaDiM, PatchCore) using PyTorch backbones.
"""

from .padim import PaDiMModel
from .patchcore import PatchCoreModel
from .ae import AEModel

__all__ = ["PaDiMModel", "PatchCoreModel", "AEModel"]