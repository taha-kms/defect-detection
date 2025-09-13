"""
Models package: contains wrappers for anomaly detection models
(PaDiM, PatchCore, AutoEncoder, FastFlow) using PyTorch backbones.
"""

from .padim import PaDiMModel
from .patchcore import PatchCoreModel
from .ae import AEModel
from .fastflow import FastFlowModel

__all__ = ["PaDiMModel", "PatchCoreModel", "AEModel", "FastFlowModel"]
