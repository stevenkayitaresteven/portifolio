"""Pluggable detectors. Each returns localized Regions; the pipeline decides."""
from .base import Detector, BaseDetector

__all__ = ["Detector", "BaseDetector"]
