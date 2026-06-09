"""Sentinel — explicit-content detection & blurring.

A small, dependency-light toolkit that finds nudity, sexual content, and
graphic wounds/gore in images and redacts (blurs) the offending regions. It
combines a box-level nudity detector (NudeNet/ONNX, fully offline), an offline
blood/wound heuristic, and an optional fine-tuned whole-image classifier behind
a single :class:`ExplicitContentFilter`.

The core (types, policy, blur engine) depends only on ``numpy`` + ``opencv``.
Heavy/optional backends (``nudenet``, ``onnxruntime``, ``torch``) are imported
lazily and degrade gracefully when absent.

Quick start::

    from safety import ExplicitContentFilter
    filt = ExplicitContentFilter()
    clean_image, verdict = filt.scan_and_redact(bgr_image)
"""
from __future__ import annotations

from .config import SafetyConfig, DEFAULT_CONFIG
from .types import Region, Verdict, Category, Severity
from .pipeline import ExplicitContentFilter
from .blur import redact, draw_boxes

__all__ = [
    "ExplicitContentFilter",
    "SafetyConfig",
    "DEFAULT_CONFIG",
    "Region",
    "Verdict",
    "Category",
    "Severity",
    "redact",
    "draw_boxes",
    "load_image",
    "save_image",
]

__version__ = "0.1.0"


def load_image(path: str):
    """Read an image file to a BGR numpy array (raises on failure)."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


def save_image(path: str, image) -> bool:
    import cv2

    return bool(cv2.imwrite(path, image))
