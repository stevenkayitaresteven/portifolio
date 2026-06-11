"""Whole-image NSFW classifier backed by a Hugging Face checkpoint.

Default checkpoint: ``Falconsai/nsfw_image_detection`` — a ViT fine-tuned for
binary nsfw/normal image classification (Apache-2.0). Like every detector it
only *localizes and scores* (here: one full-frame region); thresholds and the
blur decision stay in the policy (:mod:`safety.pipeline`), where a full-frame
``source="classifier"`` hit goes through the stricter whole-image gate.

The model downloads once on first use and is cached by ``transformers``. If
the library is missing or the checkpoint can't be fetched (offline box), the
detector reports itself unavailable / returns nothing — the offline detectors
(NudeNet, wound heuristic) keep covering.
"""
from __future__ import annotations

import numpy as np

from ..types import Category, Region, Severity
from .base import BaseDetector

# Labels various NSFW checkpoints emit for the positive class.
_NSFW_LABELS = {"nsfw", "porn", "pornography", "hentai", "sexy", "explicit"}


class HFNSFWDetector(BaseDetector):
    name = "hf_nsfw"

    def __init__(self, model_id: str = "Falconsai/nsfw_image_detection",
                 min_confidence: float = 0.25):
        self.model_id = model_id
        self.min_confidence = min_confidence
        self._pipe = None
        self._failed = False

    @property
    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return not self._failed
        except Exception:
            return False

    def _load(self):
        if self._pipe is not None or self._failed:
            return self._pipe
        try:
            from transformers import pipeline

            self._pipe = pipeline("image-classification", model=self.model_id)
        except Exception:
            self._failed = True
        return self._pipe

    def analyze(self, image: np.ndarray) -> list[Region]:
        pipe = self._load()
        if pipe is None:
            return []
        try:
            from PIL import Image

            rgb = image[:, :, ::-1]  # BGR (OpenCV) -> RGB (PIL)
            rows = pipe(Image.fromarray(rgb))
        except Exception:
            return []

        score = max((float(r["score"]) for r in rows
                     if r["label"].lower() in _NSFW_LABELS), default=0.0)
        if score < self.min_confidence:
            return []
        h, w = image.shape[:2]
        return [Region(x1=0, y1=0, x2=w, y2=h,
                       label=f"nsfw:{self.model_id}", score=score,
                       category=Category.NUDITY, severity=Severity.HIGH,
                       source="classifier")]
