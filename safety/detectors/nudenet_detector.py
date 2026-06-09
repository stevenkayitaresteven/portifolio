"""NudeNet-backed nudity detector (box-level, runs on CPU, fully offline).

NudeNet ships a small ONNX model (``320n.onnx``) inside its wheel, so this
detector needs no network access at runtime — important here because the
environment can reach PyPI but not model hubs. It emits 18 raw labels which we
normalize to our taxonomy via ``NUDENET_LABEL_MAP``.

Install the backend with::

    pip install nudenet onnxruntime

If ``nudenet`` isn't installed, the detector reports ``available is False`` and
the pipeline simply skips it (degrading gracefully to the other detectors).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from .base import BaseDetector
from ..config import NUDENET_LABEL_MAP
from ..types import Region


class NudeNetDetector(BaseDetector):
    name = "nudenet"

    def __init__(self, min_confidence: float = 0.25):
        self.min_confidence = min_confidence
        self._detector = None
        self._import_error: str | None = None
        try:
            from nudenet import NudeDetector  # noqa: F401
            self._NudeDetector = NudeDetector
        except Exception as exc:  # pragma: no cover - depends on optional extra
            self._NudeDetector = None
            self._import_error = str(exc)

    @property
    def available(self) -> bool:
        return self._NudeDetector is not None

    def _ensure_model(self):
        if self._detector is None:
            if not self.available:
                raise RuntimeError(
                    "nudenet is not installed; `pip install nudenet onnxruntime`"
                    f" (import error: {self._import_error})"
                )
            self._detector = self._NudeDetector()
        return self._detector

    def analyze(self, image: np.ndarray) -> list[Region]:
        if not self.available:
            return []
        det = self._ensure_model()
        h, w = image.shape[:2]

        # NudeNet's API takes a path; write the frame to a short-lived temp file.
        import cv2
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            cv2.imwrite(path, image)
            raw = det.detect(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        regions: list[Region] = []
        for d in raw:
            label = d.get("class", "")
            score = float(d.get("score", 0.0))
            if score < self.min_confidence:
                continue
            mapped = NUDENET_LABEL_MAP.get(label)
            if mapped is None:
                continue  # face / feet / armpits / belly-covered -> ignore
            category, severity = mapped
            bx, by, bw, bh = d.get("box", [0, 0, 0, 0])
            x1 = max(0, int(bx)); y1 = max(0, int(by))
            x2 = min(w, int(bx + bw)); y2 = min(h, int(by + bh))
            if x2 <= x1 or y2 <= y1:
                continue
            regions.append(Region(
                x1=x1, y1=y1, x2=x2, y2=y2, label=label, score=score,
                category=category, severity=severity, source=self.name,
            ))
        return regions
