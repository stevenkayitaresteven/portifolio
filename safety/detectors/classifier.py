"""Whole-image classifier detector backed by a fine-tuned model.

This consumes a checkpoint produced by ``safety/train`` (see that package's
README). It supports two formats so inference doesn't have to drag in PyTorch:

* ``.onnx``  -> run with ``onnxruntime`` (lightweight, CPU, no torch needed)
* ``.pt``    -> a TorchScript module, run with ``torch`` if available

The model is a multi-label classifier over our categories (e.g.
``safe / nudity / gore``). Unlike the box detectors, it scores the *whole frame*;
when a class crosses ``whole_image_threshold`` the pipeline blurs the entire
image (there is no localized box to redact).

The checkpoint directory / sidecar is expected to provide the class order via a
``labels.json`` next to the model file, e.g. ``["safe", "nudity", "gore"]``.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .base import BaseDetector
from ..types import Region, Category, Severity


# How the trained class names map onto our taxonomy. Unknown class names default
# to NUDITY/HIGH so an unexpected label still errs on the side of redaction.
_CLASS_TO_CATEGORY = {
    "safe": Category.SAFE,
    "neutral": Category.SAFE,
    "nudity": Category.NUDITY,
    "nsfw": Category.NUDITY,
    "porn": Category.NUDITY,
    "sexy": Category.SUGGESTIVE,
    "suggestive": Category.SUGGESTIVE,
    "gore": Category.GORE,
    "wound": Category.GORE,
    "blood": Category.GORE,
}


class ClassifierDetector(BaseDetector):
    name = "classifier"

    def __init__(self, model_path: str, input_size: int = 224):
        self.model_path = model_path
        self.input_size = input_size
        self.labels: list[str] = []
        self._session = None     # onnxruntime
        self._module = None      # torchscript
        self._backend: str | None = None
        self._load_labels()

    # --- availability / loading ---------------------------------------------
    @property
    def available(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        if self.model_path.endswith(".onnx"):
            try:
                import onnxruntime  # noqa: F401
                return True
            except Exception:
                return False
        if self.model_path.endswith((".pt", ".pth", ".ts")):
            try:
                import torch  # noqa: F401
                return True
            except Exception:
                return False
        return False

    def _load_labels(self):
        sidecar = os.path.join(os.path.dirname(self.model_path), "labels.json")
        if os.path.exists(sidecar):
            try:
                self.labels = json.load(open(sidecar))
            except Exception:
                self.labels = []

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        if self.model_path.endswith(".onnx"):
            import onnxruntime
            self._session = onnxruntime.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self._backend = "onnx"
        else:
            import torch
            self._module = torch.jit.load(self.model_path, map_location="cpu")
            self._module.eval()
            self._backend = "torch"

    # --- inference -----------------------------------------------------------
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2

        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalization (matches the training transforms).
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        chw = np.transpose(img, (2, 0, 1))[None]  # NCHW
        return chw.astype(np.float32)

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        if self._backend == "onnx":
            name = self._session.get_inputs()[0].name
            logits = self._session.run(None, {name: batch})[0]
        else:
            import torch
            with torch.no_grad():
                logits = self._module(torch.from_numpy(batch)).cpu().numpy()
        return logits[0]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def analyze(self, image: np.ndarray) -> list[Region]:
        if not self.available:
            return []
        self._ensure_loaded()
        logits = self._infer(self._preprocess(image))
        probs = self._sigmoid(logits)

        h, w = image.shape[:2]
        regions: list[Region] = []
        for i, p in enumerate(probs):
            name = self.labels[i] if i < len(self.labels) else f"class_{i}"
            category = _CLASS_TO_CATEGORY.get(name.lower(), Category.NUDITY)
            if category is Category.SAFE:
                continue
            severity = Severity.HIGH if category is Category.NUDITY else Severity.MEDIUM
            # Full-frame region: the pipeline reads source=="classifier" +
            # full-frame to mean "blur the whole image".
            regions.append(Region(
                x1=0, y1=0, x2=w, y2=h, label=name, score=float(p),
                category=category, severity=severity, source=self.name,
            ))
        return regions
