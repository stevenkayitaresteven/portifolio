"""Offline heuristic detector for wounds / blood / graphic injury.

There is no small, offline, pip-installable gore model the way NudeNet exists
for nudity, and this environment cannot reach model hubs. So for the gore
category we ship two things:

1. **This heuristic** — a classical OpenCV blood detector that localizes large,
   highly-saturated red regions with the texture characteristics of blood. It is
   a *baseline*: it will flag a bright-red wound or blood pool, but also risks
   flagging red paint, roses, or a red sofa. It is fast, deterministic, needs no
   weights, and gives bounding boxes for targeted blurring.
2. **A fine-tunable classifier** (see ``safety/train``) — when you have a
   labeled gore dataset, train the whole-image model and enable
   ``use_classifier`` for far better precision. The heuristic then becomes a
   localization assist / fallback.

Because of the false-positive risk, gore detections default to MEDIUM severity
and a relatively high score threshold, so a single red object won't blur a
listing photo unless the red region is large and blood-like.
"""
from __future__ import annotations

import numpy as np

from .base import BaseDetector
from ..types import Region, Category, Severity


# Blood-red lives at both ends of the hue wheel in OpenCV's 0-179 H space.
_LOWER_RED_1 = np.array([0, 80, 50], dtype=np.uint8)
_UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)
_LOWER_RED_2 = np.array([170, 80, 50], dtype=np.uint8)
_UPPER_RED_2 = np.array([179, 255, 255], dtype=np.uint8)


class WoundHeuristicDetector(BaseDetector):
    name = "wound_heuristic"

    def __init__(
        self,
        min_area_frac: float = 0.015,   # red blob must cover >=1.5% of the image
        min_score: float = 0.5,
        max_regions: int = 4,
    ):
        self.min_area_frac = min_area_frac
        self.min_score = min_score
        self.max_regions = max_regions

    @property
    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except Exception:  # pragma: no cover
            return False

    def analyze(self, image: np.ndarray) -> list[Region]:
        import cv2

        if image is None or image.ndim != 3:
            return []
        h, w = image.shape[:2]
        total = float(h * w)
        if total == 0:
            return []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _LOWER_RED_1, _UPPER_RED_1) | cv2.inRange(
            hsv, _LOWER_RED_2, _UPPER_RED_2
        )
        # Clean speckle, then close gaps so a wound reads as one blob.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        red_frac = float(mask.sum()) / 255.0 / total
        if red_frac < self.min_area_frac:
            return []

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[Region] = []
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            frac = area / total
            if frac < self.min_area_frac:
                break  # contours are sorted; the rest are smaller
            x, y, bw, bh = cv2.boundingRect(c)
            score = self._blood_score(hsv, mask, x, y, bw, bh, frac)
            if score < self.min_score:
                continue
            regions.append(Region(
                x1=x, y1=y, x2=min(w, x + bw), y2=min(h, y + bh),
                label="wound", score=round(score, 4),
                category=Category.GORE, severity=Severity.MEDIUM,
                source=self.name,
            ))
            if len(regions) >= self.max_regions:
                break
        return regions

    @staticmethod
    def _blood_score(hsv, mask, x, y, bw, bh, frac) -> float:
        """Heuristic confidence: blood is large, irregular, and very saturated.

        Combines (a) how much of the bounding box is actually red (fill ratio,
        which is low for a thin red line and high for a pool), (b) mean
        saturation of the red pixels, and (c) area fraction. Each term is in
        0-1; the product-ish blend keeps thin/low-saturation reds well under the
        threshold.
        """
        import numpy as np

        sub_mask = mask[y:y + bh, x:x + bw] > 0
        if sub_mask.sum() == 0:
            return 0.0
        fill = float(sub_mask.sum()) / float(bw * bh)            # 0-1
        sat = hsv[y:y + bh, x:x + bw, 1][sub_mask].astype(np.float32)
        sat_term = float(np.clip(sat.mean() / 255.0, 0, 1))      # 0-1
        area_term = float(np.clip(frac * 6.0, 0, 1))             # saturates ~16%
        # Irregularity: a perfectly rectangular fill (fill~1) is more likely a
        # design element than a wound, so favour mid fill ratios slightly.
        shape_term = 1.0 - abs(fill - 0.6)
        score = 0.45 * sat_term + 0.30 * area_term + 0.25 * max(0.0, shape_term)
        return float(np.clip(score, 0, 1))
