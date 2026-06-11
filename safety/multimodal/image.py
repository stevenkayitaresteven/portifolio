"""Image moderation: the existing detect→policy→blur pipeline, plus the
Hugging Face whole-image NSFW classifier, reported as a :class:`ModerationResult`.

The ensemble is complementary: NudeNet localizes *which* body parts are exposed
(box-level, fully offline), while ``Falconsai/nsfw_image_detection`` judges the
*whole frame* and catches compositions box detectors miss. Either one crossing
the policy threshold blurs the image.
"""
from __future__ import annotations

import numpy as np

from ..pipeline import ExplicitContentFilter
from ..types import Verdict
from .config import MultimodalConfig
from .result import ModerationResult, ACTION_BLUR, ACTION_NONE


class ImageModerator:
    def __init__(self, config: MultimodalConfig | None = None):
        self.config = config or MultimodalConfig.from_env()
        self.filter = ExplicitContentFilter(self.config.safety)
        if self.config.use_hf_image:
            from ..detectors.hf_nsfw import HFNSFWDetector

            det = HFNSFWDetector(self.config.image_model)
            if det.available:
                self.filter.detectors.append(det)

    @property
    def active_detectors(self) -> list[str]:
        return self.filter.active_detectors

    def _to_result(self, verdict: Verdict) -> ModerationResult:
        res = ModerationResult(
            modality="image",
            flagged=verdict.explicit,
            action=ACTION_BLUR if verdict.explicit else ACTION_NONE,
            categories=sorted(c.value for c in verdict.categories()),
            scores=dict(verdict.scores),
            reasons=list(verdict.reasons),
            detectors=self.active_detectors,
            scanned=bool(self.filter.detectors),
            extra={"whole_image": verdict.whole_image,
                   "num_regions": len(verdict.regions)},
        )
        return res

    def moderate_bgr(self, image: np.ndarray) -> tuple[np.ndarray, ModerationResult]:
        """Return ``(delivered_image, result)`` — blurred copy when flagged."""
        clean, verdict = self.filter.scan_and_redact(image)
        return clean, self._to_result(verdict)

    def evaluate_bgr(self, image: np.ndarray) -> ModerationResult:
        """Verdict only, no redaction (used by the video frame sampler)."""
        return self._to_result(self.filter.evaluate(image))

    def moderate_bytes(self, data: bytes) -> tuple[bytes | None, ModerationResult]:
        """Decode → moderate → re-encode. Returns ``(jpeg_bytes, result)``;
        bytes are ``None`` when the payload is not a decodable image."""
        import cv2

        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            res = ModerationResult(modality="image", scanned=False,
                                   reasons=["could not decode image"])
            return None, res
        out, res = self.moderate_bgr(img)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return (buf.tobytes() if ok else None), res
