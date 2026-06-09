"""The explicit-content filter: detect -> decide -> redact.

:class:`ExplicitContentFilter` is the one object most callers need. It builds the
configured detectors, runs them over an image, applies the policy thresholds to
turn raw detections into a :class:`~safety.types.Verdict`, and (optionally)
returns a redacted copy of the image.

Typical use::

    from safety import ExplicitContentFilter
    filt = ExplicitContentFilter()
    clean, verdict = filt.scan_and_redact(bgr_image)
    if verdict.explicit:
        log.warning("blurred %s", verdict.categories())
"""
from __future__ import annotations

import numpy as np

from .config import SafetyConfig
from .types import Region, Verdict, Category, Severity, merge_regions
from .blur import redact
from .detectors.base import Detector


class ExplicitContentFilter:
    def __init__(self, config: SafetyConfig | None = None,
                 detectors: list[Detector] | None = None):
        self.config = config or SafetyConfig()
        self.detectors: list[Detector] = (
            detectors if detectors is not None else self._build_detectors()
        )

    # --- construction --------------------------------------------------------
    def _build_detectors(self) -> list[Detector]:
        cfg = self.config
        built: list[Detector] = []
        if cfg.use_nudenet:
            from .detectors.nudenet_detector import NudeNetDetector
            built.append(NudeNetDetector(min_confidence=cfg.nudenet_min_confidence))
        if cfg.use_wound_heuristic:
            from .detectors.wound import WoundHeuristicDetector
            built.append(WoundHeuristicDetector(min_score=cfg.gore_threshold * 0.9))
        if cfg.use_classifier and cfg.classifier_path:
            from .detectors.classifier import ClassifierDetector
            built.append(ClassifierDetector(cfg.classifier_path))
        return [d for d in built if getattr(d, "available", True)]

    @property
    def active_detectors(self) -> list[str]:
        return [d.name for d in self.detectors]

    # --- detection + policy --------------------------------------------------
    def detect(self, image: np.ndarray) -> list[Region]:
        """Run every active detector and return the merged raw regions."""
        groups = []
        for d in self.detectors:
            try:
                groups.append(d.analyze(image))
            except Exception:
                # A failing detector must never take down the pipeline; the
                # remaining detectors still provide coverage.
                groups.append([])
        return merge_regions(*groups)

    def evaluate(self, image: np.ndarray) -> Verdict:
        """Detect and apply the policy, but do not modify the image."""
        all_regions = self.detect(image)
        return self._apply_policy(all_regions, image.shape[1], image.shape[0])

    def _apply_policy(self, all_regions: list[Region], w: int, h: int) -> Verdict:
        cfg = self.config
        scores: dict[str, float] = {}
        for r in all_regions:
            scores[r.category.value] = max(scores.get(r.category.value, 0.0), r.score)

        to_blur: list[Region] = []
        whole_image = False
        reasons: list[str] = []

        for r in all_regions:
            if r.category is Category.SAFE:
                continue
            threshold = cfg.threshold_for(r.category)
            if r.score < threshold:
                continue
            if r.severity < cfg.min_blur_severity:
                continue

            # A full-frame classifier hit means "blur everything", but only for
            # the categories we allow to do that, and above the stricter
            # whole-image threshold.
            is_full = r.x1 <= 0 and r.y1 <= 0 and r.x2 >= w and r.y2 >= h
            if is_full and r.source == "classifier":
                if (r.category in cfg.whole_image_categories
                        and r.score >= cfg.whole_image_threshold):
                    whole_image = True
                    to_blur.append(r)
                    reasons.append(
                        f"whole-image {r.category.value} {r.score:.2f}")
                continue

            to_blur.append(r)
            reasons.append(
                f"{r.label} ({r.category.value}/{r.severity}) {r.score:.2f}")

        # Policy: a single detection anywhere blurs the entire frame (default).
        if to_blur and cfg.whole_image_on_detection:
            whole_image = True

        return Verdict(
            explicit=bool(to_blur),
            regions=to_blur,
            all_regions=all_regions,
            scores=scores,
            whole_image=whole_image,
            reasons=reasons,
        )

    # --- one-shot convenience ------------------------------------------------
    def scan_and_redact(self, image: np.ndarray) -> tuple[np.ndarray, Verdict]:
        """Return ``(possibly_blurred_copy, verdict)``.

        If nothing is flagged the original image is returned unchanged (a copy is
        only made when redaction happens).
        """
        verdict = self.evaluate(image)
        if not verdict.explicit:
            return image, verdict
        clean = redact(image, verdict.regions, self.config,
                       whole_image=verdict.whole_image)
        return clean, verdict
