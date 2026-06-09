"""Policy/pipeline tests using a stub detector — no models, no network.

These lock down the decision logic: thresholds, min-severity gating, whole-image
classifier handling, and graceful failure of a broken detector.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from safety.pipeline import ExplicitContentFilter
from safety.config import SafetyConfig
from safety.types import Region, Category, Severity


class StubDetector:
    name = "stub"

    def __init__(self, regions):
        self._regions = regions
        self.available = True

    def analyze(self, image):
        return list(self._regions)


class BrokenDetector:
    name = "broken"
    available = True

    def analyze(self, image):
        raise RuntimeError("boom")


def _img(h=100, w=100):
    return np.zeros((h, w, 3), np.uint8)


def _filter(regions, **cfg):
    return ExplicitContentFilter(SafetyConfig(**cfg),
                                 detectors=[StubDetector(regions)])


def test_high_severity_nudity_is_flagged():
    r = Region(10, 10, 40, 40, "FEMALE_BREAST_EXPOSED", 0.9,
               Category.NUDITY, Severity.HIGH, "stub")
    v = _filter([r]).evaluate(_img())
    assert v.explicit and Category.NUDITY in v.categories()
    assert v.scores["nudity"] == 0.9


def test_below_threshold_is_not_flagged():
    r = Region(10, 10, 40, 40, "FEMALE_BREAST_EXPOSED", 0.20,
               Category.NUDITY, Severity.HIGH, "stub")
    v = _filter([r], nudity_threshold=0.35).evaluate(_img())
    assert not v.explicit


def test_min_severity_gating_skips_suggestive():
    r = Region(10, 10, 40, 40, "FEMALE_BREAST_COVERED", 0.9,
               Category.SUGGESTIVE, Severity.LOW, "stub")
    # Default min_blur_severity=MEDIUM -> a LOW (suggestive) hit is ignored.
    v = _filter([r], suggestive_threshold=0.3).evaluate(_img())
    assert not v.explicit
    # Lower the bar to LOW and it now blurs.
    v2 = _filter([r], suggestive_threshold=0.3,
                 min_blur_severity=Severity.LOW).evaluate(_img())
    assert v2.explicit


def test_safe_regions_never_blur():
    r = Region(0, 0, 100, 100, "safe", 0.99, Category.SAFE, Severity.NONE, "stub")
    v = _filter([r]).evaluate(_img())
    assert not v.explicit


def test_whole_image_classifier_hit():
    full = Region(0, 0, 100, 100, "nsfw", 0.95, Category.NUDITY,
                  Severity.HIGH, "classifier")
    v = _filter([full], whole_image_threshold=0.8).evaluate(_img())
    assert v.explicit and v.whole_image


def test_whole_image_below_threshold_does_not_blur_everything():
    full = Region(0, 0, 100, 100, "nsfw", 0.5, Category.NUDITY,
                  Severity.HIGH, "classifier")
    v = _filter([full], whole_image_threshold=0.8,
                nudity_threshold=0.35).evaluate(_img())
    # 0.5 > nudity_threshold but < whole_image_threshold -> not a full blur.
    assert not v.whole_image


def test_broken_detector_does_not_crash_pipeline():
    good = Region(10, 10, 40, 40, "MALE_GENITALIA_EXPOSED", 0.9,
                  Category.NUDITY, Severity.HIGH, "stub")
    filt = ExplicitContentFilter(
        SafetyConfig(), detectors=[BrokenDetector(), StubDetector([good])])
    v = filt.evaluate(_img())
    assert v.explicit  # the working detector still produced a verdict


def test_any_detection_blurs_whole_image_by_default():
    # Default policy: one localized detection -> the ENTIRE frame is blurred.
    rng = np.random.default_rng(1)
    img = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
    r = Region(20, 20, 80, 80, "FEMALE_GENITALIA_EXPOSED", 0.9,
               Category.NUDITY, Severity.HIGH, "stub")
    filt = ExplicitContentFilter(SafetyConfig(), detectors=[StubDetector([r])])
    clean, v = filt.scan_and_redact(img)
    assert v.explicit and v.whole_image
    # Even a far corner, nowhere near the detected box, is altered.
    assert not np.array_equal(clean[0:10, 0:10], img[0:10, 0:10])
    # Solid blur crushes detail: variance collapses far below the original.
    assert clean.var() < img.var() * 0.25


def test_localized_redaction_when_whole_image_disabled():
    # Opt out: blur only the detected region, leave the rest intact.
    rng = np.random.default_rng(1)
    img = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
    r = Region(20, 20, 80, 80, "FEMALE_GENITALIA_EXPOSED", 0.9,
               Category.NUDITY, Severity.HIGH, "stub")
    filt = ExplicitContentFilter(
        SafetyConfig(region_margin=0.0, feather=False,
                     whole_image_on_detection=False),
        detectors=[StubDetector([r])])
    clean, v = filt.scan_and_redact(img)
    assert v.explicit and not v.whole_image
    assert not np.array_equal(clean, img)                       # region redacted
    assert np.array_equal(clean[0:10, 0:10], img[0:10, 0:10])   # corner intact


def test_clean_image_returned_unchanged():
    img = _img()
    filt = ExplicitContentFilter(SafetyConfig(), detectors=[StubDetector([])])
    clean, v = filt.scan_and_redact(img)
    assert not v.explicit
    assert clean is img  # no copy made when nothing is flagged
