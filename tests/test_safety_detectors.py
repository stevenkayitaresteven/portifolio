"""Detector tests.

The wound heuristic is pure OpenCV and always runs. The NudeNet test is skipped
unless the optional `nudenet` extra is installed (and never hits the network —
its ONNX model ships in the wheel).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from safety.detectors.wound import WoundHeuristicDetector
from safety.types import Category, Severity
from safety.config import SafetyConfig, NUDENET_LABEL_MAP


def test_wound_detector_flags_large_red_blob():
    img = np.full((300, 300, 3), 200, np.uint8)        # light grey background
    # A big, saturated red region (BGR red = (0,0,255)) ~ a wound/blood pool.
    cv2.circle(img, (150, 150), 90, (0, 0, 255), -1)
    det = WoundHeuristicDetector(min_score=0.4)
    regions = det.analyze(img)
    assert regions, "expected at least one gore region"
    assert regions[0].category is Category.GORE
    assert regions[0].score >= 0.4


def test_wound_detector_ignores_clean_image():
    img = np.full((300, 300, 3), 180, np.uint8)        # plain, no red
    det = WoundHeuristicDetector()
    assert det.analyze(img) == []


def test_wound_detector_ignores_tiny_red_speck():
    img = np.full((300, 300, 3), 180, np.uint8)
    cv2.circle(img, (150, 150), 4, (0, 0, 255), -1)    # well under min area
    det = WoundHeuristicDetector()
    assert det.analyze(img) == []


def test_wound_detector_available_without_optional_deps():
    assert WoundHeuristicDetector().available is True


def test_nudenet_label_map_is_consistent():
    # Every mapped label resolves to a real Category/Severity pair.
    for label, (cat, sev) in NUDENET_LABEL_MAP.items():
        assert isinstance(cat, Category)
        assert isinstance(sev, Severity)
    # Faces are intentionally NOT mapped (never blurred).
    assert "FACE_FEMALE" not in NUDENET_LABEL_MAP
    assert "FACE_MALE" not in NUDENET_LABEL_MAP
    # The unambiguous explicit labels are HIGH severity.
    assert NUDENET_LABEL_MAP["FEMALE_GENITALIA_EXPOSED"][1] is Severity.HIGH


@pytest.mark.skipif(
    pytest.importorskip is None, reason="pytest required")
def test_nudenet_detector_if_installed():
    nudenet = pytest.importorskip("nudenet")  # noqa: F841
    from safety.detectors.nudenet_detector import NudeNetDetector

    det = NudeNetDetector()
    assert det.available is True
    # A flat grey frame should not crash and should yield a list.
    out = det.analyze(np.full((256, 256, 3), 127, np.uint8))
    assert isinstance(out, list)


def test_config_threshold_lookup():
    cfg = SafetyConfig(nudity_threshold=0.3, gore_threshold=0.6)
    assert cfg.threshold_for(Category.NUDITY) == 0.3
    assert cfg.threshold_for(Category.GORE) == 0.6


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("SAFETY_NUDITY_THRESHOLD", "0.5")
    monkeypatch.setenv("SAFETY_USE_NUDENET", "false")
    monkeypatch.setenv("SAFETY_MIN_BLUR_SEVERITY", "high")
    monkeypatch.setenv("SAFETY_BLUR_STYLE", "pixelate")
    cfg = SafetyConfig.from_env()
    assert cfg.nudity_threshold == 0.5
    assert cfg.use_nudenet is False
    assert cfg.min_blur_severity is Severity.HIGH
    assert cfg.blur_style == "pixelate"
