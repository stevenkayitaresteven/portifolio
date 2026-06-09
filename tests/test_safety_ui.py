"""Tests for the browser UI app.

The confidence-collapsing logic is tested directly (no web deps). The HTTP
endpoint tests are skipped unless FastAPI + a multipart parser are installed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from safety.ui import decision_and_confidence
from safety.types import Verdict, Region, Category, Severity


def _region(score, category=Category.NUDITY, sev=Severity.HIGH):
    return Region(0, 0, 10, 10, "x", score, category, sev, "stub")


def test_confidence_explicit_uses_top_region_score():
    v = Verdict(explicit=True,
                regions=[_region(0.91), _region(0.40)],
                all_regions=[_region(0.91), _region(0.40)])
    decision, label, conf = decision_and_confidence(v)
    assert decision == "explicit"
    assert label == "nudity"
    assert conf == pytest.approx(0.91)


def test_confidence_safe_is_one_when_nothing_detected():
    v = Verdict(explicit=False, regions=[], all_regions=[])
    decision, label, conf = decision_and_confidence(v)
    assert decision == "safe" and label == "safe"
    assert conf == pytest.approx(1.0)


def test_confidence_safe_drops_with_borderline_detection():
    # A sub-threshold detection (0.3) lowers our confidence that it's safe.
    v = Verdict(explicit=False, regions=[],
                all_regions=[_region(0.3, Category.GORE, Severity.LOW)])
    decision, _, conf = decision_and_confidence(v)
    assert decision == "safe"
    assert conf == pytest.approx(0.7)


# --- HTTP endpoint tests (need fastapi + a multipart parser) -----------------
def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")  # python-multipart
    from fastapi.testclient import TestClient

    from safety.ui import create_ui_app
    return TestClient(create_ui_app())


def _jpeg(img):
    import cv2

    return cv2.imencode(".jpg", img)[1].tobytes()


def test_index_serves_upload_page():
    c = _client()
    body = c.get("/").text
    assert "Drop an image here" in body
    assert "/process" in body  # the JS posts here


def test_health_lists_detectors():
    c = _client()
    j = c.get("/health").json()
    assert j["status"] == "ok"
    assert isinstance(j["detectors"], list)


def test_process_safe_image_returns_original_and_high_confidence():
    c = _client()
    safe = np.full((120, 120, 3), 150, np.uint8)
    r = c.post("/process", files={"file": ("s.jpg", _jpeg(safe), "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] == "safe"
    assert d["confidence"] >= 0.99
    assert d["image"].startswith("data:image/jpeg;base64,")


def test_process_rejects_non_image():
    c = _client()
    r = c.post("/process", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
