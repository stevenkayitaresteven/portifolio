"""Tests for the real-time webcam moderation app (browser front-end)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cv2")

import cv2
import numpy as np
from fastapi.testclient import TestClient

from safety.live import create_live_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_live_app())


def _jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_and_page(client):
    body = client.get("/live/health").json()
    assert body["status"] == "ok"
    assert "wound_heuristic" in body["detectors"]   # always-on detector
    assert "getUserMedia" in client.get("/").text   # the webcam page


def test_frame_flags_and_blurs_red(client):
    # a large saturated-red frame trips the wound/gore heuristic
    red = np.zeros((360, 480, 3), np.uint8)
    red[:] = (0, 0, 200)
    j = client.post("/live/frame", content=_jpeg(red),
                    headers={"Content-Type": "application/octet-stream"}).json()
    assert j["explicit"] is True
    assert "gore" in j["categories"] and j["regions"] >= 1
    assert j["image"].startswith("data:image/jpeg;base64,")
    assert isinstance(j["ms"], int)


def test_frame_passes_clean(client):
    gray = np.full((360, 480, 3), 127, np.uint8)
    j = client.post("/live/frame", content=_jpeg(gray),
                    headers={"Content-Type": "application/octet-stream"}).json()
    assert j["explicit"] is False and j["categories"] == []


def test_empty_frame_rejected(client):
    assert client.post("/live/frame", content=b"").status_code == 400
