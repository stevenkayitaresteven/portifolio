"""Multimodal dispatch, image moderation, and video frame sampling.

Everything here runs offline: the Hugging Face backends are disabled and a
stub detector stands in where a "this is explicit" signal is needed.
"""
import os

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from safety.types import Category, Region, Severity
from safety.multimodal import (ImageModerator, MultimodalConfig,
                               MultimodalModerator, VideoModerator,
                               modality_for)


def offline_config() -> MultimodalConfig:
    cfg = MultimodalConfig()
    cfg.use_hf_text = cfg.use_hf_image = cfg.use_hf_audio = False
    cfg.check_video_audio = False
    cfg.safety.use_nudenet = False           # not installed in CI
    cfg.safety.use_wound_heuristic = True    # offline, deterministic
    return cfg


class AlwaysExplicit:
    """Stub detector: flags every frame as full-frame nudity."""
    name = "stub_explicit"
    available = True

    def analyze(self, image):
        h, w = image.shape[:2]
        return [Region(0, 0, w, h, "stub", 0.99,
                       Category.NUDITY, Severity.HIGH, source="stub")]


def gray(w=96, h=64):
    return np.full((h, w, 3), 128, np.uint8)


# --- dispatch ----------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("photo.JPG", "image"), ("pic.webp", "image"),
    ("clip.mp4", "video"), ("clip.MOV", "video"),
    ("note.wav", "audio"), ("voice.ogg", "audio"),
    ("readme.txt", "text"), ("data.json", "text"),
    ("archive.zip", None), ("binary.exe", None),
])
def test_modality_for_extensions(name, expected):
    assert modality_for(name) == expected


def test_modality_for_falls_back_to_mime():
    assert modality_for("blob", "image/png") == "image"
    assert modality_for("blob", "application/zip") is None


# --- image ----------------------------------------------------------------------

def test_clean_image_passes_unchanged():
    mod = ImageModerator(offline_config())
    img = gray()
    out, res = mod.moderate_bgr(img)
    assert not res.flagged and res.action == "none"
    assert np.array_equal(out, img)


def test_explicit_image_is_blurred():
    mod = ImageModerator(offline_config())
    mod.filter.detectors.append(AlwaysExplicit())
    img = np.zeros((64, 96, 3), np.uint8)
    img[:, :48] = (40, 200, 240)  # structure so the blur visibly changes pixels
    out, res = mod.moderate_bgr(img)
    assert res.flagged and res.action == "blur"
    assert "nudity" in res.categories
    assert not np.array_equal(out, img)


def test_image_bytes_roundtrip():
    mod = ImageModerator(offline_config())
    ok, buf = cv2.imencode(".png", gray())
    assert ok
    jpeg, res = mod.moderate_bytes(buf.tobytes())
    assert jpeg is not None and not res.flagged
    assert mod.moderate_bytes(b"not an image")[0] is None


# --- text files through the facade ------------------------------------------------

def test_text_file_gets_censored_copy(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("meeting at 9\nthis fucking printer again\n")
    mod = MultimodalModerator(offline_config())
    delivered, res = mod.moderate_file(str(src))
    assert res.flagged and res.action == "mask"
    assert delivered != str(src)
    assert "f******" in open(delivered).read()


def test_clean_text_file_passes_through(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("totally normal grocery list")
    delivered, res = MultimodalModerator(offline_config()).moderate_file(str(src))
    assert delivered == str(src) and not res.flagged


# --- video --------------------------------------------------------------------------

def _write_video(path: str, frames=12, w=64, h=48):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (w, h))
    assert writer.isOpened(), "OpenCV cannot write MJPG/AVI in this environment"
    for i in range(frames):
        frame = np.full((h, w, 3), 100, np.uint8)
        frame[:, : (i * 5) % w] = (30, 60, 90)
        writer.write(frame)
    writer.release()


def test_clean_video_passes_through(tmp_path):
    src = str(tmp_path / "clip.avi")
    _write_video(src)
    mod = VideoModerator(offline_config())
    delivered, res = mod.moderate_file(src)
    assert delivered == src
    assert not res.flagged and res.action == "none"
    assert res.extra["frames_sampled"] >= 1


def test_explicit_video_is_reencoded_blurred(tmp_path):
    src = str(tmp_path / "clip.avi")
    out = str(tmp_path / "clean.avi")
    _write_video(src)
    mod = VideoModerator(offline_config())
    mod.image.filter.detectors.append(AlwaysExplicit())
    delivered, res = mod.moderate_file(src, out_path=out)
    assert res.flagged and res.action == "blur"
    assert delivered == out and os.path.getsize(out) > 0

    # The delivered clip must not contain the original frames.
    cap_a, cap_b = cv2.VideoCapture(src), cv2.VideoCapture(out)
    ok_a, fa = cap_a.read()
    ok_b, fb = cap_b.read()
    cap_a.release(), cap_b.release()
    assert ok_a and ok_b
    assert not np.array_equal(fa, fb)


def test_unreadable_video_is_withheld(tmp_path):
    bogus = tmp_path / "x.mp4"
    bogus.write_bytes(b"definitely not a video")
    delivered, res = VideoModerator(offline_config()).moderate_file(str(bogus))
    assert delivered is None and not res.scanned
