"""Blur-engine tests: the redacted region must actually change, the rest must
stay (near-)identical. Pure numpy/opencv — no model downloads."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from safety.blur import redact, draw_boxes
from safety.config import SafetyConfig
from safety.types import Region, Category, Severity


def _noisy_image(h=200, w=200):
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _region(x1, y1, x2, y2):
    return Region(x1, y1, x2, y2, "test", 0.9, Category.NUDITY, Severity.HIGH, "t")


def test_redact_changes_only_the_region():
    img = _noisy_image()
    cfg = SafetyConfig(region_margin=0.0, feather=False)
    out = redact(img, [_region(50, 50, 100, 100)], cfg)

    # Region differs (blur reduces local variance / changes pixels).
    assert not np.array_equal(out[50:100, 50:100], img[50:100, 50:100])
    # A far-away corner is untouched.
    assert np.array_equal(out[0:40, 0:40], img[0:40, 0:40])


def test_blur_reduces_local_variance():
    img = _noisy_image()
    cfg = SafetyConfig(region_margin=0.0, feather=False, blur_style="gaussian")
    out = redact(img, [_region(50, 50, 150, 150)], cfg)
    assert out[50:150, 50:150].var() < img[50:150, 50:150].var()


def test_box_style_is_solid_color():
    img = _noisy_image()
    cfg = SafetyConfig(region_margin=0.0, feather=False, blur_style="box",
                       fill_color=(0, 0, 0))
    out = redact(img, [_region(50, 50, 100, 100)], cfg)
    assert out[50:100, 50:100].max() == 0  # fully blacked out


def test_pixelate_runs_and_alters_region():
    img = _noisy_image()
    cfg = SafetyConfig(region_margin=0.0, feather=False, blur_style="pixelate")
    out = redact(img, [_region(40, 40, 160, 160)], cfg)
    assert not np.array_equal(out[40:160, 40:160], img[40:160, 40:160])


def test_whole_image_blur():
    img = _noisy_image()
    cfg = SafetyConfig()
    out = redact(img, [], cfg, whole_image=True)
    assert out.shape == img.shape
    assert out.var() < img.var()          # entire frame smoothed


def test_solid_whole_image_blur_is_unrecognizable():
    # Build a structured image (sharp halves) and confirm the solid blur
    # destroys that structure into a near-uniform wash.
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :100] = (255, 0, 0)
    img[:, 100:] = (0, 255, 0)
    solid = redact(img, [], SafetyConfig(solid_blur=True), whole_image=True)
    light = redact(img, [], SafetyConfig(solid_blur=False), whole_image=True)
    # Solid blur smears the hard center edge far more than the light blur:
    # the variance across the frame collapses dramatically.
    assert solid.var() < light.var()
    assert solid.shape == img.shape
    # No sharp seam survives: adjacent columns near the old boundary are close.
    band = solid[:, 95:105].astype(np.int16)
    assert np.abs(np.diff(band, axis=1)).max() < 40


def test_full_frame_region_triggers_whole_image():
    img = _noisy_image(120, 120)
    full = Region(0, 0, 120, 120, "nsfw", 0.95, Category.NUDITY, Severity.HIGH,
                  "classifier")
    out = redact(img, [full], SafetyConfig())
    assert out.var() < img.var()


def test_margin_expands_blurred_area():
    img = _noisy_image()
    no_margin = redact(img, [_region(80, 80, 120, 120)],
                       SafetyConfig(region_margin=0.0, feather=False))
    with_margin = redact(img, [_region(80, 80, 120, 120)],
                         SafetyConfig(region_margin=0.5, feather=False))
    # With margin, pixels just outside the original box are now altered too.
    changed_no = not np.array_equal(no_margin[70:80, 90:110], img[70:80, 90:110])
    changed_yes = not np.array_equal(with_margin[70:80, 90:110], img[70:80, 90:110])
    assert changed_yes and not changed_no


def test_draw_boxes_does_not_mutate_input():
    img = _noisy_image()
    before = img.copy()
    _ = draw_boxes(img, [_region(10, 10, 50, 50)])
    assert np.array_equal(img, before)


def test_redact_handles_empty_regions():
    img = _noisy_image()
    out = redact(img, [], SafetyConfig())
    assert np.array_equal(out, img)
