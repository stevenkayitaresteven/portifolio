"""Redaction engine: turn detected regions into a blurred/obscured image.

Given a BGR image and the regions a :class:`~safety.types.Verdict` says to
redact, produce a copy where those regions are obscured. Supported styles:

* ``gaussian``  — heavy Gaussian blur (recognizable shape, content unreadable)
* ``pixelate``  — mosaic / downscale-upscale (the classic censor look)
* ``box``       — solid filled rectangle
* ``fill``      — alias of ``box`` with a configurable colour

The blur kernel scales with region size so a small box isn't over-blurred and a
large one isn't under-blurred. When ``feather`` is on, the blurred patch is
alpha-composited through a soft-edged mask so the redaction blends in rather
than showing a hard rectangle seam.
"""
from __future__ import annotations

import numpy as np

from .types import Region
from .config import SafetyConfig


def _odd(n: int) -> int:
    n = int(n)
    return n + 1 if n % 2 == 0 else max(1, n)


def _blur_patch(patch: np.ndarray, style: str, strength: float, color) -> np.ndarray:
    import cv2

    h, w = patch.shape[:2]
    if h == 0 or w == 0:
        return patch
    if style in ("box", "fill"):
        out = np.empty_like(patch)
        out[:] = np.array(color, dtype=patch.dtype)
        return out
    if style == "pixelate":
        # Mosaic: shrink then nearest-neighbour upscale. Blocks ~ region size.
        blocks = max(2, int(min(h, w) / (12 / max(strength, 1e-3))))
        small = cv2.resize(patch, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    # gaussian (default): kernel proportional to the region's shorter side.
    k = _odd(max(7, int(min(h, w) * 0.35 * strength)))
    sigma = max(4.0, min(h, w) * 0.12 * strength)
    return cv2.GaussianBlur(patch, (k, k), sigma)


def _solid_whole_blur(image: np.ndarray, blocks: int, strength: float) -> np.ndarray:
    """Crush the whole frame to an unrecognizable wash.

    Two stages: (1) downscale to a tiny ``blocks``-wide mosaic with area
    averaging — this *destroys* all fine detail (a face/body at 6px is gone),
    then (2) a heavy Gaussian smears the macro-blocks into a smooth gradient so
    not even the blocky structure reads. The result is "solid": you can see
    something was there, but not what. Higher ``strength`` => fewer blocks =>
    more solid.
    """
    import cv2

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    blocks = max(1, int(round(blocks / max(strength, 0.1))))
    scale = blocks / float(max(h, w))
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    # Smear the blocks; kernel ~ a couple of macro-blocks so edges vanish.
    block_px = max(1, max(h, w) // max(blocks, 1))
    k = _odd(max(15, int(block_px * 2)))
    return cv2.GaussianBlur(mosaic, (k, k), max(6.0, float(block_px)))


def redact(
    image: np.ndarray,
    regions: list[Region],
    config: SafetyConfig | None = None,
    whole_image: bool = False,
) -> np.ndarray:
    """Return a redacted copy of ``image``.

    ``regions`` are blurred in place on the copy. If ``whole_image`` is set (or a
    region spans the full frame), the entire image is blurred instead.
    """
    import cv2

    cfg = config or SafetyConfig()
    if image is None or image.ndim != 3:
        return image
    out = image.copy()
    h, w = out.shape[:2]

    full_frame = whole_image or any(
        r.x1 <= 0 and r.y1 <= 0 and r.x2 >= w and r.y2 >= h for r in regions
    )
    if full_frame:
        if cfg.solid_blur:
            return _solid_whole_blur(out, cfg.solid_blur_blocks, cfg.blur_strength)
        k = _odd(max(21, int(min(h, w) * 0.18 * cfg.blur_strength)))
        return cv2.GaussianBlur(out, (k, k), max(8.0, min(h, w) * 0.06))

    for region in regions:
        r = region.expanded(cfg.region_margin, w, h)
        x1, y1, x2, y2 = r.box
        if x2 <= x1 or y2 <= y1:
            continue
        patch = out[y1:y2, x1:x2]
        blurred = _blur_patch(patch, cfg.blur_style, cfg.blur_strength, cfg.fill_color)

        if cfg.feather and cfg.blur_style not in ("box", "fill"):
            out[y1:y2, x1:x2] = _feathered(patch, blurred)
        else:
            out[y1:y2, x1:x2] = blurred
    return out


def _feathered(orig: np.ndarray, blurred: np.ndarray) -> np.ndarray:
    """Alpha-blend ``blurred`` over ``orig`` through a soft-edged mask so the
    redaction has no hard rectangular seam."""
    import cv2

    h, w = orig.shape[:2]
    mask = np.zeros((h, w), np.float32)
    pad_y = max(1, int(h * 0.12))
    pad_x = max(1, int(w * 0.12))
    mask[pad_y:h - pad_y, pad_x:w - pad_x] = 1.0
    ksize = _odd(max(3, int(min(h, w) * 0.2)))
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask = mask[..., None]
    return (blurred.astype(np.float32) * mask
            + orig.astype(np.float32) * (1.0 - mask)).astype(orig.dtype)


def draw_boxes(image: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Debug helper: draw labeled boxes (no blurring). Useful for tuning."""
    import cv2

    out = image.copy()
    colors = {
        "nudity": (0, 0, 255), "suggestive": (0, 165, 255),
        "gore": (0, 0, 139), "safe": (0, 255, 0),
    }
    for r in regions:
        c = colors.get(r.category.value, (255, 255, 255))
        cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), c, 2)
        cv2.putText(out, f"{r.label} {r.score:.2f}", (r.x1, max(12, r.y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out
