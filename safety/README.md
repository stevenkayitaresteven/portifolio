# Sentinel — explicit-content detection & blurring

A small, dependency-light toolkit that finds **nudity / sexual content** and
**wounds / gore** in images and **blurs the offending regions**. Built to run on
CPU with no network access at inference time (this matters: the environment can
reach PyPI but not model hubs, so every model used here is either bundled in its
wheel or trained by you).

```
        ┌──────────────┐   regions   ┌────────────┐   verdict   ┌────────────┐
 image ─►│  detectors   ├────────────►│   policy   ├────────────►│   blur     ├─► clean image
        │  (localize)  │             │ (thresholds│             │  engine    │   + report
        └──────────────┘             │  severity) │             └────────────┘
          nudenet · wound            └────────────┘
          · classifier(optional)
```

## Why this design

| Need | Choice | Reason |
|------|--------|--------|
| Localize nudity for **targeted** blur | **NudeNet** (ONNX) | 18 body-part boxes, model ships in the wheel → fully offline, 30 ms/img on CPU |
| Gore/wounds with **no** offline model available | HSV **blood heuristic** + a **fine-tunable** classifier | heuristic works today with zero weights; train the classifier when you have data |
| Run core anywhere | numpy + opencv only | heavy backends (nudenet, onnx, torch) are lazy & optional, degrade gracefully |
| Tune precision/recall without retraining | **per-category thresholds + severity gate** | a safety filter should err toward over-blurring |

## Install

```bash
# Core (policy + blur) — already covered by the crawler's numpy/opencv deps.
pip install numpy opencv-python-headless

# Nudity detector (recommended; offline ONNX model bundled in the wheel):
pip install nudenet onnxruntime

# Optional HTTP service / fine-tuning — see below.
```

## Use it

### Python

```python
from safety import ExplicitContentFilter, SafetyConfig
import cv2

filt = ExplicitContentFilter()              # uses default config + env overrides
img = cv2.imread("photo.jpg")
clean, verdict = filt.scan_and_redact(img)  # original returned if nothing found

if verdict.explicit:
    print(verdict.categories(), verdict.reasons)
    cv2.imwrite("clean.jpg", clean)
```

`verdict.to_dict()` is JSON-ready (categories, per-category scores, severity,
each redacted region's box/label/score).

### Browser UI (drag-and-drop)

```bash
pip install fastapi uvicorn python-multipart
python -m safety ui                 # opens http://127.0.0.1:8000 in your browser
python -m safety ui --port 9000 --no-browser
```

A local page where you drop or pick an image. It shows the **blurred** image if
the picture is flagged (nudity / gore) or the **original** image if it's safe —
and a **confidence score** either way:

- flagged → confidence = the top triggering region's score (e.g. `gore 0.78`)
- safe → confidence = `1 − max(any detection score)` (100% when nothing is seen)

The page is a single self-contained HTML file (no CDN, no build step) and runs
fully offline; everything is processed locally by the same `ExplicitContentFilter`.

### CLI

```bash
python -m safety scan   ./photos --json                 # verdicts only
python -m safety blur    ./photos -o ./clean --report r.json
python -m safety debug   photo.jpg -o boxed.jpg         # draw boxes, no blur
python -m safety blur    ./photos --style pixelate --min-severity high
```

### HTTP service (FastAPI)

```python
from fastapi import FastAPI
from safety.service import build_router
app = FastAPI()
app.include_router(build_router())
#  POST /safety/scan    -> JSON verdict
#  POST /safety/redact  -> blurred image (image/jpeg) + X-Explicit header
#  GET  /safety/health  -> active detectors
```

Needs `pip install fastapi python-multipart`.

## Configuration

Everything lives in [`SafetyConfig`](config.py); override in code or via
`SAFETY_*` env vars (`SafetyConfig.from_env()`):

| Field | Default | Meaning |
|-------|---------|---------|
| `nudity_threshold` | 0.35 | min score to act on a nudity box |
| `suggestive_threshold` | 0.55 | min score for covered/partial areas |
| `gore_threshold` | 0.55 | min score for wound/blood |
| `min_blur_severity` | `medium` | `low` also blurs suggestive; `high` only explicit |
| `whole_image_on_detection` | `true` | **any** detection blurs the **whole** frame (set `false` for localized box blur only) |
| `solid_blur` | `true` | whole-image blur is heavy/unrecognizable (set `false` for a lighter, legible blur) |
| `solid_blur_blocks` | 6 | long-side macro-blocks for solid blur; **fewer = more solid** |
| `whole_image_threshold` | 0.80 | classifier score that blurs the whole frame |
| `blur_style` | `gaussian` | per-region style: `gaussian` · `pixelate` · `box` · `fill` |
| `region_margin` | 0.12 | grow each box before blurring (localized mode) |
| `feather` | `true` | soft-edge the redaction (localized mode) |
| `use_nudenet` / `use_wound_heuristic` / `use_classifier` | on/on/off | toggle detectors |

> **Default policy is fail-safe:** if *any* region in an image is flagged, the
> **entire** image is blurred with a heavy, unrecognizable "solid" blur — so a
> single missed pixel of a localized box can't leak the rest. To instead blur
> only the detected regions, set `whole_image_on_detection=false` (and
> `solid_blur=false` for a lighter look).

The NudeNet label → category/severity table is in `config.NUDENET_LABEL_MAP`
(faces, feet, armpits, and bare midriff are intentionally **never** blurred).

## Detection taxonomy

- **nudity** — exposed genitalia/anus/female-breast (HIGH), exposed buttocks (MEDIUM)
- **suggestive** — covered intimate areas, exposed male chest, bare belly (LOW)
- **gore** — wounds / blood / graphic injury (MEDIUM)
- **safe** — explicitly clean (never blurred)

## Train your own classifier

The box detectors work out of the box. For a **learned gore model** or a
whole-image NSFW score tuned to your data, see
[`safety/train/README.md`](train/README.md):

```bash
python -m safety.train.train  --data ./data --epochs 8 --out runs/v1
python -m safety.train.export runs/v1/best.pt --out runs/v1/model.onnx
python -m safety blur ./photos --classifier runs/v1/model.onnx
```

ONNX export means production inference needs only `onnxruntime`.

## Integration with the crawler

Set `ENABLE_SAFETY_FILTER=true` (and install `nudenet onnxruntime`) to screen
every scraped listing photo: explicit regions are blurred **before** the pixels
are stored (`app/imaging.py`). If the detector extras are absent the filter is
skipped silently — a missing safety dependency never breaks a scrape.

## Limitations & responsible use

- **No explicit imagery is shipped or required to run.** You bring your own data
  to train the optional classifier; handle it lawfully and store it securely.
- The **wound heuristic is a baseline** — it keys on large, saturated red
  regions and can be fooled by red clothing/objects or miss dark/dried blood.
  Train the classifier for production-grade gore detection.
- No detector is perfect. Tune thresholds toward **recall** for a moderation use
  case, keep a human in the loop for borderline content, and treat a "safe"
  verdict as "nothing detected", not a guarantee.
- This is a **defensive / content-moderation** tool: detect and obscure
  sensitive content. It does not generate or restore it.
