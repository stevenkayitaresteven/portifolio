# Sentinel — explicit-content detection & blurring

Detect **nudity / sexual content** and **wounds / gore** in images and **blur
them out** — running on CPU with **no network access at inference time**. If any
sensitive region is found, the whole image is hidden behind a heavy, solid blur;
clean images pass through untouched. Every result comes with a **confidence
score**.

It ships as a Python library, a CLI, a drag-and-drop **browser UI**, a mountable
**HTTP service**, and a **fine-tuning suite** for training your own classifier.

```bash
pip install -e ".[detectors,serve]"
python -m safety ui          # opens a browser: drop an image, get a verdict + blur
```

> **Defensive / content-moderation tool.** It detects and obscures sensitive
> content — it does not generate it. **No explicit imagery is shipped or required
> to run.**

---

## Why it's built this way

A real design constraint shaped every choice: the target environment has **no
GPU** and can reach **PyPI but not model hubs**. So nothing is downloaded at
runtime — every model is either **bundled in its wheel** or **trained by you**.

| Capability | Approach | Offline? |
|---|---|---|
| **Nudity** | [NudeNet](https://pypi.org/project/nudenet/) ONNX — 18 body-part classes, **box-level** | ✅ model ships in the wheel, ~30 ms/img on CPU |
| **Wounds / gore** | An HSV blood heuristic **+** a fine-tunable classifier | ✅ heuristic needs zero weights |
| **Whole-image NSFW** | Your own MobileNetV3 classifier (transfer learning) | ✅ exported to ONNX → `onnxruntime` only |

The **core** (policy + blur engine) depends only on `numpy` + `opencv`. Heavy
backends (`nudenet`, `onnxruntime`, `torch`) are imported lazily and **degrade
gracefully** when absent — the pipeline simply skips a detector it can't load.

## Architecture

```
 image ─► detectors (localize) ─► policy (thresholds + severity) ─► blur engine ─► clean image + JSON report
          nudenet · wound · classifier(opt)
```

- **Detectors** (`safety/detectors/`) — pluggable, lazy, report `available`.
- **Policy** (`safety/pipeline.py`) — per-category thresholds + a severity gate
  turn raw detections into a `Verdict`. Cautious by default (over-blur a
  borderline image rather than leak it).
- **Blur engine** (`safety/blur.py`) — solid whole-image blur (mosaic + heavy
  smear → unrecognizable), or localized gaussian/pixelate/box redaction.

## Install

```bash
git clone https://github.com/stevenkayitaresteven/portifolio
cd portifolio
python -m venv .venv && source .venv/bin/activate

pip install -e .                      # core only (numpy + opencv)
pip install -e ".[detectors]"         # + offline nudity detector (recommended)
pip install -e ".[detectors,serve]"   # + browser UI / HTTP service
pip install -e ".[detectors,serve,train]"   # + fine-tuning stack
```

## Usage

### Browser UI

```bash
python -m safety ui                 # http://127.0.0.1:8000
```

Drop or pick an image → it shows the **blurred** image if flagged (nudity / gore)
or the **original** image if safe, plus a confidence score and the detected
categories.

### CLI

```bash
python -m safety scan   ./photos --json              # verdicts only, nothing written
python -m safety blur    ./photos -o ./clean --report r.json   # write redacted copies
python -m safety debug   photo.jpg -o boxed.jpg      # draw detection boxes
python -m safety blur    ./photos --style pixelate --min-severity high
# (after `pip install -e .` the `sentinel` command is an alias for `python -m safety`)
```

### Python

```python
from safety import ExplicitContentFilter
import cv2

filt = ExplicitContentFilter()                 # env-overridable config
clean, verdict = filt.scan_and_redact(cv2.imread("photo.jpg"))
if verdict.explicit:
    print(verdict.categories(), verdict.reasons)
    cv2.imwrite("clean.jpg", clean)
print(verdict.to_dict())                       # JSON-ready
```

### HTTP service

```python
from fastapi import FastAPI
from safety.service import build_router
app = FastAPI(); app.include_router(build_router())
#  POST /safety/scan   -> JSON verdict
#  POST /safety/redact -> blurred image (image/jpeg)
#  GET  /safety/health -> active detectors
```

## Configuration

All knobs live in [`SafetyConfig`](safety/config.py); override in code or via
`SAFETY_*` env vars. Highlights:

| Field | Default | Meaning |
|-------|---------|---------|
| `whole_image_on_detection` | `true` | any detection blurs the **whole** image |
| `solid_blur` / `solid_blur_blocks` | `true` / 6 | heavy, unrecognizable whole-image blur (fewer blocks = more solid) |
| `nudity_threshold` / `gore_threshold` | 0.35 / 0.55 | per-category score floors |
| `min_blur_severity` | `medium` | `low` also blurs suggestive; `high` only explicit |
| `blur_style` | `gaussian` | localized style: `gaussian` · `pixelate` · `box` · `fill` |

```bash
SAFETY_SOLID_BLUR_BLOCKS=3 python -m safety ui   # near-solid color block
SAFETY_WHOLE_IMAGE_ON_DETECTION=false ...        # localized box blur instead
```

## Train your own classifier

The detectors work out of the box. For a **learned gore model** or a whole-image
NSFW score tuned to your data, see [`safety/train/README.md`](safety/train/README.md):

```bash
pip install torch torchvision pillow onnx
python -m safety.train.train  --data ./data --epochs 8 --out runs/v1
python -m safety.train.export runs/v1/best.pt --out runs/v1/model.onnx
python -m safety blur ./photos --classifier runs/v1/model.onnx
```

ONNX export means production inference needs only `onnxruntime`.

## Tests

```bash
pip install -e ".[detectors,serve,test]"
pytest -q
```

The core/policy/blur tests need only numpy+opencv; the UI/service tests skip
cleanly when FastAPI isn't installed, and the NudeNet test skips when the
detector extra is absent.

## Limitations & responsible use

- The **wound heuristic is a baseline** — it keys on large, saturated red
  regions and can be fooled by red objects or miss dark/dried blood. Train the
  classifier for production-grade gore detection.
- No detector is perfect. Tune thresholds toward **recall** for moderation, keep
  a human in the loop for borderline content, and treat "safe" as *"nothing
  detected"*, not a guarantee.
- **No explicit imagery is shipped or required to run.** You bring your own data
  to train the optional classifier; handle it lawfully and store it securely.

See [`safety/README.md`](safety/README.md) for the deeper component reference.
