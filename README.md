# Sentinel — multimodal explicit-content detection & redaction

Detect **nudity / sexual content, gore, profanity and toxicity** across
**images, video, audio, and text** — and act on it: explicit images and video
are **blurred**, explicit audio is **removed** (with a censored transcript),
curse words are **masked** (`f***`), and anything that can't be analyzed is
**flagged** for review. Every verdict comes with a **confidence score**.

It ships as a Python library, a CLI, a **WhatsApp-style chat UI** where every
message and attachment is moderated before delivery, a drag-and-drop image UI,
a mountable **HTTP service**, and a **fine-tuning suite**.

```bash
pip install -e ".[detectors,hf,serve]"
python -m safety chat        # WhatsApp-style chat: send text / image / video / audio
python -m safety ui          # single-image drag-and-drop verdict UI
```

> **Defensive / content-moderation tool.** It detects and obscures sensitive
> content — it does not generate it. **No explicit material is shipped or
> required to run.**

---

## The models (Hugging Face + offline fallbacks)

Each modality pairs a **fine-tuned pretrained model** from the Hugging Face Hub
with an **offline fallback**, so the filter degrades gracefully instead of
failing open when a model (or the network) is unavailable:

| Modality | Hugging Face model | Offline fallback | Redaction |
|---|---|---|---|
| **Image** | [`Falconsai/nsfw_image_detection`](https://hf.co/Falconsai/nsfw_image_detection) (ViT, whole-frame) | [NudeNet](https://pypi.org/project/nudenet/) ONNX (18 body-part boxes, ships in its wheel) + HSV wound heuristic | solid whole-image **blur** |
| **Text** | [`unitary/toxic-bert`](https://hf.co/unitary/toxic-bert) (toxic · obscene · threat · insult · identity-hate) | built-in profanity lexicon w/ leetspeak handling | curse words **masked** in place |
| **Audio** | [`openai/whisper-base`](https://hf.co/openai/whisper-base) ASR → transcript → text chain | flagged as *unscanned* for human review | explicit audio **blocked**, censored transcript shown |
| **Video** | ~1 frame/s sampling → image ensemble; soundtrack → audio chain | NudeNet/heuristic on sampled frames | re-encoded fully **blurred**, or **muted** if only the audio is explicit |

The ensembles are complementary: NudeNet localizes *which* body parts are
exposed, the ViT judges the whole frame; toxic-bert scores a whole message,
the lexicon pinpoints (and masks) the individual words — including `f@ck` /
`sh1t` / `fuuuck` evasions. Fine-tuning data and the deeper model rationale
live in [`safety/multimodal/README.md`](safety/multimodal/README.md).

All HF backends are lazy: models download once on first use (cached by
`transformers`); without the `[hf]` extra or without network, the offline
fallbacks carry the load. The original core (policy + blur engine) still
depends only on `numpy` + `opencv`.

## Architecture

```
                          ┌── text ──► lexicon + toxic-bert ─► mask / flag / block
 chat message / file ──►  ├── image ─► NudeNet · ViT · heuristic ─► policy ─► blur
 (one file at a time)     ├── audio ─► whisper ─► transcript ─► text chain ─► block + transcript
                          └── video ─► frame sampler ─► image chain ─► blur ─┐
                                       soundtrack ───► audio chain ──► mute ─┴► re-encode
```

- **Detectors** (`safety/detectors/`) — pluggable, lazy, report `available`.
- **Policy** (`safety/pipeline.py`) — per-category thresholds + severity gate.
  Cautious by default: over-blur a borderline image rather than leak it.
- **Blur engine** (`safety/blur.py`) — solid whole-image blur, or localized
  gaussian/pixelate/box redaction.
- **Multimodal layer** (`safety/multimodal/`) — per-modality moderators, one
  `ModerationResult` shape, one `MultimodalModerator` front door.
- **Chat UI** (`safety/chat.py`) — the WhatsApp-style demo surface.

## Install

```bash
git clone https://github.com/stevenkayitaresteven/portifolio
cd portifolio
python -m venv .venv && source .venv/bin/activate

pip install -e .                      # core only (numpy + opencv)
pip install -e ".[detectors]"         # + offline nudity detector (recommended)
pip install -e ".[detectors,serve]"   # + chat UI / browser UI / HTTP service
pip install -e ".[detectors,hf,serve]"  # + Hugging Face text/image/audio models
```

Audio formats beyond WAV — and video soundtrack scanning — also want the
`ffmpeg` binary on PATH (it additionally makes blurred video browser-playable
H.264).

## Usage

### Chat UI (WhatsApp-style)

```bash
python -m safety chat                # http://127.0.0.1:8000
python -m safety chat --no-hf       # offline fallbacks only
```

Type messages and attach **one file at a time** (📎 → photo / video / audio /
document). Everything is scanned before it lands in the conversation, and
flagged content is delivered the way a receiver would see it: blurred image or
video, a "voice message removed" card with the censored transcript, masked
curse words — each with a category + confidence pill.

### Image UI

```bash
python -m safety ui                 # drag-and-drop: blurred if flagged + confidence
```

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
from safety.multimodal import MultimodalModerator

mod = MultimodalModerator()
res = mod.moderate_text("what the f@ck")     # res.censored_text == "what the f***"
delivered, res = mod.moderate_file("photo.jpg")   # blurred copy when explicit
delivered, res = mod.moderate_file("voice.ogg")   # None (blocked) when explicit
delivered, res = mod.moderate_file("clip.mp4")    # blurred/muted re-encode
print(res.to_dict())                              # action, categories, confidence…
```

The image-only API is unchanged:

```python
from safety import ExplicitContentFilter
import cv2

filt = ExplicitContentFilter()
clean, verdict = filt.scan_and_redact(cv2.imread("photo.jpg"))
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

The chat app is also mountable: `from safety.chat import create_chat_app`.

## Configuration

Image policy knobs live in [`SafetyConfig`](safety/config.py) (env prefix
`SAFETY_`); multimodal knobs in
[`MultimodalConfig`](safety/multimodal/config.py) (env prefix `SAFETY_MM_`).
Highlights:

| Field | Default | Meaning |
|-------|---------|---------|
| `whole_image_on_detection` | `true` | any detection blurs the **whole** image |
| `solid_blur` / `solid_blur_blocks` | `true` / 6 | heavy, unrecognizable whole-image blur |
| `nudity_threshold` / `gore_threshold` | 0.35 / 0.55 | per-category score floors |
| `min_blur_severity` | `medium` | `low` also blurs suggestive; `high` only explicit |
| `SAFETY_MM_TEXT_THRESHOLD` | 0.50 | toxic-bert score that flags a message |
| `SAFETY_MM_DELIVER_FLAGGED_TEXT` | `true` | mask & deliver vs block flagged text |
| `SAFETY_MM_FLAG_UNSCANNED` | `true` | unanalyzable content is flagged, not waved through |
| `SAFETY_MM_VIDEO_SAMPLE_FPS` | 1.0 | analyzed frames per second of video |

```bash
SAFETY_SOLID_BLUR_BLOCKS=3 python -m safety chat     # near-solid color block
SAFETY_MM_ASR_MODEL=openai/whisper-tiny python -m safety chat   # faster ASR
```

## Train your own classifier

The detectors work out of the box. For a **learned gore model** or a
whole-image NSFW score tuned to your data, see
[`safety/train/README.md`](safety/train/README.md):

```bash
pip install torch torchvision pillow onnx
python -m safety.train.train  --data ./data --epochs 8 --out runs/v1
python -m safety.train.export runs/v1/best.pt --out runs/v1/model.onnx
python -m safety blur ./photos --classifier runs/v1/model.onnx
```

The Jigsaw toxic-comment datasets on the Hub (e.g.
[`anitamaxvim/jigsaw-toxic-comments`](https://hf.co/datasets/anitamaxvim/jigsaw-toxic-comments))
are the natural starting point for specializing the text model.

## Tests

```bash
pip install -e ".[detectors,serve,test]"
pytest -q
```

The whole suite runs **offline** — HF backends are exercised through their
fallback paths and stub detectors. UI/service/chat tests skip cleanly when
FastAPI isn't installed; the NudeNet test skips when the detector extra is
absent.

## Limitations & responsible use

- The **wound heuristic is a baseline** — it keys on large, saturated red
  regions and can be fooled by red objects or miss dark/dried blood. Train the
  classifier for production-grade gore detection.
- The **profanity lexicon is English-centric**; toxic-bert is also primarily
  English. Swap `SAFETY_MM_TEXT_MODEL` for a multilingual checkpoint if you
  need broader coverage.
- **Audio/video moderation is transcript-based** — it catches what is *said*,
  not non-speech sounds.
- No detector is perfect. Tune thresholds toward **recall** for moderation,
  keep a human in the loop for borderline content, and treat "safe" as
  *"nothing detected"*, not a guarantee.
- **No explicit material is shipped or required to run.** You bring your own
  data to fine-tune; handle it lawfully and store it securely.

See [`safety/README.md`](safety/README.md) for the image-pipeline reference and
[`safety/multimodal/README.md`](safety/multimodal/README.md) for the multimodal
layer.
