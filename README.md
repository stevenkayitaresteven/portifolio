# Sentinel — multimodal content moderation across 12 safety categories

Detect, risk-score, and safely handle harmful content across **text, images,
video, audio, documents, and file uploads** — in **12 categories**: toxic,
hate, sexual, violence, self-harm, criminal, cybersecurity, spam, privacy,
extremism, misinformation, and child-safety. Serious harms are **blocked**,
explicit media is **blurred**, curse words and PII are **masked** (`f***`,
`b**@corp.io`), spam/misinformation is **flagged**, and dangerous file uploads
are **rejected**. Every verdict carries a category, action, and confidence.

It ships as a Python library, a CLI, a **WhatsApp-style chat UI** where every
message and attachment is moderated before delivery, a drag-and-drop image UI,
a mountable **HTTP service**, and a **fine-tuning suite** for the 12-category
text classifier. The full design is in
[`docs/SAFETY_BLUEPRINT.md`](docs/SAFETY_BLUEPRINT.md).

```bash
pip install -e ".[detectors,hf,serve]"
python -m safety chat        # WhatsApp-style moderated chat (text/image/video/audio/files)
python -m safety ui          # single-image drag-and-drop verdict UI
```

> **Defensive / content-moderation tool.** It detects and obscures harmful
> content — it does not generate it. **No explicit material is shipped or
> required to run.**

---

## What gets caught, and how

Every detector — HF model, lexicon, regex, file sniffer — normalizes into one
**12-category taxonomy** with one action per category (the most severe firing
category wins). Each category has an **offline floor** *and* a **fine-tuned
Hugging Face model**, so it degrades gracefully instead of failing open:

| Category | Offline floor (always on) | Hugging Face model(s) |
|---|---|---|
| **toxic / hate** | profanity lexicon (leetspeak-aware, masks in place) | [`unitary/toxic-bert`](https://hf.co/unitary/toxic-bert), [`facebook/roberta-hate-speech`](https://hf.co/facebook/roberta-hate-speech-dynabench-r4-target) |
| **sexual / child_safety** | NudeNet + wound heuristic (image) | [`KoalaAI/Text-Moderation`](https://hf.co/KoalaAI/Text-Moderation), [`Falconsai/nsfw_image_detection`](https://hf.co/Falconsai/nsfw_image_detection) |
| **violence / self_harm** | phrase lexicons (+ support note) | KoalaAI moderation, [`sentinet/suicidality`](https://hf.co/sentinet/suicidality) |
| **criminal / cybersecurity / extremism** | phrase lexicons + executable/EICAR file gate | KoalaAI moderation |
| **spam** | promo + phishing-URL signals | [`mshenoda/roberta-spam`](https://hf.co/mshenoda/roberta-spam), [`ealvaradob/bert-finetuned-phishing`](https://hf.co/ealvaradob/bert-finetuned-phishing) |
| **privacy** | PII regexes (email/card-Luhn/SSN/phone/IP/IBAN), masked in place | (pairs with [`iiiorg/piiranha-v1`](https://hf.co/iiiorg/piiranha-v1-detect-personal-information)) |
| **misinformation** | debunked-claim phrases (flag) | a [`liar2`](https://hf.co/datasets/chengxuphd/liar2)-fine-tuned head |

The text moderator runs an **ensemble** (KoalaAI moderation + toxic-bert by
default; spam/phishing/suicidality specialists opt-in) and merges every label
into the taxonomy. Models score the *whole message*; the lexicon/regex floor
*localizes* curse words and PII to mask them in place and keeps working fully
offline. All HF backends are lazy (download once, cached); without the `[hf]`
extra or network, the offline floor carries the load.

Train your own 12-category classifier with
[`safety/train/finetune_text.py`](safety/train/finetune_text.py) — 12 dataset
presets (nvidia Aegis 2.0, Jigsaw + Jigsaw Unintended Bias, Civil Comments,
HateXplain, Davidson, ToxiGen, OLID/SOLID, textdetox multilingual,
cyberbullying, GoEmotions hard negatives, LIAR2), mixable into one balanced
corpus with `--mix` — and drop it in via `SAFETY_MM_TEXT_MODELS`.

## Architecture

```
                          ┌── text ──► lexicon · PII regex · phrases · spam/URL · HF ensemble ─► 12-cat
 message / upload  ──►     ├── image ─► NudeNet · ViT · wound · OCR-of-embedded-text ─► blur
 (one item)               ├── audio ─► whisper ─► transcript ─► text chain ─► block + transcript
 router by modality       ├── video ─► frame sampler ─► image chain + soundtrack ─► blur / mute
                          └── file  ─► magic bytes · extension · EICAR ─► block (cybersecurity)
                                       │
                                       ▼  risk score ─► policy (action per category) ─►
                                       allow / flag / mask / blur / mute / block  + audit
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
mod.moderate_text("what the f@ck").censored_text       # "what the f***"   (mask, toxic)
mod.moderate_text("i will kill you").action            # "block"           (violence)
mod.moderate_text("my ssn is 123-45-6789").censored_text  # masked          (privacy)
mod.moderate_text("CLAIM your PRIZE http://x.tk").action  # "flag"          (spam)

delivered, res = mod.moderate_file("photo.jpg")   # blurred copy when explicit
delivered, res = mod.moderate_file("voice.ogg")   # None (blocked) when explicit
delivered, res = mod.moderate_file("report.pdf")  # text extracted + moderated
print(res.to_dict())                              # action, categories, confidence…

# the file gate runs before modality moderation
from safety.multimodal import check_file_safety
check_file_safety("setup.exe", data)              # -> block (cybersecurity)
```

### Fine-tune the 12-category classifier

```bash
pip install -e ".[train]"
python -m safety.train.finetune_text --preset aegis2 --out runs/aegis   # nvidia Aegis 2.0

# one balanced toxicity corpus from four Hub datasets (implicit hate via
# ToxiGen, hate-vs-profanity via Davidson, scale via Civil Comments, and
# GoEmotions benign hard negatives to cut false positives):
python -m safety.train.finetune_text \
    --mix civil_comments,davidson,toxigen,goemotions \
    --max-per-source 50000 --out runs/toxicity-v2

python -m safety.train.finetune_text --preset jigsaw --data train.csv --out runs/jigsaw
python -m safety.train.finetune_text --preset textdetox --split es \
    --model microsoft/mdeberta-v3-base --out runs/es      # multilingual

SAFETY_MM_TEXT_MODELS=runs/toxicity-v2 python -m safety chat   # use your model
```

All 12 presets: `aegis2` · `jigsaw` · `jigsaw_bias` · `civil_comments` ·
`hatexplain` · `davidson` · `toxigen` · `textdetox` · `olid` ·
`cyberbullying` · `goemotions` · `liar2` — schemas and label mappings in
[`docs/SAFETY_BLUEPRINT.md`](docs/SAFETY_BLUEPRINT.md) Layer 1.

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
| `SAFETY_MM_TEXT_MODELS` | KoalaAI + toxic-bert | comma-separated HF checkpoints (or a local fine-tune) |
| `SAFETY_MM_USE_SPECIALIST_TEXT_MODELS` | `false` | add spam / phishing / suicidality specialists |
| `SAFETY_MM_TEXT_THRESHOLD` | 0.50 | model label score that flags a message |
| `SAFETY_MM_ACTION_OVERRIDES` | — | per-category action, e.g. `spam:block,sexual:mask` |
| `SAFETY_MM_DELIVER_FLAGGED_TEXT` | `true` | mask & deliver vs block-everything-flagged |
| `SAFETY_MM_OCR_IMAGE_TEXT` | `true` | OCR text inside images (memes/screenshots) |
| `SAFETY_MM_AUDIT_LOG` | — | append decisions (metadata only) to this JSONL file |
| `SAFETY_MM_VIDEO_SAMPLE_FPS` | 1.0 | analyzed frames per second of video |

```bash
SAFETY_MM_ACTION_OVERRIDES=spam:block python -m safety chat      # block spam outright
SAFETY_MM_USE_SPECIALIST_TEXT_MODELS=1 python -m safety chat     # +3 specialist models
SAFETY_MM_AUDIT_LOG=decisions.jsonl python -m safety chat        # write an audit trail
```

## Train your own models

**Text (12-category multi-label)** —
[`safety/train/finetune_text.py`](safety/train/finetune_text.py). Presets with
verified column mappings for nvidia Aegis 2.0, Jigsaw, and LIAR2; per-category
precision/recall/F1 + micro/macro metrics; exports straight into the chat:

```bash
pip install -e ".[train]"
python -m safety.train.finetune_text --preset aegis2 --out runs/aegis
python -m safety.train.finetune_text --eval-only --preset jigsaw --data dev.csv \
    --model runs/aegis                                   # metrics only
SAFETY_MM_TEXT_MODELS=runs/aegis python -m safety chat
```

**Image (gore / NSFW)** — [`safety/train/README.md`](safety/train/README.md):

```bash
python -m safety.train.train  --data ./data --epochs 8 --out runs/v1
python -m safety.train.export runs/v1/best.pt --out runs/v1/model.onnx
python -m safety blur ./photos --classifier runs/v1/model.onnx
```

Dataset and model recommendations for every category are in
[`docs/SAFETY_BLUEPRINT.md`](docs/SAFETY_BLUEPRINT.md).

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
