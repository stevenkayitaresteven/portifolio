# `safety.multimodal` — sensitive-content moderation for text, image, audio & video

One front door (`MultimodalModerator`) routes any content to a per-modality
moderator. Each moderator pairs a **fine-tuned pretrained Hugging Face model**
with an **offline fallback**, and everything returns the same
`ModerationResult`, so a caller can act uniformly: mask, blur, mute, block,
or flag.

## Models & datasets

| Modality | Primary model (Hugging Face) | What it was trained on | Offline fallback |
|---|---|---|---|
| **Text** | [`unitary/toxic-bert`](https://hf.co/unitary/toxic-bert) — BERT, 6 labels (toxic, severe_toxic, obscene, threat, insult, identity_hate) | [Jigsaw toxic-comment datasets](https://hf.co/datasets/anitamaxvim/jigsaw-toxic-comments) | built-in profanity lexicon (`wordlist.py`) |
| **Image** | [`Falconsai/nsfw_image_detection`](https://hf.co/Falconsai/nsfw_image_detection) — ViT, nsfw/normal | proprietary 80k-image NSFW corpus | NudeNet (ONNX in the wheel) + wound heuristic |
| **Audio** | [`openai/whisper-base`](https://hf.co/openai/whisper-base) ASR → transcript → text moderator | 680k h multilingual speech | none → marked *unscanned* and flagged for review |
| **Video** | frame sampling → image ensemble; soundtrack → audio chain | (composition of the above) | NudeNet/heuristic on sampled frames |

Useful fine-tuning data if you want to specialize the models:
[jigsaw-toxic-comments](https://hf.co/datasets/anitamaxvim/jigsaw-toxic-comments)
(text, multi-label), [processed-jigsaw](https://hf.co/datasets/Koushim/processed-jigsaw-toxic-comments)
(pre-tokenized), and for images the corpora referenced by
[strangerguardhf/nsfw-image-detection](https://hf.co/strangerguardhf/nsfw-image-detection).
The repo's own training suite (`safety/train/`) fine-tunes a whole-image
classifier from a folder of labeled images.

Why two layers for text? The model scores the *whole message* and catches
hostility a wordlist can't ("go back where you came from"); the lexicon
localizes the *individual* curse words — including `f*ck`, `f@ck`, `sh1t`,
`fuuuck` evasions — so they can be masked in place, WhatsApp-style. The same
split shows up in audio (model transcribes, lexicon censors the transcript)
and images (NudeNet localizes, the ViT judges the whole frame).

## Actions

| `action` | Meaning |
|---|---|
| `none`  | clean — delivered untouched |
| `mask`  | text delivered with curse words masked (`f***`) |
| `blur`  | image/video delivered with pixels redacted (solid whole-frame blur by default) |
| `mute`  | video visuals clean but soundtrack explicit — delivered silent |
| `block` | withheld entirely (explicit audio; flagged text when `deliver_flagged_text=False`) |
| `flag`  | delivered but marked for human review (model-only text hit, unscannable media) |

## Usage

```python
from safety.multimodal import MultimodalModerator

mod = MultimodalModerator()

res = mod.moderate_text("what the f@ck")
res.flagged          # True
res.action           # "mask"
res.censored_text    # "what the f***"

delivered, res = mod.moderate_file("holiday.jpg")   # path to blurred copy if flagged
delivered, res = mod.moderate_file("voice.ogg")     # None if explicit (blocked)
delivered, res = mod.moderate_file("clip.mp4")      # blurred/muted re-encode if flagged
```

Every HF backend is lazy and optional: without the `[hf]` extra (or without
network for the first download) the moderators degrade to their offline
fallbacks, and content that *cannot* be analyzed (audio with no ASR) is
flagged for review rather than waved through (`flag_unscanned`).

## Configuration

`MultimodalConfig` — overridable via `SAFETY_MM_*` env vars; the embedded
image policy (`.safety`, a `SafetyConfig`) keeps its `SAFETY_*` vars.

| Field | Default | Meaning |
|---|---|---|
| `text_model` / `image_model` / `asr_model` | toxic-bert / Falconsai ViT / whisper-base | HF checkpoints |
| `use_hf_text` / `use_hf_image` / `use_hf_audio` | `true` | disable to run offline-only |
| `text_threshold` | `0.50` | toxic-bert label score that flags a message |
| `deliver_flagged_text` | `true` | mask & deliver vs block flagged text |
| `flag_unscanned` | `true` | unanalyzable content is flagged, not waved through |
| `video_sample_fps` / `video_max_samples` | `1.0` / `32` | frame-sampling density |
| `check_video_audio` | `true` | also scan video soundtracks (needs ffmpeg) |

```bash
SAFETY_MM_ASR_MODEL=openai/whisper-tiny python -m safety chat   # faster ASR
SAFETY_MM_USE_HF_IMAGE=0 python -m safety chat                  # NudeNet only
```

## The chat UI

`python -m safety chat` serves a WhatsApp-style page (`safety/chat.py`):
type messages, attach **one file at a time** (photo / video / audio /
document) via the 📎 menu. Flagged content arrives the way a receiver would
see it: blurred image, blurred or muted video, a "voice message removed" card
with the censored transcript, masked text — each with a category + confidence
pill and the detector reasons.

Endpoints: `POST /chat/send` (`text` + ≤1 `files`), `GET /chat/messages`,
`GET /chat/media/{id}`, `GET /chat/health`.
