# `safety.multimodal` — 12-category moderation for text, image, audio, video & files

One front door (`MultimodalModerator`) routes any content to a per-modality
moderator. Every detector — HF model, lexicon, regex, file sniffer — normalizes
its labels into **one 12-category taxonomy**, and everything returns the same
`ModerationResult`, so a caller acts uniformly: mask, blur, mute, flag, or block.

## The 12 categories

`toxic` · `hate` · `sexual` · `violence` · `self_harm` · `criminal` ·
`cybersecurity` · `spam` · `privacy` · `extremism` · `misinformation` ·
`child_safety` (see [`taxonomy.py`](taxonomy.py)). Each maps to a default
action; the most severe firing category wins. Overridable via
`SAFETY_MM_ACTION_OVERRIDES="spam:block,sexual:mask"`.

## How each category is detected

| Category | Offline floor (always on) | Hugging Face model |
|---|---|---|
| toxic / hate | profanity lexicon (mask in place) | `unitary/toxic-bert`, `facebook/roberta-hate-speech-dynabench-r4` |
| sexual / child_safety | NudeNet + wound heuristic (image) | `KoalaAI/Text-Moderation` (text), `Falconsai/nsfw_image_detection` (image) |
| violence / self_harm | phrase lexicons | KoalaAI moderation, `sentinet/suicidality` (opt-in) |
| criminal / cybersecurity / extremism | phrase lexicons + executable/EICAR file gate | KoalaAI moderation |
| spam | promo + phishing-URL signals | `mshenoda/roberta-spam`, `ealvaradob/bert-finetuned-phishing` (opt-in) |
| privacy | PII regexes (email/card-Luhn/SSN/phone/IP/IBAN), masked in place | (pairs with `iiiorg/piiranha-v1` token model) |
| misinformation | debunked-claim phrases (flag only) | a `liar2`-fine-tuned head |

The text moderator runs an **ensemble**: `KoalaAI/Text-Moderation` (DeBERTa, the
OpenAI moderation taxonomy: sexual/hate/violence/harassment/self-harm/minors) +
`unitary/toxic-bert` (Jigsaw labels) by default; spam, phishing, and
suicidality specialists are one flag away (`use_specialist_text_models`). A
checkpoint you fine-tune with
[`safety/train/finetune_text.py`](../train/finetune_text.py) slots in via
`SAFETY_MM_TEXT_MODELS=runs/your-model`.

**Datasets** for specializing the models (schemas verified, presets in
`finetune_text.py`): [`nvidia/Aegis-AI-Content-Safety-2.0`](https://hf.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
(12+ categories), [`jigsaw-toxic-comments`](https://hf.co/datasets/anitamaxvim/jigsaw-toxic-comments),
[`chengxuphd/liar2`](https://hf.co/datasets/chengxuphd/liar2) (misinformation),
[`ai4privacy/pii-masking-400k`](https://hf.co/datasets/ai4privacy/pii-masking-400k).
The full dataset/model/deployment rationale is in
[`docs/SAFETY_BLUEPRINT.md`](../../docs/SAFETY_BLUEPRINT.md).

Why layered? Models score the *whole message* and catch hostility a wordlist
can't ("go back where you came from"); the lexicon/regex floor *localizes* curse
words and PII so they can be masked in place (`f***`, `b**@corp.io`) — including
`f@ck` / `sh1t` / `fuuuck` evasions — and keeps working fully offline.

## Actions

| `action` | Meaning |
|---|---|
| `none`  | clean — delivered untouched |
| `flag`  | delivered with a warning badge (spam, misinformation, unscannable media) |
| `mask`  | text delivered with curse words / PII masked (`f***`, `b**@corp.io`) |
| `blur`  | image/video delivered with pixels redacted (solid whole-frame blur) |
| `mute`  | video visuals clean but soundtrack explicit — delivered silent |
| `block` | withheld entirely (hate/violence/sexual/self-harm/criminal/cyber/extremism/child-safety, dangerous files) |

Default action per category lives in `taxonomy.DEFAULT_ACTIONS`. Self-harm
blocks **and** attaches a support-resources note. Blocked content is never
echoed back; the audit log (`/chat/audit`) records the decision, not the body.

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
