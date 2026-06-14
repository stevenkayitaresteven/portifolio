# Sentinel

**A content-moderation engine for chat apps.** It looks at everything a
conversation can carry — text, images, video, voice notes, documents, file
uploads — and decides, per message, whether to let it through, mask it, blur
it, mute it, or block it. Every decision comes back with a category, an action,
and a confidence score.

I wrapped it in a WhatsApp-style chat so you can actually *see* it work:
type a threat and it never sends; paste your SSN and it ships as `***-**-6789`;
drop in an explicit image and your friend receives it blurred; rename `virus.exe`
to `photo.jpg` and it still gets caught.

> 🟢 **Try it live (no install):** _deploy in one command — see
> [Live demo](#live-demo)._ &nbsp;|&nbsp; **Source:** you're reading it.

> This is a **defensive** tool. It detects and obscures harmful content; it does
> not generate any. No explicit material is shipped with the repo or needed to
> run it.

---

## Why I built it

I wanted to understand how platforms like WhatsApp or Discord keep their chats
safe, so I tried to build the thing myself instead of reading about it. The
naive version — "run text through a toxicity model" — falls apart the moment
you remember that people send images, voice notes, PDFs, and `.exe` files
renamed to look like photos. So the project grew into a single moderation
surface that handles *all* of those and folds them into one consistent verdict.

Two constraints shaped most of the design:

1. **It can't fail open.** If a model won't load or the network is down, the
   system still has to catch the obvious stuff. So every category has an
   always-on offline floor (lexicons, regexes, a file sniffer, an image
   detector) underneath the smarter Hugging Face models.
2. **"Safe" should mean *nothing was detected*, not *this is fine*.** Borderline
   image? Blur it rather than risk leaking it. Tune thresholds toward recall.

## What it catches

Everything — a Hugging Face model, a word list, a regex, a file's magic bytes —
normalizes into the same **12-category taxonomy**, and the most severe category
that fires wins:

| Category | Always-on offline floor | Hugging Face model(s) it pairs with |
|---|---|---|
| toxic / hate | leetspeak-aware profanity lexicon (masks in place) | `unitary/toxic-bert`, `facebook/roberta-hate-speech` |
| sexual / child-safety | NudeNet + a red-region "wound" heuristic | `KoalaAI/Text-Moderation`, `Falconsai/nsfw_image_detection` |
| violence / self-harm | phrase lists (self-harm adds a support note) | KoalaAI moderation, `sentinet/suicidality` |
| criminal / cyber / extremism | phrase lists + executable/EICAR file gate | KoalaAI moderation |
| spam | promo + phishing-URL signals | `mshenoda/roberta-spam`, `bert-finetuned-phishing` |
| privacy | PII regexes (email, card+Luhn, SSN, phone, IP, IBAN) | pairs with `iiiorg/piiranha` |
| misinformation | debunked-claim phrases (flagged, not blocked) | a LIAR2-fine-tuned head |

The text path runs an **ensemble** and merges every label into the taxonomy.
The models judge the whole message; the lexicon/regex floor *localizes* the bad
span so it can be masked in place — and it keeps working with no network at all.

## How it fits together

```
                       ┌── text  → lexicon · PII regex · phrases · spam/URL · HF ensemble
 message / upload  →   ├── image → NudeNet · ViT · wound · OCR of embedded text
 (one item)            ├── audio → Whisper transcript → text path
 routed by modality    ├── video → frame sampler → image path  + soundtrack
                       └── file  → magic bytes · extension · EICAR
                                    │
                                    ▼   risk score → policy → allow / flag / mask
                                                              / blur / mute / block  + audit
```

- **`safety/detectors/`** — pluggable, lazy-loaded detectors that report whether
  they're `available`, so the pipeline adapts to whatever is installed.
- **`safety/pipeline.py`** — per-category thresholds and the severity gate.
- **`safety/blur.py`** — whole-image blur or localized gaussian/pixelate/box.
- **`safety/multimodal/`** — one moderator per modality, all returning the same
  `ModerationResult`, behind one `MultimodalModerator` front door.
- **`safety/chat.py`** — the WhatsApp-style demo (one self-contained HTML file,
  no build step, no CDN).

A deeper write-up of the design — including the dataset/model choices for each
category — is in [`docs/SAFETY_BLUEPRINT.md`](docs/SAFETY_BLUEPRINT.md).

## Live demo

The repo ships ready to deploy. The hosted demo runs the **offline floor** only
(no GPU, no model downloads), so it boots on a free CPU box in a couple of
minutes and anyone with the link can use it.

**Hugging Face Spaces — one command:**

```bash
pip install huggingface_hub
export HF_TOKEN=hf_xxx                  # a write token from hf.co/settings/tokens
python deploy/deploy_to_hf.py           # → https://huggingface.co/spaces/<you>/sentinel-chat
```

**Or any container host** (Render, Railway, Fly, Cloud Run) — the root
`Dockerfile` + `app.py` bind to `$PORT` with zero config. Full instructions and
how to turn the Hugging Face models back on: [`deploy/README.md`](deploy/README.md).

```bash
docker build -t sentinel . && docker run -p 7860:7860 sentinel   # then open :7860
```

## Run it locally

```bash
git clone https://github.com/stevenkayitaresteven/portifolio && cd portifolio
python -m venv .venv && source .venv/bin/activate

pip install -e ".[detectors,serve]"     # offline engine + chat UI (fast, no torch)
python -m safety chat                    # → http://127.0.0.1:8000

# add the Hugging Face models when you want them (heavier):
pip install -e ".[detectors,hf,serve]"
python -m safety chat                    # full ensemble
python -m safety chat --no-hf            # force offline-only
```

A few things you can do beyond the chat:

```bash
python -m safety scan ./photos --json            # verdicts only, nothing written
python -m safety blur ./photos -o ./clean        # write blurred copies + a report
```

```python
from safety.multimodal import MultimodalModerator
mod = MultimodalModerator()

mod.moderate_text("what the f@ck").censored_text        # "what the f***"   (mask)
mod.moderate_text("i will kill you").action             # "block"           (violence)
mod.moderate_text("my ssn is 123-45-6789").censored_text  # masked          (privacy)

delivered, res = mod.moderate_file("photo.jpg")   # a blurred copy if explicit
delivered, res = mod.moderate_file("voice.ogg")   # None if the audio is explicit
```

It's also a mountable FastAPI service (`POST /safety/scan`, `POST /safety/redact`)
— see [`safety/service.py`](safety/service.py). Config knobs live in
[`SafetyConfig`](safety/config.py) and [`MultimodalConfig`](safety/multimodal/config.py),
all overridable from `SAFETY_*` env vars.

## The part that was actually hard

The fine-tuning suite ([`safety/train/finetune_text.py`](safety/train/finetune_text.py))
lets you train the 12-category classifier from 12 public datasets — nvidia
Aegis 2.0, Jigsaw, Civil Comments, HateXplain, ToxiGen, and more — and mix
several into one balanced corpus with `--mix`. Getting the label schemas to line
up across datasets that all disagree about what "toxic" means was fiddly. The
nastier problem was memory: mixing a few large corpora happily OOM-killed the
trainer, so I had to stream and cap per-source sampling instead of loading
everything into RAM. That fix is in the history, and it's the kind of bug you
only meet once you run the thing on real data.

```bash
python -m safety.train.finetune_text --preset aegis2 --out runs/aegis
python -m safety.train.finetune_text --mix civil_comments,davidson,toxigen,goemotions \
    --max-per-source 50000 --out runs/toxicity-v2
SAFETY_MM_TEXT_MODELS=runs/toxicity-v2 python -m safety chat   # drop your model in
```

## Skills

**Applied here:** Python (typed, packaged, `pyproject` + extras), computer
vision with OpenCV + ONNX, NLP / transformer fine-tuning with Hugging Face
(`transformers`, `datasets`, multi-label heads, per-class P/R/F1), audio ASR
(Whisper), FastAPI services and a from-scratch web UI, Docker, system design
(pluggable detectors, graceful degradation, one normalized result shape), and a
100+-test suite that runs fully offline.

**What I picked up along the way:** that real moderation is a *policy* problem as
much as an ML one — thresholds, per-category actions, and human-in-the-loop
matter more than squeezing another F1 point; how to make a system degrade
gracefully instead of crashing when a dependency is missing; the discipline of
never logging the harmful content itself, only the decision; and how much
careful data work hides behind a single "toxic / not toxic" label.

## What I'd build next

- **Real-time streaming** so a video is moderated frame-by-frame as it plays
  rather than re-encoded up front.
- **Multilingual coverage** — the lexicon and toxic-bert are English-first;
  swapping in `mdeberta` and the textdetox multilingual data is the obvious next
  step (the training code already accepts it).
- **A reviewer dashboard** over the audit log so a human can sample borderline
  decisions and feed corrections back into the next fine-tune (active learning).
- **Perceptual-hash matching** against known-bad media (an industry standard I
  modelled the file gate after but haven't implemented).
- **Calibrated confidence + per-tenant policy**, so different communities can
  set their own thresholds instead of sharing one global default.

## Honest limitations

- The wound/gore heuristic is a baseline — it keys on big saturated-red regions
  and can be fooled by a red shirt or miss dried blood. Train the classifier for
  anything production-grade.
- Audio/video moderation is **transcript-based**: it catches what's *said*, not
  non-speech sounds.
- No detector is perfect. Keep a human in the loop for the borderline cases.

## Tests

```bash
pip install -e ".[detectors,serve,test]" && pytest -q   # 100+ tests, all offline
```

The whole suite runs without a network — the Hugging Face paths are exercised
through their offline fallbacks, and UI/service tests skip cleanly when FastAPI
isn't installed.

---

*Built by **Steven Kayitare**.* Image-pipeline details are in
[`safety/README.md`](safety/README.md); the multimodal layer in
[`safety/multimodal/README.md`](safety/multimodal/README.md).
