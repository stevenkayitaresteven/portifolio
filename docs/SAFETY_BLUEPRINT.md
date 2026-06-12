# Sentinel Safety & Moderation — Engineering Blueprint

A production blueprint for detecting, risk-scoring, and safely handling harmful
content across every surface of a chat product: 1:1, group, channels, DMs,
uploads (image / audio / video / document / code), URLs, profiles, usernames,
and generated AI responses.

This document is the *design*. The repository already ships a **working
reference implementation** of Layers 3–9 and 11 — see
[`safety/multimodal/`](../safety/multimodal/) and
[`safety/train/finetune_text.py`](../safety/train/finetune_text.py). Where code
exists, it is linked inline.

---

## 0. Taxonomy

All detectors normalize to **12 canonical categories**
([`taxonomy.py`](../safety/multimodal/taxonomy.py)). The 30 harm types in the
brief collapse onto them:

| Category | Absorbs |
|---|---|
| `toxic` | toxicity, profanity, harassment, bullying, abusive language |
| `hate` | hate speech, protected-group attacks |
| `sexual` | explicit sexual content, NSFW, sexual solicitation |
| `violence` | violence, threats |
| `self_harm` | self-harm, suicide references |
| `criminal` | illegal activity, fraud, scams |
| `cybersecurity` | malware distribution, social engineering, harmful uploads, cyber threats |
| `spam` | spam, phishing |
| `privacy` | PII exposure, doxxing |
| `extremism` | extremist / terrorist propaganda, manipulation/radicalization |
| `misinformation` | false/manipulated claims |
| `child_safety` | child exploitation / CSAM risk |

Impersonation and unsafe-AI-response are handled as **policy contexts** (see
§9, §14) rather than standalone categories, because they reuse the same
detectors on a different surface.

---

## LAYER 1 — Data collection

Recommended public datasets (verified on the Hub). License column matters:
`*-NC` datasets are fine for training a model you self-host but read the terms
before commercial redistribution of weights.

| Need | Dataset | License | Size | Label schema | Strengths | Weaknesses | Use |
|---|---|---|---|---|---|---|---|
| Multi-label moderation | [`nvidia/Aegis-AI-Content-Safety-2.0`](https://hf.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | CC-BY-4.0 | 33.4K | `violated_categories` (13 incl. criminal, fraud, malware, suicide, PII) | LLM-interaction context, broad coverage, safe/unsafe + categories | English-only, prompt/response not chat | **Primary** multi-label backbone (`--preset aegis2`) |
| Toxicity / hate / threat | [`jigsaw-toxic-comments`](https://hf.co/datasets/anitamaxvim/jigsaw-toxic-comments) | CC0/MIT | ~160K | 6 binary: toxic, severe, obscene, threat, insult, identity_hate | Large, well-studied, multi-label | Comment-style, some label noise | Toxicity/hate/violence head (`--preset jigsaw`) |
| LLM-prompt toxicity | [`lmsys/toxic-chat`](https://hf.co/datasets/lmsys/toxic-chat) | CC-BY-NC-4.0 | 10K | toxicity + jailbreak | Real user→LLM prompts, jailbreak labels | Small, NC | Calibration / jailbreak eval |
| Hate speech | [`ucberkeley-dlab/measuring-hate-speech`] | CC-BY-4.0 | 135K | continuous hate score + 10 target groups | Graded, demographic targets | Annotation subjectivity | Hate head fine-tune |
| Spam | [`mshenoda/spam-messages`](https://hf.co/datasets/mshenoda/spam-messages) | MIT | 17K | spam/ham | Clean, SMS-style | Domain-narrow | Spam head |
| Phishing | [`ealvaradob/phishing-dataset`](https://hf.co/datasets/ealvaradob/phishing-dataset) | Apache-2.0 | ~80K | phishing/benign (URL+text+email) | Multi-source | Mixed input types | Phishing/URL head |
| PII | [`ai4privacy/pii-masking-400k`](https://hf.co/datasets/ai4privacy/pii-masking-400k) | (custom) | 400K | token spans, 19 PII classes, 6 langs | Huge, multilingual, span-level | NER not classification | PII token model |
| Misinformation | [`chengxuphd/liar2`](https://hf.co/datasets/chengxuphd/liar2) | Apache-2.0 | 23K | 6-way truthfulness | Fact-checker labeled | Politics-skewed | Misinfo head (`--preset liar2`) |
| NSFW image | corpora behind [`Falconsai/nsfw_image_detection`](https://hf.co/Falconsai/nsfw_image_detection) | — | — | nsfw/normal | Battle-tested | Binary only | Image head (already wired) |
| Multilingual toxicity | [`textdetox/multilingual_toxicity_dataset`] | OpenRAIL | 9 langs | toxic/neutral | Cross-lingual | Per-lang size varies | XLM-R fine-tune (§12) |

> **Child safety:** never collect, store, or train on CSAM. Use *hash-matching*
> against known-illegal-content databases (PhotoDNA / NCMEC, Apple/Google CSAI
> APIs) and the `child_safety` *behavioral* signals from Aegis (grooming
> language). Report per legal obligation. This blueprint treats `child_safety`
> as block-on-detection + escalate-to-human + mandated reporting, **not** an
> ML training target on raw media.

---

## LAYER 2 — Model selection

| Model | Params | ~Mem (fp16) | Fine-tune fit | Multiling. | Speed (CPU) | Best for |
|---|---:|---:|---|---|---|---|
| DistilBERT | 66M | 130MB | ✅ easy | ✗ | fastest | spam, latency-critical heads |
| **DistilRoBERTa** | 82M | 160MB | ✅ easy | ✗ | fast | **default multi-label head** (this repo) |
| BERT-base | 110M | 220MB | ✅ | ✗ | med | toxicity baselines |
| RoBERTa-base | 125M | 250MB | ✅ | ✗ | med | hate (facebook dynabench) |
| **DeBERTa-v3-base** | 184M | 370MB | ✅ best acc | ✗ | med-slow | **highest-accuracy moderation** (KoalaAI) |
| ModernBERT-base | 150M | 300MB | ✅ | ✗ | fast (8k ctx) | long messages, code |
| mDeBERTa-v3 | 280M | 560MB | ✅ | ✅ | slow | multilingual moderation |
| XLM-R-base | 270M | 540MB | ✅ | ✅ (100 langs) | slow | multilingual (§12) |
| Piiranha (mDeBERTa) | 280M | 560MB | token-cls | ✅ 6 langs | slow | PII spans |
| Llama-3.1-8B / Mistral-7B / Gemma-2-9B / Qwen2.5-7B (Guard variants) | 7–9B | 16GB+ | LoRA/QLoRA | ✅ | GPU-only | nuanced/zero-shot policy, escalation tier |

**Recommended assignment**

| Task | Model | Rationale |
|---|---|---|
| Multi-label moderation (primary) | DeBERTa-v3 or DistilRoBERTa fine-tuned on Aegis2+Jigsaw | accuracy vs latency; ship both, route by load |
| Toxicity / hate / violence | `unitary/toxic-bert` + `facebook/roberta-hate-speech-dynabench-r4` | strong off-the-shelf |
| Spam / phishing | `mshenoda/roberta-spam`, `ealvaradob/bert-finetuned-phishing` | specialist, cheap |
| NSFW image | `Falconsai/nsfw_image_detection` (ViT); CLIP/SigLIP zero-shot for symbols/gore | wired in repo |
| PII | `iiiorg/piiranha-v1` (token) + regex floor | spans + offline floor |
| Self-harm | `sentinet/suicidality` | dedicated, high recall |
| Escalation / ambiguous | 7–8B Guard model (LoRA) | reserve for low-confidence band |

This repo runs an **ensemble** (KoalaAI moderation + toxic-bert by default;
spam/phishing/suicidality opt-in) and normalizes every label through the
taxonomy — see [`text.py`](../safety/multimodal/text.py).

---

## LAYER 3 — Multi-model moderation pipeline

```
                          ┌─────────────────────────────────────────────┐
  message / upload  ──►   │  ROUTER  (modality_for: text/image/audio/    │
  (one item)              │          video/document/file)                │
                          └───────┬───────────┬──────────┬───────────────┘
                       text │  image │  audio │  video │  file/url
                            ▼        ▼        ▼        ▼        ▼
            ┌──────────┐ ┌────────┐ ┌──────┐ ┌──────────┐ ┌──────────┐
            │ heuristic│ │ NudeNet│ │whisper│ │ sampler  │ │magic+ext │
            │ +ensemble│ │ +ViT   │ │ →text │ │ →img+aud │ │ +EICAR   │
            │ (12-cat) │ │ +OCR   │ │ chain │ │ aggregate│ │ →cyber   │
            └────┬─────┘ └───┬────┘ └──┬───┘ └────┬─────┘ └────┬─────┘
                 └───────────┴─────────┴──────────┴────────────┘
                                     ▼
                        ┌────────────────────────┐
                        │  RISK SCORING           │  per-category max score,
                        │  ModerationResult       │  confidence, strongest action
                        └───────────┬────────────┘
                                    ▼
                        ┌────────────────────────┐
                        │  POLICY ENGINE          │  category→action map,
                        │  (action_for, overrides)│  thresholds, escalation band
                        └───────────┬────────────┘
                                    ▼
                ┌──────────────────────────────────────────┐
                │ RESPONSE ENGINE: allow / mask / blur /    │
                │ mute / flag / block  + audit + support    │
                └──────────────────────────────────────────┘
```

- **Confidence** = top category score when flagged, `1 − max(score)` when clean
  ([`ModerationResult.confidence`](../safety/multimodal/result.py)).
- **Decision thresholds**: `text_threshold` (model label floor, default 0.50);
  heuristic phrases carry per-phrase scores; spam/URL gates at 0.50.
- **Escalation logic**: scores in a configurable *uncertainty band*
  (e.g. 0.40–0.60) should route to the heavy Guard model or a human queue
  rather than auto-acting. (Band routing is a documented extension point;
  the policy hook is `MultimodalConfig.action_for`.)

---

## LAYER 4 — Text moderation

Implemented in [`text.py`](../safety/multimodal/text.py) +
[`heuristics.py`](../safety/multimodal/heuristics.py) +
[`wordlist.py`](../safety/multimodal/wordlist.py). Strategy:

- **Multi-label** classification (a message can be toxic *and* a threat).
- **Hierarchical** handling via action severity: `none < flag < mask < blur/mute < block`; the most severe firing category wins.
- **Confidence calibration**: thresholds per source; lexicon/regex hits are
  treated as high-precision (they *localize* and mask in place), models as
  recall-boosters. For production, temperature-scale model logits on a held-out
  calibration set and store per-category thresholds.

Layers, strongest-action-wins:
1. profanity lexicon → `toxic`, masked in place (`f***`, leetspeak-aware)
2. PII regex (email/card-Luhn/SSN/phone/IP/IBAN) → `privacy`, masked in place
3. phrase lexicons → self_harm / violence / criminal / cybersecurity / extremism / misinformation
4. spam + phishing-URL signals → `spam`
5. HF ensemble (KoalaAI + toxic-bert; +spam/phishing/suicide opt-in) → all 12

Training architecture: [`finetune_text.py`](../safety/train/finetune_text.py) —
multi-label head over the 12 categories, dataset presets with verified column
mappings, local metrics (no sklearn).

---

## LAYER 5 — File analysis

Implemented: [`filecheck.py`](../safety/multimodal/filecheck.py) (gate) +
`extract_document_text` in [`pipeline.py`](../safety/multimodal/pipeline.py).

| Format | Handling |
|---|---|
| TXT/MD/CSV/JSON/LOG | read → text moderation |
| PDF | `pypdf` extract → text moderation (optional dep) |
| DOCX | stdlib zip+XML extract → text moderation |
| PNG/JPEG | image moderation + OCR of embedded text |
| MP4/MOV/… | video pipeline (Layer 8) |
| WAV/MP3/… | audio pipeline (Layer 7) |
| EXE/DLL/ELF/Mach-O/scripts | **blocked** by extension *and* magic bytes |
| ZIP/RAR/7z/ISO | **blocked** — opaque containers |
| EICAR signature | **blocked** — AV pipeline check |

Production add-ons (documented hooks): **OCR** via `pytesseract` (wired for
images), **malware scanning** via a ClamAV/`clamd` sidecar before relay,
**metadata/EXIF** stripping on delivered media, **embedded-content** (macros in
DOCX, JS in PDF) flagged as `cybersecurity`.

---

## LAYER 6 — Image safety

Implemented: [`detectors/hf_nsfw.py`](../safety/detectors/hf_nsfw.py) +
NudeNet + wound heuristic, ensembled in
[`image.py`](../safety/multimodal/image.py). Plus **OCR** so harmful *text* in
memes/screenshots is judged by the text moderator.

- **Nudity/sexual** — NudeNet (18 body-part boxes) + Falconsai ViT (whole-frame).
- **Violence/gore** — HSV wound heuristic now; for production, **CLIP/SigLIP
  zero-shot** against prompts (`"graphic violence"`, `"weapon"`,
  `"extremist flag/symbol"`) gives open-set coverage without new training, and
  a fine-tuned ViT for precision.
- **Extremist symbols** — SigLIP zero-shot prompt set + a curated symbol
  classifier.
- Flagged → whole-image **solid blur**; original never relayed.

---

## LAYER 7 — Audio safety

Implemented: [`audio.py`](../safety/multimodal/audio.py).
`audio → Whisper ASR → transcript → text moderation`. Default
`openai/whisper-base`; use `whisper-large-v3` for accuracy or a distilled
variant (`distil-whisper`) for throughput (`SAFETY_MM_ASR_MODEL`). Explicit
audio is **blocked**; the censored transcript is surfaced so context survives.
Non-speech audio is out of scope (documented limitation).

---

## LAYER 8 — Video safety

Implemented: [`video.py`](../safety/multimodal/video.py).
`video → ~1fps frame sampling → image moderation` + `soundtrack → audio chain`,
then **risk aggregation**: any explicit frame → re-encode fully blurred; clean
visuals + explicit audio → delivered muted. Frame density and sample cap are
configurable.

---

## LAYER 9 — Policy engine

Implemented: [`taxonomy.py`](../safety/multimodal/taxonomy.py) `DEFAULT_ACTIONS`
+ `MultimodalConfig.action_for` + `action_overrides`.

Actions: **ALLOW** (`none`) · **WARN/FLAG** (`flag`) · **MASK** (`mask`) ·
**REDACT** (`blur`/`mute`) · **BLOCK** (`block`) · **REVIEW** (escalation band →
human queue).

- **Threshold tuning** — `text_threshold`, per-phrase scores, gate thresholds.
- **Admin controls** — `action_overrides="spam:block,sexual:mask"` (env or DB).
- **Audit logs** — [`/chat/audit`](../safety/chat.py): every decision recorded
  as metadata (category, action, confidence, detectors) — **never the content**,
  so logs don't re-amplify harm.
- **User appeals** — schema in §DB below (`appeals` table); a blocked message
  keeps its decision id so a user can contest it without exposing the body.

---

## LAYER 10 — Training strategy

`finetune_text.py` covers preprocessing → multi-label head → eval. Recipe:

- **Cleaning** — dedupe, language-filter, strip PII from training text (train on
  *masked* PII so the model learns shape not values), drop near-duplicates.
- **Class balancing** — categories are skewed; use class-weighted BCE or
  focal loss, and oversample rare categories (child_safety, extremism).
- **Augmentation** — leetspeak/character-swap (mirrors the lexicon evasions),
  back-translation for multilingual, synonym swap. Keep an un-augmented eval set.
- **Fine-tuning tiers**
  - *Full FT* — DistilRoBERTa/DeBERTa heads (this repo's default path).
  - *LoRA* — adapt a 7–8B Guard model cheaply; rank 16–32, α 32, dropout 0.05,
    target `q_proj,v_proj`.
  - *QLoRA* — 4-bit NF4 base for single-GPU 7B training.
- **Hyperparameters (encoder multi-label)**: lr `2e-5`, batch 16–32, epochs 3–4,
  warmup 6%, weight-decay 0.01, max-len 256 (512 for documents), fp16,
  `problem_type="multi_label_classification"`, threshold tuned per category on
  validation PR curves.

---

## LAYER 11 — Evaluation

`multilabel_metrics()` reports per-category **precision/recall/F1**, confusion
counts (tp/fp/fn), and **micro/macro-F1**. Extend with **ROC-AUC / PR-AUC** per
category (store probabilities, sweep thresholds). Moderation should optimize
**recall** for block-tier categories (missing a threat is worse than a false
positive) and **precision** for flag-tier (don't cry wolf on spam/misinfo).
`--eval-only` scores any checkpoint against a labeled set without training.

---

## LAYER 12 — Multilingual support

Target: en, ko, fr, ar, es, zh, sw.

- **Models** — `mDeBERTa-v3-base` / `XLM-R` for text; `Piiranha` already covers
  6 langs for PII; Whisper is multilingual for audio.
- **Datasets** — `textdetox/multilingual_toxicity_dataset`, `jigsaw-multilingual`,
  per-language sets (Korean: `apeach`/`kmhas`; Arabic: `OSACT`; Spanish:
  `hateval`; Chinese: `COLD`; Swahili: AfriHate / `swahili-safety`).
- The lexicon/regex floor is English-centric — for other languages, lean on the
  multilingual model and a localized wordlist (`extra_words`).

---

## LAYER 13 — Real-time deployment

```
                 ┌─────────┐     ┌──────────────────────┐
  clients ─────► │  API GW │ ──► │ FastAPI moderation    │  (stateless, N replicas)
  (chat, upload) │ (auth,  │     │ svc  /moderate /chat  │
                 │  rate)  │     └────┬─────────────┬────┘
                 └─────────┘          │             │
                          sync (text, fast)   async (media, slow)
                              │                     │
                       ┌──────▼─────┐        ┌──────▼──────┐
                       │ Redis cache│        │ Redis queue │──► GPU workers
                       │ (hash→     │        │ (Celery/RQ) │    (image/audio/video,
                       │  verdict)  │        └──────┬──────┘     ONNX/TensorRT)
                       └────────────┘               │
                       ┌────────────────────────────▼─────────────┐
                       │ PostgreSQL: decisions, appeals, policies   │
                       └────────────────────────────────────────────┘
```

- **Microservices** — `moderation-api` (sync text/URL), `media-worker` (GPU,
  image/audio/video), `policy-svc` (config + audit), `csam-matcher` (isolated,
  hash-only). Models served via **ONNX Runtime** (CPU) / **TensorRT** (GPU);
  export with the existing `safety/train/export.py` pattern.
- **Queue** — text is synchronous (<50ms target); media goes async on Redis +
  Celery so a 30s video doesn't block the socket.
- **Caching** — Redis keyed by content hash → verdict (identical forwards/memes
  are scored once). TTL + versioned by model hash so a model update invalidates.
- **Monitoring** — Prometheus (latency p50/p95, queue depth, per-category fire
  rate, model-unavailable counter) + Grafana; alert on detector-down and on
  category-rate anomalies (spam wave / coordinated attack).
- **Containers** — Docker per service; Kubernetes with HPA on queue depth and
  GPU node pool for media workers.

---

## LAYER 14 — Safe response generation

Implemented behaviors: blocked content is **not echoed** (text body dropped,
audit stores metadata only); self-harm carries a **support resource** note;
PII is **masked, not repeated**. Principles:

- Don't amplify — never quote the harmful span back; show category + reason.
- Don't repeat — logs and admin views get metadata, not bodies.
- Safe alternatives / de-escalation — for toxic (not blocked) messages, a
  rewrite suggestion ("this may read as hostile") is a documented hook on the
  flag path.
- Protect privacy — redaction is the default for PII even when delivering.
- For **generated AI responses**, run the *same* moderator on model output
  before it reaches the user (output guardrail), blocking unsafe completions.

---

## Output: folder structure

```
safety/
├── config.py, pipeline.py, blur.py, types.py     # image core (existing)
├── detectors/         nudenet · wound · classifier · hf_nsfw
├── multimodal/
│   ├── taxonomy.py        12 categories + label maps + actions
│   ├── result.py          ModerationResult
│   ├── config.py          MultimodalConfig (SAFETY_MM_*)
│   ├── wordlist.py        profanity lexicon (mask in place)
│   ├── heuristics.py      PII regex · phrase lexicons · spam · URL risk
│   ├── filecheck.py       executable/archive/EICAR gate
│   ├── text.py image.py audio.py video.py
│   └── pipeline.py        MultimodalModerator + doc extraction + dispatch
├── train/
│   ├── finetune_text.py   multi-label fine-tune + eval (presets)
│   └── train.py export.py model.py dataset.py   # image classifier (existing)
└── chat.py                moderated chat app + /chat/audit
docs/SAFETY_BLUEPRINT.md    (this file)
tests/                      4 suites, 115+ offline tests
```

## Output: database schema (PostgreSQL)

```sql
CREATE TABLE decisions (
  id            UUID PRIMARY KEY,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  surface       TEXT,            -- dm | group | channel | upload | ai_output
  modality      TEXT NOT NULL,   -- text | image | audio | video | file
  flagged       BOOLEAN NOT NULL,
  action        TEXT NOT NULL,   -- none|flag|mask|blur|mute|block
  categories    TEXT[] NOT NULL,
  confidence    REAL,
  detectors     TEXT[],
  content_hash  TEXT,            -- for cache + dedupe; NEVER the content
  actor_id      UUID,            -- sender (FK users)
  reasons       JSONB
);
CREATE INDEX ON decisions (created_at);
CREATE INDEX ON decisions USING GIN (categories);

CREATE TABLE policies (         -- admin-tunable, hot-reloaded
  category   TEXT PRIMARY KEY,
  action     TEXT NOT NULL,
  threshold  REAL NOT NULL DEFAULT 0.5,
  updated_by UUID, updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE appeals (
  id          UUID PRIMARY KEY,
  decision_id UUID REFERENCES decisions(id),
  user_id     UUID, reason TEXT,
  status      TEXT DEFAULT 'open',   -- open|upheld|overturned
  created_at  TIMESTAMPTZ DEFAULT now(), resolved_at TIMESTAMPTZ
);
```

## Output: API design

```
POST /moderate            {text?, url?}                 → ModerationResult JSON
POST /moderate/file       multipart (one file)          → ModerationResult (+ media id)
GET  /moderate/health                                    → loaded detectors/models
POST /chat/send           text + ≤1 files               → moderated message(s)
GET  /chat/messages                                      → conversation
GET  /chat/media/{id}                                    → delivered (redacted) media
GET  /chat/audit?limit=                                  → decisions (metadata only)
GET  /policies / PUT /policies/{category}   (admin)      → tune action/threshold
POST /appeals             {decision_id, reason}          → appeal
```

`ModerationResult` JSON: `modality, flagged, action, categories[], scores{},
confidence, reasons[], detectors[], scanned, censored_text?, transcript?`.

## Monitoring, security, scalability, future

- **Monitoring** — per-category fire-rate dashboards, latency SLOs, detector-up
  health, drift alarms (category distribution shift), human-review queue depth.
- **Security** — models are untrusted-input parsers: sandbox media decoders,
  cap upload size, time-box ffmpeg/whisper, isolate the CSAM matcher, never log
  raw harmful content, sign model artifacts, pin model hashes.
- **Scalability** — stateless API + Redis cache + async media workers; cache hit
  on forwarded/duplicate content is the biggest win; shard GPU workers by
  modality; ONNX/TensorRT for 3–5× throughput.
- **Future** — multimodal LMs (judge image+caption jointly), graph signals for
  coordinated spam/extremism rings, on-device pre-filter for E2E-encrypted
  chats, active-learning loop feeding `/appeals` overturns back into training,
  per-community policy profiles.
```
