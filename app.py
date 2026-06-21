"""Entrypoint for hosted deployments (Hugging Face Spaces, Render, Railway, Fly,
or any plain container host).

It launches the WhatsApp-style moderated chat UI bound to ``0.0.0.0`` on the
port the host hands us (``$PORT``, falling back to 7860 — the Hugging Face
Spaces default).

Two toggles decide how smart the moderation is:

* ``SENTINEL_ENABLE_HF=1`` turns on the Hugging Face **text + image** models
  (toxic-bert, KoalaAI moderation, NSFW ViT). This is what catches implicit
  toxicity/hate the offline word list can't — e.g. "you are ugly". Needs the
  ``hf`` extra and a few GB of RAM (a free CPU Space is enough).
* ``SENTINEL_ENABLE_HF_AUDIO=1`` additionally turns on Whisper for voice
  notes. Off by default: CPU transcription is slow and not the point of the
  text/image demo.

With neither set, it runs the fully-offline floor (lexicon, PII regexes, file
gate, NudeNet) — fast, but lexicon-limited on text.

    python app.py            # local
    # container: CMD ["python", "app.py"]
"""
from __future__ import annotations

import os


def _on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


from safety.chat import create_chat_app
from safety.multimodal import MultimodalConfig

# Build config from SAFETY_MM_* env vars, then apply the deploy toggles.
config = MultimodalConfig.from_env()
enable_hf = _on("SENTINEL_ENABLE_HF")
# Text is the cheap, high-value model and is on whenever HF is enabled. The
# image ViT and Whisper are heavier, so each is opt-in — on a free CPU box the
# offline NudeNet covers images and audio falls back to "flag for review".
config.use_hf_text = enable_hf
config.use_hf_image = enable_hf and _on("SENTINEL_ENABLE_HF_IMAGE")
config.use_hf_audio = enable_hf and _on("SENTINEL_ENABLE_HF_AUDIO")

# `app` is what Gunicorn/Uvicorn import (`app:app`); running this file directly
# starts Uvicorn itself.
app = create_chat_app(config)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
