"""Entrypoint for hosted deployments (Hugging Face Spaces, Render, Railway, Fly,
or any plain container host).

It launches the WhatsApp-style moderated chat UI bound to ``0.0.0.0`` on the
port the host hands us (``$PORT``, falling back to 7860 — the Hugging Face
Spaces default). By default it runs the fully-offline moderation floor
(profanity lexicon, PII regexes, file gate, NudeNet) so the demo boots in
seconds on a free CPU box. Set ``SENTINEL_ENABLE_HF=1`` to additionally pull
the Hugging Face models when the host has the RAM and the ``hf`` extra.

    python app.py            # local
    # container: CMD ["python", "app.py"]
"""
from __future__ import annotations

import os

from safety.chat import create_chat_app
from safety.multimodal import MultimodalConfig

# Build config from SAFETY_MM_* env vars, then make it safe-by-default for a
# public box: keep the heavy transformer downloads opt-in.
config = MultimodalConfig.from_env()
if os.environ.get("SENTINEL_ENABLE_HF", "").lower() not in ("1", "true", "yes"):
    config.use_hf_text = config.use_hf_image = config.use_hf_audio = False

# `app` is what Gunicorn/Uvicorn import (`app:app`); running this file directly
# starts Uvicorn itself.
app = create_chat_app(config)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
