# Get a public link

Everything here exists so you (or a recruiter) can open Sentinel in a browser
without installing anything. Pick one:

### Hugging Face Spaces — recommended, free, one command

```bash
pip install huggingface_hub
export HF_TOKEN=hf_xxx          # write token: https://huggingface.co/settings/tokens
python deploy/deploy_to_hf.py   # -> https://huggingface.co/spaces/<you>/sentinel-chat
```

That creates a Docker Space, uploads the repo, and gives back a public URL.
The Space's `Dockerfile` installs CPU-only torch + the Hugging Face text and
image models and bakes them into the image, so the demo moderates as well as a
local full install (it catches implicit toxicity/hate the word list misses).
First build takes a few minutes; after that, messages are scanned instantly.

### Any container host (Render / Railway / Fly / Cloud Run)

The repo root has a `Dockerfile` and `app.py` that bind to `$PORT`. On
[Render](https://render.com), for example: New → Web Service → point at this
repo → Docker → deploy. No config needed.

```bash
# or just run the container locally:
docker build -t sentinel . && docker run -p 7860:7860 sentinel
# open http://localhost:7860
```

### Toggles

The container sets `SENTINEL_ENABLE_HF=1`, which turns on the Hugging Face
**text + image** models. To also moderate voice notes with Whisper, set
`SENTINEL_ENABLE_HF_AUDIO=1` (off by default — CPU transcription is slow). To
run a purely offline, ultra-light box instead, build without the `hf` extra and
unset `SENTINEL_ENABLE_HF`.
