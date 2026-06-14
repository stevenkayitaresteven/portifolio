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
The Space runs the offline moderation floor, so it boots on a free CPU box in
a couple of minutes.

### Any container host (Render / Railway / Fly / Cloud Run)

The repo root has a `Dockerfile` and `app.py` that bind to `$PORT`. On
[Render](https://render.com), for example: New → Web Service → point at this
repo → Docker → deploy. No config needed.

```bash
# or just run the container locally:
docker build -t sentinel . && docker run -p 7860:7860 sentinel
# open http://localhost:7860
```

### Turn on the Hugging Face models

The hosted demo is offline-only by default for speed. To enable the
transformer ensemble (toxic-bert, NSFW ViT, Whisper), build with the `hf`
extra and set `SENTINEL_ENABLE_HF=1`. This needs more RAM than a free tier
usually gives — use a paid instance or run it locally.
