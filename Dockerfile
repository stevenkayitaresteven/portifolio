# Container for the public Sentinel chat demo, with the Hugging Face text +
# image models baked in so it moderates as well as a local full install
# (toxic-bert + KoalaAI moderation catch implicit toxicity/hate the offline
# word list misses). Sized for a free CPU box: CPU-only torch, models cached
# into the image at build time.
FROM python:3.11-slim

# opencv-python-headless needs libGL/libglib at runtime even on headless boxes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, from the CPU wheel index — keeps the image small and
# avoids downloading multi-GB CUDA libraries that a CPU Space can't use.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY . .

# detectors = offline NudeNet, hf = transformers (text/image models),
# serve = FastAPI/uvicorn for the chat UI.
RUN pip install --no-cache-dir -e ".[detectors,hf,serve]"

# Turn the Hugging Face text + image models on (Whisper audio stays opt-in).
ENV SENTINEL_ENABLE_HF=1
# Cache HF weights inside the image so the first message doesn't wait on a
# download. Must be writable at runtime — Spaces runs as a non-root user.
ENV HF_HOME=/app/.hf_cache
RUN mkdir -p /app/.hf_cache && chmod -R 777 /app/.hf_cache && \
    python -c "from transformers import pipeline; \
[pipeline('text-classification', model=m, top_k=None, truncation=True) \
 for m in ('KoalaAI/Text-Moderation', 'unitary/toxic-bert')]; \
pipeline('image-classification', model='Falconsai/nsfw_image_detection')" && \
    chmod -R 777 /app/.hf_cache

# Hugging Face Spaces routes to 7860; other hosts inject $PORT (handled in app.py).
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
